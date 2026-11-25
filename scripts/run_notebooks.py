"""Run the notebooks one after another without supervision (e.g. overnight).

    python scripts/run_notebooks.py                      # 01 → 02 → 03 → 04
    python scripts/run_notebooks.py 03 04                # only these
    python scripts/run_notebooks.py --from 02            # 02 and everything after it
    python scripts/run_notebooks.py --set QUICK=True --set N_JOBS=4

Each notebook is executed in place (or saved to --out DIR), so its outputs (tables, plots) are kept as if run in Jupyter. The file is saved
after every cell: open it during the run to watch progress. On an error the run stops (unless --keep-going), the
notebook is saved up to the failing cell and the exit code is 1.

--set adds a cell right after the notebook's "parameters" cell, like papermill. The cell stays in the saved notebook,
so it is visible which settings were overridden. It is applied to every selected notebook; names a notebook does not
use are harmless. Needs requirements-notebooks.txt (nbclient, ipykernel).

Console noise from the worker processes is filtered (--raw-output turns this off): joblib's harmless clean-up
tracebacks on Windows ("KeyError: ...joblib_memmapping_folder...") and leak notices are dropped, repeated warnings
(e.g. ConvergenceWarning) are shown once and counted. Everything inside the notebooks is unaffected.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import glob
import os
import re
import sys
import threading
import time
from collections import Counter

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")
OVERRIDE_TAG = "injected-parameters"


class KernelOutputFilter:
    """Reads the kernel's stderr (and that of its worker processes) from a pipe and passes on what matters.

    Dropped: tracebacks of loky's resource tracker (a known, harmless joblib clean-up race on Windows), its "leaked
    ... objects" notices and the IPKernelApp transport warning. Warnings: the first occurrence of each message is
    shown, repeats are counted and summarised by summary().
    """

    WARNING = re.compile(r"^\S.*?:\d+: (\w*Warning): (.*)$")

    def __init__(self, out=sys.stderr):
        """Create the pipe (pass write_fd to the kernel as stderr) and start the reader thread."""
        self.out = out
        self.read_fd, self.write_fd = os.pipe()
        self.dropped, self.warnings = 0, Counter()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        """Reader thread: line by line, collect tracebacks, drop clean-up noise, de-duplicate warnings."""
        block, skip_next_warn_line = [], False
        with os.fdopen(self.read_fd, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                text = line.rstrip("\n")
                if block:                                   # inside a traceback: collect until the exception line
                    block.append(text)
                    if text and not text.startswith((" ", "\t")):
                        self._flush_block(block)
                        block = []
                    continue
                if text.startswith("Traceback (most recent call last):"):
                    block = [text]
                    continue
                if skip_next_warn_line and text.strip().startswith("warnings.warn("):
                    skip_next_warn_line = False
                    continue
                skip_next_warn_line = False
                if "resource_tracker: There appear to be" in text or "Kernel is running over TCP" in text:
                    self.dropped += 1
                    skip_next_warn_line = True
                    continue
                m = self.WARNING.match(text)
                if m:
                    key = f"{m.group(1)}: {m.group(2)}"
                    self.warnings[key] += 1
                    skip_next_warn_line = True
                    if self.warnings[key] > 1:
                        continue
                self.out.write(line)
                self.out.flush()
        if block:
            self._flush_block(block)

    def _flush_block(self, block):
        """Print a collected traceback unless it is the resource tracker's clean-up race."""
        text = "\n".join(block)
        if "resource_tracker" in text and ("KeyError" in text or "FileNotFoundError" in text):
            self.dropped += 1
            return
        self.out.write(text + "\n")
        self.out.flush()

    def close(self, wait: float = 5.0) -> str:
        """Close our end of the pipe, wait for the workers' output to drain, return a one-line summary."""
        os.close(self.write_fd)
        self._thread.join(timeout=wait)
        parts = [f"{n}× {w[:90]}" for w, n in self.warnings.items() if n > 1]
        if self.dropped:
            parts.append(f"{self.dropped} harmless joblib clean-up messages hidden")
        return ("  (console: " + "; ".join(parts) + ")") if parts else ""


def select(names: list[str], start: str | None) -> list[str]:
    """Notebook paths in order, filtered by name prefixes and/or a start notebook."""
    paths = sorted(glob.glob(os.path.join(NOTEBOOKS, "[0-9][0-9]_*.ipynb")))
    if names:
        paths = [p for p in paths if any(os.path.basename(p).startswith(n) for n in names)]
    if start:
        paths = [p for p in paths if os.path.basename(p) >= start]
    return paths


def inject(nb, overrides: list[str]) -> None:
    """Replace a cell injected by an earlier run, then add the current overrides after the parameters cell."""
    nb.cells = [c for c in nb.cells if OVERRIDE_TAG not in c.metadata.get("tags", [])]
    if not overrides:
        return
    i = next((i for i, c in enumerate(nb.cells) if "parameters" in c.metadata.get("tags", [])), None)
    if i is None:
        raise ValueError("no cell tagged 'parameters'")
    cell = nbformat.v4.new_code_cell("# set by scripts/run_notebooks.py\n" + "\n".join(overrides))
    cell.metadata["tags"] = [OVERRIDE_TAG]
    nb.cells.insert(i + 1, cell)


def fmt(seconds: float) -> str:
    """Seconds as H:MM:SS."""
    return str(dt.timedelta(seconds=round(seconds)))


def run(path: str, overrides: list[str], timeout: int | None, kernel: str, out_dir: str | None = None,
        raw_output: bool = False) -> bool:
    """Execute one notebook; True if all cells ran. Saves after every cell and on failure."""
    name = os.path.basename(path)
    target = os.path.join(out_dir, name) if out_dir else path
    nb = nbformat.read(path, as_version=4)
    inject(nb, overrides)
    code = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    t0 = time.perf_counter()
    last = [t0]

    def on_cell_executed(cell, cell_index, **_):
        """Progress line and save after each cell."""
        now = time.perf_counter()
        n = code.index(cell_index) + 1 if cell_index in code else "?"
        print(f"  {name}: cell {n}/{len(code)} done in {fmt(now - last[0])} (total {fmt(now - t0)})", flush=True)
        last[0] = now
        nbformat.write(nb, target)  # progress is visible in the file during the run

    print(f"\n[{dt.datetime.now():%H:%M:%S}] {name}", flush=True)
    client = NotebookClient(nb, timeout=timeout, kernel_name=kernel, resources={"metadata": {"path": NOTEBOOKS}},
                            on_cell_executed=on_cell_executed)
    filt = None if raw_output else KernelOutputFilter()
    try:
        client.execute(**({"stderr": filt.write_fd} if filt else {}))
        ok = True
    except CellExecutionError as e:
        nbformat.write(nb, target)
        print(f"  FAILED after {fmt(time.perf_counter() - t0)}:\n{e}", file=sys.stderr, flush=True)
        ok = False
    finally:
        summary = filt.close() if filt else ""
    if ok:
        nbformat.write(nb, target)
        print(f"  finished in {fmt(time.perf_counter() - t0)}", flush=True)
    if summary:
        print(summary, flush=True)
    return ok


def main(argv=None) -> int:
    """Command line (see the module docstring). Returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebooks", nargs="*", help="name prefixes, e.g. 02 03 (default: all)")
    ap.add_argument("--from", dest="start", help="start at this notebook, e.g. 02")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a setting (Python expression), e.g. QUICK=True; repeatable")
    ap.add_argument("--timeout", type=int, default=None, help="seconds per cell (default: no limit)")
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--out", metavar="DIR", help="save the executed notebooks here instead of in place")
    ap.add_argument("--raw-output", action="store_true", help="do not filter the kernel's console output")
    ap.add_argument("--keep-going", action="store_true", help="continue with the next notebook after an error")
    args = ap.parse_args(argv)
    for o in args.set:
        if "=" not in o:
            ap.error(f"--set expects NAME=VALUE, got {o!r}")
    if sys.platform == "win32":  # the kernel connection (zmq) needs the selector event loop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if args.out:
        os.makedirs(args.out, exist_ok=True)
    paths = select(args.notebooks, args.start)
    if not paths:
        ap.error("no notebook matches")
    print("Running:", ", ".join(os.path.basename(p) for p in paths))
    t0, failed = time.perf_counter(), []
    for p in paths:
        if not run(p, args.set, args.timeout, args.kernel, args.out, args.raw_output):
            failed.append(os.path.basename(p))
            if not args.keep_going:
                break
    print(f"\nTotal {fmt(time.perf_counter() - t0)}. " + (f"FAILED: {', '.join(failed)}" if failed else "All notebooks ran."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
