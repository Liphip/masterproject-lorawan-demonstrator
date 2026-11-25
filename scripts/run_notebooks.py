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
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import glob
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")
OVERRIDE_TAG = "injected-parameters"


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


def run(path: str, overrides: list[str], timeout: int | None, kernel: str, out_dir: str | None = None) -> bool:
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
    try:
        client.execute()
    except CellExecutionError as e:
        nbformat.write(nb, target)
        print(f"  FAILED after {fmt(time.perf_counter() - t0)}:\n{e}", file=sys.stderr, flush=True)
        return False
    nbformat.write(nb, target)
    print(f"  finished in {fmt(time.perf_counter() - t0)}", flush=True)
    return True


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
        if not run(p, args.set, args.timeout, args.kernel, args.out):
            failed.append(os.path.basename(p))
            if not args.keep_going:
                break
    print(f"\nTotal {fmt(time.perf_counter() - t0)}. " + (f"FAILED: {', '.join(failed)}" if failed else "All notebooks ran."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
