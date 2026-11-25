"""Replay of recorded curing runs at accelerated speed, with live CPU inference.

A curing run takes days; replay compresses it so the audience can watch the
models' phase predictions evolve. For each labeled training sausage the fold
model that never saw that sausage is used, so replayed predictions are honest
held-out predictions rather than memorised training data.
"""
from __future__ import annotations

import glob
import logging
import os
import threading
import time

import numpy as np
import pandas as pd

from ml_core.data import FEATURE_ORDER, is_lfs_pointer, load_recording, recording_name, window_labels

from .inference import ModelRegistry

log = logging.getLogger(__name__)
TICK_S = 0.25
MAX_PREDICTIONS_PER_TICK = 4
SERIES_POINTS = 1500


_SPANS: dict[str, tuple[float, tuple]] = {}  # path -> (mtime, (start, end, labeled))


def _span(path: str) -> tuple[str | None, str | None, bool]:
    """(first timestamp, last timestamp, has a label column) of a converted CSV without reading it all.

    Reads the header, the first data line and the last line only; cached per file modification time.
    Used for readable names in the UI ("Run 1 · sausage 3 · Mar–Aug 2023").
    """
    mtime = os.path.getmtime(path)
    if path in _SPANS and _SPANS[path][0] == mtime:
        return _SPANS[path][1]
    start = end = None
    labeled = False
    try:
        with open(path, "rb") as fh:
            header = fh.readline().decode("utf-8", "replace").strip().split(",")
            first = fh.readline().decode("utf-8", "replace").strip().split(",")
            fh.seek(0, os.SEEK_END)
            fh.seek(max(fh.tell() - 4096, 0))
            last = fh.read().decode("utf-8", "replace").strip().splitlines()[-1].split(",")
        i = header.index("timestamp")
        start, end = (first[i] or None), (last[i] or None)
        labeled = "label" in header
    except (OSError, ValueError, IndexError):
        pass
    _SPANS[path] = (mtime, (start, end, labeled))
    return start, end, labeled


def window_with_history(feats: pd.DataFrame, ts_ns: np.ndarray, k: int, ws: int, history: pd.Timedelta):
    """Rows of the window ending at row k plus the context history before it (models with context features)."""
    start = k + 1 - ws
    if history > pd.Timedelta(0):
        start = min(start, int(np.searchsorted(ts_ns, ts_ns[k] - history.value, side="right")))
    return feats.iloc[max(start, 0):k + 1]


def list_recordings(dirs: list[str]) -> list[dict]:
    """All CSV recordings in `dirs` (converted training data and live recordings), with a Git LFS check.

    Each entry: group (file name without .csv), path, lfs_pointer, kind, start, end (ISO timestamps or None).
    """
    out = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
            lfs = is_lfs_pointer(p)
            start, end, _ = (None, None, False) if lfs else _span(p)
            out.append({"group": recording_name(p), "path": p, "lfs_pointer": lfs, "start": start, "end": end,
                        "kind": "live recording" if os.path.basename(p).startswith("live_") else "training data"})
    return out


class Decimated:
    """Keeps a bounded, evenly thinned record of an entire run."""

    def __init__(self, cap: int):
        """Keep at most `cap` items; when full, keep every second one and halve the sampling rate."""
        self.cap, self.every, self.n, self.items = cap, 1, 0, []

    def append(self, item):
        """Add an item if it falls on the current sampling step."""
        if self.n % self.every == 0:
            self.items.append(item)
            if len(self.items) > self.cap:
                self.items = self.items[::2]
                self.every *= 2
        self.n += 1

    def __iter__(self):
        """Iterate over the kept items."""
        return iter(self.items)

    def __len__(self):
        """Number of kept items."""
        return len(self.items)


class ReplayEngine:
    """Plays one recording back in a background thread at `speed` data seconds per real second.

    Every TICK_S it advances the data clock, adds the new rows to the plotted series and predicts at the stride
    boundaries that were passed (at most MAX_PREDICTIONS_PER_TICK, to bound CPU load at high speed). Labeled
    recordings get a running accuracy against the butcher's annotation.
    """
    def __init__(self, models: ModelRegistry, dirs: list[str]):
        """No replay running yet; `dirs` are searched for recordings."""
        self.models, self.dirs = models, dirs
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self.reset()

    def reset(self):
        """Clear status, plotted series, prediction history and running score."""
        self.status = {"running": False, "group": None, "speed": None, "progress": 0.0,
                       "data_time": None, "start": None, "end": None, "labeled": False,
                       "rows": 0, "error": None, "finished": False, "paused": False, "complete": False}
        self.series = Decimated(SERIES_POINTS * 3)  # rows of all sensors interleaved
        self.history = Decimated(SERIES_POINTS)
        self.latest: dict | None = None
        self.latest_window: pd.DataFrame | None = None
        self.running_score = {}

    def recordings(self):
        """Recordings available for replay."""
        return list_recordings(self.dirs)

    def start(self, group: str, speed: float = 1800.0, start_fraction: float = 0.0, complete: bool = False):
        """Stop any running replay and start `group` at `speed`, from `start_fraction` (0..1) of the run.

        complete=True skips the animation: the whole run is predicted at once (batch, a few seconds) and shown as
        finished, with the full timeline.
        """
        rec = next((r for r in self.recordings() if r["group"] == group), None)
        if rec is None:
            raise ValueError(f"unknown recording {group}")
        if rec["lfs_pointer"]:
            raise ValueError(f"{group} is a Git LFS pointer – run `git lfs pull`")
        self.stop()
        df = load_recording(rec["path"])
        labeled = "label" in df.columns and df["label"].notna().any()
        with self.lock:
            self.reset()
            self.status.update(running=True, group=group, speed=float(speed), labeled=bool(labeled),
                               start=df["timestamp"].iloc[0].isoformat(),
                               end=df["timestamp"].iloc[-1].isoformat(), kind=rec["kind"], complete=bool(complete))
        self._stop.clear()
        self._paused.clear()
        target = self._complete if complete else self._run
        args = (df, group) if complete else (df, group, float(speed), float(start_fraction))
        self._thread = threading.Thread(target=target, args=args, daemon=True, name="replay")
        self._thread.start()

    def stop(self):
        """Stop the replay thread (waits up to 3 s)."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        with self.lock:
            self.status["running"] = False

    def set_paused(self, paused: bool):
        """Pause or resume; the data clock does not advance while paused."""
        (self._paused.set if paused else self._paused.clear)()
        with self.lock:
            self.status["paused"] = bool(paused) and self.status["running"]

    def current_window(self):
        """(rows, group, latest prediction) of the window on screen, for /api/explain."""
        with self.lock:
            return (None if self.latest_window is None else self.latest_window.copy(),
                    self.status["group"], self.latest)

    def _run(self, df: pd.DataFrame, group: str, speed: float, start_fraction: float):
        """Thread entry point: run the loop, report errors in the status instead of crashing silently."""
        try:
            self._loop(df, group, speed, start_fraction)
        except Exception as e:  # noqa: BLE001
            log.exception("replay failed")
            with self.lock:
                self.status.update(running=False, error=repr(e))

    def _complete(self, df: pd.DataFrame, group: str):
        """Whole run at once: batch prediction of every window, then the same state as a finished replay."""
        try:
            ws, st = self.models.window_size, self.models.stride
            feats = df[FEATURE_ORDER]
            n_win = (len(df) - ws) // st + 1 if len(df) >= ws else 0
            if n_win <= 0:
                raise ValueError(f"{group} has fewer than {ws} uplinks")
            ends = np.arange(n_win) * st + ws - 1                     # last row of each window
            times = df["timestamp"].iloc[ends].reset_index(drop=True)
            truth = (window_labels(df["label"], ws, st) if "label" in df.columns and df["label"].notna().any()
                     else np.full(n_win, None, dtype=object))
            preds = {}
            for key in self.models.keys:
                pipe, _ = self.models.model_for(key, group)
                proba = pipe.predict_proba(feats)                      # one row per window, same windows as replay
                preds[key] = np.asarray(pipe.classes_, dtype=object)[proba.argmax(axis=1)]
            with self.lock:
                for k in range(len(df)):
                    r = df.iloc[k] if k % 4 == 0 else None             # thinning: every 4th row is plenty here
                    if r is None or r["sensor"] == "ws202":
                        continue
                    self.series.append({"t": r["timestamp"].isoformat(), "sensor": r["sensor"],
                                        **{c: float(r[c]) for c in FEATURE_ORDER[2:] if pd.notna(r[c])}})
                for w in range(n_win):
                    t = truth[w] if isinstance(truth[w], str) else None
                    self.history.append({"t": times[w].isoformat(), "truth": t,
                                         **{key: preds[key][w] for key in preds}})
                scored = [w for w in range(n_win) if isinstance(truth[w], str)]
                if scored:
                    self.running_score = {key: float(np.mean([preds[key][w] == truth[w] for w in scored]))
                                          for key in preds} | {"n": len(scored)}
            ts_ns = pd.to_datetime(df["timestamp"], utc=True).astype("int64").to_numpy()
            last = window_with_history(feats, ts_ns, int(ends[-1]), ws, self.models.history)
            res = self.models.predict(last, group=group)
            res.update(time=times.iloc[-1].isoformat(), truth=truth[-1] if isinstance(truth[-1], str) else None)
            with self.lock:
                self.latest, self.latest_window = res, last.copy()
                self.status.update(running=False, finished=True, progress=1.0, rows=len(df), complete=True,
                                   data_time=df["timestamp"].iloc[-1].isoformat())
        except Exception as e:  # noqa: BLE001
            log.exception("complete replay failed")
            with self.lock:
                self.status.update(running=False, error=repr(e))

    def _loop(self, df, group, speed, start_fraction):
        """The replay loop (see the class docstring)."""
        ws, st = self.models.window_size, self.models.stride
        ts = df["timestamp"]
        t_start, t_end = ts.iloc[0], ts.iloc[-1]
        total = (t_end - t_start).total_seconds() or 1.0
        i = int(np.searchsorted(ts.values, (t_start + pd.Timedelta(seconds=total * start_fraction)).to_datetime64()))
        i = max(i, 0)
        # history of the first window so the first prediction can be made right away
        data_time = ts.iloc[max(i - 1, 0)]
        feats = df[FEATURE_ORDER]
        ts_ns = pd.to_datetime(ts, utc=True).astype("int64").to_numpy()
        history = self.models.history
        labels = df["label"] if "label" in df.columns else None
        n = len(df)
        hits = {k: 0 for k in self.models.keys}
        scored = 0
        while not self._stop.is_set() and i < n:
            if self._paused.is_set():
                time.sleep(TICK_S)
                continue
            data_time = data_time + pd.Timedelta(seconds=speed * TICK_S)
            j = int(np.searchsorted(ts.values, data_time.to_datetime64(), side="right"))
            new_idx = range(i, min(j, n))
            # predictions at stride boundaries, capped per tick to bound CPU load
            pred_at = [k for k in new_idx if k + 1 >= ws and (k + 1 - ws) % st == 0][-MAX_PREDICTIONS_PER_TICK:]
            with self.lock:
                for k in new_idx:
                    r = df.iloc[k]
                    if r["sensor"] == "ws202":
                        continue  # environment rows carry no plotted values
                    self.series.append({"t": r["timestamp"].isoformat(), "sensor": r["sensor"],
                                        **{c: float(r[c]) for c in FEATURE_ORDER[2:] if pd.notna(r[c])}})
            for k in pred_at:
                window = window_with_history(feats, ts_ns, k, ws, history)
                res = self.models.predict(window, group=group)
                if not res.get("ready"):
                    continue
                truth = None
                if labels is not None:
                    lw = labels.iloc[k + 1 - ws:k + 1].dropna()
                    truth = lw.mode().iloc[0] if len(lw) else None
                res.update(time=df["timestamp"].iloc[k].isoformat(), truth=truth)
                if truth is not None:
                    scored += 1
                    for key, m in res["models"].items():
                        hits[key] += int(m["label"] == truth)
                with self.lock:
                    self.latest = res
                    self.latest_window = window.copy()
                    self.history.append({"t": res["time"], "truth": truth,
                                         **{key: m["label"] for key, m in res["models"].items()}})
                    if scored:
                        self.running_score = {key: hits[key] / scored for key in hits} | {"n": scored}
            i = max(i, min(j, n))
            with self.lock:
                self.status.update(progress=(data_time - t_start).total_seconds() / total,
                                   data_time=min(data_time, t_end).isoformat(), rows=i)
            time.sleep(TICK_S)
        with self.lock:
            self.status.update(running=False, paused=False, finished=i >= n, progress=min(self.status["progress"], 1.0))

    def snapshot(self, series_points: int = 600) -> dict:
        """Status, latest prediction, thinned series and history for /api/state."""
        with self.lock:
            s = list(self.series)
            h = list(self.history)
            step_s = max(1, len(s) // series_points)
            step_h = max(1, len(h) // series_points)
            return {"status": dict(self.status), "latest": self.latest,
                    "series": s[::step_s], "history": h[::step_h],
                    "running_score": dict(self.running_score)}
