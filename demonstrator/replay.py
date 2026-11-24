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

from ml_core.data import FEATURE_ORDER, is_lfs_pointer, load_recording, recording_name

from .inference import ModelRegistry

log = logging.getLogger(__name__)
TICK_S = 0.25
MAX_PREDICTIONS_PER_TICK = 4
SERIES_POINTS = 1500


def list_recordings(dirs: list[str]) -> list[dict]:
    out = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
            out.append({"group": recording_name(p), "path": p, "lfs_pointer": is_lfs_pointer(p),
                        "kind": "live recording" if os.path.basename(p).startswith("live_") else "training data"})
    return out


class Decimated:
    """Keeps a bounded, evenly thinned record of an entire run."""

    def __init__(self, cap: int):
        self.cap, self.every, self.n, self.items = cap, 1, 0, []

    def append(self, item):
        if self.n % self.every == 0:
            self.items.append(item)
            if len(self.items) > self.cap:
                self.items = self.items[::2]
                self.every *= 2
        self.n += 1

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class ReplayEngine:
    def __init__(self, models: ModelRegistry, dirs: list[str]):
        self.models, self.dirs = models, dirs
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self.reset()

    def reset(self):
        self.status = {"running": False, "group": None, "speed": None, "progress": 0.0,
                       "data_time": None, "start": None, "end": None, "labeled": False,
                       "rows": 0, "error": None, "finished": False, "paused": False}
        self.series = Decimated(SERIES_POINTS * 3)  # rows of all sensors interleaved
        self.history = Decimated(SERIES_POINTS)
        self.latest: dict | None = None
        self.latest_window: pd.DataFrame | None = None
        self.running_score = {}

    def recordings(self):
        return list_recordings(self.dirs)

    def start(self, group: str, speed: float = 1800.0, start_fraction: float = 0.0):
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
                               end=df["timestamp"].iloc[-1].isoformat(), kind=rec["kind"])
        self._stop.clear()
        self._paused.clear()
        self._thread = threading.Thread(target=self._run, args=(df, group, float(speed), float(start_fraction)),
                                        daemon=True, name="replay")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        with self.lock:
            self.status["running"] = False

    def set_paused(self, paused: bool):
        (self._paused.set if paused else self._paused.clear)()
        with self.lock:
            self.status["paused"] = bool(paused) and self.status["running"]

    def current_window(self):
        with self.lock:
            return (None if self.latest_window is None else self.latest_window.copy(),
                    self.status["group"], self.latest)

    def _run(self, df: pd.DataFrame, group: str, speed: float, start_fraction: float):
        try:
            self._loop(df, group, speed, start_fraction)
        except Exception as e:  # noqa: BLE001
            log.exception("replay failed")
            with self.lock:
                self.status.update(running=False, error=repr(e))

    def _loop(self, df, group, speed, start_fraction):
        ws, st = self.models.window_size, self.models.stride
        ts = df["timestamp"]
        t_start, t_end = ts.iloc[0], ts.iloc[-1]
        total = (t_end - t_start).total_seconds() or 1.0
        i = int(np.searchsorted(ts.values, (t_start + pd.Timedelta(seconds=total * start_fraction)).to_datetime64()))
        i = max(i, 0)
        # history of the first window so the first prediction can be made right away
        data_time = ts.iloc[max(i - 1, 0)]
        feats = df[FEATURE_ORDER]
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
                window = feats.iloc[k + 1 - ws:k + 1]
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
        with self.lock:
            s = list(self.series)
            h = list(self.history)
            step_s = max(1, len(s) // series_points)
            step_h = max(1, len(h) // series_points)
            return {"status": dict(self.status), "latest": self.latest,
                    "series": s[::step_s], "history": h[::step_h],
                    "running_score": dict(self.running_score)}
