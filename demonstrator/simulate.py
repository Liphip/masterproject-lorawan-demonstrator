"""Virtual sausage: what-if simulation on top of a recorded curing run.

There is no real sausage in the case, so live sensor values cannot show a
curing phase. The simulator takes a real recorded window (any day of a
recorded run) and lets the presenter shift the measured values with sliders.
The models re-predict immediately. Everything it returns is marked SIMULATED.

Adjustments are applied to the recorded readings, so the shape of the data
(noise, uplink timing, sensor interleaving) stays realistic. Only the levels
change.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from ml_core.data import FEATURE_ORDER, add_temperature_difference, load_recording

from .inference import ModelRegistry
from .replay import list_recordings

LIMITS = {  # slider ranges, enforced server-side too
    "moisture_delta": (-30.0, 30.0),       # percentage points, LSE01
    "conductivity_factor": (0.3, 2.0),     # multiplier, LSE01
    "temperature_delta": (-10.0, 10.0),    # °C, LSE01 core + LTC-FS probe (air temperature stays as recorded)
}


class Simulator:
    """Virtual sausage: predictions on a recorded window with shifted levels (see the module docstring)."""
    def __init__(self, models: ModelRegistry, dirs: list[str]):
        """`dirs` are searched for recordings; the last simulation is kept for /api/explain."""
        self.models, self.dirs = models, dirs
        self._cache: dict[str, pd.DataFrame] = {}
        self.lock = threading.Lock()
        self.last: tuple | None = None   # (window, group, result) for /api/explain

    def _load(self, group: str) -> pd.DataFrame:
        """Load a recording (cached; at most ~8 kept in memory)."""
        with self.lock:
            if group not in self._cache:
                rec = next((r for r in list_recordings(self.dirs) if r["group"] == group), None)
                if rec is None:
                    raise ValueError(f"unknown recording {group}")
                if rec["lfs_pointer"]:
                    raise ValueError(f"{group} is a Git LFS pointer – run `git lfs pull`")
                if len(self._cache) > 8:
                    self._cache.clear()
                self._cache[group] = load_recording(rec["path"])
            return self._cache[group]

    @staticmethod
    def _clip(name, v):
        """Clamp a slider value to its range in LIMITS."""
        lo, hi = LIMITS[name]
        return float(min(max(float(v), lo), hi))

    def run(self, group: str, position: float, adjust: dict | None = None) -> dict:
        """Predict on the window at `position` (0..1 of the run) of `group`, with the adjustments applied:
        moisture_delta (points), conductivity_factor, temperature_delta (°C on both probes; the recorded air
        temperature stays, so the difference to the air changes). Returns the prediction marked simulated=True, the
        recorded and simulated mean values, and the annotation of the unmodified window (if labeled).
        """
        adjust = adjust or {}
        dm = self._clip("moisture_delta", adjust.get("moisture_delta", 0))
        cf = self._clip("conductivity_factor", adjust.get("conductivity_factor", 1))
        dt = self._clip("temperature_delta", adjust.get("temperature_delta", 0))
        df = self._load(group)
        ws = self.models.window_size
        ts = df["timestamp"]
        t0, t1 = ts.iloc[0], ts.iloc[-1]
        position = min(max(float(position), 0.0), 1.0)
        target = t0 + (t1 - t0) * position
        k = int(np.searchsorted(ts.values, target.to_datetime64(), side="right")) - 1
        k = min(max(k, ws - 1), len(df) - 1)
        base = df[FEATURE_ORDER + ["ambient_temperature"]].iloc[k + 1 - ws:k + 1].copy()
        win = base.copy()
        win["lse01_moisture"] = (win["lse01_moisture"] + dm).clip(0, 100)
        win["lse01_conductivity"] = (win["lse01_conductivity"] * cf).clip(lower=0)
        win["lse01_temperature"] = win["lse01_temperature"] + dt
        win["ltcfs_temperature"] = win["ltcfs_temperature"] + dt
        add_temperature_difference(win)  # recorded air temperature, shifted probe temperatures

        res = self.models.predict(win, group=group)
        truth = None
        if "label" in df.columns:
            lw = df["label"].iloc[k + 1 - ws:k + 1].dropna()
            truth = lw.mode().iloc[0] if len(lw) else None
        cols = ["lse01_moisture", "lse01_conductivity", "lse01_temperature", "ltcfs_temperature",
                "ambient_temperature", "lse01_temperature_diff", "ltcfs_temperature_diff"]
        mean = lambda d: {c: (None if d[c].isna().all() else float(d[c].mean())) for c in cols}
        res.update(simulated=True, group=group, position=position,
                   time=ts.iloc[k].isoformat(), day=(ts.iloc[k] - t0).total_seconds() / 86400,
                   run_days=(t1 - t0).total_seconds() / 86400,
                   adjust={"moisture_delta": dm, "conductivity_factor": cf, "temperature_delta": dt},
                   recorded_means=mean(base), simulated_means=mean(win),
                   truth=truth, truth_is_for_unmodified=bool(dm or cf != 1 or dt))
        with self.lock:
            self.last = (win.reset_index(drop=True), group, res)
        return res
