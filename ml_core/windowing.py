"""Sliding-window feature extraction over interleaved sensor uplinks."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WindowFeatures(BaseEstimator, TransformerMixin):
    """Turn raw uplink rows into one feature row per sliding window.

    Input: rows with ``timestamp``, ``sensor`` and value columns named
    ``<sensor>_<quantity>`` (e.g. ``lse01_moisture``), sorted by time. Rows of
    different sensors are interleaved, as they arrive from the gateway.

    For every window of ``window_size`` rows (step ``stride``) it computes
    - per value column, over the rows of its own sensor only:
      min, max, mean, std, skew, slope (first to last valid reading, per row step)
    - timing of the uplinks: count, duration, mean/min/max gap (seconds).
    - optionally, longer context per value column (``context``, e.g. ("24h", "7d")): over the readings of
      the last 24 h / 7 d before the window's last uplink, ``<col>_ctxmean_<h>``, ``<col>_ctxstd_<h>`` and
      ``<col>_ctxtrend_<h>`` (least-squares slope per day). A window of 10 uplinks spans minutes, so without
      context the models only see the current level; the trend shows how the sausage has been changing.
      Only the past of the same recording is used. At the start of a run (and in the case after a restart)
      less history exists; the statistics then use what is there.

    Stateless: ``fit`` does nothing, so windows can be computed once and reused.
    """

    def __init__(self, window_size: int = 10, stride: int = 2, output_columns: list[str] | None = None,
                 context: tuple[str, ...] | list[str] = ()):
        """window_size: uplinks per window; stride: step between window starts; output_columns: see below;
        context: horizons for the context features as pandas offsets ("24h", "7d"), () = none."""
        self.window_size = window_size
        self.stride = stride
        # Fixed output schema. If set, the result is reindexed to these columns so that
        # windows in which a sensor did not report (common live) still match the models.
        self.output_columns = output_columns
        self.context = tuple(context)

    @property
    def history(self) -> pd.Timedelta:
        """How much history before a window the features need (the longest context horizon, else 0)."""
        ctx = getattr(self, "context", ())  # models pickled before context existed
        return max((pd.Timedelta(h) for h in ctx), default=pd.Timedelta(0))

    def transform_last(self, X: pd.DataFrame) -> pd.DataFrame:
        """Features of the last window only (one row): the last window_size rows plus, for the context, the
        history before them. What live, replay and explanation need; much cheaper than transform() on the history."""
        X = X.reset_index(drop=True)
        last = X.tail(self.window_size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = self._window_stats(last)
            if len(out) and getattr(self, "context", ()):
                ctx = self._context_stats(X, self._times(last)[-1:])
                out = pd.concat([out.reset_index(drop=True), ctx], axis=1)
        if self.output_columns is not None:
            out = out.reindex(columns=list(self.output_columns))
        return out

    def fit(self, X: pd.DataFrame, y=None):
        """Nothing to learn (stateless); returns self."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """One feature row per window of the raw rows X (see the class docstring)."""
        # A window in which a sensor did not report yields NaN statistics by design;
        # silence numpy's "empty slice" warnings for exactly this computation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self._transform(X)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Window statistics for every window, plus the context statistics at each window's last uplink."""
        out = self._window_stats(X)
        if len(out) and getattr(self, "context", ()):
            X = X.reset_index(drop=True)
            ends = np.arange(len(out)) * self.stride + self.window_size - 1
            out = pd.concat([out, self._context_stats(X, self._times(X)[ends])], axis=1)
        if self.output_columns is not None:
            out = out.reindex(columns=list(self.output_columns))
        return out

    @staticmethod
    def _times(X: pd.DataFrame) -> np.ndarray:
        """Timestamps as int64 ns (UTC, naive). Normalised to ns first: pandas >= 3 may parse at µs resolution,
        which would scale every interval feature by 1000."""
        ts = X["timestamp"]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts, utc=True, format="ISO8601")
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        return ts.astype("datetime64[ns]").values.astype("int64")

    def _context_stats(self, X: pd.DataFrame, at: np.ndarray) -> pd.DataFrame:
        """Mean, std and trend (per day) of every value column over each context horizon ending at the times `at`.

        Rolling sums over each sensor's own readings (time-based windows), looked up at the last reading at or
        before each time. The slope is the least-squares slope of value over time in days.
        """
        t_all = self._times(X)
        sensors = X["sensor"].to_numpy()
        out = {}
        for sensor in sorted(pd.unique(sensors).tolist()):
            own = sensors == sensor
            for col in (c for c in X.columns if c.startswith(sensor) and c not in ("sensor", "timestamp")):
                v = X[col].to_numpy(dtype=float)
                ok = own & ~np.isnan(v)
                t = t_all[ok]
                y = v[ok]
                idx = np.searchsorted(t, at, side="right") - 1          # last reading at or before each time
                for h in self.context:
                    names = [f"{col}_ctx{stat}_{h}" for stat in ("mean", "std", "trend")]
                    if not len(t):
                        for n in names:
                            out[n] = np.full(len(at), np.nan)
                        continue
                    d = (t - t[0]) / 86400e9                             # days since the first reading (stable sums)
                    frame = pd.DataFrame({"n": 1.0, "y": y, "yy": y * y, "d": d, "dd": d * d, "dy": d * y},
                                         index=pd.DatetimeIndex(t))
                    r = frame.rolling(pd.Timedelta(h)).sum().to_numpy()
                    n, sy, syy, sd, sdd, sdy = (r[:, i] for i in range(6))
                    mean = sy / n
                    var = np.maximum(syy / n - mean ** 2, 0) * n / np.maximum(n - 1, 1)
                    den = n * sdd - sd ** 2
                    trend = np.where(den > 1e-12, (n * sdy - sd * sy) / np.where(den > 1e-12, den, 1), np.nan)
                    valid = idx >= 0
                    pick = np.where(valid, idx, 0)
                    for name, arr in zip(names, (mean, np.sqrt(var), trend)):
                        out[name] = np.where(valid, arr[pick], np.nan)
        return pd.DataFrame(out)

    def _window_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """Vectorised computation over all windows at once (index array of shape windows × window_size)."""
        for col in ("timestamp", "sensor"):
            if col not in X.columns:
                raise ValueError(f"Input must contain a '{col}' column.")
        ws, st = self.window_size, self.stride
        if len(X) < ws:
            return pd.DataFrame()

        X = X.reset_index(drop=True)
        starts = np.arange(0, len(X) - ws + 1, st)
        n_windows = len(starts)
        idx = starts[:, None] + np.arange(ws)[None, :]          # (n_windows, ws)

        # ── timing ──────────────────────────────────────────────────────────
        ts_w = self._times(X)[idx]
        gaps = np.diff(ts_w, axis=1) / 1e9
        features: dict[str, np.ndarray] = {}
        timing = {
            "timestamp_count": np.full(n_windows, ws, dtype=np.float32),
            "timestamp_duration": (ts_w[:, -1] - ts_w[:, 0]) / 1e9,
            "timestamp_interval": gaps.mean(axis=1),
            "timestamp_min_interval": gaps.min(axis=1),
            "timestamp_max_interval": gaps.max(axis=1),
        }

        # ── per sensor value column ─────────────────────────────────────────
        sensor_arr = X["sensor"].values
        rows = np.arange(n_windows)
        for sensor in sorted(X["sensor"].unique().tolist()):
            own_w = (sensor_arr == sensor)[idx]                  # rows of this sensor
            for col in (c for c in X.columns if c.startswith(sensor) and c not in ("sensor", "timestamp")):
                v = np.where(own_w, X[col].values.astype(np.float32)[idx], np.nan)
                n = own_w.sum(axis=1).astype(np.float32)
                mean = np.nanmean(v, axis=1)
                centered = v - mean[:, None]
                std = np.sqrt(np.nansum(centered ** 2, axis=1) / np.maximum(n - 1, 1))
                std_safe = np.where(std > 0, std, np.nan)

                valid = own_w & ~np.isnan(v)
                first = np.argmax(valid, axis=1)
                last = ws - 1 - np.argmax(valid[:, ::-1], axis=1)
                slope = (v[rows, last] - v[rows, first]) / np.maximum(last - first, 1)

                features[f"{col}_min"] = np.nanmin(v, axis=1)
                features[f"{col}_max"] = np.nanmax(v, axis=1)
                features[f"{col}_mean"] = mean
                features[f"{col}_std"] = std
                features[f"{col}_skew"] = np.nanmean(centered ** 3, axis=1) / std_safe ** 3
                features[f"{col}_slope"] = np.where(valid.sum(axis=1) >= 2, slope, np.nan)

        return pd.DataFrame({**features, **timing}, copy=False)
