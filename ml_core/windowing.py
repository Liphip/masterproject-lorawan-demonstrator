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

    Stateless: ``fit`` does nothing, so windows can be computed once and reused.
    """

    def __init__(self, window_size: int = 10, stride: int = 2, output_columns: list[str] | None = None):
        self.window_size = window_size
        self.stride = stride
        # Fixed output schema. If set, the result is reindexed to these columns so that
        # windows in which a sensor did not report (common live) still match the models.
        self.output_columns = output_columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # A window in which a sensor did not report yields NaN statistics by design;
        # silence numpy's "empty slice" warnings for exactly this computation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self._transform(X)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in ("timestamp", "sensor"):
            if col not in X.columns:
                raise ValueError(f"Input must contain a '{col}' column.")
        ws, st = self.window_size, self.stride
        if len(X) < ws:
            return pd.DataFrame(columns=self.output_columns)

        X = X.reset_index(drop=True)
        starts = np.arange(0, len(X) - ws + 1, st)
        n_windows = len(starts)
        idx = starts[:, None] + np.arange(ws)[None, :]          # (n_windows, ws)

        # ── timing ──────────────────────────────────────────────────────────
        # Normalise to ns before the int64 cast: pandas >= 3 may parse at µs
        # resolution, which would scale every interval feature by 1000.
        ts = X["timestamp"]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts, utc=True, format="ISO8601")
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        ts_w = ts.astype("datetime64[ns]").values.astype("int64")[idx]
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

        out = pd.DataFrame({**features, **timing}, copy=False)
        if self.output_columns is not None:
            out = out.reindex(columns=list(self.output_columns))
        return out
