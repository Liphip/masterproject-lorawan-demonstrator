import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .feature_extraction import FeatureExtractor


class WindowingMethod(BaseEstimator, TransformerMixin):
    def __init__(self, method: "dict[str, FeatureExtractor] | FeatureExtractor" = None,
                 window_size: int = 10, stride: int = 2):
        self.method = method if method is not None else FeatureExtractor()
        self.window_size = window_size
        self.stride = stride

    def __str__(self):
        return f"WindowingMethod(method={self.method}, window_size={self.window_size}, stride={self.stride})"

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if "sensor" not in X.columns:
            raise ValueError("Input DataFrame must contain a 'sensor' column.")
        if "timestamp" not in X.columns:
            raise ValueError("Input DataFrame must contain a 'timestamp' column.")
        if isinstance(self.method, dict) and 'timestamp' not in self.method:
            raise ValueError(
                "When method is a dictionary, it must contain a 'timestamp' key."
            )

        ws = self.window_size
        st = self.stride
        n_rows = len(X)
        if n_rows < ws:
            return pd.DataFrame()

        X = X.reset_index(drop=True)
        starts = np.arange(0, n_rows - ws + 1, st)       # (n_windows,)
        n_windows = int(len(starts))
        idx = starts[:, None] + np.arange(ws)[None, :]   # (n_windows, ws)

        # ── timestamp features ────────────────────────────────────────────────
        ts_ns = X["timestamp"].values.astype("int64")     # (n_rows,)
        ts_w  = ts_ns[idx]                                # (n_windows, ws)

        intervals_s = np.diff(ts_w, axis=1) / 1e9        # (n_windows, ws-1)
        ts_features = {
            "timestamp_count":        np.full(n_windows, ws, dtype=np.float32),
            "timestamp_duration":     (ts_w[:, -1] - ts_w[:, 0]) / 1e9,
            "timestamp_interval":     intervals_s.mean(axis=1),
            "timestamp_min_interval": intervals_s.min(axis=1),
            "timestamp_max_interval": intervals_s.max(axis=1),
        }

        # ── per-sensor / per-column features ─────────────────────────────────
        sensors = sorted(X["sensor"].unique().tolist())
        sensor_arr = X["sensor"].values
        sensor_features: dict = {}

        for sensor in sensors:
            mask = sensor_arr == sensor                   # (n_rows,) bool
            sensor_mask_w = mask[idx]                    # (n_windows, ws) bool

            value_cols = [
                c for c in X.columns
                if c.startswith(sensor) and c not in ("sensor", "timestamp")
            ]

            for col in value_cols:
                dtype = X[col].dtype

                if (pd.api.types.is_numeric_dtype(dtype)
                        or pd.api.types.is_bool_dtype(dtype)):
                    col_w = X[col].values.astype(np.float32)[idx]    # (n_windows, ws)
                    col_w = np.where(sensor_mask_w, col_w, np.nan)

                    col_mean = np.nanmean(col_w, axis=1)
                    count    = sensor_mask_w.sum(axis=1).astype(np.float32)
                    sq_diff  = np.where(sensor_mask_w,
                                        (col_w - col_mean[:, None]) ** 2,
                                        np.nan)
                    col_std  = np.sqrt(np.nansum(sq_diff, axis=1)
                                       / np.maximum(count - 1, 1))
                    std_safe = np.where(col_std > 0, col_std, np.nan)
                    centered = np.where(sensor_mask_w,
                                        col_w - col_mean[:, None],
                                        np.nan)

                    sensor_features[f"{col}_min"]   = np.nanmin(col_w,  axis=1)
                    sensor_features[f"{col}_max"]   = np.nanmax(col_w,  axis=1)
                    sensor_features[f"{col}_mean"]  = col_mean
                    sensor_features[f"{col}_std"]   = col_std
                    sensor_features[f"{col}_skew"]  = (
                        np.nanmean(centered ** 3, axis=1) / (std_safe ** 3)
                    )
                    sensor_features[f"{col}_slope"] = np.where(
                        ws > 1,
                        (col_w[:, -1] - col_w[:, 0]) / (ws - 1),
                        np.zeros(n_windows, dtype=np.float32),
                    )

                else:
                    # categorical — Python loop fallback (rare, low-cardinality)
                    str_w  = X[col].values[idx]
                    modes         = np.empty(n_windows, dtype=object)
                    unique_counts = np.empty(n_windows, dtype=np.float32)
                    for i in range(n_windows):
                        vals = str_w[i][sensor_mask_w[i]]
                        if len(vals) == 0:
                            modes[i], unique_counts[i] = None, 0.0
                        else:
                            uniq, cnts = np.unique(vals, return_counts=True)
                            modes[i]         = uniq[np.argmax(cnts)]
                            unique_counts[i] = float(len(uniq))
                    sensor_features[f"{col}_mode"]         = modes
                    sensor_features[f"{col}_unique_count"] = unique_counts

        return pd.DataFrame({**sensor_features, **ts_features}, copy=False)

# import pandas as pd
# from typing import Callable
# from sklearn.base import BaseEstimator, TransformerMixin
# from .feature_extraction import FeatureExtractor

# class WindowingMethod(BaseEstimator, TransformerMixin):
#     def __init__(self, method: dict[str, FeatureExtractor] | FeatureExtractor, window_size: int = 10, stride: int = 2):
#         self.method = method
#         self.window_size = window_size
#         self.stride = stride
    
#     def __str__(self):
#         return f"WindowingMethod(method={self.method}, window_size={self.window_size}, stride={self.stride})"

#     def fit(self, X: pd.DataFrame, y=None):
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:    
#         """
#         Create windows of data for time series analysis.

#         Parameters:
#         - data: Input DataFrame containing the time series data.
#         - method: A dictionary mapping column names to FeatureExtractor instances for processing each column, or a single FeatureExtractor instance to apply to all columns.
#         - window_size: The size of each window (number of rows).
#         - stride: The number of rows to move the window forward.

#         Returns:
#         - A DataFrame containing the windows of data.
#         """
#         if "sensor" not in X.columns:
#             raise ValueError("Input DataFrame must contain a 'sensor' column.")

#         if "timestamp" not in X.columns:
#             raise ValueError("Input DataFrame must contain a 'timestamp' column.")

#         if isinstance(self.method, dict) and 'timestamp' not in self.method:
#             raise ValueError("When method is a dictionary, it must contain a 'timestamp' key for processing the timestamp column.")
        
#         sensors = sorted(X['sensor'].unique())
#         windows = []

#         for start in range(0, len(X) - self.window_size + 1, self.stride):
#             end = start + self.window_size
#             window = X.iloc[start:end]
#             window_data = dict()
#             for sensor in sensors:
#                 sensor_window = window[window['sensor'] == sensor]
#                 if len(sensor_window) == 0:
#                     continue
#                 columns = [col for col in sensor_window.columns if col.startswith(sensor) and col not in ['sensor', 'timestamp']]
#                 features = dict()
#                 if isinstance(self.method, dict):
#                     for col in columns:
#                         if col in self.method and col not in ['sensor', 'timestamp']:
#                             col_val = self.method[col](col, sensor_window[col])
#                             features.update(col_val.to_dict())
#                 else:
#                     for col in columns:
#                         if col not in ['sensor', 'timestamp']:
#                             col_val = self.method(col, sensor_window[col])
#                             features.update(col_val.to_dict())
#                 window_data.update(features)
#             timestamp_val = self.method['timestamp']('timestamp', window['timestamp']) if isinstance(self.method, dict) else self.method('timestamp', window['timestamp'])
#             window_data.update(timestamp_val.to_dict())
#             windows.append(window_data)
#         return pd.DataFrame(windows)