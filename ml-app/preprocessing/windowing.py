import pandas as pd
from typing import Callable
from sklearn.base import BaseEstimator, TransformerMixin
from .feature_extraction import FeatureExtractor

class WindowingMethod(BaseEstimator, TransformerMixin):
    def __init__(self, method: dict[str, FeatureExtractor] | FeatureExtractor, window_size: int = 10, stride: int = 2):
        self.method = method
        self.window_size = window_size
        self.stride = stride

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:    
        """
        Create windows of data for time series analysis.

        Parameters:
        - data: Input DataFrame containing the time series data.
        - method: A dictionary mapping column names to FeatureExtractor instances for processing each column, or a single FeatureExtractor instance to apply to all columns.
        - window_size: The size of each window (number of rows).
        - stride: The number of rows to move the window forward.

        Returns:
        - A DataFrame containing the windows of data.
        """
        if "sensor" not in X.columns:
            raise ValueError("Input DataFrame must contain a 'sensor' column.")

        if "timestamp" not in X.columns:
            raise ValueError("Input DataFrame must contain a 'timestamp' column.")

        if isinstance(self.method, dict) and 'timestamp' not in self.method:
            raise ValueError("When method is a dictionary, it must contain a 'timestamp' key for processing the timestamp column.")
        
        sensors = X['sensor'].unique()
        windows = []

        for start in range(0, len(X) - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = X.iloc[start:end]
            window_data = dict()
            for sensor in sensors:
                sensor_window = window[window['sensor'] == sensor]
                if len(sensor_window) == self.window_size:
                    features = dict()
                    if isinstance(self.method, dict):
                        for col in sensor_window.columns:
                            if col in self.method and col not in ['sensor', 'timestamp']:
                                col_val = self.method[col](col, sensor_window[col])
                                features.update({f'{sensor}_{k}': v for k, v in col_val.items()})
                    else:
                        for col in sensor_window.columns:
                            if col not in ['sensor', 'timestamp']:
                                col_val = self.method(col, sensor_window[col])
                                features.update({f'{sensor}_{k}': v for k, v in col_val.items()})
                    window_data.update(features)
            timestamp_val = self.method['timestamp']('timestamp', window['timestamp']) if isinstance(self.method, dict) else self.method('timestamp', window['timestamp'])
            window_data.update(timestamp_val.to_dict())
            windows.append(window_data)
        return pd.DataFrame(windows)