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
        
        sensors = X['sensor'].unique()
        windows = []

        for sensor in sensors:
            sensor_data = X[X['sensor'] == sensor].sort_values('timestamp')
            for start in range(0, len(sensor_data) - self.window_size + 1, self.stride):
                end = start + self.window_size
                window = sensor_data.iloc[start:end]
                if isinstance(self.method, dict):
                    features = {col: extractor(col, window[col]) for col, extractor in self.method.items()}
                else:
                    features = {col: self.method(col, window[col]) for col in window.columns if col not in ['sensor', 'timestamp']}
                features['sensor'] = sensor
                features['timestamp'] = window['timestamp'].iloc[-1]  # Use the last timestamp of the window
                windows.append(features)
        return pd.DataFrame(windows)