import pandas as pd


class FeatureExtractor:
    def __init__(self):
        pass

    def __call__(self, column_name: str, data: pd.Series) -> pd.Series:
        """
        Extract features from a given column of data.

        Parameters:
        - column_name: The name of the column to extract features from.
        - data: A pandas Series containing the data for the column.

        Returns:
        - A pandas Series containing the extracted features.
        """
        if isinstance(data.dtype, pd.api.types.CategoricalDtype) or pd.api.types.is_object_dtype(data) or pd.api.types.is_string_dtype(data):
            return CategoricalExtractor()(column_name, data)
        elif pd.api.types.is_numeric_dtype(data) or pd.api.types.is_bool_dtype(data):
            return NumericalExtractor()(column_name, data)
        elif pd.api.types.is_datetime64_any_dtype(data) or pd.api.types.is_timedelta64_dtype(data):
            return TimestampExtractor()(column_name, data)
        else:
            raise NotImplementedError(f"FeatureExtractor is an abstract class. Please implement the __call__ method for type {type(data.dtype)} for column '{column_name}'.")
    
    def __str__(self):
        return f"{self.__class__.__name__}()"

class NumericalExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def __call__(self, column_name: str, data: pd.Series) -> pd.Series:
        """
        Extract features from a given column of data.

        Parameters:
        - column_name: The name of the column to extract features from.
        - data: A pandas Series containing the data for the column.

        Returns:
        - A pandas Series containing the extracted features.
        """
        if not pd.api.types.is_numeric_dtype(data) and not pd.api.types.is_bool_dtype(data):
            raise ValueError(f"NumericExtractor can only be applied to numeric or boolean data for column '{column_name}'.")
        data_min = data.min()
        data_max = data.max()
        data_mean = data.mean()
        data_std = data.std()
        data_skew = data.skew()
        data_slope = (data.iloc[-1] - data.iloc[0]) / (len(data) - 1) if len(data) > 1 else 0
        return pd.Series({
            f"{column_name}_min": data_min,
            f"{column_name}_max": data_max,
            f"{column_name}_mean": data_mean,
            f"{column_name}_std": data_std,
            f"{column_name}_skew": data_skew,
            f"{column_name}_slope": data_slope,
        })

class TimestampExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def __call__(self, column_name: str, data: pd.Series) -> pd.Series:
        """
        Extract features from a given column of data.

        Parameters:
        - column_name: The name of the column to extract features from.
        - data: A pandas Series containing the data for the column.

        Returns:
        - A pandas Series containing the extracted features.
        """
        if not pd.api.types.is_datetime64_any_dtype(data) and not pd.api.types.is_timedelta64_dtype(data):
            raise ValueError(f"TimestampExtractor can only be applied to datetime or timedelta data for column '{column_name}'.")
        data_count = data.count()
        data_duration = (data.max() - data.min()).total_seconds() if data_count > 1 else 0
        data_interval = data.diff().dt.total_seconds().mean() if data_count > 1 else 0
        data_min_interval = data.diff().dt.total_seconds().min() if data_count > 1 else 0
        data_max_interval = data.diff().dt.total_seconds().max() if data_count > 1 else 0
        return pd.Series({
            f"{column_name}_count": data_count,
            f"{column_name}_duration": data_duration,
            f"{column_name}_interval": data_interval,
            f"{column_name}_min_interval": data_min_interval,
            f"{column_name}_max_interval": data_max_interval,
        })

class CategoricalExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def __call__(self, column_name: str, data: pd.Series) -> pd.Series:
        """
        Extract features from a given column of data.

        Parameters:
        - column_name: The name of the column to extract features from.
        - data: A pandas Series containing the data for the column.

        Returns:
        - A pandas Series containing the extracted features.
        """
        if not isinstance(data.dtype, pd.api.types.CategoricalDtype) and not pd.api.types.is_object_dtype(data) and not pd.api.types.is_string_dtype(data):
            raise ValueError(f"CategoricalExtractor can only be applied to categorical, object, or string data for column '{column_name}'.")
        data_mode = data.mode().iloc[0] if not data.mode().empty else None
        data_unique_count = data.nunique()
        return pd.Series({
            f"{column_name}_mode": data_mode,
            f"{column_name}_unique_count": data_unique_count,
        })