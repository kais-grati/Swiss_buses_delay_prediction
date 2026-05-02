import pandas as pd
from ml.preprocessors.base import BasePreprocessor


class TemporalFeatureExtractor(BasePreprocessor):
    """Extracts hour, dow, month integers from a timestamp column and drops it.

    Must be placed before any scaler since these are integer features, not floats.
    """

    def fit(self, X: pd.DataFrame) -> "TemporalFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        ts = pd.to_datetime(X["timestamp"])
        X["hour"] = ts.dt.hour.astype("int32")
        X["dow"] = ts.dt.dayofweek.astype("int32")   # 0=Monday, 6=Sunday
        X["month"] = ts.dt.month.astype("int32")
        return X.drop(columns=["timestamp"])
