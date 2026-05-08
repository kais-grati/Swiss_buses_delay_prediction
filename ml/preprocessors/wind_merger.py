import pandas as pd
from ml.preprocessors.base import BasePreprocessor


class WindMerger(BasePreprocessor):
    """Merge wind_speed and wind_gust (r=0.956) into a single wind field.

    Uses the mean of both columns since gust is almost always >= speed
    and they carry near-identical information.
    """

    def fit(self, X: pd.DataFrame) -> "WindMerger":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["wind"] = (X["wind_speed"] + X["wind_gust"]) / 2
        return X.drop(columns=["wind_speed", "wind_gust"])
