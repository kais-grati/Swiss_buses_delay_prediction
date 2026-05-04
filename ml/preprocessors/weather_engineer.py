import pandas as pd
from ml.preprocessors.base import BasePreprocessor


class WeatherFeatureEngineer(BasePreprocessor):
    def fit(self, X: pd.DataFrame) -> "WeatherFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["wind_chill"] = X["temperature"] - 0.7 * X["wind_speed"]
        X["adverse_weather"] = (
            (X["precipitation"] > 1.0) | (X["snow_depth"] > 0)
        ).astype("int8")
        return X
