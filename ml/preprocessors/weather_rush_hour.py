import pandas as pd
from ml.preprocessors.base import BasePreprocessor

class WeatherRushHourPreprocessor(BasePreprocessor):
    """
    Creates interaction features between weather conditions and rush hour windows.

    Rush hour is defined as:
    - Morning: 07:00 - 09:00
    - Evening: 16:00 - 19:00
    """
    def fit(self, X: pd.DataFrame) -> "WeatherRushHourPreprocessor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Extract hour from timestamp if not already present
        if "hour" not in X.columns and "timestamp" in X.columns:
            ts = pd.to_datetime(X["timestamp"])
            hour = ts.dt.hour
        elif "hour" in X.columns:
            hour = X["hour"]
        else:
            raise KeyError("Neither 'hour' nor 'timestamp' column found for rush hour calculation")

        # Define rush hour boolean
        is_rush_hour = (
            ((hour >= 7) & (hour <= 9)) |
            ((hour >= 16) & (hour <= 19))
        )

        # Weather stress indicators
        #Severe weather is defined as precipitation > 1.0 or snow_depth > 0
        is_adverse = (X["precipitation"] > 1.0) | (X["snow_depth"] > 0)

        X["is_rush_hour"] = is_rush_hour.astype("int8")
        X["weather_rush_hour_stress"] = (is_rush_hour & is_adverse).astype("int8")

        return X
