import numpy as np
import pandas as pd
from ml.preprocessors.base import BasePreprocessor


class WindDirectionEncoder(BasePreprocessor):
    def fit(self, X: pd.DataFrame) -> "WindDirectionEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        radians = np.deg2rad(X["wind_dir"])
        X["wind_dir_sin"] = np.sin(radians)
        X["wind_dir_cos"] = np.cos(radians)
        return X.drop(columns=["wind_dir"])
