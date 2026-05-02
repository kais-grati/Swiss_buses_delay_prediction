from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ml.preprocessors.base import BasePreprocessor


class FeatureScaler(BasePreprocessor):
    def __init__(self, cols: List[str]):
        self.cols = cols
        self._scaler = StandardScaler()

    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        self._scaler.fit(X[self.cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.cols] = self._scaler.transform(X[self.cols])
        return X
