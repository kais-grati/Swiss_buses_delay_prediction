from typing import List
import pandas as pd
from ml.preprocessors.base import BasePreprocessor


class HistoricalMeanEncoder(BasePreprocessor):
    """Target encodes a combination of columns as the mean of y per group.

    Must be fit with y to avoid leakage — only fit on training data.
    Groups with no training examples fall back to the global training mean.
    """

    def __init__(self, group_cols: List[str], output_col: str = "hist_mean_delay"):
        self.group_cols = group_cols
        self.output_col = output_col
        self._lookup: dict = {}
        self._global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "HistoricalMeanEncoder":
        assert y is not None, "HistoricalMeanEncoder.fit requires y"
        df = X[self.group_cols].copy()
        df["_y"] = y.values
        grouped = df.groupby(self.group_cols)["_y"].mean()
        self._lookup = {k: float(v) for k, v in grouped.items()}
        self._global_mean = float(y.mean())
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if len(self.group_cols) == 1:
            col = self.group_cols[0]
            X[self.output_col] = X[col].map(self._lookup).fillna(self._global_mean)
        else:
            keys = list(zip(*[X[col] for col in self.group_cols]))
            X[self.output_col] = [self._lookup.get(k, self._global_mean) for k in keys]
        return X
