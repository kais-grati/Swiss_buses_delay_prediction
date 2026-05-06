from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from ml.preprocessors.base import BasePreprocessor


class PolyTrigExpander(BasePreprocessor):
    """
    Combines polynomial and trigonometric expansions on selected columns.

    Output for each expanded column set:
      - Degree-1 through degree-n polynomial terms (includes originals + interactions)
      - sin(k * x) and cos(k * x) for k = 1 .. n_trig harmonics

    Non-selected columns are passed through unchanged.
    """

    def __init__(self, cols: List[str], degree: int = 2, n_trig: int = 1):
        self.cols = cols
        self.degree = degree
        self.n_trig = n_trig
        self._poly = PolynomialFeatures(degree=degree, include_bias=False)

    def fit(self, X: pd.DataFrame) -> "PolyTrigExpander":
        self._poly.fit(X[self.cols])
        self._poly_names = list(self._poly.get_feature_names_out(self.cols))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        poly_df = pd.DataFrame(
            self._poly.transform(X[self.cols]),
            columns=self._poly_names,
            index=X.index,
        )

        trig_cols = {}
        for col in self.cols:
            for k in range(1, self.n_trig + 1):
                suffix = f"_{k}" if self.n_trig > 1 else ""
                trig_cols[f"{col}_sin{suffix}"] = np.sin(k * X[col])
                trig_cols[f"{col}_cos{suffix}"] = np.cos(k * X[col])
        trig_df = pd.DataFrame(trig_cols, index=X.index)

        passthrough = X.drop(columns=self.cols)
        return pd.concat([passthrough, poly_df, trig_df], axis=1)
