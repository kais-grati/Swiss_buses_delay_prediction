from typing import List
import numpy as np
import pandas as pd
from ml.preprocessors.base import BasePreprocessor


class SinusoidalExpander(BasePreprocessor):
    def __init__(self, cols: List[str], n_components: int = 3):
        self.cols = cols
        self.n_components = n_components

    def fit(self, X: pd.DataFrame) -> "SinusoidalExpander":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new_cols = {}
        for col in self.cols:
            for k in range(1, self.n_components + 1):
                new_cols[f"{col}_sin{k}"] = np.sin(k * X[col])
                new_cols[f"{col}_cos{k}"] = np.cos(k * X[col])
        return pd.concat(
            [X.drop(columns=self.cols), pd.DataFrame(new_cols, index=X.index)],
            axis=1,
        )
