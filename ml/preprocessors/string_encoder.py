"""Encode categorical string columns as integer labels.

Fits a {value → int} mapping per column.  Unseen values at transform time
receive a shared <UNK> token so the preprocessor never crashes on new data.
"""

from typing import Dict, List
import pandas as pd
from ml.preprocessors.base import BasePreprocessor


class StringEncoder(BasePreprocessor):
    def __init__(self, cols: List[str]):
        self.cols = cols
        self._mappings: Dict[str, Dict[str, int]] = {}

    def fit(self, X: pd.DataFrame) -> "StringEncoder":
        for col in self.cols:
            unique = X[col].dropna().unique()
            self._mappings[col] = {str(v): i for i, v in enumerate(unique)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            mapping = self._mappings[col]
            unk = len(mapping)  # <UNK> token = one past the last known index
            X[col] = X[col].map(lambda v: mapping.get(str(v), unk) if pd.notna(v) else unk)
        return X
