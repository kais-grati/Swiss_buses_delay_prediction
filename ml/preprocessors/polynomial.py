from typing import List
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from ml.preprocessors.base import BasePreprocessor


class PolynomialExpander(BasePreprocessor):
    def __init__(self, cols: List[str], degree: int = 2):
        self.cols = cols
        self.degree = degree
        self._poly = PolynomialFeatures(degree=degree, include_bias=False)

    def fit(self, X: pd.DataFrame) -> "PolynomialExpander":
        self._poly.fit(X[self.cols])
        self._feature_names = self._poly.get_feature_names_out(self.cols)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        poly_array = self._poly.transform(X[self.cols])
        poly_df = pd.DataFrame(poly_array, columns=self._feature_names, index=X.index)
        passthrough = X.drop(columns=self.cols)
        return pd.concat([passthrough, poly_df], axis=1)
