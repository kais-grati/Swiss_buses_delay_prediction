from typing import Optional
import pandas as pd
from sklearn.decomposition import PCA
from ml.preprocessors.base import BasePreprocessor


class PCAReducer(BasePreprocessor):
    def __init__(self, n_components: Optional[int] = None, variance_threshold: Optional[float] = None):
        if n_components is None and variance_threshold is None:
            raise ValueError("Provide either n_components or variance_threshold")
        self._pca = PCA(n_components=n_components if n_components is not None else variance_threshold)

    def fit(self, X: pd.DataFrame) -> "PCAReducer":
        self._pca.fit(X)
        self._cols = [f"pc{i+1}" for i in range(self._pca.n_components_)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self._pca.transform(X), columns=self._cols, index=X.index)
