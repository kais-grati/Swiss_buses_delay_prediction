from typing import Optional
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from ml.preprocessors.base import BasePreprocessor


class NystroemExpander(BasePreprocessor):
    def __init__(
        self,
        n_components: int = 100,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        random_state: int = 42,
    ):
        self._nystroem = Nystroem(
            kernel=kernel,
            gamma=gamma,
            n_components=n_components,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame) -> "NystroemExpander":
        self._nystroem.fit(X)
        n = self._nystroem.components_.shape[0]
        self._cols = [f"nys{i+1}" for i in range(n)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self._nystroem.transform(X), columns=self._cols, index=X.index)
