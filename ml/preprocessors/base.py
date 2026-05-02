from abc import ABC, abstractmethod
import pandas as pd


class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BasePreprocessor":
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
