from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


class ClassifierModel(BaseModel):
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...
