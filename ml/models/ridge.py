import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from ml.models.base import BaseModel


class RidgeModel(BaseModel):
    def __init__(self, alpha: float = 1.0):
        self._model = Ridge(alpha=alpha)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
