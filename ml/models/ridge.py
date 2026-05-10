import joblib
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

    def save(self, path):
        joblib.dump(self._model, str(path))

    @classmethod
    def load(cls, path, **init_kwargs):
        model = cls(**init_kwargs)
        model._model = joblib.load(str(path))
        return model


from ml.models.base import _register
_register("RidgeModel", RidgeModel)
