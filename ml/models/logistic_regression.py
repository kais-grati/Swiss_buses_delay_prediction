import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ml.models.base import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: str | None = "balanced",
    ):
        self._model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=42
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
