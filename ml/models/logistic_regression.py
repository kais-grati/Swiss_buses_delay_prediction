import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ml.models.base import ClassifierModel


class LogisticRegressionModel(ClassifierModel):
    def __init__(
        self,
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 2000,
        class_weight: str | None = "balanced",
        l1_ratio: float | None = None,
    ):
        # saga is required for ElasticNet (l1_ratio between 0 and 1)
        if l1_ratio is not None and solver != "saga":
            solver = "saga"

        kwargs = dict(C=C, solver=solver, max_iter=max_iter,
                      class_weight=class_weight, random_state=42)
        if l1_ratio is not None:
            kwargs["l1_ratio"] = l1_ratio
        self._model = LogisticRegression(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)
