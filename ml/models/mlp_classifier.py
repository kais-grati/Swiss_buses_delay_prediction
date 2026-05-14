import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from ml.models.base import ClassifierModel


class MLPClassifierModel(ClassifierModel):
    def __init__(
        self,
        hidden_layer_sizes: tuple = (128, 64, 32),
        activation: str = "relu",
        alpha: float = 0.001,
        learning_rate_init: float = 0.001,
        learning_rate: str = "adaptive",
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        max_iter: int = 500,
        batch_size: int = 64,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        random_state: int = 42,
    ):
        self._model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            max_iter=max_iter,
            batch_size=batch_size,
            beta_1=beta_1,
            beta_2=beta_2,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPClassifierModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, path):
        joblib.dump(self._model, str(path))

    @classmethod
    def load(cls, path, **init_kwargs):
        model = cls(**init_kwargs)
        model._model = joblib.load(str(path))
        return model


from ml.models.base import _register
_register("MLPClassifierModel", MLPClassifierModel)
