import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.models.base import ClassifierModel


class RandomForestClassifierModel(ClassifierModel):
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float = "sqrt",
        bootstrap: bool = True,
        class_weight: str | None = "balanced_subsample",
        early_stopping_rounds: int = 0,
        val_fraction: float = 0.1,
        random_state: int = 42,
    ):
        self._early_stopping_rounds = early_stopping_rounds
        self._val_fraction = val_fraction
        self._n_estimators = n_estimators
        self._rf_kwargs = dict(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
        )
        self._model = RandomForestClassifier(
            n_estimators=n_estimators if early_stopping_rounds == 0 else 1,
            warm_start=(early_stopping_rounds > 0),
            **self._rf_kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestClassifierModel":
        if self._early_stopping_rounds > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=self._val_fraction, random_state=42, stratify=y
            )

            rf = RandomForestClassifier(
                n_estimators=1, warm_start=True, **self._rf_kwargs
            )

            best_score = -np.inf
            best_n = 1
            no_improve = 0

            for i in range(1, self._n_estimators + 1):
                rf.set_params(n_estimators=i)
                rf.fit(X_tr, y_tr)
                val_score = rf.score(X_val, y_val)

                if val_score > best_score:
                    best_score = val_score
                    best_n = i
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= self._early_stopping_rounds:
                    break

            self._model = RandomForestClassifier(
                n_estimators=best_n, warm_start=False, **self._rf_kwargs
            )
            self._model.fit(X, y)
        else:
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
_register("RandomForestClassifierModel", RandomForestClassifierModel)
