import copy
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ml.models.base import ClassifierModel


class ClassificationStackingModel(ClassifierModel):
    def __init__(
        self,
        base_models: List[ClassifierModel],
        meta_model: ClassifierModel,
        n_folds: int = 5,
    ):
        if not base_models:
            raise ValueError("base_models must not be empty")
        self._base_models = base_models
        self._meta_model = meta_model
        self._n_folds = n_folds
        self._n_classes: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClassificationStackingModel":
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        self._n_classes = len(np.unique(y))
        n_meta_cols = len(self._base_models) * self._n_classes
        oof = np.zeros((len(X), n_meta_cols))

        skf = StratifiedKFold(n_splits=self._n_folds, shuffle=True, random_state=42)

        for i, model in enumerate(self._base_models):
            col_start = i * self._n_classes
            col_end = col_start + self._n_classes
            for train_idx, val_idx in skf.split(X, y):
                fold_model = copy.deepcopy(model)
                fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
                oof[val_idx, col_start:col_end] = fold_model.predict_proba(X.iloc[val_idx])

        for model in self._base_models:
            model.fit(X, y)

        self._meta_model.fit(pd.DataFrame(oof), y)
        return self

    def _meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        probas = [m.predict_proba(X) for m in self._base_models]
        return pd.DataFrame(np.hstack(probas))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._meta_model.predict(self._meta_features(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._meta_model.predict_proba(self._meta_features(X))
