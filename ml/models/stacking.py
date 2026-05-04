import copy
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ml.models.base import BaseModel


class StackingModel(BaseModel):
    def __init__(
        self,
        base_models: List[BaseModel],
        meta_model: BaseModel,
        n_folds: int = 5,
    ):
        self._base_models = base_models
        self._meta_model = meta_model
        self._n_folds = n_folds

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingModel":
        oof = np.zeros((len(X), len(self._base_models)))
        kf = KFold(n_splits=self._n_folds, shuffle=True, random_state=42)
        X_arr = X.reset_index(drop=True)
        y_arr = y.reset_index(drop=True)

        for i, model in enumerate(self._base_models):
            for train_idx, val_idx in kf.split(X_arr):
                fold_model = copy.deepcopy(model)
                fold_model.fit(X_arr.iloc[train_idx], y_arr.iloc[train_idx])
                oof[val_idx, i] = fold_model.predict(X_arr.iloc[val_idx])

        # Refit each base model on the full training data
        for model in self._base_models:
            model.fit(X_arr, y_arr)

        self._meta_model.fit(pd.DataFrame(oof), y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = np.column_stack([m.predict(X) for m in self._base_models])
        return self._meta_model.predict(pd.DataFrame(base_preds))
