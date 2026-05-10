import copy
import json
import joblib
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ml.models.base import BaseModel, _MODEL_REGISTRY, _lookup


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

    def save(self, path):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "type": "StackingModel",
            "n_folds": self._n_folds,
            "base_models": [],
            "meta_model": None,
        }
        for i, m in enumerate(self._base_models):
            sub = root / f"base_{i}"
            sub.mkdir(exist_ok=True)
            m.save(sub / "model")
            manifest["base_models"].append(type(m).__name__)
        meta_type = type(self._meta_model).__name__
        if meta_type in _MODEL_REGISTRY:
            self._meta_model.save(root / "meta")
        else:
            joblib.dump(self._meta_model, root / "meta.joblib")
        manifest["meta_model"] = meta_type
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path):
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        base_models = [
            _lookup(name).load(root / f"base_{i}" / "model")
            for i, name in enumerate(manifest["base_models"])
        ]
        meta_name = manifest["meta_model"]
        if meta_name in _MODEL_REGISTRY:
            meta_model = _lookup(meta_name).load(root / "meta")
        else:
            meta_model = joblib.load(root / "meta.joblib")
        instance = cls(base_models=base_models, meta_model=meta_model,
                       n_folds=manifest.get("n_folds", 5))
        return instance


from ml.models.base import _register
_register("StackingModel", StackingModel)
