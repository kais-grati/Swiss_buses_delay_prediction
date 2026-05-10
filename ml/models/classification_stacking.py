import copy
import json
import joblib
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ml.models.base import ClassifierModel, _MODEL_REGISTRY, _lookup


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

    def save(self, path):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "type": "ClassificationStackingModel",
            "n_folds": self._n_folds,
            "n_classes": self._n_classes,
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
        instance._n_classes = manifest.get("n_classes")
        return instance


from ml.models.base import _register
_register("ClassificationStackingModel", ClassificationStackingModel)
