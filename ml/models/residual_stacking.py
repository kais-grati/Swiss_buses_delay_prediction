import copy
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ml.models.base import BaseModel, _MODEL_REGISTRY, _lookup


class ResidualStackingModel(BaseModel):
    """Two-stage residual stacking regressor.

    Stage 1 predicts the target. Stage 2 predicts the residuals (errors)
    of stage 1, using K-fold out-of-fold predictions to avoid overfitting.

    Final prediction = stage1.predict(X) + stage2.predict(X)
    """

    def __init__(
        self,
        stage1_model: BaseModel,
        stage2_model: BaseModel,
        n_folds: int = 5,
    ):
        self._stage1_model = stage1_model
        self._stage2_model = stage2_model
        self._n_folds = n_folds

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ResidualStackingModel":
        kf = KFold(n_splits=self._n_folds, shuffle=True, random_state=42)
        X_arr = X.reset_index(drop=True)
        y_arr = y.reset_index(drop=True)
        residuals = np.zeros(len(X_arr))

        # Generate OOF predictions from stage 1
        for train_idx, val_idx in kf.split(X_arr):
            fold_model = copy.deepcopy(self._stage1_model)
            fold_model.fit(X_arr.iloc[train_idx], y_arr.iloc[train_idx])
            residuals[val_idx] = (
                y_arr.iloc[val_idx].values - fold_model.predict(X_arr.iloc[val_idx])
            )

        # Fit stage 1 on full training data
        self._stage1_model.fit(X_arr, y_arr)

        # Fit stage 2 on the OOF residuals
        self._stage2_model.fit(X_arr, pd.Series(residuals, index=y_arr.index))

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred_stage1 = self._stage1_model.predict(X)
        pred_stage2 = self._stage2_model.predict(X)
        return pred_stage1 + pred_stage2

    def save(self, path):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "type": "ResidualStackingModel",
            "n_folds": self._n_folds,
            "stage1_type": type(self._stage1_model).__name__,
            "stage2_type": type(self._stage2_model).__name__,
        }
        if type(self._stage1_model).__name__ in _MODEL_REGISTRY:
            self._stage1_model.save(root / "stage1")
        else:
            joblib.dump(self._stage1_model, root / "stage1.joblib")
        if type(self._stage2_model).__name__ in _MODEL_REGISTRY:
            self._stage2_model.save(root / "stage2")
        else:
            joblib.dump(self._stage2_model, root / "stage2.joblib")
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path):
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        s1_name = manifest["stage1_type"]
        s2_name = manifest["stage2_type"]
        stage1 = (
            _lookup(s1_name).load(root / "stage1")
            if s1_name in _MODEL_REGISTRY
            else joblib.load(root / "stage1.joblib")
        )
        stage2 = (
            _lookup(s2_name).load(root / "stage2")
            if s2_name in _MODEL_REGISTRY
            else joblib.load(root / "stage2.joblib")
        )
        return cls(
            stage1_model=stage1,
            stage2_model=stage2,
            n_folds=manifest.get("n_folds", 5),
        )


from ml.models.base import _register
_register("ResidualStackingModel", ResidualStackingModel)
