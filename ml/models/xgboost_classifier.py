import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from ml.models.base import ClassifierModel


class XGBoostClassifierModel(ClassifierModel):
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float | None = None,
        class_weight: str | None = "balanced",
        early_stopping_rounds: int = 50,
        val_fraction: float = 0.1,
    ):
        self._early_stopping_rounds = early_stopping_rounds
        self._val_fraction = val_fraction
        self._class_weight = class_weight
        xgb_kwargs = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            verbosity=0,
            n_jobs=-1,
        )
        if scale_pos_weight is not None:
            xgb_kwargs["scale_pos_weight"] = scale_pos_weight
        self._model = xgb.XGBClassifier(**xgb_kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostClassifierModel":
        # XGBoost doesn't natively support class_weight="balanced" for multi-class;
        # compute sample weights manually instead.
        sample_weight = (
            compute_sample_weight("balanced", y) if self._class_weight == "balanced" else None
        )

        if self._early_stopping_rounds > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=self._val_fraction, random_state=42, stratify=y
            )
            w_tr = (
                compute_sample_weight("balanced", y_tr)
                if self._class_weight == "balanced"
                else None
            )
            self._model.set_params(early_stopping_rounds=self._early_stopping_rounds)
            self._model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self._model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, path):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self._model.get_booster().save_model(str(root / "booster.ubj"))
        meta = {"n_classes": self._model.n_classes_}
        (root / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path, **init_kwargs):
        root = Path(path)
        meta = json.loads((root / "meta.json").read_text())
        model = cls(**init_kwargs)
        model._model._Booster = xgb.Booster(model_file=str(root / "booster.ubj"))
        model._model._n_features_in = model._model._Booster.num_features()
        model._model.n_classes_ = meta["n_classes"]
        return model


from ml.models.base import _register
_register("XGBoostClassifierModel", XGBoostClassifierModel)
