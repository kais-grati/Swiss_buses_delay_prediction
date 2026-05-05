import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from ml.models.base import BaseModel


class XGBoostClassifierModel(BaseModel):
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
        class_weight: str | None = "balanced",
        early_stopping_rounds: int = 0,
        val_fraction: float = 0.1,
    ):
        self._early_stopping_rounds = early_stopping_rounds
        self._val_fraction = val_fraction
        self._class_weight = class_weight
        self._model = xgb.XGBClassifier(
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
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )

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
