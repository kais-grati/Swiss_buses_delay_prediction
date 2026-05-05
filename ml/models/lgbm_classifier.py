import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from ml.models.base import BaseModel


class LightGBMClassifierModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        min_sum_hessian_in_leaf: float = 1e-3,
        max_depth: int = -1,
        max_bin: int = 255,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        feature_fraction_bynode: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        min_gain_to_split: float = 0.0,
        path_smooth: float = 0.0,
        class_weight: str | None = "balanced",
        early_stopping_rounds: int = 0,
        val_fraction: float = 0.1,
        log_every: int = 0,
    ):
        self._early_stopping_rounds = early_stopping_rounds
        self._val_fraction = val_fraction
        self._log_every = log_every
        self._model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
            max_depth=max_depth,
            max_bin=max_bin,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            feature_fraction_bynode=feature_fraction_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_gain_to_split,
            path_smooth=path_smooth,
            class_weight=class_weight,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMClassifierModel":
        callbacks = []
        eval_set = None

        if self._early_stopping_rounds > 0 or self._log_every > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=self._val_fraction, random_state=42, stratify=y
            )
            eval_set = [(X_val, y_val)]
            if self._early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(self._early_stopping_rounds, verbose=False))
            if self._log_every > 0:
                callbacks.append(lgb.log_evaluation(period=self._log_every))
        else:
            X_tr, y_tr = X, y

        self._model.fit(X_tr, y_tr, eval_set=eval_set, callbacks=callbacks)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
