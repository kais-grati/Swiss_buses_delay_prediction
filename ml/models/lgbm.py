import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from ml.models.base import BaseModel


class LightGBMModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        max_depth: int = -1,
        max_bin: int = 255,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        min_gain_to_split: float = 0.0,
        num_threads: int = -1,
        early_stopping_rounds: int = 0,
        val_fraction: float = 0.1,
        log_every: int = 0,
    ):
        self._early_stopping_rounds = early_stopping_rounds
        self._val_fraction = val_fraction
        self._log_every = log_every
        self._model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            max_depth=max_depth,
            max_bin=max_bin,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_gain_to_split,
            n_jobs=num_threads,
            metric="rmse",
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        callbacks = []
        eval_set = None

        if self._early_stopping_rounds > 0 or self._log_every > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=self._val_fraction, random_state=42
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
