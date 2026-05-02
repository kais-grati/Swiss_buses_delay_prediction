import numpy as np
import pandas as pd
import xgboost as xgb
from ml.models.base import BaseModel


class XGBoostModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
    ):
        self._model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            verbosity=0,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
