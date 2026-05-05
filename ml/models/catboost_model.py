import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from ml.models.base import BaseModel

class CatBoostModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 1.0,
        border_count: int = 254,
        min_data_in_leaf: int = 1,
        early_stopping_rounds: int = 0,
        val_fraction: float = 0.1,
        num_threads: int = -1,
    ):
        self._early_stopping_rounds = early_stopping_rounds
        self._val_fraction = val_fraction
        self._model = CatBoostRegressor(
            iterations=n_estimators,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            bagging_temperature=bagging_temperature,
            border_count=border_count,
            min_data_in_leaf=min_data_in_leaf,
            loss_function="RMSE",
            eval_metric="RMSE",
            verbose=False,
            thread_count=num_threads,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostModel":
        if self._early_stopping_rounds > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=self._val_fraction, random_state=42
            )
            self._model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self._early_stopping_rounds,
                use_best_model=True
            )
        else:
            self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
