from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluator:
    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = ""
    ) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_true, y_pred)
        label = f"{model_name} | " if model_name else ""
        print(f"{label}MSE: {mse:.2f} | RMSE: {rmse:.2f}s | R²: {r2:.4f}")
        return {"mse": mse, "rmse": rmse, "r2": r2}
