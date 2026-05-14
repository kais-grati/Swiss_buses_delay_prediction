import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from ml.models.base import BaseModel, _MODEL_REGISTRY, _lookup


class LogTargetModel(BaseModel):
    """Wraps any regressor to predict on log-transformed target.

    Fits the inner model on y' = log(y + offset), then reverse-transforms
    predictions via exp(ŷ') - offset.

    The offset is auto-computed during fit to ensure y + offset > 0 for
    all training values, handling negative delays (early arrivals).
    """

    def __init__(self, model: BaseModel, offset: float = 1.0):
        self._model = model
        self._offset = offset

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogTargetModel":
        y_arr = y.values
        y_min = y_arr.min()
        if y_min <= -self._offset:
            self._offset = float(-y_min + 1.0)
        y_log = np.log(y_arr + self._offset)
        self._model.fit(X, pd.Series(y_log, index=y.index))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred_log = self._model.predict(X)
        return np.exp(pred_log) - self._offset

    def save(self, path):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "type": "LogTargetModel",
            "offset": self._offset,
            "inner_type": type(self._model).__name__,
        }
        if type(self._model).__name__ in _MODEL_REGISTRY:
            self._model.save(root / "inner")
        else:
            joblib.dump(self._model, root / "inner.joblib")
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path):
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        inner_name = manifest["inner_type"]
        inner = (
            _lookup(inner_name).load(root / "inner")
            if inner_name in _MODEL_REGISTRY
            else joblib.load(root / "inner.joblib")
        )
        return cls(model=inner, offset=manifest.get("offset", 1.0))


from ml.models.base import _register
_register("LogTargetModel", LogTargetModel)
