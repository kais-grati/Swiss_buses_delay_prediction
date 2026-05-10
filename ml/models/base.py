from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def save(self, path: str | Path) -> None:
        """Serialize model to disk.  Override with native format where possible."""
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: str | Path, **init_kwargs) -> "BaseModel":
        """Deserialize model from disk.  Override with native format where possible."""
        return joblib.load(str(path))


# Registry for polymorphic load() dispatch in composite models
_MODEL_REGISTRY: dict[str, type] = {}

def _register(name: str, cls: type) -> None:
    _MODEL_REGISTRY[name] = cls

def _lookup(name: str) -> type:
    return _MODEL_REGISTRY[name]


class ClassifierModel(BaseModel):
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...
