import json
from pathlib import Path
from typing import List
import joblib
import numpy as np
import pandas as pd
from ml.preprocessors.base import BasePreprocessor
from ml.models.base import BaseModel, _lookup


class MLPipeline:
    def __init__(self, preprocessors: List[BasePreprocessor], model: BaseModel):
        self.preprocessors = preprocessors
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPipeline":
        for preprocessor in self.preprocessors:
            X = preprocessor.fit_transform(X, y)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        for preprocessor in self.preprocessors:
            X = preprocessor.transform(X)
        return self.model.predict(X)

    def save(self, path):
        """Save pipeline (preprocessors + model) to a directory."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(self.preprocessors):
            joblib.dump(p, root / f"preprocessor_{i}.joblib")
        self.model.save(root / "model")
        manifest = {
            "model_type": type(self.model).__name__,
            "n_preprocessors": len(self.preprocessors),
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path):
        """Load a previously saved pipeline."""
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        preprocessors = [
            joblib.load(root / f"preprocessor_{i}.joblib")
            for i in range(manifest["n_preprocessors"])
        ]
        model = _lookup(manifest["model_type"]).load(root / "model")
        return cls(preprocessors=preprocessors, model=model)
