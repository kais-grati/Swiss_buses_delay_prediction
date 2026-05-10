import json
from pathlib import Path
from typing import List
import joblib
import numpy as np
import pandas as pd
from ml.models.base import ClassifierModel, _lookup
from ml.preprocessors.base import BasePreprocessor


class PipelinedClassifierModel(ClassifierModel):
    """ClassifierModel that applies its own preprocessor chain before fit/predict.

    Useful as a base model inside ClassificationStackingModel when one base
    model needs different feature transformations than the others.
    """

    def __init__(self, preprocessors: List[BasePreprocessor], classifier: ClassifierModel):
        self._preprocessors = preprocessors
        self._classifier = classifier

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for p in self._preprocessors:
            X = p.transform(X)
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PipelinedClassifierModel":
        for p in self._preprocessors:
            X = p.fit_transform(X, y)
        self._classifier.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._classifier.predict(self._transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._classifier.predict_proba(self._transform(X))

    def save(self, path):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(self._preprocessors):
            joblib.dump(p, root / f"preprocessor_{i}.joblib")
        self._classifier.save(root / "classifier")
        manifest = {
            "type": "PipelinedClassifierModel",
            "classifier_type": type(self._classifier).__name__,
            "n_preprocessors": len(self._preprocessors),
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path):
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        preprocessors = [
            joblib.load(root / f"preprocessor_{i}.joblib")
            for i in range(manifest["n_preprocessors"])
        ]
        classifier = _lookup(manifest["classifier_type"]).load(root / "classifier")
        return cls(preprocessors=preprocessors, classifier=classifier)


from ml.models.base import _register
_register("PipelinedClassifierModel", PipelinedClassifierModel)
