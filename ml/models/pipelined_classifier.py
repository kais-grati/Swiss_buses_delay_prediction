from typing import List
import numpy as np
import pandas as pd
from ml.models.base import ClassifierModel
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
