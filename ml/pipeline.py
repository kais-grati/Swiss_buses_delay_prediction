from typing import List
import numpy as np
import pandas as pd
from ml.preprocessors.base import BasePreprocessor
from ml.models.base import BaseModel


class MLPipeline:
    def __init__(self, preprocessors: List[BasePreprocessor], model: BaseModel):
        self.preprocessors = preprocessors
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPipeline":
        for preprocessor in self.preprocessors:
            X = preprocessor.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        for preprocessor in self.preprocessors:
            X = preprocessor.transform(X)
        return self.model.predict(X)
