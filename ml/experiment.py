from typing import Dict
from ml.data import DataLoader
from ml.pipeline import MLPipeline
from ml.evaluation import Evaluator


class Experiment:
    def __init__(
        self, loader: DataLoader, pipeline: MLPipeline, evaluator: Evaluator
    ):
        self.loader = loader
        self.pipeline = pipeline
        self.evaluator = evaluator

    def run(self) -> Dict[str, float]:
        X_train, X_test, y_train, y_test = self.loader.load()
        self.pipeline.fit(X_train, y_train)
        predictions = self.pipeline.predict(X_test)
        return self.evaluator.evaluate(
            y_test,
            predictions,
            model_name=type(self.pipeline.model).__name__,
        )
