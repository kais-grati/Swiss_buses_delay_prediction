from typing import Dict, Union
from ml.data import DataLoader
from ml.pipeline import MLPipeline
from ml.evaluation import Evaluator
from ml.preprocessors.class_encoder import ClassEncoder

class Experiment:
    def __init__(
        self, loader: DataLoader, pipeline: MLPipeline, evaluator: Evaluator
    ):
        self.loader = loader
        self.pipeline = pipeline
        self.evaluator = evaluator

    def run(self) -> Dict[str, Union[float, str]]:
        X_train, X_test, y_train, y_test = self.loader.load()
        self.pipeline.fit(X_train, y_train)
        predictions = self.pipeline.predict(X_test)
        return self.evaluator.evaluate(
            y_test,
            predictions,
            model_name=type(self.pipeline.model).__name__,
        )

class ClassificationExperiment(Experiment):
    def __init__(
        self,
        loader: DataLoader,
        pipeline: MLPipeline,
        evaluator: Evaluator,
        encoder: ClassEncoder
    ):
        super().__init__(loader, pipeline, evaluator)
        self.encoder = encoder

    def run(self) -> Dict[str, Union[float, str]]:
        X_train, X_test, y_train, y_test = self.loader.load()

        # Bin the target variables
        y_train_encoded = self.encoder.encode(y_train)
        y_test_encoded = self.encoder.encode(y_test)

        self.pipeline.fit(X_train, y_train_encoded)
        predictions = self.pipeline.predict(X_test)

        return self.evaluator.evaluate(
            y_test_encoded,
            predictions,
            model_name=type(self.pipeline.model).__name__,
            is_classification=True,
        )
