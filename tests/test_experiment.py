import numpy as np
import pytest
from unittest.mock import MagicMock
from ml.experiment import Experiment
from ml.pipeline import MLPipeline
from ml.preprocessors.wind_encoder import WindDirectionEncoder
from ml.preprocessors.scaler import FeatureScaler
from ml.models.ridge import RidgeModel
from ml.models.lgbm import LightGBMModel
from ml.evaluation import Evaluator

NUMERIC_COLS = [
    "temperature", "precipitation", "humidity",
    "wind_speed", "wind_gust", "pressure", "snow_depth",
]


def _make_loader(sample_X, sample_y):
    loader = MagicMock()
    X_train = sample_X.iloc[:40].reset_index(drop=True)
    X_test = sample_X.iloc[40:].reset_index(drop=True)
    y_train = sample_y.iloc[:40].reset_index(drop=True)
    y_test = sample_y.iloc[40:].reset_index(drop=True)
    loader.load.return_value = (X_train, X_test, y_train, y_test)
    return loader


def test_experiment_run_returns_mse_and_rmse(sample_X, sample_y):
    experiment = Experiment(
        loader=_make_loader(sample_X, sample_y),
        pipeline=MLPipeline(
            preprocessors=[WindDirectionEncoder()],
            model=RidgeModel(alpha=1.0),
        ),
        evaluator=Evaluator(),
    )
    metrics = experiment.run()
    assert "mse" in metrics
    assert "rmse" in metrics
    assert metrics["mse"] >= 0


def test_experiment_with_lgbm(sample_X, sample_y):
    experiment = Experiment(
        loader=_make_loader(sample_X, sample_y),
        pipeline=MLPipeline(
            preprocessors=[
                WindDirectionEncoder(),
                FeatureScaler(cols=NUMERIC_COLS),
            ],
            model=LightGBMModel(n_estimators=10),
        ),
        evaluator=Evaluator(),
    )
    metrics = experiment.run()
    assert "mse" in metrics
    assert "rmse" in metrics
    assert metrics["mse"] >= 0


def test_experiment_run_calls_collaborators_in_order(sample_X, sample_y):
    X_train = sample_X.iloc[:40].reset_index(drop=True)
    X_test = sample_X.iloc[40:].reset_index(drop=True)
    y_train = sample_y.iloc[:40].reset_index(drop=True)
    y_test = sample_y.iloc[40:].reset_index(drop=True)

    loader = MagicMock()
    loader.load.return_value = (X_train, X_test, y_train, y_test)

    pipeline = MagicMock()
    pipeline.predict.return_value = np.zeros(len(X_test))
    pipeline.model.__class__.__name__ = "MockModel"

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"mse": 0.0, "rmse": 0.0}

    experiment = Experiment(loader=loader, pipeline=pipeline, evaluator=evaluator)
    result = experiment.run()

    loader.load.assert_called_once()
    pipeline.fit.assert_called_once_with(X_train, y_train)
    pipeline.predict.assert_called_once_with(X_test)
    evaluator.evaluate.assert_called_once()
    assert result == {"mse": 0.0, "rmse": 0.0}
