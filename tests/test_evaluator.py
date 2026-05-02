import numpy as np
import pytest
from ml.evaluation import Evaluator


def test_mse_is_correct():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_true, y_pred, model_name="Test")
    expected_mse = (10**2 + 10**2 + 10**2) / 3
    assert metrics["mse"] == pytest.approx(expected_mse)


def test_rmse_is_sqrt_of_mse():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_true, y_pred, model_name="Test")
    assert metrics["rmse"] == pytest.approx(metrics["mse"] ** 0.5)


def test_returns_dict_with_mse_rmse_r2():
    evaluator = Evaluator()
    metrics = evaluator.evaluate(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), model_name="X"
    )
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics


def test_r2_perfect_predictions():
    y = np.array([100.0, 200.0, 300.0])
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y, y, model_name="Test")
    assert metrics["r2"] == pytest.approx(1.0)


def test_r2_baseline_predictions():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.full(3, y_true.mean())
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_true, y_pred, model_name="Test")
    assert metrics["r2"] == pytest.approx(0.0)
