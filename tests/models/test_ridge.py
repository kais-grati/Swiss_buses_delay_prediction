import numpy as np
import pytest
from ml.models.ridge import RidgeModel


def test_predict_returns_correct_shape(sample_X, sample_y):
    model = RidgeModel(alpha=1.0)
    model.fit(sample_X, sample_y)
    predictions = model.predict(sample_X)
    assert predictions.shape == (len(sample_X),)


def test_predictions_are_floats(sample_X, sample_y):
    model = RidgeModel(alpha=1.0)
    model.fit(sample_X, sample_y)
    predictions = model.predict(sample_X)
    assert predictions.dtype in [np.float32, np.float64]


def test_fit_returns_self(sample_X, sample_y):
    model = RidgeModel(alpha=1.0)
    result = model.fit(sample_X, sample_y)
    assert result is model
