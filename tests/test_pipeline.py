import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from ml.pipeline import MLPipeline
from ml.preprocessors.wind_encoder import WindDirectionEncoder
from ml.models.ridge import RidgeModel


def test_fit_calls_fit_transform_on_all_preprocessors(sample_X, sample_y):
    p1 = MagicMock()
    p1.fit_transform.return_value = sample_X
    p2 = MagicMock()
    p2.fit_transform.return_value = sample_X
    model = MagicMock()

    pipeline = MLPipeline(preprocessors=[p1, p2], model=model)
    pipeline.fit(sample_X, sample_y)

    p1.fit_transform.assert_called_once()
    p2.fit_transform.assert_called_once()
    model.fit.assert_called_once()


def test_predict_calls_transform_not_fit_transform(sample_X, sample_y):
    p1 = MagicMock()
    p1.fit_transform.return_value = sample_X
    p1.transform.return_value = sample_X
    model = MagicMock()
    model.predict.return_value = np.zeros(len(sample_X))

    pipeline = MLPipeline(preprocessors=[p1], model=model)
    pipeline.fit(sample_X, sample_y)
    pipeline.predict(sample_X)

    p1.fit_transform.assert_called_once()   # only during fit
    p1.transform.assert_called_once()       # only during predict


def test_end_to_end_with_real_components(sample_X, sample_y):
    pipeline = MLPipeline(
        preprocessors=[WindDirectionEncoder()],
        model=RidgeModel(alpha=1.0),
    )
    pipeline.fit(sample_X, sample_y)
    predictions = pipeline.predict(sample_X)
    assert predictions.shape == (len(sample_X),)
