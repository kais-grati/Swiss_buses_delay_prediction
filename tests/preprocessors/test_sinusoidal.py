import numpy as np
import pandas as pd
import pytest
from ml.preprocessors.sinusoidal import SinusoidalExpander

COLS = ["temperature", "precipitation"]


def test_sin_cos_columns_added(sample_X):
    expander = SinusoidalExpander(cols=COLS, n_components=2)
    result = expander.fit_transform(sample_X)
    for col in COLS:
        for k in range(1, 3):
            assert f"{col}_sin{k}" in result.columns
            assert f"{col}_cos{k}" in result.columns


def test_original_cols_dropped(sample_X):
    expander = SinusoidalExpander(cols=COLS, n_components=2)
    result = expander.fit_transform(sample_X)
    for col in COLS:
        assert col not in result.columns


def test_output_column_count(sample_X):
    expander = SinusoidalExpander(cols=COLS, n_components=3)
    result = expander.fit_transform(sample_X)
    # each col → 2*n_components new cols, originals dropped → net +2*n_components - len(cols)
    expected = sample_X.shape[1] + (2 * 3 - 1) * len(COLS)
    assert result.shape[1] == expected


def test_sin_cos_values_correct(sample_X):
    expander = SinusoidalExpander(cols=["temperature"], n_components=2)
    result = expander.fit_transform(sample_X)
    np.testing.assert_allclose(result["temperature_sin1"].values, np.sin(sample_X["temperature"].values), rtol=1e-5)
    np.testing.assert_allclose(result["temperature_cos2"].values, np.cos(2 * sample_X["temperature"].values), rtol=1e-5)


def test_passthrough_cols_unchanged(sample_X):
    expander = SinusoidalExpander(cols=COLS, n_components=1)
    result = expander.fit_transform(sample_X)
    for col in sample_X.columns:
        if col not in COLS:
            pd.testing.assert_series_equal(result[col], sample_X[col])


def test_fit_is_stateless(sample_X):
    expander = SinusoidalExpander(cols=COLS, n_components=2)
    result1 = expander.fit_transform(sample_X)
    result2 = expander.transform(sample_X)
    pd.testing.assert_frame_equal(result1, result2)
