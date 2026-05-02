import numpy as np
import pandas as pd
import pytest
from ml.preprocessors.wind_encoder import WindDirectionEncoder


def test_wind_dir_replaced_by_sin_cos(sample_X):
    enc = WindDirectionEncoder()
    result = enc.fit_transform(sample_X)
    assert "wind_dir" not in result.columns
    assert "wind_dir_sin" in result.columns
    assert "wind_dir_cos" in result.columns


def test_sin_cos_values_correct(sample_X):
    enc = WindDirectionEncoder()
    result = enc.fit_transform(sample_X)
    radians = np.deg2rad(sample_X["wind_dir"])
    np.testing.assert_allclose(result["wind_dir_sin"].values, np.sin(radians).values, rtol=1e-5)
    np.testing.assert_allclose(result["wind_dir_cos"].values, np.cos(radians).values, rtol=1e-5)


def test_other_columns_unchanged(sample_X):
    enc = WindDirectionEncoder()
    result = enc.fit_transform(sample_X)
    for col in sample_X.columns:
        if col != "wind_dir":
            pd.testing.assert_series_equal(result[col], sample_X[col])


def test_fit_is_stateless(sample_X):
    enc = WindDirectionEncoder()
    result1 = enc.fit_transform(sample_X)
    result2 = enc.transform(sample_X)
    pd.testing.assert_frame_equal(result1, result2)
