import pandas as pd
import pytest
from ml.preprocessors.polynomial import PolynomialExpander

COLS = ["temperature", "precipitation"]


def test_squared_and_interaction_terms_added(sample_X):
    expander = PolynomialExpander(cols=COLS, degree=2)
    result = expander.fit_transform(sample_X)
    assert "temperature^2" in result.columns
    assert "temperature precipitation" in result.columns
    assert "precipitation^2" in result.columns


def test_output_wider_than_input(sample_X):
    expander = PolynomialExpander(cols=COLS, degree=2)
    result = expander.fit_transform(sample_X)
    # 2 original COLS → 5 poly features (t, p, t^2, t*p, p^2), net +3 columns
    assert result.shape[1] == sample_X.shape[1] + 3


def test_passthrough_cols_unchanged(sample_X):
    expander = PolynomialExpander(cols=COLS, degree=2)
    result = expander.fit_transform(sample_X)
    for col in sample_X.columns:
        if col not in COLS:
            pd.testing.assert_series_equal(result[col], sample_X[col])


def test_degree_1_same_column_count(sample_X):
    expander = PolynomialExpander(cols=COLS, degree=1)
    result = expander.fit_transform(sample_X)
    assert result.shape[1] == sample_X.shape[1]
