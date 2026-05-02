import numpy as np
import pandas as pd
import pytest
from ml.preprocessors.scaler import FeatureScaler

COLS = ["temperature", "precipitation"]


def test_scaled_cols_have_zero_mean(sample_X):
    scaler = FeatureScaler(cols=COLS)
    result = scaler.fit_transform(sample_X)
    for col in COLS:
        assert abs(result[col].mean()) < 1e-10


def test_scaled_cols_have_unit_std(sample_X):
    scaler = FeatureScaler(cols=COLS)
    result = scaler.fit_transform(sample_X)
    for col in COLS:
        assert abs(result[col].std(ddof=0) - 1.0) < 1e-5


def test_unscaled_cols_unchanged(sample_X):
    scaler = FeatureScaler(cols=COLS)
    result = scaler.fit_transform(sample_X)
    for col in sample_X.columns:
        if col not in COLS:
            pd.testing.assert_series_equal(result[col], sample_X[col])


def test_transform_uses_train_statistics(sample_X):
    train = sample_X.iloc[:40].copy().reset_index(drop=True)
    test = sample_X.iloc[40:].copy().reset_index(drop=True)
    scaler = FeatureScaler(cols=["temperature"])
    scaler.fit(train)
    result = scaler.transform(test)
    train_mean = train["temperature"].mean()
    train_std = train["temperature"].std(ddof=0)
    expected = (test["temperature"] - train_mean) / train_std
    pd.testing.assert_series_equal(
        result["temperature"], expected, check_names=False, rtol=1e-5
    )
