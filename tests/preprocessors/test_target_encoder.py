import numpy as np
import pandas as pd
import pytest
from ml.preprocessors.target_encoder import HistoricalMeanEncoder


@pytest.fixture
def X_groups():
    return pd.DataFrame({
        "hour": [8, 8, 17, 17, 8],
        "dow":  [0, 0,  0,  0, 1],
    })


@pytest.fixture
def y_groups():
    return pd.Series([100.0, 200.0, 50.0, 150.0, 80.0])


def test_output_col_added(X_groups, y_groups):
    enc = HistoricalMeanEncoder(group_cols=["hour", "dow"])
    result = enc.fit_transform(X_groups, y_groups)
    assert "hist_mean_delay" in result.columns


def test_mean_values_correct(X_groups, y_groups):
    enc = HistoricalMeanEncoder(group_cols=["hour", "dow"])
    result = enc.fit_transform(X_groups, y_groups)
    # (hour=8, dow=0) mean = (100+200)/2 = 150
    assert result.loc[0, "hist_mean_delay"] == pytest.approx(150.0)
    assert result.loc[1, "hist_mean_delay"] == pytest.approx(150.0)
    # (hour=17, dow=0) mean = (50+150)/2 = 100
    assert result.loc[2, "hist_mean_delay"] == pytest.approx(100.0)


def test_unseen_group_falls_back_to_global_mean(X_groups, y_groups):
    enc = HistoricalMeanEncoder(group_cols=["hour", "dow"])
    enc.fit(X_groups, y_groups)
    X_new = pd.DataFrame({"hour": [23], "dow": [6]})  # unseen group
    result = enc.transform(X_new)
    assert result.loc[0, "hist_mean_delay"] == pytest.approx(y_groups.mean())


def test_single_group_col():
    X = pd.DataFrame({"hour": [8, 8, 17]})
    y = pd.Series([100.0, 200.0, 60.0])
    enc = HistoricalMeanEncoder(group_cols=["hour"])
    result = enc.fit_transform(X, y)
    assert result.loc[0, "hist_mean_delay"] == pytest.approx(150.0)
    assert result.loc[2, "hist_mean_delay"] == pytest.approx(60.0)


def test_original_cols_unchanged(X_groups, y_groups):
    enc = HistoricalMeanEncoder(group_cols=["hour", "dow"])
    result = enc.fit_transform(X_groups, y_groups)
    pd.testing.assert_series_equal(result["hour"], X_groups["hour"])
    pd.testing.assert_series_equal(result["dow"], X_groups["dow"])


def test_requires_y_in_fit():
    enc = HistoricalMeanEncoder(group_cols=["hour"])
    with pytest.raises(AssertionError):
        enc.fit(pd.DataFrame({"hour": [8]}))
