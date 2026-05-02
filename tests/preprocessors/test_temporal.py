import pandas as pd
import pytest
from ml.preprocessors.temporal import TemporalFeatureExtractor


@pytest.fixture
def X_with_timestamp():
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-03-10 08:30:00",  # Monday, hour=8, dow=0, month=3
            "2025-03-15 17:45:00",  # Saturday, hour=17, dow=5, month=3
            "2025-07-04 00:15:00",  # Friday, hour=0, dow=4, month=7
        ]),
        "temperature": [10.0, 20.0, 30.0],
    })


def test_timestamp_dropped(X_with_timestamp):
    result = TemporalFeatureExtractor().fit_transform(X_with_timestamp)
    assert "timestamp" not in result.columns


def test_hour_extracted(X_with_timestamp):
    result = TemporalFeatureExtractor().fit_transform(X_with_timestamp)
    assert list(result["hour"]) == [8, 17, 0]


def test_dow_extracted(X_with_timestamp):
    result = TemporalFeatureExtractor().fit_transform(X_with_timestamp)
    assert list(result["dow"]) == [0, 5, 4]


def test_month_extracted(X_with_timestamp):
    result = TemporalFeatureExtractor().fit_transform(X_with_timestamp)
    assert list(result["month"]) == [3, 3, 7]


def test_other_columns_unchanged(X_with_timestamp):
    result = TemporalFeatureExtractor().fit_transform(X_with_timestamp)
    pd.testing.assert_series_equal(result["temperature"], X_with_timestamp["temperature"])


def test_fit_is_stateless(X_with_timestamp):
    enc = TemporalFeatureExtractor()
    r1 = enc.fit_transform(X_with_timestamp)
    r2 = enc.transform(X_with_timestamp)
    pd.testing.assert_frame_equal(r1, r2)
