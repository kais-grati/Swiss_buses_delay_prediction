import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ml.data import DataLoader

DROP_COLS = [
    "timestamp", "stop_id", "stop_name", "operator", "line", "departure_delay_s"
]


@pytest.fixture
def sample_parquet(tmp_path):
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "time_sin": np.random.uniform(-1, 1, n),
        "time_cos": np.random.uniform(-1, 1, n),
        "dow_sin": np.random.uniform(-1, 1, n),
        "dow_cos": np.random.uniform(-1, 1, n),
        "month_sin": np.random.uniform(-1, 1, n),
        "month_cos": np.random.uniform(-1, 1, n),
        "is_weekend": np.random.choice([True, False], n),
        "additional_trip": np.random.choice([True, False], n),
        "is_public_holiday": np.random.choice([True, False], n),
        "operator": ["MBC Auto"] * n,
        "line": ["705"] * n,
        "stop_id": [8592244] * n,
        "stop_name": ["Echandens, Chocolatière"] * n,
        "sunshine": np.random.uniform(0, 1, n),
        "temperature": np.random.uniform(-10, 35, n),
        "precipitation": np.random.uniform(0, 20, n),
        "humidity": np.random.uniform(20, 100, n),
        "wind_speed": np.random.uniform(0, 10, n),
        "wind_gust": np.random.uniform(0, 30, n),
        "pressure": np.random.uniform(950, 1013, n),
        "snow_depth": np.random.uniform(0, 5, n),
        "wind_dir": np.random.uniform(0, 360, n),
        "arrival_delay_s": np.random.randint(-120, 1800, n).astype(float),
        "departure_delay_s": np.random.randint(-120, 1800, n).astype(float),
    })
    path = tmp_path / "test_dataset.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return str(path)


def test_load_returns_correct_split_sizes(sample_parquet):
    loader = DataLoader(
        path=sample_parquet,
        target="arrival_delay_s",
        drop_cols=DROP_COLS,
        test_size=0.2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = loader.load()
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_dropped_columns_absent(sample_parquet):
    loader = DataLoader(
        path=sample_parquet,
        target="arrival_delay_s",
        drop_cols=DROP_COLS,
        test_size=0.2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = loader.load()
    for col in DROP_COLS:
        assert col not in X_train.columns
        assert col not in X_test.columns


def test_target_not_in_features(sample_parquet):
    loader = DataLoader(
        path=sample_parquet,
        target="arrival_delay_s",
        drop_cols=DROP_COLS,
        test_size=0.2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = loader.load()
    assert "arrival_delay_s" not in X_train.columns
    assert "arrival_delay_s" not in X_test.columns


def test_y_series_name_matches_target(sample_parquet):
    loader = DataLoader(
        path=sample_parquet,
        target="arrival_delay_s",
        drop_cols=DROP_COLS,
        test_size=0.2,
        random_state=42,
    )
    _, _, y_train, _ = loader.load()
    assert y_train.name == "arrival_delay_s"
