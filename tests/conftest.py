import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def sample_X():
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "time_sin": np.random.uniform(-1, 1, n),
        "time_cos": np.random.uniform(-1, 1, n),
        "dow_sin": np.random.uniform(-1, 1, n),
        "dow_cos": np.random.uniform(-1, 1, n),
        "month_sin": np.random.uniform(-1, 1, n),
        "month_cos": np.random.uniform(-1, 1, n),
        "is_weekend": np.random.choice([True, False], n),
        "additional_trip": np.random.choice([True, False], n),
        "is_public_holiday": np.random.choice([True, False], n),
        "sunshine": np.random.uniform(0, 1, n),
        "temperature": np.random.uniform(-10, 35, n),
        "precipitation": np.random.uniform(0, 20, n),
        "humidity": np.random.uniform(20, 100, n),
        "wind_speed": np.random.uniform(0, 10, n),
        "wind_gust": np.random.uniform(0, 30, n),
        "pressure": np.random.uniform(950, 1013, n),
        "snow_depth": np.random.uniform(0, 5, n),
        "wind_dir": np.random.uniform(0, 360, n),
    })

@pytest.fixture
def sample_y(sample_X):
    np.random.seed(42)
    return pd.Series(
        np.random.randint(-120, 1800, len(sample_X)),
        name="arrival_delay_s",
        dtype=float,
    )
