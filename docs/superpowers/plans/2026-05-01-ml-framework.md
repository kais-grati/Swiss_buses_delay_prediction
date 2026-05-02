# ML Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular OOP regression framework with swappable preprocessors and models, targeting `arrival_delay_s` prediction on the Echandens dataset.

**Architecture:** Thin wrappers around sklearn/LightGBM behind abstract base classes (`BasePreprocessor`, `BaseModel`). An `MLPipeline` chains a list of preprocessors then a model. An `Experiment` wires a `DataLoader`, `MLPipeline`, and `Evaluator` into a single `run()` call.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, lightgbm, pyarrow, pytest

---

## File Map

```
ml/
  __init__.py
  data.py                      # DataLoader
  preprocessors/
    __init__.py
    base.py                    # abstract BasePreprocessor
    wind_encoder.py            # WindDirectionEncoder
    scaler.py                  # FeatureScaler
    polynomial.py              # PolynomialExpander
  models/
    __init__.py
    base.py                    # abstract BaseModel
    ridge.py                   # RidgeModel
    lgbm.py                    # LightGBMModel
  pipeline.py                  # MLPipeline
  evaluation.py                # Evaluator
  experiment.py                # Experiment

tests/
  __init__.py
  conftest.py                  # shared fixtures
  test_data.py
  test_pipeline.py
  test_evaluator.py
  test_experiment.py
  preprocessors/
    __init__.py
    test_base.py
    test_wind_encoder.py
    test_scaler.py
    test_polynomial.py
  models/
    __init__.py
    test_base.py
    test_ridge.py
    test_lgbm.py
```

---

## Task 1: Project Setup

**Files:**
- Create: `ml/__init__.py`, `ml/preprocessors/__init__.py`, `ml/models/__init__.py`
- Create: `tests/__init__.py`, `tests/preprocessors/__init__.py`, `tests/models/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p ml/preprocessors ml/models tests/preprocessors tests/models
touch ml/__init__.py ml/preprocessors/__init__.py ml/models/__init__.py
touch tests/__init__.py tests/preprocessors/__init__.py tests/models/__init__.py
```

- [ ] **Step 2: Verify lightgbm is installed**

```bash
source venv/bin/activate && python3 -c "import lightgbm; print(lightgbm.__version__)"
```

Expected: a version number (e.g. `4.x.x`). If ImportError: `pip install lightgbm`.

- [ ] **Step 3: Verify pytest is installed**

```bash
source venv/bin/activate && python3 -c "import pytest; print(pytest.__version__)"
```

Expected: a version number. If ImportError: `pip install pytest`.

- [ ] **Step 4: Write shared test fixtures**

Create `tests/conftest.py`:

```python
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
```

- [ ] **Step 5: Verify conftest is discoverable**

```bash
source venv/bin/activate && pytest tests/ --collect-only 2>&1 | head -5
```

Expected: no import errors (even if no tests found yet).

---

## Task 2: BasePreprocessor

**Files:**
- Create: `ml/preprocessors/base.py`
- Create: `tests/preprocessors/test_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/preprocessors/test_base.py`:

```python
import pandas as pd
import pytest
from ml.preprocessors.base import BasePreprocessor


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        BasePreprocessor()


def test_fit_transform_calls_fit_then_transform(sample_X):
    class DummyPreprocessor(BasePreprocessor):
        def fit(self, X):
            self.fitted = True
            return self

        def transform(self, X):
            assert hasattr(self, "fitted"), "transform called before fit"
            return X

    p = DummyPreprocessor()
    result = p.fit_transform(sample_X)
    assert p.fitted
    pd.testing.assert_frame_equal(result, sample_X)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source venv/bin/activate && pytest tests/preprocessors/test_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'ml'`

- [ ] **Step 3: Implement BasePreprocessor**

Create `ml/preprocessors/base.py`:

```python
from abc import ABC, abstractmethod
import pandas as pd


class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BasePreprocessor":
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && pytest tests/preprocessors/test_base.py -v
```

Expected: `2 passed`

---

## Task 3: WindDirectionEncoder

**Files:**
- Create: `ml/preprocessors/wind_encoder.py`
- Create: `tests/preprocessors/test_wind_encoder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/preprocessors/test_wind_encoder.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/preprocessors/test_wind_encoder.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement WindDirectionEncoder**

Create `ml/preprocessors/wind_encoder.py`:

```python
import numpy as np
import pandas as pd
from ml.preprocessors.base import BasePreprocessor


class WindDirectionEncoder(BasePreprocessor):
    def fit(self, X: pd.DataFrame) -> "WindDirectionEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        radians = np.deg2rad(X["wind_dir"])
        X["wind_dir_sin"] = np.sin(radians)
        X["wind_dir_cos"] = np.cos(radians)
        return X.drop(columns=["wind_dir"])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && pytest tests/preprocessors/test_wind_encoder.py -v
```

Expected: `4 passed`

---

## Task 4: FeatureScaler

**Files:**
- Create: `ml/preprocessors/scaler.py`
- Create: `tests/preprocessors/test_scaler.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/preprocessors/test_scaler.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/preprocessors/test_scaler.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement FeatureScaler**

Create `ml/preprocessors/scaler.py`:

```python
from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ml.preprocessors.base import BasePreprocessor


class FeatureScaler(BasePreprocessor):
    def __init__(self, cols: List[str]):
        self.cols = cols
        self._scaler = StandardScaler()

    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        self._scaler.fit(X[self.cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.cols] = self._scaler.transform(X[self.cols])
        return X
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && pytest tests/preprocessors/test_scaler.py -v
```

Expected: `4 passed`

---

## Task 5: PolynomialExpander

**Files:**
- Create: `ml/preprocessors/polynomial.py`
- Create: `tests/preprocessors/test_polynomial.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/preprocessors/test_polynomial.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/preprocessors/test_polynomial.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement PolynomialExpander**

Create `ml/preprocessors/polynomial.py`:

```python
from typing import List
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from ml.preprocessors.base import BasePreprocessor


class PolynomialExpander(BasePreprocessor):
    def __init__(self, cols: List[str], degree: int = 2):
        self.cols = cols
        self.degree = degree
        self._poly = PolynomialFeatures(degree=degree, include_bias=False)

    def fit(self, X: pd.DataFrame) -> "PolynomialExpander":
        self._poly.fit(X[self.cols])
        self._feature_names = self._poly.get_feature_names_out(self.cols)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        poly_array = self._poly.transform(X[self.cols])
        poly_df = pd.DataFrame(poly_array, columns=self._feature_names, index=X.index)
        passthrough = X.drop(columns=self.cols)
        return pd.concat([passthrough, poly_df], axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && pytest tests/preprocessors/test_polynomial.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Run all preprocessor tests**

```bash
source venv/bin/activate && pytest tests/preprocessors/ -v
```

Expected: `12 passed`

---

## Task 6: BaseModel

**Files:**
- Create: `ml/models/base.py`
- Create: `tests/models/test_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_base.py`:

```python
import pytest
from ml.models.base import BaseModel


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        BaseModel()
```

- [ ] **Step 2: Run to verify it fails**

```bash
source venv/bin/activate && pytest tests/models/test_base.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement BaseModel**

Create `ml/models/base.py`:

```python
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

```bash
source venv/bin/activate && pytest tests/models/test_base.py -v
```

Expected: `1 passed`

---

## Task 7: RidgeModel

**Files:**
- Create: `ml/models/ridge.py`
- Create: `tests/models/test_ridge.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_ridge.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/models/test_ridge.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement RidgeModel**

Create `ml/models/ridge.py`:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from ml.models.base import BaseModel


class RidgeModel(BaseModel):
    def __init__(self, alpha: float = 1.0):
        self._model = Ridge(alpha=alpha)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && pytest tests/models/test_ridge.py -v
```

Expected: `3 passed`

---

## Task 8: LightGBMModel

**Files:**
- Create: `ml/models/lgbm.py`
- Create: `tests/models/test_lgbm.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_lgbm.py`:

```python
import numpy as np
import pytest
from ml.models.lgbm import LightGBMModel


def test_predict_returns_correct_shape(sample_X, sample_y):
    model = LightGBMModel(n_estimators=10)
    model.fit(sample_X, sample_y)
    predictions = model.predict(sample_X)
    assert predictions.shape == (len(sample_X),)


def test_predictions_are_floats(sample_X, sample_y):
    model = LightGBMModel(n_estimators=10)
    model.fit(sample_X, sample_y)
    predictions = model.predict(sample_X)
    assert predictions.dtype in [np.float32, np.float64]


def test_fit_returns_self(sample_X, sample_y):
    model = LightGBMModel(n_estimators=10)
    result = model.fit(sample_X, sample_y)
    assert result is model
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/models/test_lgbm.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement LightGBMModel**

Create `ml/models/lgbm.py`:

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from ml.models.base import BaseModel


class LightGBMModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
    ):
        self._model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && pytest tests/models/test_lgbm.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Run all model tests**

```bash
source venv/bin/activate && pytest tests/models/ -v
```

Expected: `7 passed`

---

## Task 9: DataLoader

**Files:**
- Create: `ml/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/test_data.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement DataLoader**

Create `ml/data.py`:

```python
from typing import List, Tuple
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(
        self,
        path: str,
        target: str,
        drop_cols: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.path = path
        self.target = target
        self.drop_cols = drop_cols
        self.test_size = test_size
        self.random_state = random_state

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        df = pq.read_table(self.path).to_pandas()
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns])
        X = df.drop(columns=[self.target])
        y = df[self.target]
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && pytest tests/test_data.py -v
```

Expected: `4 passed`

---

## Task 10: MLPipeline

**Files:**
- Create: `ml/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/test_pipeline.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement MLPipeline**

Create `ml/pipeline.py`:

```python
from typing import List
import numpy as np
import pandas as pd
from ml.preprocessors.base import BasePreprocessor
from ml.models.base import BaseModel


class MLPipeline:
    def __init__(self, preprocessors: List[BasePreprocessor], model: BaseModel):
        self.preprocessors = preprocessors
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPipeline":
        for preprocessor in self.preprocessors:
            X = preprocessor.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        for preprocessor in self.preprocessors:
            X = preprocessor.transform(X)
        return self.model.predict(X)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && pytest tests/test_pipeline.py -v
```

Expected: `3 passed`

---

## Task 11: Evaluator

**Files:**
- Create: `ml/evaluation.py`
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluator.py`:

```python
import numpy as np
import pytest
from ml.evaluation import Evaluator


def test_mse_is_correct():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_true, y_pred, model_name="Test")
    expected_mse = (10**2 + 10**2 + 10**2) / 3
    assert metrics["mse"] == pytest.approx(expected_mse)


def test_rmse_is_sqrt_of_mse():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_true, y_pred, model_name="Test")
    assert metrics["rmse"] == pytest.approx(metrics["mse"] ** 0.5)


def test_returns_dict_with_mse_and_rmse():
    evaluator = Evaluator()
    metrics = evaluator.evaluate(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), model_name="X"
    )
    assert "mse" in metrics
    assert "rmse" in metrics
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/test_evaluator.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement Evaluator**

Create `ml/evaluation.py`:

```python
from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error


class Evaluator:
    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = ""
    ) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        label = f"{model_name} | " if model_name else ""
        print(f"{label}MSE: {mse:.2f} | RMSE: {rmse:.2f}s")
        return {"mse": mse, "rmse": rmse}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && pytest tests/test_evaluator.py -v
```

Expected: `3 passed`

---

## Task 12: Experiment + End-to-End

**Files:**
- Create: `ml/experiment.py`
- Create: `tests/test_experiment.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_experiment.py`:

```python
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from ml.experiment import Experiment
from ml.pipeline import MLPipeline
from ml.preprocessors.wind_encoder import WindDirectionEncoder
from ml.preprocessors.scaler import FeatureScaler
from ml.models.ridge import RidgeModel
from ml.models.lgbm import LightGBMModel
from ml.evaluation import Evaluator

NUMERIC_COLS = [
    "temperature", "precipitation", "humidity",
    "wind_speed", "wind_gust", "pressure", "snow_depth",
]


def _make_loader(sample_X, sample_y):
    loader = MagicMock()
    X_train = sample_X.iloc[:40].reset_index(drop=True)
    X_test = sample_X.iloc[40:].reset_index(drop=True)
    y_train = sample_y.iloc[:40].reset_index(drop=True)
    y_test = sample_y.iloc[40:].reset_index(drop=True)
    loader.load.return_value = (X_train, X_test, y_train, y_test)
    return loader


def test_experiment_run_returns_mse_and_rmse(sample_X, sample_y):
    experiment = Experiment(
        loader=_make_loader(sample_X, sample_y),
        pipeline=MLPipeline(
            preprocessors=[WindDirectionEncoder()],
            model=RidgeModel(alpha=1.0),
        ),
        evaluator=Evaluator(),
    )
    metrics = experiment.run()
    assert "mse" in metrics
    assert "rmse" in metrics
    assert metrics["mse"] >= 0


def test_experiment_with_lgbm(sample_X, sample_y):
    experiment = Experiment(
        loader=_make_loader(sample_X, sample_y),
        pipeline=MLPipeline(
            preprocessors=[
                WindDirectionEncoder(),
                FeatureScaler(cols=NUMERIC_COLS),
            ],
            model=LightGBMModel(n_estimators=10),
        ),
        evaluator=Evaluator(),
    )
    metrics = experiment.run()
    assert metrics["mse"] >= 0


def test_two_experiments_produce_different_mse(sample_X, sample_y):
    ridge_exp = Experiment(
        loader=_make_loader(sample_X, sample_y),
        pipeline=MLPipeline(
            preprocessors=[WindDirectionEncoder(), FeatureScaler(cols=NUMERIC_COLS)],
            model=RidgeModel(alpha=1.0),
        ),
        evaluator=Evaluator(),
    )
    lgbm_exp = Experiment(
        loader=_make_loader(sample_X, sample_y),
        pipeline=MLPipeline(
            preprocessors=[WindDirectionEncoder(), FeatureScaler(cols=NUMERIC_COLS)],
            model=LightGBMModel(n_estimators=10),
        ),
        evaluator=Evaluator(),
    )
    ridge_metrics = ridge_exp.run()
    lgbm_metrics = lgbm_exp.run()
    # models are different so MSEs should differ
    assert ridge_metrics["mse"] != lgbm_metrics["mse"]
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/test_experiment.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement Experiment**

Create `ml/experiment.py`:

```python
from typing import Dict
from ml.data import DataLoader
from ml.pipeline import MLPipeline
from ml.evaluation import Evaluator


class Experiment:
    def __init__(
        self, loader: DataLoader, pipeline: MLPipeline, evaluator: Evaluator
    ):
        self.loader = loader
        self.pipeline = pipeline
        self.evaluator = evaluator

    def run(self) -> Dict[str, float]:
        X_train, X_test, y_train, y_test = self.loader.load()
        self.pipeline.fit(X_train, y_train)
        predictions = self.pipeline.predict(X_test)
        return self.evaluator.evaluate(
            y_test.values,
            predictions,
            model_name=type(self.pipeline.model).__name__,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && pytest tests/test_experiment.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Run the full test suite**

```bash
source venv/bin/activate && pytest tests/ -v
```

Expected: `all tests passed` (29 total)

- [ ] **Step 6: Smoke test on real Echandens data**

Run this from the project root:

```python
# paste into a Python shell or notebook
import sys; sys.path.insert(0, ".")
from ml.data import DataLoader
from ml.pipeline import MLPipeline
from ml.preprocessors.wind_encoder import WindDirectionEncoder
from ml.preprocessors.scaler import FeatureScaler
from ml.preprocessors.polynomial import PolynomialExpander
from ml.models.ridge import RidgeModel
from ml.models.lgbm import LightGBMModel
from ml.evaluation import Evaluator
from ml.experiment import Experiment

NUMERIC_COLS = [
    "temperature", "precipitation", "humidity",
    "wind_speed", "wind_gust", "pressure", "snow_depth",
]
DROP_COLS = [
    "timestamp", "stop_id", "stop_name", "operator", "line", "departure_delay_s"
]

loader = DataLoader(
    path="data/dataset_705_echandens.parquet",
    target="arrival_delay_s",
    drop_cols=DROP_COLS,
)

# Ridge with polynomial expansion
Experiment(
    loader=loader,
    pipeline=MLPipeline(
        preprocessors=[
            WindDirectionEncoder(),
            FeatureScaler(cols=NUMERIC_COLS),
            PolynomialExpander(cols=NUMERIC_COLS + ["wind_dir_sin", "wind_dir_cos"], degree=2),
        ],
        model=RidgeModel(alpha=1.0),
    ),
    evaluator=Evaluator(),
).run()

# LightGBM
Experiment(
    loader=loader,
    pipeline=MLPipeline(
        preprocessors=[
            WindDirectionEncoder(),
            FeatureScaler(cols=NUMERIC_COLS),
        ],
        model=LightGBMModel(n_estimators=500),
    ),
    evaluator=Evaluator(),
).run()
```

Expected: two lines printed like:
```
RidgeModel | MSE: XXXXX.XX | RMSE: XXX.XXs
LightGBMModel | MSE: XXXXX.XX | RMSE: XXX.XXs
```
