# ML Framework Design

**Date:** 2026-05-01  
**Project:** Swiss Buses Delay Prediction  
**Scope:** Modular OOP regression framework, starting with Echandens dataset

---

## Goal

Build a modular, OOP-based ML framework that makes it easy to swap preprocessing techniques and models without changing any surrounding code. Initial target: predict `arrival_delay_s` (regression) on `data/dataset_705_echandens.parquet` using Ridge regression and LightGBM.

---

## Directory Structure

```
ml/
  __init__.py
  data.py              # DataLoader
  preprocessors/
    __init__.py
    base.py            # abstract BasePreprocessor
    scaler.py          # FeatureScaler (wraps StandardScaler)
    wind_encoder.py    # WindDirectionEncoder (wind_dir → sin/cos)
    polynomial.py      # PolynomialExpander (wraps PolynomialFeatures)
  models/
    __init__.py
    base.py            # abstract BaseModel
    ridge.py           # RidgeModel (wraps sklearn Ridge)
    lgbm.py            # LightGBMModel (wraps lgb.LGBMRegressor)
  pipeline.py          # MLPipeline
  evaluation.py        # Evaluator
  experiment.py        # Experiment (entry point)
```

---

## Components

### DataLoader (`ml/data.py`)

Loads a parquet file, drops specified columns, and splits into train/test sets before any preprocessing occurs.

```python
loader = DataLoader(
    path="data/dataset_705_echandens.parquet",
    target="arrival_delay_s",
    drop_cols=["timestamp", "stop_id", "stop_name", "operator", "line",
               "departure_delay_s"],
    test_size=0.2,
    random_state=42,
)
X_train, X_test, y_train, y_test = loader.load()
```

**Dropped columns rationale:**
- `timestamp`, `stop_id`, `stop_name`, `operator`, `line` — constant across all Echandens rows, zero signal
- `departure_delay_s` — data leakage: not available before arrival in a real prediction scenario

The train/test split happens before preprocessing to prevent leakage from the scaler fitting on test data.

---

### BasePreprocessor (`ml/preprocessors/base.py`)

Abstract base class enforcing a consistent interface across all preprocessors:

```python
class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BasePreprocessor": ...
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
```

---

### WindDirectionEncoder (`ml/preprocessors/wind_encoder.py`)

Converts `wind_dir` (0–360°, circular) into `wind_dir_sin` and `wind_dir_cos`, then drops `wind_dir`. Stateless — `fit` is a no-op.

---

### FeatureScaler (`ml/preprocessors/scaler.py`)

Wraps `sklearn.preprocessing.StandardScaler` on a configurable list of columns. Columns not in the list pass through untouched (cyclical features and booleans are left alone).

```python
FeatureScaler(cols=["temperature", "precipitation", "humidity",
                    "wind_speed", "wind_gust", "pressure", "snow_depth"])
```

---

### PolynomialExpander (`ml/preprocessors/polynomial.py`)

Wraps `sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)` on a configurable list of columns. Applied after scaling. Degree is configurable. Used for Ridge only — omitted from LightGBM pipelines.

```python
PolynomialExpander(cols=["temperature", "precipitation", "humidity",
                         "wind_speed", "wind_gust", "pressure", "snow_depth",
                         "wind_dir_sin", "wind_dir_cos"], degree=2)
```

---

### BaseModel (`ml/models/base.py`)

Abstract base class for all models:

```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel": ...
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
```

---

### RidgeModel (`ml/models/ridge.py`)

Wraps `sklearn.linear_model.Ridge`. Alpha configurable at instantiation:

```python
RidgeModel(alpha=1.0)
```

---

### LightGBMModel (`ml/models/lgbm.py`)

Wraps `lgb.LGBMRegressor`. Key parameters configurable at instantiation:

```python
LightGBMModel(n_estimators=500, learning_rate=0.05, num_leaves=31)
```

No polynomial expansion in its pipeline — trees handle non-linearities natively.

---

### MLPipeline (`ml/pipeline.py`)

Chains a list of preprocessors followed by a model. During `fit`, calls `fit_transform` on each preprocessor in order then `fit` on the model. During `predict`, calls `transform`-only on each preprocessor then `predict` on the model.

```python
pipeline = MLPipeline(preprocessors=[...], model=RidgeModel(alpha=1.0))
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

### Evaluator (`ml/evaluation.py`)

Computes MSE and RMSE (RMSE included as it's in seconds — more interpretable than MSE). Prints a labelled report. Easily extended with more metrics later.

```python
evaluator = Evaluator()
evaluator.evaluate(y_test, predictions, model_name="Ridge")
# → Ridge | MSE: 4821.3 | RMSE: 69.4s
```

---

### Experiment (`ml/experiment.py`)

Top-level entry point. Wires DataLoader, MLPipeline, and Evaluator together. Multiple experiments can be instantiated to compare models side by side.

```python
experiment = Experiment(
    loader=DataLoader(...),
    pipeline=MLPipeline(...),
    evaluator=Evaluator(),
)
experiment.run()
```

---

## Example Usage

**Ridge with polynomial expansion:**
```python
Experiment(
    loader=DataLoader(path="data/dataset_705_echandens.parquet", ...),
    pipeline=MLPipeline(
        preprocessors=[
            WindDirectionEncoder(),
            FeatureScaler(cols=["temperature", "precipitation", "humidity", "wind_speed", "wind_gust", "pressure", "snow_depth"]),
            PolynomialExpander(cols=["temperature", "precipitation", "humidity", "wind_speed", "wind_gust", "pressure", "snow_depth"], degree=2),
        ],
        model=RidgeModel(alpha=1.0),
    ),
    evaluator=Evaluator(),
).run()
```

**LightGBM (no polynomial expansion):**
```python
Experiment(
    loader=DataLoader(path="data/dataset_705_echandens.parquet", ...),
    pipeline=MLPipeline(
        preprocessors=[
            WindDirectionEncoder(),
            FeatureScaler(cols=["temperature", "precipitation", "humidity", "wind_speed", "wind_gust", "pressure", "snow_depth"]),
        ],
        model=LightGBMModel(n_estimators=500),
    ),
    evaluator=Evaluator(),
).run()
```

---

## Constraints

- Python 3.x, dependencies: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `pyarrow`
- Target dataset for initial implementation: `data/dataset_705_echandens.parquet` (27,809 rows)
- Target variable: `arrival_delay_s` (regression)
- Metric: MSE (+ RMSE for interpretability)
- Framework must extend cleanly to the full 705 dataset (502k rows) and eventually 495M rows
