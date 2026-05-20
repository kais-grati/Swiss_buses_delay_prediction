# Architecture — ML Pipeline Design

## Overview

The project follows a clean **preprocessor → model** pipeline architecture where each component is independently testable and serializable.

```
DataLoader  →  [Preprocessor Chain]  →  Model  →  Predictions
   │              │                       │
   │         TemporalFeatureExtractor    CatBoostModel
   │         WindMerger                  LightGBMModel
   │         StringEncoder               XGBoostModel
   │         FeatureScaler               RidgeModel
   │         PolynomialExpander          StackingModel
   │         PCAReducer                  ...
   │         NystroemExpander
   │         HistoricalMeanEncoder
```

## Core Abstractions

### DataLoader (`ml/data.py`)

Loads parquet data via DuckDB (streaming) or PyArrow, handles NaN imputation, and produces train/test splits.

```python
loader = DataLoader(
    path="data/705_bus_2025_weather_traffic.parquet",
    target="arrival_delay_s",
    drop_cols=["stop_name", "departure_delay_s", "trip_id"],
    sample_n=None,          # Optional: reservoir sample N rows
    test_size=0.2,
)
X_train, X_test, y_train, y_test = loader.load()
```

Key behaviors:
- `dropna(subset=[target])` — removes rows where target is NaN
- `dist_to_prev_stop` NaN → 0.0 imputation (first stop of each trip)
- DuckDB `USING SAMPLE ... (reservoir)` for memory-efficient sampling
- `.stream()` method for row-group-at-a-time iteration

### MLPipeline (`ml/pipeline.py`)

Sequential preprocessor chain feeding into a model. Handles fit/transform ordering correctly (fit_transform on train, transform only on test).

```python
pipeline = MLPipeline(
    preprocessors=[TemporalFeatureExtractor(), WindMerger(), StringEncoder(...)],
    model=CatBoostModel(...),
)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
pipeline.save("path/to/model_dir/")
pipeline = MLPipeline.load("path/to/model_dir/")
```

### Preprocessors (`ml/preprocessors/`)

All preprocessors implement:
```python
class BasePreprocessor:
    def fit(self, X, y=None) -> "BasePreprocessor": ...
    def transform(self, X) -> pd.DataFrame: ...
    def fit_transform(self, X, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
```

| Preprocessor | Purpose | Adds Columns |
|-------------|---------|--------------|
| `TemporalFeatureExtractor` | Extracts hour, dow, month from timestamp; creates sin/cos cyclical encodings | `hour`, `dow`, `month`, `time_sin`, `time_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `is_weekend` |
| `WindMerger` | Merges `wind_speed` + `wind_gust` → `wind` (mean) | `wind` (removes `wind_speed`, `wind_gust`) |
| `StringEncoder` | Label-encodes categorical columns (operator, line) | Replaces string cols with integer codes |
| `FeatureScaler` | StandardScaler on specified numeric columns | None (in-place) |
| `PolynomialExpander` | PolynomialFeatures (degree 2) on specified columns | `col1^2`, `col1*col2`, etc. |
| `PCAReducer` | PCA with variance threshold | `pc1`, `pc2`, ... |
| `NystroemExpander` | RBF kernel approximation | `nys1`, `nys2`, ... |
| `HistoricalMeanEncoder` | Target encoding by group | `hist_mean_delay` |
| `DelayBinner` | Bins continuous delay into classes | N/A (target encoder) |

### Models (`ml/models/`)

All models implement `fit(X, y)` and `predict(X)`. Classifier models additionally implement `predict_proba(X)`.

| Model | Type | Library | Key Hyperparameters |
|-------|------|---------|---------------------|
| `CatBoostModel` | Regression | CatBoost | `n_estimators`, `depth`, `learning_rate`, `l2_leaf_reg` |
| `LightGBMModel` | Regression | LightGBM | `n_estimators`, `num_leaves`, `learning_rate`, `subsample` |
| `XGBoostModel` | Regression | XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample` |
| `RidgeModel` | Regression | sklearn | `alpha` |
| `RandomForestRegressorModel` | Regression | sklearn | `n_estimators`, `max_depth` |
| `StackingModel` | Regression | custom | `base_models`, `meta_model`, `n_folds` |
| `ResidualStackingModel` | Regression | custom | `stage1_model`, `stage2_model`, `n_folds` |
| `LogTargetModel` | Regression | custom | `model` (wraps log(y) → exp(pred)) |
| `CatBoostClassifierModel` | Classification | CatBoost | `n_estimators`, `depth`, `auto_class_weights` |
| `LightGBMClassifierModel` | Classification | LightGBM | `n_estimators`, `num_leaves`, `scale_pos_weight` |
| `XGBoostClassifierModel` | Classification | XGBoost | `n_estimators`, `max_depth` |
| `RandomForestClassifierModel` | Classification | sklearn | `n_estimators`, `max_depth` |
| `ClassificationStackingModel` | Classification | custom | `base_models`, `meta_model`, `n_folds` |
| `LogisticRegressionModel` | Classification | sklearn | `C`, `l1_ratio` |

All models have `save(path)` and `load(path)` for serialization.

## Experiment System

### Experiment (`ml/experiment.py`)

Encapsulates a full train→evaluate cycle:

```python
exp = Experiment(loader, pipeline, evaluator)
metrics = exp.run()  # → {"mse": ..., "rmse": ..., "r2": ...}
```

### ClassificationExperiment

Extends Experiment with target binning:

```python
exp = ClassificationExperiment(loader, pipeline, evaluator, encoder=DelayBinner(bins=[60, 120, 300]))
metrics = exp.run()  # → {"f1": ..., "report": {...}, "cm": ...}
```

### Evaluator (`ml/evaluation.py`)

Produces rich-formatted output using the `rich` library:
- Regression: MSE, RMSE, R²
- Classification: per-class precision/recall/F1, confusion matrix, macro/weighted averages

## Optimization System (`ml/optimizer.py`)

Bayesian hyperparameter optimization using Optuna with TPE sampler:

```
Optimizer._load_once()      # Load data once, reuse across trials
    ↓
Optimizer._objective(trial)  # Inner train/val split → fit → evaluate
    ↓
Optuna Study.optimize()     # TPE-guided search, early pruning
```

12 optimizer classes covering every model type, with built-in early stopping support and stratified splits for classification.

The external script `optimize_classifiers.py` adds a two-phase workflow:
1. Phase 1: Optuna search on 50K stratified sample (fast trials)
2. Phase 2: Best params evaluated on full dataset

## Feature Engineering

### Precomputed (in parquet)

| Feature | Source | Computation |
|---------|--------|------------|
| `prev_stop_delay` | `add_lag_delay.py` | `LAG(arrival_delay_s) OVER (PARTITION BY trip_id)` |
| `dist_to_prev_stop` | `add_stop_distance.py` | Euclidean distance between consecutive stop LV95 coordinates |
| `trip_stop_index` | `add_trip_stop_index.py` | `ROW_NUMBER() OVER (PARTITION BY trip_id ORDER BY timestamp)` |
| `temperature`, `precipitation`, ... | `add_weather.py` | Spatial join: stop → nearest MeteoSwiss station → Open-Meteo hourly data |
| `traffic_dtv`, `traffic_peak`, ... | `add_traffic_features.py` | Spatial join: stop → nearest road segment → ASTRA traffic counts |
| `is_public_holiday` | `build_dataset.py` | Stop canton → Swiss holiday calendar |
| `time_sin`, `time_cos`, ... | `build_dataset.py` | sin/cos cyclical encoding of time/day/month |

### Runtime (in preprocessors)

| Feature | Preprocessor | Computation |
|---------|-------------|------------|
| `hour`, `dow`, `month` | `TemporalFeatureExtractor` | Extracted from timestamp |
| `wind` | `WindMerger` | Mean of `wind_speed` + `wind_gust` |
| `hist_mean_delay` | `HistoricalMeanEncoder` | Grouped mean of target by (operator, line) |

## Design Decisions

**Why precompute features?** The `prev_stop_delay` lag computation requires a `GROUP BY trip_id` → `SHIFT` window operation. Doing this at runtime on a 400K-row DataFrame in pandas would take minutes and risk OOM. Precomputing it in DuckDB (which spills to disk) keeps training fast and memory-bounded.

**Why DuckDB for data processing?** DuckDB's streaming query engine processes parquet files larger than RAM by spilling to disk. It's used in every data pipeline script with `PRAGMA memory_limit='4GB'` to prevent memory exhaustion. For ML training, PyArrow reads the much smaller filtered subsets directly into pandas.

**Why keep `trip_stop_index`?** This was dropped for months. The ablation study (FEATURE_ANALYSIS.md) proved it adds +0.022 R² and +0.036 F1 — more than any other feature besides `prev_stop_delay`. It encodes the physical reality that delays compound along a bus route: later stops have systematically larger delays.

**Why stacking helps only sometimes.** On 705 with `trip_stop_index`, CatBoost and Stack-CB-Ridge are essentially tied (R²=0.8800 vs 0.8801). The stacking premium shrinks when both base models have access to the same strong features. Stacking adds more value when the base models are complementary (e.g., tree + linear on Lausanne 50k where Ridge and CatBoost capture different patterns).

**Why logistic regression fails.** `StringEncoder` creates hundreds of sparse one-hot columns for operator and line values. sklearn's `LogisticRegression(lbfgs)` cannot converge on this high-dimensional sparse feature space. A `HistoricalMeanEncoder` instead of `StringEncoder` would fix this, but was not pursued since tree models handle raw categorical features natively and outperform linear models anyway.
