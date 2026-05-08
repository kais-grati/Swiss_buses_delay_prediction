# Dataset Pipeline

This document describes every step required to go from the raw SBB open data
CSV files to the final `dataset_with_weather.parquet` used for model training.

---

## Overview

```
data/*.csv  (raw, German headers, all transport modes)
    │
    ├─ 1. translate_headers.py   → rename German columns to English
    ├─ 2. keep_only_buses.py     → filter to bus rows only  →  cleaned_data/*.csv
    ├─ 3. prepare_features.py    → drop unused columns      →  cleaned_data/*.csv (in-place)
    ├─ 4. to_parquet.py          → feature engineering      →  dataset.parquet
    │
    ├─ 5. fetch_weather_hourly.py→ hourly Open-Meteo weather→  weather_hourly.parquet
    │                                                           station_metadata.parquet
    ├─ 6. add_weather.py         → join weather to dataset  →  dataset_with_weather.parquet
    ├─ 7. add_holidays.py        → add is_public_holiday    →  dataset_with_weather.parquet (in-place)
    ├─ 8. drop_outlier_delays.py → remove delay > 30 min   →  dataset_with_weather.parquet (in-place)
    ├─ 9. drop_early_outliers.py  → remove delay < -2 min  →  dataset_with_weather.parquet (in-place)
    └─10. drop_missing_weather.py → remove missing weather →  dataset_with_weather.parquet (in-place)
```

---

## Data Sources

| Source | URL | Content |
|--------|-----|---------|
| SBB Istdaten | https://data.opentransportdata.swiss/dataset/istdaten | One CSV per day with all public transport departures/arrivals in Switzerland and their actual vs scheduled times |
| MeteoSwiss OGD | https://data.geo.admin.ch/api/stac/v0.9/collections/ch.meteoschweiz.ogd-smn | Daily climate observations for 158 automatic weather stations |
| Open-Meteo Archive | https://archive-api.open-meteo.com/v1/archive | Hourly historical weather for any lat/lon (free, no key) |
| Swiss Stop Coordinates | `station_data.parquet` (pre-downloaded) | 28,982 Swiss public transport stops with LV95 coordinates and BPUIC identifiers |

---

## Step 1 — Translate Headers

**Script:** `translate_headers.py`  
**Input:** `data/*.csv` (365 daily files, German column names)  
**Output:** same files, header rewritten in English  
**Reads into memory:** one file at a time

Renames German SBB column names to English equivalents, e.g.:
- `BETRIEBSTAG` → `DATE`
- `HALTESTELLEN_NAME` → `STOP_NAME`
- `AN_PROGNOSE` → `ARRIVAL_FORECAST`

Run:
```bash
python translate_headers.py
```

---

## Step 2 — Keep Only Buses

**Script:** `keep_only_buses.py`  
**Input:** `data/*.csv`  
**Output:** `cleaned_data/*.csv` (new files, bus rows only)

Filters each daily CSV to rows where `PRODUCT_ID == "Bus"`, discarding trains,
trams, boats, etc. Processes files in parallel with all available CPU cores.

Run:
```bash
python keep_only_buses.py
```

---

## Step 3 — Drop Unused Columns

**Script:** `prepare_features.py`  
**Input:** `cleaned_data/*.csv`  
**Output:** same files, rewritten in-place with only the needed columns

Keeps only these 14 columns, dropping everything else:

| Column | Description |
|--------|-------------|
| `DATE` | Service date (DD.MM.YYYY) |
| `OPERATOR_ABB` | Operator abbreviation |
| `LINE_NAME` | Line identifier |
| `BPUIC` | Stop identifier (numeric) |
| `STOP_NAME` | Stop name |
| `ARRIVAL_TIME` | Scheduled arrival (DD.MM.YYYY HH:MM) |
| `ARRIVAL_FORECAST` | Actual arrival (DD.MM.YYYY HH:MM:SS) |
| `ARRIVAL_FORECAST_STATUS` | `REAL` / `GESCHAETZT` / etc. |
| `DEPARTURE_TIME` | Scheduled departure |
| `DEPARTURE_FORECAST` | Actual departure |
| `DEPARTURE_FORECAST_STATUS` | `REAL` / `GESCHAETZT` / etc. |
| `ADDITIONAL_TRIP` | Extra unscheduled trip flag |
| `CANCELLED` | Trip cancelled flag |
| `PASS_THROUGH` | Bus passed without stopping |

Skips files already processed (idempotent). Writes atomically via `.tmp` files.

Run:
```bash
python prepare_features.py
```

---

## Step 4 — Feature Engineering → dataset.parquet

**Script:** `to_parquet.py`  
**Input:** `cleaned_data/*.csv`  
**Output:** `dataset.parquet` (~3.5 GB, snappy-compressed)

This is the main processing step. For each daily CSV:

1. **Drop cancelled trips** (`CANCELLED == true`)
2. **Keep only REAL observations** — rows where at least one of
   `ARRIVAL_FORECAST_STATUS` or `DEPARTURE_FORECAST_STATUS` is `"REAL"`.
   Estimated (`GESCHAETZT`) forecasts are excluded.
3. **Compute delays** — `(actual_time − scheduled_time)` in seconds, using
   time-of-day only to avoid ±24h errors from date mismatches. Wrapped to
   (−12h, +12h].
4. **Encode cyclical time features:**
   - `time_sin / time_cos` — minute of day mapped onto unit circle
   - `dow_sin / dow_cos` — day of week (0=Mon … 6=Sun)
   - `month_sin / month_cos` — month (1–12)
   - `is_weekend` — bool
5. **Preserve raw timestamp** — scheduled arrival (falling back to scheduled
   departure) kept as `timestamp` for weather joining.

Output schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | timestamp[s] | Scheduled arrival/departure (Swiss local, naive) |
| `time_sin` / `time_cos` | float32 | Cyclical time of day |
| `dow_sin` / `dow_cos` | float32 | Cyclical day of week |
| `month_sin` / `month_cos` | float32 | Cyclical month |
| `is_weekend` | bool | Saturday or Sunday |
| `operator` | string | Operator abbreviation |
| `line` | string | Line name |
| `stop_id` | int32 | BPUIC stop number |
| `stop_name` | string | Stop name |
| `additional_trip` | bool | Unscheduled extra trip |
| `pass_through` | bool | No stop made (filtered out in Step 6) |
| `arrival_delay_s` | int32 | Arrival delay in seconds (nullable) |
| `departure_delay_s` | int32 | Departure delay in seconds (nullable) |

Processes files in parallel, writes per-file parquets to `cleaned_data_parquet_tmp/`,
then merges into `dataset.parquet`. Supports resume (skips already-converted files).

Run:
```bash
python to_parquet.py --workers 4
# or test on 3 files first:
python to_parquet.py --test
```

---

## Step 5 — Hourly Open-Meteo Weather

**Script:** `fetch_weather_hourly.py`  
**Input:** Open-Meteo Archive API, MeteoSwiss STAC API (station discovery)  
**Output:**
- `weather_hourly.parquet` — hourly observations for 158 station locations
- `station_metadata.parquet` — MeteoSwiss station coordinates (generated once if missing)
- `weather_hourly_cache/` — per-station parquets (resume cache)

Queries Open-Meteo's free historical archive API for each of the 158
MeteoSwiss station lat/lon coordinates. Returns 8,760 rows per station
(one per hour of 2025). Timestamps are in UTC.

Weather variables (hourly):

| Column | Description | Unit |
|--------|-------------|------|
| `temperature` | Air temperature 2m | °C |
| `precipitation` | Rainfall + snowfall | mm |
| `sunshine` | Fraction of the hour with sunshine | 0–1 |
| `humidity` | Relative humidity 2m | % |
| `wind_speed` | Wind speed at 10m | m/s |
| `wind_gust` | Wind gust at 10m | m/s |
| `wind_dir` | Wind direction at 10m | ° |
| `pressure` | Surface pressure | hPa |
| `snow_depth` | Snow depth | m |

Runs sequentially (1 worker) with a 1-second gap between requests to respect
the free-tier rate limit. ~3 minutes total for all 158 stations.

Run:
```bash
python fetch_weather_hourly.py
```

---

## Step 6 — Join Weather → dataset_with_weather.parquet

**Script:** `add_weather.py`  
**Input:** `dataset.parquet`, `weather_hourly.parquet`, `station_data.parquet`, `station_metadata.parquet`  
**Output:** `dataset_with_weather.parquet`

Three sub-steps:

**6a. Stop → nearest MeteoSwiss station mapping**

Loads all 28,982 bus stop coordinates from `station_data.parquet` (Swiss LV95
grid, EPSG:2056), converts them to WGS84 (lat/lon) using `pyproj`, then uses
a `scipy` KDTree to find the nearest of the 158 MeteoSwiss stations for each
stop. Result: a `{bpuic → station_id}` dictionary held in memory.

**6b. Weather lookup table**

Loads `weather_hourly.parquet` (~80 MB) fully into RAM, indexed by
`(station_id, timestamp)` where timestamp is UTC-naive, floored to the hour.

**6c. Filtered join**

Reads `dataset.parquet` via DuckDB streaming:
1. **Excludes pass-through rows** (`WHERE NOT pass_through`) — stops where the
   bus never opens its doors have no meaningful delay to predict.
2. **Drops the `pass_through` column** from the output (all remaining rows are
   `pass_through = FALSE`).
3. Maps `stop_id` → `meteoswiss_station_id` via the dictionary from 6a.
4. Converts `timestamp` from Swiss local time (`Europe/Zurich`) to UTC, floors to hour.
5. Left-joins with the weather table on `(station_id, utc_hour)`.

Adds these columns to the dataset (all float32):
`temperature`, `precipitation`, `sunshine`, `humidity`,
`wind_speed`, `wind_gust`, `wind_dir`, `pressure`, `snow_depth`

Run:
```bash
pip install pyproj scipy   # first time only
python add_weather.py
```

---

## Step 7 — Add Public Holidays → dataset_with_weather.parquet

**Script:** `add_holidays.py`  
**Input:** `dataset_with_weather.parquet`, `station_data.parquet`  
**Output:** `dataset_with_weather.parquet` (in-place, adds `is_public_holiday` column)

Three sub-steps:

1. **Stop → canton mapping** — loads `station_data.parquet`, maps each `stop_id` (BPUIC) to its Swiss canton abbreviation (e.g. `VD`, `ZH`). Writes a temporary `_stop_map_tmp.parquet`.
2. **Holiday table** — uses the `holidays` library to generate all canton-specific Swiss public holidays for each year present in the dataset. Writes `_holidays_tmp.parquet`.
3. **Join** — DuckDB streaming join: `dataset ⟕ stop_map ⟕ holidays` on `(stop_id → canton, DATE(timestamp) = holiday_date)`. Sets `is_public_holiday = TRUE` for matching rows, `FALSE` otherwise. Verifies row count before atomically replacing the original file.

Temporary files are cleaned up on exit (even on failure).

Run:
```bash
pip install holidays   # first time only
python add_holidays.py
```

---

## Step 8 — Drop Outlier Delays → dataset_with_weather.parquet

**Script:** `drop_outlier_delays.py`  
**Input:** `dataset_with_weather.parquet`  
**Output:** `dataset_with_weather.parquet` (in-place, outlier rows removed)

Drops rows where `arrival_delay_s > 1800` **or** `departure_delay_s > 1800` (30-minute threshold). These represent ~15.6% of the dataset and are likely corrupt records rather than real delays.

Processes the file one PyArrow row group at a time — RAM usage stays bounded regardless of file size. Writes to `.tmp`, then atomically replaces the original.

Run:
```bash
python drop_outlier_delays.py
```

---

## Step 9 — Drop Implausibly Early Arrivals → dataset_with_weather.parquet

**Script:** `drop_early_outliers.py`  
**Input:** `dataset_with_weather.parquet`  
**Output:** `dataset_with_weather.parquet` (in-place, early outlier rows removed)

Drops rows where `arrival_delay_s < -120` **or** `departure_delay_s < -120` (2-minute early threshold). Swiss transit regulations prohibit early departures, so values beyond -120s are data artifacts rather than real early arrivals (~0.60% of rows). Dropping is preferred over capping to avoid creating an artificial spike at -120s in the delay distribution.

Same row-group streaming approach as Step 8.

Run:
```bash
python drop_early_outliers.py
```

---

## Step 10 — Drop Missing Weather → dataset_with_weather.parquet

**Script:** `drop_missing_weather.py`  
**Input:** `dataset_with_weather.parquet`  
**Output:** `dataset_with_weather.parquet` (in-place, incomplete rows removed)

Drops rows where any of the 9 weather columns (`temperature`, `precipitation`, `sunshine`, `humidity`, `wind_speed`, `wind_gust`, `wind_dir`, `pressure`, `snow_depth`) is NULL. These arise from stops that could not be matched to a MeteoSwiss station or timestamps outside the weather data coverage (~0.47% of rows).

Same row-group streaming approach as previous steps.

Run:
```bash
python drop_missing_weather.py
```

---

## Full Run Order

```bash
# 1–3: one-time data cleaning (already done if cleaned_data/ exists)
python translate_headers.py
python keep_only_buses.py
python prepare_features.py

# 4: build dataset (re-run if to_parquet.py was modified)
python to_parquet.py --workers 4

# 5: fetch hourly weather (run once, resume-safe)
python fetch_weather_hourly.py

# 6: join everything (pass-through rows excluded automatically)
python add_weather.py

# 7: add canton-aware public holiday flag
python add_holidays.py

# 8: drop outlier delays (> 30 min)
python drop_outlier_delays.py

# 9: drop implausibly early arrivals/departures (< -2 min — ~0.60% of rows)
python drop_early_outliers.py

# 10: drop rows with any missing weather field (~0.47% of rows)
python drop_missing_weather.py
```

### Utility: create a line/stop sub-dataset

To extract a filtered subset without touching the main file:

```bash
# Example: line 705, Echandens Chocolatière only
python filter_705_echandens.py
# → data/dataset_705_echandens.parquet (27,959 rows)
```

Uses DuckDB predicate pushdown — scans only relevant row groups.

### Utility: clean an existing dataset_with_weather.parquet

If `dataset_with_weather.parquet` was generated before the pass-through filter
was added to `add_weather.py`, run this once to remove those rows in-place:

```bash
python drop_pass_through.py
```

Uses DuckDB streaming — the full file is never loaded into RAM. Writes to a
`.tmp` file first, verifies the row count, then atomically replaces the original.

## Output Files

| File | Size | Description |
|------|------|-------------|
| `dataset.parquet` | ~3.5 GB | Bus trip features without weather |
| `station_metadata.parquet` | ~15 KB | MeteoSwiss station locations |
| `weather_hourly.parquet` | ~50–80 MB | Hourly weather per station |
| `station_data.parquet` | ~2.6 MB | Swiss bus stop coordinates |
| `dataset_with_weather.parquet` | ~5–6 GB | Final training dataset |

---

# ML Pipeline

This section describes the modular ML framework in `ml/` used to train and evaluate models on the prepared dataset.

## Architecture

```
DataLoader  →  MLPipeline (preprocessors → model)  →  Evaluator
                                                           ↑
                                             Experiment / ClassificationExperiment
```

`Experiment` wires the three components together. `ClassificationExperiment` adds a target-encoding step via a `ClassEncoder` before fitting.

## Directory Structure

```
ml/
  data.py                         # DataLoader
  pipeline.py                     # MLPipeline
  experiment.py                   # Experiment, ClassificationExperiment
  evaluation.py                   # Evaluator (regression + classification)
  optimizer.py                    # LGBMRegressorOptimizer, XGBoostRegressorOptimizer, CatBoostRegressorOptimizer,
                                    #   LGBMClassifierOptimizer, OrdinalLGBMClassifierOptimizer
  logger.py                       # ExperimentLogger (appends results to results/experiment_log.jsonl)
  preprocessors/
    base.py                       # abstract BasePreprocessor
    class_encoder.py              # abstract ClassEncoder (base for target binning)
    delay_binner.py               # DelayBinner (continuous delay → 4 classes)
    scaler.py                     # FeatureScaler (StandardScaler on selected cols)
    temporal.py                   # TemporalFeatureExtractor (hour, dow from timestamp)
    target_encoder.py             # HistoricalMeanEncoder (lag feature by group)
    weather_engineer.py           # WeatherFeatureEngineer (wind chill, adverse flag)
    weather_rush_hour.py          # WeatherRushHourPreprocessor
    wind_encoder.py               # WindDirectionEncoder (wind_dir → sin/cos)
    polynomial.py                 # PolynomialExpander (degree-n polynomial + interaction terms)
    sinusoidal.py                 # SinusoidalExpander (sin/cos harmonics per feature)
    poly_trig.py                  # PolyTrigExpander (polynomial + trig combined)
    pca.py                        # PCAReducer (n_components or variance_threshold)
    nystroem.py                   # NystroemExpander (RBF kernel approximation via Nyström)
  models/
    base.py                       # BaseModel (fit/predict), ClassifierModel (+ predict_proba)
    ridge.py                      # RidgeModel
    lgbm.py                       # LightGBMModel (GBDT + DART, early stopping)
    xgboost_model.py              # XGBoostModel (early stopping)
    catboost_model.py             # CatBoostModel (early stopping)
    stacking.py                   # StackingModel (k-fold OOF meta-learner, regression)
    logistic_regression.py        # LogisticRegressionModel (multi-class)
    lgbm_classifier.py            # LightGBMClassifierModel
    xgboost_classifier.py         # XGBoostClassifierModel
    catboost_classifier.py        # CatBoostClassifierModel
    random_forest_classifier.py   # RandomForestClassifierModel
    ordinal_classifier.py         # OrdinalClassifierModel (Frank & Hall K-1 binary decomposition)
    classification_stacking.py    # ClassificationStackingModel (probability OOF stacking)
    pipelined_classifier.py       # PipelinedClassifierModel (preprocessors + classifier as one unit)
```

---

## Regression Pipeline

Standard path: predict `arrival_delay_s` as a continuous value.

```python
experiment = Experiment(
    loader=DataLoader(path=DATASET, target="arrival_delay_s", drop_cols=DROP_COLS),
    pipeline=MLPipeline(
        preprocessors=[
            TemporalFeatureExtractor(),
            FeatureScaler(cols=NUMERIC_COLS),
        ],
        model=LightGBMModel(n_estimators=3000, early_stopping_rounds=50),
    ),
    evaluator=Evaluator(),
)
results = experiment.run()
# returns: {"mse": ..., "rmse": ..., "r2": ...}
```

**Evaluator metrics (regression):** MSE, RMSE (seconds), R²

---

## Classification Pipeline

Converts the continuous delay target into discrete classes before training.

### Target Encoding: `ClassEncoder` and `DelayBinner`

`ClassEncoder` is the abstract base class. `DelayBinner` is the concrete implementation that bins `arrival_delay_s` into classes using configurable thresholds:

| Class | Range | Meaning |
|-------|-------|---------|
| 0 | delay ≤ 0s | On-time or early |
| 1 | 0s < delay ≤ 60s | Slight delay (within 1 min) |
| 2 | 60s < delay ≤ 100s | Minor delay |
| 3 | 100s < delay ≤ 160s | Moderate delay |
| 4 | delay > 160s | Significant delay |

The bin thresholds are configurable. `class_names` is auto-generated from `self.bins` and flows through to the evaluator for labeled output.

```python
binner = DelayBinner()                        # default bins: [0, 60, 100, 160]  → 5 classes
binner = DelayBinner(bins=[60, 180, 600])     # operationally-grounded 4-class variant
binner = DelayBinner(bins=[90])               # binary split at 90s  → 2 classes
print(binner.class_names)
# → ['≤0s', '0–60s', '60–100s', '100–160s', '>160s']
```

### `ClassificationExperiment`

Extends `Experiment` with a target-encoding step. Encodes both `y_train` and `y_test` before fitting, then passes `class_names` to the evaluator automatically.

```python
experiment = ClassificationExperiment(
    loader=DataLoader(path=DATASET, target="arrival_delay_s", drop_cols=DROP_COLS_KEEP_TS),
    pipeline=MLPipeline(
        preprocessors=[
            TemporalFeatureExtractor(),
            FeatureScaler(cols=NUMERIC_COLS),
        ],
        model=LogisticRegressionModel(class_weight="balanced"),
    ),
    evaluator=Evaluator(),
    encoder=DelayBinner(),
)
results = experiment.run()
# returns: {"f1": ..., "report": ..., "cm": ...}
```

**Evaluator output (classification):**
- Macro-F1 summary panel
- Per-class precision / recall / F1 / support table
- Labeled confusion matrix (rows = actual class, columns = predicted class, using `class_names`)

### Stacking Ensemble: `ClassificationStackingModel`

`ClassificationStackingModel` trains multiple base classifiers using out-of-fold (OOF) cross-validation, then fits a meta-classifier on the resulting probability outputs. It inherits from `ClassifierModel` and is a drop-in replacement for any single classifier in a `ClassificationExperiment`.

**Training flow:**
1. Stratified 5-fold CV — for each fold × base model, `predict_proba()` fills an OOF matrix of shape `(n_train, n_base_models × n_classes)`
2. Each base model is refit on the full training set
3. The meta-model (LogisticRegression) is fit on the OOF matrix

**Inference:** base models call `predict_proba()` in parallel → horizontal stack → meta-model `predict()`

```python
model = ClassificationStackingModel(
    base_models=[
        LightGBMClassifierModel(...),
        XGBoostClassifierModel(...),
        CatBoostClassifierModel(...),
    ],
    meta_model=LogisticRegressionModel(C=1.0),
    n_folds=5,
)
```

The model also exposes `predict_proba()`, making it composable (e.g. as a base model in a deeper stack).

---

### `PipelinedClassifierModel`

A `ClassifierModel` that bundles its own preprocessor chain alongside a wrapped classifier. Useful when one base model in a stacking ensemble needs different feature transformations than the others — the internal preprocessors run privately during `fit`, `predict`, and `predict_proba`, leaving the shared pipeline unchanged.

```python
PipelinedClassifierModel(
    preprocessors=[NystroemExpander(n_components=100, kernel="rbf")],
    classifier=LogisticRegressionModel(C=1.0, max_iter=5000),
)
```

The pipeline's shared preprocessors run first (e.g. `FeatureScaler`), then `PipelinedClassifierModel` applies its own chain on top before passing the result to the inner classifier. This avoids re-applying transformations that are already done by the outer pipeline.

Since it inherits from `ClassifierModel`, it is a valid base model for `ClassificationStackingModel` and exposes both `predict()` and `predict_proba()`.

---

### Adding a new encoder

Subclass `ClassEncoder` and implement `encode`, `decode`, and optionally `class_names`:

```python
class QuartileEncoder(ClassEncoder):
    def encode(self, y): ...
    def decode(self, y_encoded): ...

    @property
    def class_names(self):
        return ["Q1", "Q2", "Q3", "Q4"]
```

No other changes needed — `ClassificationExperiment` and `Evaluator` pick up `class_names` automatically.

---

## Feature Expansion Preprocessors

These preprocessors transform the feature matrix before it reaches the model. All follow the `BasePreprocessor` interface (`fit` / `transform` / `fit_transform`).

### `PolynomialExpander`

Replaces selected columns with their degree-n polynomial expansion using sklearn's `PolynomialFeatures` (`include_bias=False`). The output includes all degree-1 through degree-n terms (originals, interactions, powers).

```python
PolynomialExpander(cols=NUMERIC_COLS_LOGREG, degree=2)
# 11 inputs → 77 output columns (degree-1 + degree-2 terms)
```

### `SinusoidalExpander`

Replaces selected columns with `sin(k·x)` and `cos(k·x)` harmonics for `k = 1..n_components`. Useful for periodic signals like time-of-day or cyclical weather patterns.

```python
SinusoidalExpander(cols=["hour", "dow"], n_components=3)
```

### `PolyTrigExpander`

Combines polynomial and trigonometric expansions in a single step. Produces degree-n polynomial terms plus `sin(k·x)` / `cos(k·x)` harmonics for each selected column. Non-selected columns pass through unchanged.

```python
PolyTrigExpander(cols=NUMERIC_COLS_LOGREG, degree=2, n_trig=1)
# 11 inputs → 77 poly terms + 22 trig terms = 99 output columns
# n_trig > 1 adds _1, _2, ... suffix to trig column names
```

Best used after `FeatureScaler` so polynomial and trig terms stay bounded.

### `PCAReducer`

Wraps sklearn's `PCA`. Replaces the entire feature matrix with `pc1, pc2, ...` principal components. Accepts either a fixed component count or a variance threshold.

```python
PCAReducer(n_components=20)           # keep exactly 20 components
PCAReducer(variance_threshold=0.95)   # keep however many explain 95% variance
```

Useful after high-dimensional expansions (`PolynomialExpander`, `PolyTrigExpander`) to decorrelate features before a linear model.

### `NystroemExpander`

Approximates an RBF (or other) kernel via Nyström sampling, projecting the feature matrix into a `n_components`-dimensional kernel feature space. Fitting a linear model (LogisticRegression) on top is equivalent to a nonlinear classifier in the original space — with fundamentally different decision boundaries than tree-based models, making it a strong diversity contributor in stacking ensembles.

```python
NystroemExpander(n_components=100, kernel="rbf", gamma=None)
# gamma=None → sklearn default: 1/n_features
# actual n_components capped to n_train_samples if smaller
```

Always apply `FeatureScaler` before `NystroemExpander` — RBF distance is sensitive to feature scale.

---

## Experiment Logger

`ExperimentLogger` appends one JSON line per experiment run to `results/experiment_log.jsonl`. The `results/` directory is created automatically on first use.

```python
logger = ExperimentLogger()                          # default path
logger = ExperimentLogger("custom/path/log.jsonl")  # custom path
logger.log(name, results, kind="regression")
logger.log(name, results, kind="classification")
```

Each entry includes `timestamp`, `name`, `kind`, and the relevant metrics:
- **Regression:** `mse`, `rmse`, `r2`
- **Classification:** `macro_f1`, `f1_class_0`, `f1_class_1`, ... (per-class F1, skipping averages)

Load all runs for comparison:
```python
import pandas as pd
df = pd.read_json("results/experiment_log.jsonl", lines=True)
```

---

## Hyperparameter Optimization

`ml/optimizer.py` provides Optuna-based Bayesian optimizers for all model types. Naming convention distinguishes regression from classification:

| Optimizer | Target | Tuned model |
|-----------|--------|-------------|
| `LGBMRegressorOptimizer` | RMSE | `LightGBMModel` |
| `XGBoostRegressorOptimizer` | RMSE | `XGBoostModel` |
| `CatBoostRegressorOptimizer` | RMSE | `CatBoostModel` |
| `LGBMClassifierOptimizer` | Macro-F1 | `LightGBMClassifierModel` |
| `OrdinalLGBMClassifierOptimizer` | Macro-F1 | `OrdinalClassifierModel(LightGBMClassifierModel)` |

All optimizers tune `n_estimators` alongside architecture-specific hyperparameters via `trial.suggest_int("n_estimators", 100, self.n_estimators)`, where the constructor's `n_estimators` parameter acts as the upper bound.

```python
from ml.optimizer import LGBMRegressorOptimizer, LGBMClassifierOptimizer

# Regression
optimizer = LGBMRegressorOptimizer(loader=loader_enhanced, numeric_cols=NUMERIC_COLS, n_trials=100)
study = optimizer.optimize()
# prints best params and validation RMSE after each trial

# Classification
clf_optimizer = LGBMClassifierOptimizer(
    loader=loader_enhanced, binner=DelayBinner(), n_trials=60, n_estimators=2000,
)
clf_study = clf_optimizer.optimize()
# prints best params and validation macro-F1 after each trial
```

Optimizers cache the loaded dataset on first call (`_load_once`) and use TPE sampling. Best parameters are printed at the end and can be copy-pasted directly into an `Experiment` definition in `main.py`.
