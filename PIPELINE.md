# Dataset Pipeline

This document describes every step required to go from the raw SBB open data
CSV files to the final `dataset_with_weather.parquet` used for model training.

---

## Overview

```
data/compressed_data/*.zip  (12 monthly ZIP archives, daily CSVs with German headers)
    │
    ├─ 1. build_dataset.py       → translate, filter, features, holidays  →  dataset.parquet
    │
    ├─ 2. fetch_weather_hourly.py→ hourly Open-Meteo weather→  weather_hourly.parquet
    │                                                           station_metadata.parquet
    ├─ 3. add_weather.py         → join weather to dataset  →  dataset_with_weather.parquet
    ├─ 4. drop_missing_weather.py → remove missing weather →  dataset_with_weather.parquet (in-place)
    └─ 5. add_lag_delay.py        → precompute lag feature  →  dataset_with_weather.parquet (in-place)
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

## Step 1 — Build Dataset

**Script:** `build_dataset.py`  
**Input:** `data/compressed_data/*.zip` (12 monthly ZIP archives, each containing ~31 daily CSVs with German headers)  
**Output:** `data/dataset.parquet` (~3.5 GB, snappy-compressed)

This single script replaces the old 7-step pipeline (`translate_headers.py` →
`keep_only_buses.py` → `prepare_features.py` → `to_parquet.py` →
`add_holidays.py` → `drop_outlier_delays.py` → `drop_early_outliers.py`).
It reads daily CSVs directly from ZIP archives (no disk extraction), processes
them in parallel, and writes the final parquet in one shot.

For each daily CSV the script:

1. **Reads from ZIP** — raw bytes loaded into memory via `pyarrow.BufferReader`,
   parsed with PyArrow's CSV reader (11× faster than pandas)
2. **Translates headers** — 21 German column names mapped to English
3. **Filters to buses** — `PRODUCT_ID == "Bus"` (discards trains, trams, boats)
4. **Keeps 15 columns** — drops irrelevant columns immediately to save RAM
5. **Drops cancelled, pass-through, and additional trips** — `CANCELLED == true`,
   `PASS_THROUGH == true`, `ADDITIONAL_TRIP == true`
6. **Keeps only REAL observations** — rows where at least one of
   `ARRIVAL_FORECAST_STATUS` or `DEPARTURE_FORECAST_STATUS` is `"REAL"`.
   Estimated (`GESCHAETZT`) forecasts are excluded.
7. **Computes delays** — `(actual_time − scheduled_time)` in seconds, using
   time-of-day only to avoid ±24h errors from date mismatches. Wrapped to
   (−12h, +12h].
8. **Drops outlier delays** — rows where `arrival_delay_s ∉ [-120, 1800]` or
   `departure_delay_s ∉ [-120, 1800]`. Early departures < 2 min are impossible
   under Swiss regulations (~0.6% of rows). Delays > 30 min are likely data
   corruption (~15.6% of rows).
9. **Adds public holiday flag** — maps each `stop_id` to its canton via
   `station_data.parquet`, then checks against canton-specific Swiss public
   holidays for 2025. Sets `is_public_holiday = TRUE` for stops in a canton
   on a holiday date.
10. **Encodes cyclical time features:**
    - `time_sin / time_cos` — minute of day mapped onto unit circle
    - `dow_sin / dow_cos` — day of week (0=Mon … 6=Sun)
    - `month_sin / month_cos` — month (1–12)
    - `is_weekend` — bool
11. **Preserves raw timestamp** — scheduled arrival (falling back to scheduled
    departure) kept as `timestamp` for weather joining.

Per-file parquets are written to `data/build_dataset_tmp/`, then merged into
`dataset.parquet` with an atomic rename (no corrupt output on crash).

**RAM:** one CSV in memory per worker at a time (~50–200 MB raw, ~10–30 MB
after filtering). DataFrame freed immediately after writing.

**Resume:** skips CSVs whose temp parquet already exists — safe to interrupt
and restart.

**Output schema:**

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | timestamp[ms] | Scheduled arrival/departure (Swiss local, naive) |
| `time_sin` / `time_cos` | float32 | Cyclical time of day |
| `dow_sin` / `dow_cos` | float32 | Cyclical day of week |
| `month_sin` / `month_cos` | float32 | Cyclical month |
| `is_weekend` | bool | Saturday or Sunday |
| `operator` | string | Operator abbreviation |
| `line` | string | Line name |
| `stop_id` | int32 | BPUIC stop number |
| `stop_name` | string | Stop name |
| `additional_trip` | bool | Unscheduled extra trip (filtered out in Step 1) |
| `pass_through` | bool | No stop made (filtered out in Step 1) |
| `arrival_delay_s` | int32 | Arrival delay in seconds (nullable, within [-120, 1800]) |
| `departure_delay_s` | int32 | Departure delay in seconds (nullable, within [-120, 1800]) |
| `is_public_holiday` | bool | Stop is in a canton on a Swiss public holiday |
| `trip_id` | string | Trip identifier (for lag feature computation) |
| `prev_stop_delay` | int32 | Delay at previous stop within same trip (nullable — first stop of each trip is NaN) |

Run:
```bash
python build_dataset.py --workers 4
# or test on the first CSV:
python build_dataset.py --test
```

---

## Step 2 — Hourly Open-Meteo Weather

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

## Step 3 — Join Weather → dataset_with_weather.parquet

**Script:** `add_weather.py`  
**Input:** `dataset.parquet`, `weather_hourly.parquet`, `station_data.parquet`, `station_metadata.parquet`  
**Output:** `dataset_with_weather.parquet`

Three sub-steps:

**3a. Stop → nearest MeteoSwiss station mapping**

Loads all 28,982 bus stop coordinates from `station_data.parquet` (Swiss LV95
grid, EPSG:2056), converts them to WGS84 (lat/lon) using `pyproj`, then uses
a `scipy` KDTree to find the nearest of the 158 MeteoSwiss stations for each
stop. Result: a `{bpuic → station_id}` dictionary held in memory.

**3b. Weather lookup table**

Loads `weather_hourly.parquet` (~80 MB) fully into RAM, indexed by
`(station_id, timestamp)` where timestamp is UTC-naive, floored to the hour.

**3c. Filtered join**

Reads `dataset.parquet` via DuckDB streaming:
1. **Drops the `pass_through` column** — all rows are already `pass_through = FALSE`
   (filtered in Step 1), so the column carries no signal.
2. Maps `stop_id` → `meteoswiss_station_id` via the dictionary from 3a.
3. Converts `timestamp` from Swiss local time (`Europe/Zurich`) to UTC, floors to hour.
4. Left-joins with the weather table on `(station_id, utc_hour)`.

Adds these columns to the dataset (all float32):
`temperature`, `precipitation`, `sunshine`, `humidity`,
`wind_speed`, `wind_gust`, `wind_dir`, `pressure`, `snow_depth`

Run:
```bash
pip install pyproj scipy   # first time only
python add_weather.py
```

---

## Step 4 — Drop Missing Weather → dataset_with_weather.parquet

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

## Step 5 — Add Lag Delay Feature

**Script:** `add_lag_delay.py`  
**Input:** `dataset_with_weather.parquet` (or any dataset with `trip_id` + `timestamp` + `arrival_delay_s`)  
**Output:** `dataset_with_weather.parquet` (in-place, `prev_stop_delay` column added)

Computes `prev_stop_delay = LAG(arrival_delay_s, 1) OVER (PARTITION BY CAST(timestamp AS DATE), trip_id ORDER BY timestamp)` using DuckDB's streaming window operator. Each partition is a single trip on a single day (~20–30 rows), so memory is bounded regardless of dataset size.

The first stop of each trip gets `NULL`. The `LagDelayEncoder` preprocessor fills these with the training median delay at ML time — no need to pick a fill value at the data level.

This step is optional but recommended: precomputing `prev_stop_delay` lets `LagDelayEncoder` take its fast path (O(1) fillna), avoiding the O(n) groupby that would otherwise run on the full in-memory DataFrame during training.

Run:
```bash
python add_lag_delay.py
# or on a specific file:
python add_lag_delay.py --path data/dataset.parquet
```

---

## Full Run Order

```bash
# 1: build dataset directly from ZIP archives (translate, filter, features,
#    outlier removal, holidays — all in one step, resume-safe)
python build_dataset.py --workers 4

# 2: fetch hourly weather (run once, resume-safe)
python fetch_weather_hourly.py

# 3: join weather to dataset (pass-through rows already filtered out in step 1)
python add_weather.py

# 4: drop rows with any missing weather field (~0.47% of rows)
python drop_missing_weather.py

# 5: precompute prev_stop_delay (lag feature) so ML training doesn't OOM
python add_lag_delay.py
```

### Utility: create a line/stop sub-dataset

To extract a filtered subset without touching the main file:

```bash
# Example: line 705, Echandens Chocolatière only
python filter_705_echandens.py
# → data/dataset_705_echandens.parquet (27,959 rows)
```

Uses DuckDB predicate pushdown — scans only relevant row groups.

### Utility: create a representative sample

`sample_dataset.py` extracts a smaller representative subset from any parquet file
without loading the full dataset into memory.

Two sampling modes:

**Reservoir** (default) — uniform random sample in a single pass via DuckDB.
Works on files of any size with memory bounded by the sample size, not the input.

**Stratified** (`--stratify-on COL`) — preserves the distribution of a column
by sampling proportionally within quantile bins. Reads only the stratify column
to compute bin edges, then uses a single `row_number() OVER (PARTITION BY stratum
ORDER BY random())` query to pick the exact number of rows per stratum.

```bash
# 1M uniform random rows
python sample_dataset.py data/dataset.parquet -o data/sample.parquet -n 1_000_000

# 10% stratified — preserves arrival_delay_s distribution
python sample_dataset.py data/dataset.parquet -o data/sample.parquet --frac 0.1 --stratify-on arrival_delay_s

# Exact count with custom bins and seed
python sample_dataset.py data/dataset.parquet -o data/sample.parquet -n 50000 --stratify-on arrival_delay_s --n-bins 20 --seed 123
```

| Flag | Description |
|------|-------------|
| `-o`, `--output` | Output parquet path (required) |
| `-n`, `--rows` | Exact number of rows to sample |
| `--frac` | Fraction of rows (e.g. `0.1` for 10%) |
| `--stratify-on` | Column to stratify by (preserves its distribution) |
| `--n-bins` | Quantile bins for stratification (default: 10) |
| `--seed` | Random seed (default: 42) |

Either `-n` or `--frac` is required (mutually exclusive). `-n` accepts
underscores as thousands separators (e.g. `1_000_000`).

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
    lag_delay.py                  # LagDelayEncoder (prev_stop_delay within trip)
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
    mlp_classifier.py             # MLPClassifierModel (sklearn MLPClassifier wrapper)
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

## Lag Delay Preprocessor

`LagDelayEncoder` provides the strongest causal signal available: how late the bus was
at the previous stop.

**Fast path (precomputed):** If `prev_stop_delay` is already in the dataset (added by
`add_lag_delay.py` or `build_dataset.py`), the preprocessor only fills NaN (first stop
of each trip) with the training median delay and drops `trip_id` / `arrival_delay_s`.
No groupby needed — O(1) operation.

**Slow path (runtime):** If `prev_stop_delay` is missing, computes it on the fly via
`groupby(_date, trip_id).shift(arrival_delay_s)`. Requires `timestamp`, `trip_id`, and
`arrival_delay_s` in X (`keep_target_in_X=True`).

Must be placed after `TemporalFeatureExtractor` (needs `timestamp` in the slow path).
Drops `trip_id` and `arrival_delay_s` after extracting the feature so the model never
sees the raw target.

```python
from ml.preprocessors.lag_delay import LagDelayEncoder

LagDelayEncoder()   # fillna uses global median delay from fit()
```

When `trip_id` is not in the dataset, the column is filled with the median delay
(constant feature — no signal, but keeps the pipeline consistent). This avoids
coupling pipeline state to experiment config.

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
| `OrdinalXGBoostClassifierOptimizer` | Macro-F1 | `OrdinalClassifierModel(XGBoostClassifierModel)` |
| `XGBoostClassifierOptimizer` | Macro-F1 | `XGBoostClassifierModel` |
| `CatBoostClassifierOptimizer` | Macro-F1 | `CatBoostClassifierModel` |
| `RandomForestClassifierOptimizer` | Macro-F1 | `RandomForestClassifierModel` |
| `MLPClassifierOptimizer` | Macro-F1 | `MLPClassifierModel` |

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

---

## Model Persistence (Save / Load)

Every model and pipeline can be serialized to disk and reloaded for inference without retraining.

### Per-model serialization

`BaseModel` provides default `save()` / `load()` via `joblib`, but tree-based models override it with their native formats for speed and smaller files:

| Model | Format | Method |
|-------|--------|--------|
| `LightGBMModel` | `booster_.save_model()` | Native LightGBM text/binary |
| `LightGBMClassifierModel` | Directory: booster + `meta.json` | Native LightGBM + class metadata |
| `XGBoostModel` | `.save_model()` | Native XGBoost UBJSON |
| `XGBoostClassifierModel` | Directory: booster + `meta.json` | Native XGBoost + `n_classes_` |
| `CatBoostModel` | `.save_model()` | Native CatBoost binary |
| `CatBoostClassifierModel` | `.save_model()` | Native CatBoost binary |
| `RidgeModel` | `joblib.dump()` | Implicit sklearn serialization |
| `LogisticRegressionModel` | `joblib.dump()` | Implicit sklearn serialization |
| `RandomForestClassifierModel` | `joblib.dump()` | Implicit sklearn serialization |
| `OrdinalClassifierModel` | `joblib.dump()` | Wraps base model's format |
| `StackingModel` | `joblib.dump()` | Serializes meta-model + base models |
| `ClassificationStackingModel` | `joblib.dump()` | Serializes meta-model + base models |
| `PipelinedClassifierModel` | `joblib.dump()` | Serializes inner classifier + preprocessors |
| `MLPClassifierModel` | `joblib.dump()` | Implicit sklearn serialization |

On `load()`, tree models reconstruct only the sklearn wrapper around the pre-trained booster — the trees themselves are loaded verbatim from disk. For LightGBM and XGBoost sklearn wrappers, internal attributes (`_Booster`, `_n_features`, `_n_features_in`, `fitted_`, and for classifiers `_classes`, `_le`, `_class_map`) are re-injected so `.predict()` and `.predict_proba()` work identically to a freshly fitted model.

### Model registry

A decorator-based registry in `ml/models/base.py` maps class names to types so composite models (stacking, ordinal) can polymorphically deserialize their base models:

```python
from ml.models.base import _register, _lookup

_register("LightGBMModel", LightGBMModel)          # called at import time
cls = _lookup("LightGBMModel")                     # → LightGBMModel class
```

All 14 model classes register themselves at the bottom of their respective modules. The registry is populated when `ml.models` is imported (via `ml/models/__init__.py`).

### Pipeline save/load

`MLPipeline.save(path)` persists the entire pipeline — preprocessors + model — into a directory:

```
saved_models/LGBM-v1/
  manifest.json           # {"model_type": "LightGBMClassifierModel", "n_preprocessors": 3}
  preprocessor_0.joblib    # TemporalFeatureExtractor (fitted)
  preprocessor_1.joblib    # FeatureScaler (fitted)
  preprocessor_2.joblib    # WeatherRushHourPreprocessor (fitted)
  model/                   # model-specific format
```

Preprocessors are serialized with `joblib` (all sklearn-compatible). The model is saved in its native format inside the `model/` subdirectory.

`MLPipeline.load(path)` reads the manifest, deserializes preprocessors in order, dispatches model loading via the registry, and returns a ready-to-use pipeline:

```python
pipeline = MLPipeline.load("saved_models/LGBM-v1")
y_pred = pipeline.predict(X)        # preprocessors transform() then model.predict()
```

### Training with save

In `main.py`, each experiment appends a `model.save()` call after fitting:

```python
pipeline = experiment.run(pipeline_only=True)
pipeline.save("saved_models/LGBM-v1")
```

---

## Streaming Inference (`inference.py`)

`inference.py` loads a saved pipeline and runs prediction on a parquet dataset using row-group-at-a-time streaming — the full dataset is never held in memory.

### Why streaming

Parquet files are organized into row groups (typically ~100K–1M rows each). `pq.ParquetFile.read_row_group(i)` reads one group at a time. The expensive wide feature DataFrame is discarded per chunk; only 1D prediction arrays accumulate.

### Usage

```bash
# Evaluate on full dataset
python inference.py -m saved_models/LGBM-v1 -d data/dataset_lausanne.parquet --ts

# Inference only (no target column needed), save predictions
python inference.py -m saved_models/LGBM-v1 -d data/dataset_lausanne.parquet --ts --no-eval -o predictions.parquet
```

**Arguments:**

| Flag | Description |
|------|-------------|
| `-m`, `--model` | Path to saved model directory (required) |
| `-d`, `--dataset` | Path to parquet dataset (required) |
| `-t`, `--target` | Target column name (default: `arrival_delay_s`) |
| `--ts` | Keep timestamp column (required if model uses `TemporalFeatureExtractor`) |
| `--drop-cols` | Override columns to drop |
| `--binner` | Delay binner for classification (default: `v1` = 90s binary split) |
| `--no-eval` | Skip evaluation — prediction only |
| `--no-class-names` | Use numeric labels in confusion matrix |
| `-o`, `--output` | Save predictions to `.parquet` or `.csv` |

### How it works

`DataLoader.stream()` opens the parquet file with `ParquetFile`, iterates over row groups, and yields `(X_chunk, y_chunk)` tuples. `inference.py` loops over chunks, calls `pipeline.predict(X_chunk)` on each, and accumulates results:

```
Row group 0  →  predict()  →  y_pred₀
Row group 1  →  predict()  →  y_pred₁
...
Row group N  →  predict()  →  y_predₙ

np.concatenate([y_pred₀, y_pred₁, ..., y_predₙ])  →  final predictions
```

Target labels (if available) are accumulated the same way for evaluation. Peak memory is ~one row group + one preprocessor transform buffer — the 1D prediction arrays cost only ~8 bytes per row (e.g., 5M rows ≈ 40 MB).
