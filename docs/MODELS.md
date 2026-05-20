# Model & Preprocessor Catalogue

Every model and preprocessor tested in this project, with performance benchmarks and implementation details.

---

## 1. Regression Models

All evaluated on the 705 dataset (386K train / 97K test, 38 features including `trip_stop_index`).

### Gradient-Boosted Trees

| Model | Hyperparameters | RMSE | R² | Fit Time | Status |
|-------|----------------|------|-----|----------|--------|
| **CatBoost** | `n_estimators=1410, lr=0.05149, depth=10, l2_leaf_reg=1.97, bagging_temp=0.77, min_data_in_leaf=72` | **32.06s** | **0.8800** | ~150s | ✅ Active |
| **LightGBM** | `n_estimators=1062, lr=0.01103, num_leaves=65, min_child_samples=76, subsample=0.93, colsample_bytree=0.58` | 32.72s | 0.8750 | ~42s | ✅ Active |
| **XGBoost** | `n_estimators=1102, lr=0.02089, max_depth=7, min_child_weight=6.07, subsample=0.96` | 33.05s | 0.8725 | ~61s | ✅ Active |
| **RandomForest** | `n_estimators=393, max_depth=26, min_samples_split=10, max_features=0.5` | 48.72s | 0.8847* | ~30s | ✅ Active |

\* *RandomForest R² from earlier dataset version without trip_stop_index. Current performance may differ.*

### Linear Models

| Model | Hyperparameters | RMSE | R² | Fit Time | Status |
|-------|----------------|------|-----|----------|--------|
| **Ridge** | `alpha=19.18` | 43.78s | 0.7762 | ~2s | ✅ Active |
| Ridge-PCA | `alpha=2.88, variance_threshold=0.99` | 54.21s | 0.7023 | ~5s | Active |
| Ridge-Poly | `alpha=2.88, degree=2` | 43.74s | 0.7767 | ~30s | Active |
| Ridge-Poly2 | `alpha=2.88, PolyTrig expander` | 43.72s | 0.7768 | ~30s | Active |
| Ridge-HistMean | `alpha=2.88, HistMeanEncoder` | 43.76s | 0.7764 | ~5s | Active |
| Ridge-Nystroem | `alpha=2.88, n_components=100, rbf` | 55.74s | 0.6852 | ~15s | Active |

### Ensemble & Stacking

| Model | Description | RMSE | R² | Fit Time | Status |
|-------|-------------|------|-----|----------|--------|
| **Stack-CB-Ridge** | CatBoost + Ridge base, Ridge meta, 5-fold | **32.05s** | **0.8801** | ~1124s | ✅ Best |
| Stack-LGBM-XGB-CB | 3 boosting models base, Ridge meta, 5-fold | 35.35s | 0.8491 | ~600s | Active |
| Stack-LGBM-CB | LightGBM + CatBoost base, Ridge meta, 5-fold | 35.36s | 0.8490 | ~400s | Active |
| Residual-LGBM-Ridge | LightGBM stage1, Ridge fits residuals, 5-fold | 36.64s | 0.8639 | ~200s | Active |
| Residual-LGBM-XGB | LightGBM stage1, XGBoost fits residuals, 5-fold | 35.47s | 0.8531 | ~150s | Active |

### Target-Transformed

| Model | Description | RMSE | R² | Status |
|-------|-------------|------|-----|--------|
| Log-LGBM | log(y) → LightGBM → exp(pred) | 37.48s | 0.8360 | Active |
| Log-XGB | log(y) → XGBoost → exp(pred) | 37.79s | 0.8333 | Active |
| Log-Ridge | log(y) → Ridge → exp(pred) | 234.91s | -4.59 | Active (broken) |

### Hierarchical & Ordinal

| Model | Description | RMSE | R² | Status |
|-------|-------------|------|-----|--------|
| Hier-LGBM-LGBM | Classifier → Regressor, bins=[60,120,300] | 40.53s | 0.8082 | Active |
| Hier-LGBM-Ridge | Classifier → Ridge regressor | 43.54s | 0.7787 | Active |
| Hier-LGBM-XGB | Classifier → XGBoost regressor | 40.33s | 0.8101 | Active |
| Ordinal-LGBM | K-1 binary LGBM classifiers | 54.58s | 0.6403 | Active |
| Ordinal-LGBM-fine | 5-class ordinal LGBM | 52.80s | 0.6746 | Active |
| Ordinal-XGB | K-1 binary XGBoost classifiers | 113.04s | 0.6034 | Active |

---

## 2. Classification Models

All evaluated on 4-class `[≤60s, 60–120s, 120–300s, >300s]` task unless noted.

### 705 Dataset (4-class)

| Model | Macro-F1 | Accuracy | Fit Time | Status |
|-------|----------|----------|----------|--------|
| **CatBoost** | **0.7936** | 0.7915 | ~771s | ✅ Best |
| **LightGBM** | 0.7798 | 0.7847 | ~155s | ✅ Active |
| **XGBoost** | 0.7778 | 0.7814 | ~224s | ✅ Active |

### Lausanne 50k Dataset (4-class)

| Model | Macro-F1 | Accuracy | Status |
|-------|----------|----------|--------|
| **CatBoost** | **0.7337** | 0.7228 | ✅ Best |
| LightGBM | 0.7320 | 0.7203 | Active |
| XGBoost | 0.7297 | 0.7186 | Active |

### Earlier Binary Classification (≤180s / >180s, historical)

| Model | Macro-F1 | Notes |
|-------|----------|-------|
| LGBM-Lag | 0.9148 | Best binary result (different task, not comparable to 4-class) |
| Ordinal-LGBM | 0.7421 | Frank & Hall K-1 threshold approach |
| Stacking-LGBM-RF | 0.7375 | 2-model stacking |
| Stacking-LGBM-RF-LogReg | 0.7365 | 3-model stacking with LogReg pipeline |
| LGBM-v1 | 0.7361 | Early LightGBM with basic hyperparams |
| RandomForest-v1 | 0.7337 | Early RandomForest |
| XGBoost-v1 | 0.7306 | Early XGBoost |
| CatBoost-v1 | 0.7177 | Early CatBoost |

### Logistic Regression (failed — lbfgs convergence)

| Model | Macro-F1 | Reason |
|-------|----------|--------|
| LogReg-Scaled | 0.163 | lbfgs failed to converge |
| LogReg-PCA | 0.526 | Converged but poor |
| LogReg-Nystroem | 0.552 | Best LR, still far behind trees |
| LogReg-PolyTrig | 0.537 | Polynomial trig features |
| LogReg-ElasticNet | 0.527 | saga solver, elasticnet penalty |

### MLP (To experiment with)

| Model | Macro-F1 | Notes |
|-------|----------|-------|
| MLP-Lag |  |  |
| MLP | |  |

---

## 3. Preprocessor Catalogue

All preprocessors in `ml/preprocessors/`. Each implements `fit(X, y=None)` and `transform(X)`.

### Temporal & Cyclical

| Preprocessor | File | Purpose |
|-------------|------|---------|
| **TemporalFeatureExtractor** | `temporal.py` | Extracts `hour`, `dow`, `month` from timestamp. Creates sin/cos cyclical encodings (`time_sin`, `time_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`) and `is_weekend` flag. Removes `timestamp` when done. |
| **SinusoidalEncoder** | `sinusoidal.py` | Generic cyclical encoding for any column (sin + cos mapping). Used internally by TemporalFeatureExtractor. |

### Weather

| Preprocessor | File | Purpose |
|-------------|------|---------|
| **WindMerger** | `wind_merger.py` | Merges `wind_speed` + `wind_gust` → `wind` (element-wise mean). Drops the original two columns. Solves the collinearity problem for linear models. |
| **WindEncoder** | `wind_encoder.py` | Encodes wind direction via sin/cos decomposition (`wind_x`, `wind_y`). |
| **WeatherFeatureEngineer** | `weather_engineer.py` | Creates `wind_chill` and interaction features from weather variables. |
| **WeatherRushHour** | `weather_rush_hour.py` | Creates weather × rush-hour interaction features. |

### Encoding

| Preprocessor | File | Purpose |
|-------------|------|---------|
| **StringEncoder** | `string_encoder.py` | Label-encodes categorical string columns (e.g., `operator`, `line`) to integer codes. Simple, fast, works well with tree models. |
| **HistoricalMeanEncoder** | `target_encoder.py` | Target encoding: replaces categorical values with the historical mean of the target for that group. Leakage-free via cross-validation folds. |
| **DelayBinner** | `delay_binner.py` | Bins continuous delay into discrete classes for classification. Default: `bins=[180]` (binary ≤180s/>180s). Our choice: `bins=[60, 120, 300]` (4-class). |
| **ClassEncoder** | `class_encoder.py` | Abstract base for target encoders in classification. |

### Scaling & Dimensionality

| Preprocessor | File | Purpose |
|-------------|------|---------|
| **FeatureScaler** | `scaler.py` | `StandardScaler` on specified numeric columns. Required for linear models (Ridge, LogReg). |
| **PCAReducer** | `pca.py` | PCA dimensionality reduction. Accepts `n_components` or `variance_threshold` (e.g., 0.99 to keep 99% variance). |
| **NystroemExpander** | `nystroem.py` | RBF kernel approximation via Nystroem method. Creates non-linear features for linear models. Default: 100 components. |
| **PolynomialExpander** | `polynomial.py` | `PolynomialFeatures(degree=2)` on specified columns. Creates interaction terms and squared features. |
| **PolyTrigExpander** | `poly_trig.py` | Hybrid: polynomial features on some columns + sin/cos cyclical encoding on others. |

### Lag & Spatial

| Preprocessor | File | Purpose |
|-------------|------|---------|
| **LagDelayEncoder** | `lag_delay.py` | Originally computed `prev_stop_delay` at runtime via `groupby(trip_id).shift()`. Now takes the fast path: `prev_stop_delay` is precomputed in the parquet, so it just fills NaN with the training median. |

---

## 4. Model Implementation Catalogue

All models in `ml/models/`. Each implements `fit(X, y)`, `predict(X)`, and serialization via `save(path)` / `load(path)`.

### Regression Models

| File | Class | Library | Key Feature |
|------|-------|---------|-------------|
| `catboost_model.py` | `CatBoostModel` | CatBoost | Native categorical support, GPU optional, early stopping |
| `lgbm.py` | `LightGBMModel` | LightGBM | Built-in validation split, early stopping callbacks |
| `xgboost_model.py` | `XGBoostModel` | XGBoost | Manual sample weights, early stopping |
| `ridge.py` | `RidgeModel` | sklearn | L2-regularized linear regression, fast |
| `random_forest_regressor.py` | `RandomForestRegressorModel` | sklearn | Custom early stopping via warm_start |
| `stacking.py` | `StackingModel` | custom | K-fold out-of-fold predictions, meta-model, polymorphic save/load |
| `residual_stacking.py` | `ResidualStackingModel` | custom | Stage 1 model → residuals → Stage 2 model |
| `log_target.py` | `LogTargetModel` | custom | Wraps any regressor with log(y) → exp(pred) transform |
| `hierarchical.py` | `HierarchicalRegressor` | custom | Classifier bins target → separate regressor per bin |
| `ordinal_regressor.py` | `OrdinalRegressorModel` | custom | Frank & Hall ordinal regression via K-1 binary classifiers |

### Classification Models

| File | Class | Library | Key Feature |
|------|-------|---------|-------------|
| `catboost_classifier.py` | `CatBoostClassifierModel` | CatBoost | Multi-class, auto_class_weights, early stopping |
| `lgbm_classifier.py` | `LightGBMClassifierModel` | LightGBM | scale_pos_weight, stratified validation split, save/load with metadata |
| `xgboost_classifier.py` | `XGBoostClassifierModel` | XGBoost | sample_weight for class balancing, early stopping |
| `random_forest_classifier.py` | `RandomForestClassifierModel` | sklearn | warm_start early stopping, class_weight options |
| `classification_stacking.py` | `ClassificationStackingModel` | custom | Stratified K-fold, predict_proba meta-features, polymorphic save/load |
| `logistic_regression.py` | `LogisticRegressionModel` | sklearn | saga/elasticnet solver, L1/L2 mix |
| `ordinal_classifier.py` | `OrdinalClassifierModel` | custom | K-1 binary decomposition (Frank & Hall) |
| `mlp_classifier.py` | `MLPClassifierModel` | sklearn | Configurable hidden layers, early stopping |
| `pipelined_classifier.py` | `PipelinedClassifierModel` | custom | Inline preprocessor chain within a classifier (used in stacking) |

---

## 5. Optimizer Catalogue

All optimizers in `ml/optimizer.py`. Each uses Optuna TPE sampling with built-in early stopping.

| Optimizer Class | Target Model | Direction | Trials Default | Notes |
|----------------|-------------|-----------|----------------|-------|
| `LGBMRegressorOptimizer` | LightGBM regressor | Minimize RMSE | 100 | 15-param search space |
| `XGBoostRegressorOptimizer` | XGBoost regressor | Minimize RMSE | 100 | 11-param search space |
| `CatBoostRegressorOptimizer` | CatBoost regressor | Minimize RMSE | 100 | 6-param search space |
| `RidgeRegressorOptimizer` | Ridge regressor | Minimize RMSE | 50 | 1-param (alpha) |
| `RandomForestRegressorOptimizer` | RandomForest regressor | Minimize RMSE | 100 | 5-param search space |
| `LGBMClassifierOptimizer` | LightGBM classifier | Maximize Macro-F1 | 100 | Stratified inner split |
| `XGBoostClassifierOptimizer` | XGBoost classifier | Maximize Macro-F1 | 100 | Stratified inner split |
| `CatBoostClassifierOptimizer` | CatBoost classifier | Maximize Macro-F1 | 100 | Stratified, auto_class_weights tuning |
| `RandomForestClassifierOptimizer` | RandomForest classifier | Maximize Macro-F1 | 100 | class_weight tuning |
| `OrdinalLGBMClassifierOptimizer` | Ordinal LGBM | Maximize Macro-F1 | 75 | No inner early stopping |
| `OrdinalXGBoostClassifierOptimizer` | Ordinal XGBoost | Maximize Macro-F1 | 75 | No inner early stopping |
| `MLPClassifierOptimizer` | MLP classifier | Maximize Macro-F1 | 100 | Layer count/size tuning |

---

## 6. Low-End Hardware Engineering

The entire project was developed and tested on a machine with **8 GB of RAM**, yet it processes a **16 GB full-Switzerland dataset** (509 million rows) and trains models on 386K-row feature matrices. This section documents every memory optimization that makes this possible.

### 6.1 Core Principle: Out-of-Core Processing

**Never load the full dataset into memory.** Every script that touches a parquet file uses DuckDB's streaming query engine, which reads row groups sequentially and spills intermediate results to disk when they exceed `memory_limit`. The peak RAM usage is bounded by `PRAGMA memory_limit` — typically 2–4 GB — regardless of input file size.

This applies to every data pipeline script:

| Script | Input Size | Memory Limit | Peak RAM |
|--------|-----------|-------------|----------|
| `build_dataset.py` | 12 × 3GB ZIP archives | 8 GB | ~300 MB |
| `add_weather.py` | 3.5 GB dataset + 80 MB weather | 8 GB | ~1 GB |
| `add_lag_delay.py` | 16 GB / 499M rows | 3–4 GB | ~300 MB |
| `add_trip_stop_index.py` | 16 GB / 499M rows | 3–4 GB | ~300 MB |
| `add_traffic_features.py` | 16 GB / 509M rows | 8 GB | ~1 GB |
| `add_stop_distance.py` | 16 GB / 509M rows | 8 GB | ~2 GB |
| `filter_lausanne.py` | 16 GB / 509M rows | 2 GB | ~500 MB |
| `sample_dataset.py` | Any size | N/A (reservoir) | O(sample_n) |

### 6.2 Date-by-Date Partitioning

For operations that require sorted window functions (lag delay, trip stop index), processing the full dataset in one shot would require sorting 509M rows — an operation that easily exceeds 8 GB of RAM. The solution:

```python
# Instead of: sort 509M rows → window function
# We do:
for each distinct date:
    filter to that date (reads only relevant row groups)
    sort ~1M rows (fits in RAM)
    apply ROW_NUMBER() / LAG()
    write to temp parquet
merge all date files
```

Each date is ~1–2 million rows. Sorting one date's worth of data fits comfortably in a few hundred MB. After all dates are processed independently, the per-date parquet files are merged with a simple `COPY (SELECT * FROM read_parquet('date_*.parquet'))` — a streaming operation that never holds more than one row group in memory.

This is used in:
- `add_lag_delay.py` — `LAG(arrival_delay_s) OVER (PARTITION BY trip_id ORDER BY timestamp)` computed per-date
- `add_trip_stop_index.py` — `ROW_NUMBER() OVER (PARTITION BY trip_id ORDER BY timestamp)` computed per-date

### 6.3 Reservoir Sampling

For ML experiments that don't need the full dataset, `DataLoader(sample_n=N)` uses DuckDB's reservoir sampling:

```python
SELECT * FROM read_parquet('file.parquet')
USING SAMPLE 50000 ROWS (reservoir, 42)
```

Reservoir sampling scans the file once and keeps exactly N rows in memory — the memory footprint is O(N) regardless of input size. A 50K sample from a 16 GB file uses ~50 MB of RAM. This is central to the Optuna optimization workflow, where 50 trials each train on a fresh 50K sample instead of 386K full rows (12.5× speedup, 8× less RAM).

### 6.4 Precomputed Features

Computing `prev_stop_delay` at runtime during ML training would require a pandas `groupby(['trip_id']).shift()` on the full training DataFrame (386K rows × 34 columns ≈ 100 MB). This was the original design with `LagDelayEncoder` and it worked for small datasets, but it risked OOM on larger ones. Worse, the stacking model's 5-fold CV would recompute it 5 times.

By precomputing `prev_stop_delay` (and `dist_to_prev_stop`, `trip_stop_index`, weather, traffic, holidays) directly in the parquet file, the ML training code just reads columns it needs. No runtime groupby, no temporary copies, no memory spikes.

### 6.5 Row-Group-at-a-Time Streaming

`DataLoader.stream()` reads one parquet row group at a time for memory-efficient iteration:

```python
def stream(self):
    pf = pq.ParquetFile(self.path)
    for i in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(i).to_pandas()
        # process chunk, free, continue
        yield X_chunk, y_chunk
```

Each row group is typically ~500K rows (set by `ROW_GROUP_SIZE 500000` in all pipeline scripts). This means even on a 509M-row file, the per-chunk memory is bounded to ~500K rows × 38 columns ≈ 150 MB maximum.

### 6.6 Parquet Compression

All parquet files use **Snappy compression** (or Zstd for distance data). Snappy prioritizes decompression speed over compression ratio, giving ~2–3× size reduction with negligible CPU overhead. The full 509M-row dataset compresses from ~40 GB raw to 16 GB on disk, and decompresses transparently during DuckDB scans.

### 6.7 PyArrow CSV Parsing (Not Pandas)

`build_dataset.py` uses PyArrow's native CSV reader instead of pandas:

```python
table = pcsv.read_csv(io.BytesIO(raw_bytes), read_options=ro, parse_options=po)
```

PyArrow's CSV parser is implemented in C++ and is ~11× faster than `pd.read_csv()`. More importantly for memory: it produces Arrow columnar format directly, avoiding the pandas intermediate DataFrame copy. One CSV (~200 MB raw) becomes ~30 MB after filtering, and the Python reference is freed immediately after writing the temp parquet.

### 6.8 DuckDB Thread Limiting

All data pipeline scripts limit DuckDB to 2 threads (`PRAGMA threads=2`). This is intentional:

- **Memory:** Each thread allocates its own hash tables and sort buffers. On a 4-core machine with 8 GB RAM, 4 threads could allocate 4 × 2 GB = 8 GB just for DuckDB, leaving nothing for the OS and other processes.
- **Spilling:** With 2 threads and a 3 GB memory limit, DuckDB is forced to spill to disk early. This is actually desirable — graceful spilling is better than hitting the OS OOM killer.
- **ML training doesn't need parallelism anyway** — LightGBM/CatBoost/XGBoost all use `n_jobs=-1` internally, so DuckDB threads would compete for CPU.

### 6.9 Explicit `del` and Garbage Collection

In `to_parquet.py` and the early pipeline scripts, large objects are explicitly deleted after use:

```python
del table  # free Arrow memory immediately
del result # free pandas memory before writing
```

While Python's GC would eventually reclaim these, explicit `del` ensures the memory is freed *before* the next allocation, preventing transient double-accounting that pushes total usage past the 8 GB limit.

### 6.10 The 50K Stratified Sample for Optuna

Hyperparameter optimization with 50 trials on the full 386K training set would take ~12 hours and risk OOM from accumulated model objects. The solution:

1. **Phase 1:** Create a 50K stratified sample via `sample_dataset.py --stratify-on arrival_delay_s` — preserves delay distribution in a 50K subset (~30 MB in memory)
2. **Run all 50 Optuna trials on the sample** — each trial trains on 40K rows in ~3–4 seconds (binary) or ~30 seconds (4-class)
3. **Phase 2:** Take the best hyperparameters and evaluate once on the full 386K training set

This gives a 12.5× speedup (97s → 7.8s per LightGBM fit) and keeps peak memory during optimization under 1 GB.

### 6.11 Single-Load Pattern in Optimizers

All Optuna optimizers in `ml/optimizer.py` use a `_load_once()` cache:

```python
def _load_once(self):
    if self._data is None:
        self._data = self.loader.load()
    return self._data
```

The dataset is loaded once and reused across all trials. Without this, each of 50 trials would re-read the parquet, re-apply StringEncoder, re-extract temporal features — wasting both time and memory.

### 6.12 Memory Profile Summary

| Operation | Dataset Size | Peak RAM | Technique |
|-----------|-------------|----------|-----------|
| Build dataset from ZIP | 12 × 3 GB | ~300 MB | Per-CSV streaming, Arrow CSV |
| Join weather | 3.5 GB + 80 MB | ~1 GB | DuckDB hash join with spill |
| Add lag delay | 16 GB / 499M | ~300 MB | Date-by-date partitioning |
| Add trip stop index | 16 GB / 499M | ~300 MB | Date-by-date partitioning |
| Add stop distance | 16 GB / 509M | ~2 GB | DuckDB external sort |
| Filter to Lausanne | 16 GB / 509M | ~500 MB | Predicate pushdown |
| Stratified sample | 16 GB / 509M | ~50 MB | Per-stratum reservoir |
| ML training (full) | 386K rows | ~3 GB | CatBoost/LightGBM native |
| ML training (sample) | 50K rows | ~500 MB | Reservoir sampling |
| Stacking 5-fold CV | 386K rows | ~4 GB | Model serial + del |
| Optuna 50 trials | 50K sample | ~1 GB | _load_once cache |

The project never exceeded the 8 GB limit during development. The 16 GB full-Switzerland dataset was processed, filtered, and modeled entirely on a machine with 8 GB of RAM.

---

## 7. Performance Summary

### Best Overall

| Task | Dataset | Best Model | Metric | Score |
|------|---------|-----------|--------|-------|
| Regression | 705 | Stack-CB-Ridge | R² | **0.8801** |
| Regression | 705 (fast) | CatBoost | R² | **0.8800** |
| Regression | Lausanne 50k | Stack-CB-Ridge | R² | **0.8491** |
| Classification | 705 (4-class) | CatBoost | Macro-F1 | **0.7936** |
| Classification | Lausanne 50k (4-class) | CatBoost | Macro-F1 | **0.7337** |
| Classification | 705 (binary) | LightGBM | Macro-F1 | **0.9148** |

### What Works Best

1. **CatBoost is the best single model** across both regression and classification, on both datasets
2. **Stacking CatBoost + Ridge** provides marginal gains (0.0001 R²) when features are strong, and larger gains when base models are complementary
3. **LightGBM is the fastest** — trains 3.5× faster than CatBoost for 99.4% of the performance
4. **Ridge is the best linear model** but only competitive on small datasets (<50K rows)
5. **Feature engineering beyond the basic pipeline adds nothing** — PCA, Nystroem, and polynomial expansions don't improve tree models

### What Doesn't Work

| Approach | Outcome | Root Cause |
|----------|---------|------------|
| Logistic Regression | F1 < 0.55 | lbfgs fails on sparse StringEncoder features |
| MLP Neural Network | F1 = 0.45 | Only predicts majority class |
| Log-Ridge (log transform) | R² = -4.59 | Log transform fails when y has negative values |
| Target encoding | Zero gain | Trees handle categorical natively |
| Ordinal regression | R² < 0.68 | Binning loses information vs direct regression |
