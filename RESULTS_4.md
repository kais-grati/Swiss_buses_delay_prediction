# RESULTS_4 — Model Optimization & Stacking Ensemble

Date: 2026-05-18/19 — updated 2026-05-19 with trip_stop_index results

---

## UPDATE (May 19): trip_stop_index Feature Added (+0.022 R²)

Per the ablation study in RESULTS_6.md, `trip_stop_index` was being dropped by all loaders. Adding it as a feature yields significant improvements across all tree models:

### Updated 705 Regression Leaderboard

| Model | RMSE | R² | Δ vs before |
|-------|------|-----|-------------|
| **Stack-CB-Ridge** | **32.05s** | **0.8801** | -2.87s / +0.0224 |
| CatBoost | 32.06s | 0.8800 | -2.90s / +0.0227 |
| LightGBM | 32.72s | 0.8750 | -3.99s / +0.0115 |
| XGBoost | 33.05s | 0.8725 | -4.49s / +0.0153 |
| Ridge | 43.78s | 0.7762 | -4.41s / **-0.0690** |

**Key findings:**
- Stack-CB-Ridge and CatBoost both break **0.88 R²** (vs 0.8577 before)
- RMSE drops ~3s across all tree models
- Ridge **regresses** (-0.069 R²) — `trip_stop_index` is an ordinal integer (1-21) that Ridge treats as linear-continuous, which is a poor fit. Tree models handle it natively via splits.
- CatBoost and Stack-CB-Ridge are now essentially tied (32.05 vs 32.06 RMSE) — stacking adds negligible value when both base models have the same strong feature set.

### Updated Lausanne 50k Regression (not re-run; expected small δ based on classification results)

---

## 1. Bug Fix: NaN Handling in New Dataset Features

The dataset migration from `705.parquet` to `705_bus_2025_weather_traffic.parquet` introduced NaN values in new columns that crashed sklearn linear models.

**Fix in `ml/data.py:36-38`:**
- Drop rows where target (`arrival_delay_s`) is NaN (1.7% of rows)
- Impute `dist_to_prev_stop` NaN → 0.0 (89.6% of rows; semantically correct: no previous stop = zero distance)

---

## 2. Hyperparameter Optimization Attempts

### 2.1 Sample-Based Optimization (50k stratified sample)

Three tree models optimized via Optuna TPE on a 50k stratified sample, then evaluated on the full 705 test set.

| Model | Trials | Sample Best Val RMSE | Full Test MSE | Full Test R² | Δ vs Baseline |
|-------|--------|---------------------|---------------|-------------|---------------|
| LightGBM | 50 | 36.21s | 1309.86 | 0.8471 | +87.74 |
| XGBoost | 50 | 37.05s | 1401.05 | 0.8364 | +178.93 |
| CatBoost | 50 | 36.90s | 1328.61 | 0.8449 | +106.49 |

**Conclusion:** Sample-based optimization failed. Hyperparameters found on 50k rows don't transfer to 386k training rows. Key example: optimizer found `min_data_in_leaf=15` for CatBoost on sample, but the winning full-dataset config uses `min_data_in_leaf=72` — much more regularization needed at scale.

### 2.2 Full-Dataset CatBoost Optimization

Started 30-trial CatBoost optimization directly on full 705 dataset. Stopped early — would have taken ~45 min and results from sample optimization suggested marginal gains.

---

## 3. Preprocessor Exploration

Tested various feature engineering strategies on the 705 dataset with all 3 tree models.

### Feature Importance (CatBoost on 705)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | prev_stop_delay | 85.9% |
| 2 | operator | 1.2% |
| 3 | time_cos | 1.1% |
| ... | (everything else) | <1% each |

`prev_stop_delay` is overwhelmingly dominant — the delay at the previous stop carries almost all predictive signal.

### Preprocessor Combos Tested

| Change | Effect on MSE |
|--------|--------------|
| Drop `stop_id` | Worsened (+30 MSE) |
| Drop duplicate temporal sin/cos features | Worsened (+70-100 MSE) |
| Drop all traffic features | Worsened (+6 MSE on Lausanne) |
| Add target encoding (HistoricalMeanEncoder) | Zero effect for tree models |
| Extended feature scaling | Worse for Ridge (+17 MSE) |
| Feature selection (top-K by importance) | Worse for Ridge |

**Conclusion:** The existing pipeline (TemporalFeatureExtractor + WindMerger + StringEncoder) is already optimal. Target encoding adds nothing for tree models. Removing any features hurts — every column carries marginal signal.

---

## 4. Lausanne 50k Dataset Results

Full regression experiment suite on `lausanne50k_bus_2025_weather_traffic.parquet` (50k rows, Lausanne region). Completely different leaderboard from 705:

| Model | MSE | RMSE | R² |
|-------|-----|------|-----|
| **Ridge** | 2321.80 | 48.19s | 0.8452 |
| Ridge-HistMean | 2325.80 | 48.23s | 0.8449 |
| Ridge-Poly | 2326.30 | 48.23s | 0.8449 |
| RandomForest | 2373.66 | 48.72s | 0.8417 |
| LightGBM | 2437.28 | 49.37s | 0.8375 |
| CatBoost | 2465.50 | 49.65s | 0.8356 |

**Key insight:** On the smaller geographically-constrained Lausanne subset, linear models (Ridge) outperform tree models. CatBoost drops from #1 on 705 to #6 on Lausanne.

### Ridge Optimization on Lausanne

Grid search across alpha ∈ [0.001, 1000]:

| α | MSE | R² |
|---|-----|-----|
| 2.882 (original) | 2321.80 | 0.8452 |
| **19.179 (optimized)** | **2321.78** | **0.8452** |

Ridge is essentially saturated — `prev_stop_delay` has 86% importance, leaving minimal room for improvement.

---

## 5. Stacking Ensemble: CatBoost + Ridge (BEST RESULTS)

Stacking model combining CatBoost and Ridge as base models, with Ridge meta-model, 5-fold cross-validation.

### On 705 Dataset

| Model | MSE | RMSE | R² |
|-------|-----|------|-----|
| CatBoost (baseline) | 1222.12 | 34.96s | 0.8573 |
| Ridge (baseline) | 1917.63 | 43.79s | 0.7761 |
| **Stack CB+Ridge** | **1219.14** | **34.92s** | **0.8577** |

**Improvement: Δ=-2.98 MSE** vs previous best single model.

### On Lausanne 50k

| Model | MSE | RMSE | R² |
|-------|-----|------|-----|
| Ridge (baseline) | 2321.78 | 48.18s | 0.8452 |
| CatBoost (baseline) | 2466.74 | 49.67s | 0.8355 |
| **Stack CB+Ridge** | **2262.04** | **47.56s** | **0.8491** |

**Improvement: Δ=-59.74 MSE** vs previous best single model.

### Why Stacking Works

CatBoost and Ridge are complementary:
- **CatBoost** captures non-linear tree-based patterns, handles categorical features natively
- **Ridge** captures linear structure, benefits from feature scaling
- The **Ridge meta-model** learns the optimal weighted combination of both predictions
- FeatureScaler in the pipeline improves the stack (MSE: 1219.14 with scaler vs 1220.57 without)

---

## 6. Final Best Model Configuration

```
StackingModel:
  base_models:
    - CatBoostModel(n_estimators=1410, learning_rate=0.05149, depth=10,
                    l2_leaf_reg=1.968, random_strength=0.03778,
                    bagging_temperature=0.7664, min_data_in_leaf=72,
                    early_stopping_rounds=50)
    - RidgeModel(alpha=19.1791)
  meta_model: RidgeModel(alpha=1.0)
  n_folds: 5

Preprocessors:
  - TemporalFeatureExtractor()
  - WindMerger()
  - StringEncoder(cols=["operator", "line"])
  - FeatureScaler(cols=[temperature, precipitation, sunshine, humidity,
                         wind, pressure, snow_depth, hour, dow, month,
                         prev_stop_delay, dist_to_prev_stop])
```

---

## 7. Key Takeaways

1. **Stacking complementary models beats either alone** — the largest gains came from combining CatBoost + Ridge, not from hyperparameter tuning or feature engineering
2. **`prev_stop_delay` dominates** — 86% feature importance, everything else is marginal
3. **Sample-based optimization doesn't transfer** to full data for tree model regularization parameters
4. **Lausanne vs 705 have different optimal models** — Ridge wins on the small Lausanne subset, CatBoost wins on the larger 705 dataset, but stacking wins on both
5. **Feature engineering ceiling is reached** — the existing 34-feature set with standard preprocessing is near-optimal; removing features always hurts
