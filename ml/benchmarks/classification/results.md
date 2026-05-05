# Classification Benchmarks — Echandens 705

Dataset: `data/dataset_705_echandens.parquet`  
Target: `arrival_delay_s` → 4 classes via `DelayBinner(bins=[60, 180, 600])`  
Split: 80/20 time-ordered  
Primary metric: **Macro-F1** (punishes poor minority-class performance equally)  
Date range: April–May 2026

## Class Definitions

| Class | Label | Delay range | Approx. frequency |
|---|---|---|---|
| 0 | ≤60s | on-time / negligible | ~55% |
| 1 | 60–180s | minor delay | ~28% |
| 2 | 180–600s | moderate delay (SBB threshold) | ~16.7% |
| 3 | >600s | severe delay | ~0.26% (71 samples in test set) |

**Key constraint:** Class 3 has only ~71 test samples — macro-F1 is extremely sensitive to class-3 precision/recall. A model that gets 0 class-3 predictions scores macro-F1 ≈ 0.33.

---

## LogReg-L2 (Baseline)

**Preprocessors:** `TemporalFeatureExtractor` → `FeatureScaler(NUMERIC_COLS + TEMPORAL_COLS)`  
**No HistoricalMeanEncoder**

**Hyperparameters:**
```
C:             1.0
solver:        lbfgs
max_iter:      2000
class_weight:  balanced
```

**Results:**
```
Macro-F1:  ~0.17
```

**Finding:** Without hist_mean_delay, logistic regression cannot distinguish classes 0 and 1 (only 60s apart).

---

## LogReg-C0.1

**Preprocessors:** `TemporalFeatureExtractor` → `WeatherFeatureEngineer` → `HistoricalMeanEncoder(hour+dow)` → `FeatureScaler(NUMERIC_COLS + TEMPORAL_COLS + [wind_chill, hist_mean_delay])`

**Hyperparameters:**
```
C:             0.1
solver:        lbfgs
max_iter:      5000
class_weight:  balanced
```

**Results:**
```
Macro-F1:  0.1959
```

---

## LogReg-C1.0

**Preprocessors:** Same as LogReg-C0.1

**Hyperparameters:**
```
C:             1.0
solver:        lbfgs
max_iter:      5000
class_weight:  balanced
```

**Results:**
```
Macro-F1:  0.2071
```

---

## LogReg-C10.0

**Preprocessors:** Same as LogReg-C0.1

**Hyperparameters:**
```
C:             10.0
solver:        lbfgs
max_iter:      5000
class_weight:  balanced
```

**Results:**
```
Macro-F1:  0.2107
```

---

## LogReg-L1

**Preprocessors:** Same as LogReg-C0.1  

**Hyperparameters:**
```
C:             1.0
solver:        saga  (auto-switched for L1)
max_iter:      5000
class_weight:  balanced
l1_ratio:      1.0  (pure L1)
```

**Results:**
```
Macro-F1:  0.2050
```

---

## LogReg-ElasticNet ← Best Linear Model

**Preprocessors:** Same as LogReg-C0.1  

**Hyperparameters:**
```
C:             1.0
solver:        saga  (auto-switched for ElasticNet)
max_iter:      5000
class_weight:  balanced
l1_ratio:      0.5
```

**Results:**
```
Macro-F1:  0.2284
```

**Finding:** ElasticNet slightly best, but all LogReg variants plateau below 0.23. Classes 0 and 1 overlap in feature space — linear separation is insufficient.

---

## LGBM-Classifier-Round1 (historical reference)

**Preprocessors:** `TemporalFeatureExtractor` → `WeatherFeatureEngineer` → `HistoricalMeanEncoder(hour+dow)`  
**Note:** Non-stratified validation split for early stopping (may have inflated F1 due to chance)

**Hyperparameters:**
```
n_estimators:          1000
early_stopping_rounds: 50
learning_rate:         0.05
num_leaves:            63
min_child_samples:     10
subsample:             0.8
subsample_freq:        1
colsample_bytree:      0.8
reg_alpha:             0.1
reg_lambda:            0.1
class_weight:          balanced
```

**Results:**
```
Macro-F1:  0.4275
```

---

## LGBM-v1 ← Best Classifier (stratified split)

**Preprocessors:** `TemporalFeatureExtractor` → `WeatherFeatureEngineer` → `HistoricalMeanEncoder(hour+dow)`  
**Fix vs Round1:** Stratified validation split for early stopping (ensures class 3 is always in val set)

**Hyperparameters:**
```
n_estimators:          1000
early_stopping_rounds: 50
learning_rate:         0.05
num_leaves:            63
min_child_samples:     10
subsample:             0.8
subsample_freq:        1
colsample_bytree:      0.8
reg_alpha:             0.1
reg_lambda:            0.1
class_weight:          balanced
```

**Results:**
```
Macro-F1:  0.4215
```

---

## LGBM-v2-LowChild

**Preprocessors:** Same as LGBM-v1  
**Motivation:** Lower min_child_samples to allow splits on the 71-sample class 3

**Hyperparameters:**
```
n_estimators:              2000
early_stopping_rounds:     50
learning_rate:             0.03
num_leaves:                63
min_child_samples:         1       ← changed from 10
min_sum_hessian_in_leaf:   1e-5    ← added
subsample:                 0.8
subsample_freq:            1
colsample_bytree:          0.8
reg_alpha:                 0.1
reg_lambda:                0.1
class_weight:              balanced
```

**Results:**
```
Macro-F1:  0.4186
```

**Finding:** Lowering min_child_samples risks overfitting on the minority class; marginal degradation.

---

## LGBM-v3-DeepHist

**Preprocessors:** `TemporalFeatureExtractor` → `WeatherFeatureEngineer` → `HistoricalMeanEncoder(hour+dow)` → `HistoricalMeanEncoder(hour)` → `HistoricalMeanEncoder(dow)`  
**Motivation:** Richer historical lag features — per-hour and per-dow mean delays as separate columns

**Hyperparameters:**
```
n_estimators:              2000
early_stopping_rounds:     50
learning_rate:             0.03
num_leaves:                127     ← deeper trees
min_child_samples:         5
subsample:                 0.8
subsample_freq:            1
colsample_bytree:          0.8
feature_fraction_bynode:   0.9
reg_alpha:                 0.05
reg_lambda:                0.05
class_weight:              balanced
```

**Results:**
```
Macro-F1:  0.4156
```

**Finding:** Additional granular hist features don't improve over combined hour+dow encoder.

---

## LGBM-v4-HeavyClass3

**Preprocessors:** Same as LGBM-v1  
**Motivation:** Manual class weights with extreme boost on class 3 to force the model to learn it

**Hyperparameters:**
```
n_estimators:          1000
early_stopping_rounds: 50
learning_rate:         0.05
num_leaves:            63
min_child_samples:     1
min_sum_hessian_in_leaf: 1e-5
subsample:             0.8
subsample_freq:        1
colsample_bytree:      0.8
reg_alpha:             0.1
reg_lambda:            0.1
class_weight:          {0: 1.0, 1: 1.08, 2: 3.32, 3: 100.0}
```

**Results:**
```
Macro-F1:  0.4214
```

**Finding:** 100× weight on class 3 achieves similar F1 to balanced weights — not a meaningful gain.

---

## XGBoost-Classifier

**Preprocessors:** `TemporalFeatureExtractor` → `WeatherFeatureEngineer` → `HistoricalMeanEncoder(hour+dow)`  
**Note:** XGBoost doesn't support `class_weight` natively — uses `compute_sample_weight("balanced", y)` at fit time

**Hyperparameters:**
```
n_estimators:          1000
early_stopping_rounds: 50
learning_rate:         0.05
max_depth:             6
min_child_weight:      5.0
subsample:             0.8
colsample_bytree:      0.8
reg_alpha:             0.1
reg_lambda:            1.0
class_weight:          balanced  (→ compute_sample_weight internally)
```

**Results:**
```
Macro-F1:  0.4102
```

---

## CatBoost-Balanced (broken)

**Motivation:** Test CatBoost with `auto_class_weights="Balanced"`  
**Finding:** Assigns 175× weight to class 3, causing catastrophic over-prediction (model predicts class 3 everywhere). Not usable.

**Results:** Not recorded (catastrophic failure).

---

## CatBoost-NoWeights

**Preprocessors:** Same as LGBM-v1  
**Note:** `auto_class_weights=None` — natural class distribution

**Hyperparameters:**
```
n_estimators:          1000
early_stopping_rounds: 50
learning_rate:         0.05
depth:                 6
l2_leaf_reg:           3.0
min_data_in_leaf:      5
auto_class_weights:    None
eval_metric:           Accuracy
```

**Results:**
```
Macro-F1:  0.3366
```

**Finding:** Without class balancing, CatBoost ignores class 3 almost entirely. CatBoost is unsuitable for this extreme imbalance with its current API.

---

## Summary Table

| Experiment | Macro-F1 | Notes |
|---|---|---|
| LogReg-L2 (no hist features) | ~0.17 | baseline |
| LogReg-C0.1 | 0.1959 | |
| LogReg-L1 | 0.2050 | |
| LogReg-C1.0 | 0.2071 | |
| LogReg-C10.0 | 0.2107 | |
| LogReg-ElasticNet | 0.2284 | best linear |
| CatBoost-NoWeights | 0.3366 | no class balancing |
| LGBM-v2-LowChild | 0.4186 | min_child_samples=1 |
| LGBM-v3-DeepHist | 0.4156 | 3× HistoricalMeanEncoder |
| XGBoost-Classifier | 0.4102 | sample_weight |
| LGBM-v4-HeavyClass3 | 0.4214 | 100× class-3 weight |
| **LGBM-v1** | **0.4215** | balanced, stratified val |
| LGBM-Classifier-Round1 | 0.4275 | non-stratified val (likely inflated) |

**Best reproducible result:** LGBM-v1 — Macro-F1=0.4215  

**Performance ceiling note:** Class 3 has only ~71 test samples. A model that correctly predicts 50% of them while maintaining other classes already scores ~0.42. The marginal gap between all LGBM variants (0.41–0.43) suggests this is the data ceiling — more samples of severe delays would be needed to push further.

## Key Findings

1. **HistoricalMeanEncoder is essential for classification** — without it, logistic regression cannot distinguish classes 0 and 1 (mean delays 30s vs 120s).
2. **Tree models dominate** — LGBM outperforms LogReg by ~0.19 F1 points.  
3. **CatBoost is incompatible** with this imbalance — neither "Balanced" weights nor no weights works.
4. **Stratified early-stopping split matters** — prevents the val set from having zero class-3 samples, which caused early stopping to fire on wrong criteria in Round 1.
5. **Class weights beyond "balanced" don't help** — the bottleneck is data volume, not model focus.
