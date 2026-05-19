# RESULTS 5 — Multi-Class Classification on 705 & Lausanne 50k

**Date:** 2026-05-19 — updated with trip_stop_index results
**Goal:** Best possible classification with precise delay range bins

---

## UPDATE (May 19): trip_stop_index Feature Added (+0.036 F1 on 705)

Per the ablation study in RESULTS_6.md, `trip_stop_index` was being dropped by all loaders. Adding it improves classification substantially on 705, minimally on Lausanne 50k:

### Updated 705 Classification Leaderboard (4-class)

| Model | Macro-F1 | Accuracy | Δ vs before |
|-------|----------|----------|-------------|
| **CatBoost** | **0.7936** | 0.7915 | **+0.0365** |
| LightGBM | 0.7798 | 0.7847 | +0.0339 |
| XGBoost | 0.7778 | 0.7814 | +0.0358 |

### Updated Lausanne 50k Classification (4-class)

| Model | Macro-F1 | Accuracy | Δ vs before |
|-------|----------|----------|-------------|
| **CatBoost** | **0.7337** | 0.7228 | -0.0007 |
| LightGBM | 0.7320 | 0.7203 | +0.0029 |
| XGBoost | 0.7297 | 0.7186 | new |

**Key findings:**
- 705: All models gain ~0.035 F1 — trip_stop_index provides strong complementary signal to prev_stop_delay
- Lausanne 50k: Near-zero impact — the smaller dataset (40K train) is data-limited, not feature-limited
- CatBoost remains dominant on both datasets

---

---

## 1. Binning Strategy

Chose **4-class bins `[60, 120, 300]`** — operationally meaningful delay ranges:

| Class | Range | Meaning | 705 % | Lausanne 50k % |
|-------|-------|---------|-------|----------------|
| 0 | ≤60s | On-time | 43.0% | 43.5% |
| 1 | 60–120s | Slight delay | 30.0% | 27.7% |
| 2 | 120–300s | Moderate delay | 24.6% | 24.4% |
| 3 | >300s | Severe delay | 2.5% | 4.3% |

Rejected alternatives:
- **Binary (≤60s/>60s):** Not precise enough per user request
- **7-class `[0, 30, 60, 120, 180, 300]`:** >300s class too small (2.5%) for reliable learning
- **5-class `[30, 60, 120, 300]`:** Good balance, deferred for future investigation

---

## 2. Optimization Methodology

Two-phase approach (in `optimize_classifiers.py`):
1. **Phase 1:** Optuna (TPE sampler, 50 trials) on 50K stratified sample — fast hyperparameter search
2. **Phase 2:** Train best params on full dataset, evaluate on held-out test set

Optimized models: LightGBM, XGBoost, CatBoost — all with `TemporalFeatureExtractor + WindMerger + StringEncoder(["operator", "line"])`.

---

## 3. 705 Dataset Results (386K train / 97K test)

### Summary

| Model | Macro-F1 | Accuracy | Fit Time |
|-------|----------|----------|----------|
| **CatBoost** | **0.7571** | 0.7516 | ~717s |
| LightGBM | 0.7459 | 0.7463 | ~137s |
| XGBoost | 0.7420 | 0.7396 | ~203s |

### CatBoost — Per-Class Metrics (best model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| ≤60s | 0.79 | 0.85 | 0.82 | 41,532 |
| 60–120s | 0.63 | 0.60 | 0.62 | 28,974 |
| 120–300s | 0.80 | 0.75 | 0.78 | 23,678 |
| >300s | **0.88** | 0.75 | 0.81 | 2,485 |
| **Macro Avg** | 0.78 | 0.74 | **0.76** | 96,669 |

### CatBoost Confusion Matrix

| Actual ↓ / Pred → | ≤60s | 60–120s | 120–300s | >300s |
|-------------------|------|---------|----------|-------|
| ≤60s | 35,471 | 5,594 | 454 | 13 |
| 60–120s | 8,122 | 17,452 | 3,399 | 1 |
| 120–300s | 1,135 | 4,435 | 17,869 | 239 |
| >300s | 44 | 10 | 565 | 1,866 |

**Key insight:** CatBoost has 0.88 precision on severe delays (>300s) — only 44+10+13=67 false positives out of 96,669 predictions. When it flags a severe delay, it's almost always correct.

### CatBoost Best Hyperparameters

```
n_estimators: 1263
learning_rate: 0.01960
depth: 11
l2_leaf_reg: 0.03972
random_strength: 0.3489
bagging_temperature: 0.5924
min_data_in_leaf: 5
auto_class_weights: None
```

---

## 4. Lausanne 50k Dataset Results (40K train / 10K test)

705-optimized CatBoost and LGBM params transferred directly.

| Model | Macro-F1 | Accuracy |
|-------|----------|----------|
| **CatBoost** | **0.7344** | 0.7228 |
| LightGBM | 0.7291 | 0.7181 |

### CatBoost — Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| ≤60s | 0.77 | 0.83 | 0.80 | 4,354 |
| 60–120s | 0.58 | 0.52 | 0.55 | 2,848 |
| 120–300s | 0.76 | 0.76 | 0.76 | 2,359 |
| >300s | **0.87** | 0.80 | 0.83 | 439 |

Lower performance than 705 expected — only 40K training samples vs 386K.

---

## 5. What Didn't Work

### Logistic Regression
All LogReg variants failed (Macro-F1 ≤ 0.18):
- `LogReg-Scaled`: lbfgs failed to converge in 5,000 iterations — F1=0.178
- `LogReg-PCA`: F1=0.132
- `LogReg-Nystroem`: F1=0.110
- `LogReg-ElasticNet`: (did not complete)

**Root cause:** `StringEncoder` creates one-hot columns for every operator and line value (~hundreds of sparse columns). sklearn's `LogisticRegression(lbfgs)` cannot converge on this high-dimensional sparse feature space. Switching to `saga` solver + `HistoricalMeanEncoder` might fix this, but was not pursued due to time.

### 5-fold Stacking
`ClassificationStackingModel(CatBoost+LGBM, n_folds=5)` was killed after 42+ min of fitting with no results. Each base model (CatBoost with 1263 trees) must be refit 5× on different folds — too expensive for the current hyperparameters.

---

## 6. Key Takeaways

1. **CatBoost dominates** 4-class bus delay classification — best on both datasets, especially for rare severe delays (F1=0.81–0.83 on >300s class)
2. **60–120s is the hardest class** across all models (F1=0.55–0.64) — the boundary between "on-time" and "slightly late" is inherently fuzzy
3. **prev_stop_delay is the dominant feature** (per prior regression analysis at 86% importance) — the lag feature carries most predictive signal
4. **Lausanne 50k needs its own optimization** — transferring 705 params works but direct Optuna on Lausanne would likely find better local minima for the smaller dataset
5. **Logistic regression is not viable** with the current StringEncoder → sparse feature pipeline; would need HistoricalMeanEncoder or target encoding

---

## 7. Files Modified

| File | Change |
|------|--------|
| `config.py` | Added `loader_lausanne_50k` |
| `optimize_classifiers.py` | New — two-phase Optuna optimization script |
| `experiments_classification.py` | Rewritten with optimized 4-class experiments + LogReg stubs |
| `RESULTS_5.md` | This report |
| `results/cls_opt_705_b60_120_300.json` | Saved optimization results |
