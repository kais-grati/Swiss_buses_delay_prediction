# ML Benchmarks — Echandens 705 Dataset

All experiments run on `data/dataset_705_echandens.parquet`.  
Train/test split: 80/20, time-ordered (no shuffle).  
Target: `arrival_delay_s`.

## Dataset Statistics

- ~27,000 rows after outlier removal  
- Stop: Echandens, Chocolatière — Bus line 705  
- Delay range: [-120s, +1800s]  
- Class imbalance (4-class): class 3 (>600s) ≈ 0.26% of data

## Results Overview

### Regression (target: `arrival_delay_s` in seconds)

| Experiment | RMSE (s) | R² | Notes |
|---|---|---|---|
| LGBM-optuna-v1 | 81.10 | 0.3233 | Optuna 100 trials, no temporal extractor |
| LGBM-optuna-v2 | 80.94 | 0.3259 | + TemporalFeatureExtractor |
| LGBM-optuna-v3 | 81.02 | 0.3246 | + WeatherFeatureEngineer |
| LGBM-Optuna-Full | 81.05 | 0.3241 | v3 params, deployed in main.py |
| LGBM-HistMean | 81.30 | 0.3199 | + HistoricalMeanEncoder — no gain |
| CatBoost-optuna | 81.52 | 0.3163 | Optuna 100 trials |
| XGBoost-optuna | 81.85 | 0.3107 | Optuna 100 trials |
| Stacking2 | 80.76 | 0.3290 | LGBM-v2 + XGBoost + Ridge meta, 10-fold CV |

**Best:** Stacking2 — RMSE=80.76s, R²=0.3290  
**Ceiling note:** R²≈0.32 is a data constraint (single stop, no trip-level features).

### Classification (4 classes: ≤60s | 60–180s | 180–600s | >600s)

| Experiment | Macro-F1 | Notes |
|---|---|---|
| LogReg-L2 | ~0.17 | Baseline, no hist features |
| LogReg-ElasticNet | 0.2284 | Best linear model |
| LGBM-Classifier-Round1 | 0.4275 | Non-stratified val split |
| LGBM-v1 | 0.4215 | Stratified val split |
| LGBM-v2-LowChild | 0.4186 | min_child_samples=1 |
| LGBM-v3-DeepHist | 0.4156 | 3× HistoricalMeanEncoder |
| LGBM-v4-HeavyClass3 | 0.4214 | Manual weights, 100× class 3 |
| XGBoost-Classifier | 0.4102 | sample_weight="balanced" |
| CatBoost-NoWeights | 0.3366 | eval_metric=Accuracy |

**Best:** LGBM-v1 (stratified) — Macro-F1=0.4215  
**Ceiling note:** Class 3 (71 samples) is statistically very thin — macro-F1 ≈ 0.42 appears to be a data ceiling.

## Directory Structure

```
benchmarks/
├── README.md              ← this file
├── regression/
│   ├── results.md         ← all regression experiments, full params + metrics
│   ├── optuna_lgbm_results.txt
│   ├── optuna_xgboost_results.txt
│   ├── optuna_catb_results.txt
│   └── stacking_results.txt
└── classification/
    └── results.md         ← all classification experiments, full params + metrics
```
