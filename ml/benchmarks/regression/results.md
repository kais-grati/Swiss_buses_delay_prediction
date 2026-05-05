# Regression Benchmarks — Echandens 705

Dataset: `data/dataset_705_echandens.parquet`  
Target: `arrival_delay_s`  
Split: 80/20 time-ordered  
Date range: April–May 2026

---

## LGBM-optuna-v1

**Preprocessors:** `FeatureScaler(NUMERIC_COLS)` → `TemporalFeatureExtractor`  
**Optimization:** Optuna 100 trials, TPE sampler, early stopping RMSE  

**Hyperparameters:**
```
n_estimators:              3000
early_stopping_rounds:     50
learning_rate:             0.011164245970961571
num_leaves:                116
min_child_samples:         43
min_sum_hessian_in_leaf:   0.9756353974751673
subsample:                 0.9363929416897576
subsample_freq:            1
colsample_bytree:          0.9988094388849174
feature_fraction_bynode:   0.9450155202549205
reg_alpha:                 0.27245190090589816
reg_lambda:                0.78391786930204
path_smooth:               4.405998450654837
```

**Results:**
```
MSE:   6577.13
RMSE:  81.10s
R²:    0.3233
```

---

## LGBM-optuna-v2

**Preprocessors:** `FeatureScaler(NUMERIC_COLS)` → `TemporalFeatureExtractor`  
**Optimization:** Optuna 100 trials, TPE sampler — re-run with temporal extractor included in pipeline  

**Hyperparameters:**
```
n_estimators:              3000
early_stopping_rounds:     50
learning_rate:             0.024864985683759746
num_leaves:                78
min_child_samples:         86
min_sum_hessian_in_leaf:   0.013837807248223857
subsample:                 0.9940870655183781
subsample_freq:            1
colsample_bytree:          0.6580004324956996
feature_fraction_bynode:   0.9019956230179453
reg_alpha:                 0.29733122973301546
reg_lambda:                0.006579169405573432
path_smooth:               5.094559948404132
```

**Results:**
```
MSE:   6551.84
RMSE:  80.94s
R²:    0.3259
```

---

## LGBM-optuna-v3 (LGBM-Optuna-Full in main.py)

**Preprocessors:** `TemporalFeatureExtractor` → `WeatherFeatureEngineer` → `FeatureScaler(NUMERIC_COLS_ENHANCED)`  
**Optimization:** Optuna 100 trials, added WeatherFeatureEngineer (wind_chill + adverse_weather_flag)  

**Hyperparameters:**
```
n_estimators:              3000
early_stopping_rounds:     50
learning_rate:             0.007126947042377407
num_leaves:                168
min_child_samples:         40
min_sum_hessian_in_leaf:   0.0899176805301358
subsample:                 0.9998756827358574
subsample_freq:            1
colsample_bytree:          0.5757330744549699
feature_fraction_bynode:   0.9734799081519098
reg_alpha:                 1.7021369826715338
reg_lambda:                7.970747211586931
path_smooth:               4.671171717084908
```

**Results:**
```
MSE:   6564.12
RMSE:  81.02s
R²:    0.3246
```

---

## LGBM-HistMean

**Preprocessors:** `TemporalFeatureExtractor` → `WeatherFeatureEngineer` → `HistoricalMeanEncoder(hour+dow)` → `FeatureScaler(NUMERIC_COLS_ENHANCED + hist_mean_delay)`  
**Note:** Same model params as LGBM-optuna-v3. Tests whether adding historical mean delay helps regression.

**Hyperparameters:** Same as LGBM-optuna-v3 above.

**Results:**
```
MSE:   6609.69
RMSE:  81.30s
R²:    0.3199
```

**Finding:** HistoricalMeanEncoder does NOT help regression (−0.0047 R²). The optuna-tuned model already captures temporal patterns via cyclical sin/cos features from TemporalFeatureExtractor.

---

## CatBoost-optuna

**Preprocessors:** `TemporalFeatureExtractor` → `FeatureScaler(NUMERIC_COLS)`  
**Optimization:** Optuna 100 trials, TPE sampler  

**Hyperparameters:**
```
n_estimators:           2000
early_stopping_rounds:  50
learning_rate:          0.05543418338273171
depth:                  5
l2_leaf_reg:            0.540507962054327
random_strength:        0.16061584861239986
bagging_temperature:    0.49356279590945856
min_data_in_leaf:       62
```

**Results:**
```
MSE:   6644.90
RMSE:  81.52s
R²:    0.3163
```

---

## XGBoost-optuna

**Preprocessors:** `FeatureScaler(NUMERIC_COLS)`  
**Optimization:** Optuna 100 trials, TPE sampler  

**Hyperparameters:**
```
n_estimators:       500
early_stopping_rounds: 50
learning_rate:      0.03809720744535876
max_depth:          9
min_child_weight:   90.28669329772264
gamma:              0.016744637853508566
subsample:          0.9335838518925651
colsample_bytree:   0.9389806288470189
colsample_bylevel:  0.8337241352904957
reg_alpha:          0.00035666207579412883
reg_lambda:         1.8707902692338567
```

**Results:**
```
MSE:   ~6699
RMSE:  81.85s
R²:    0.3107
```

---

## Stacking2 ← Best Overall

**Preprocessors:** `TemporalFeatureExtractor` → `FeatureScaler(NUMERIC_COLS)`  
**Architecture:** 2 base models (LGBM-v2 + XGBoost-optuna) → Ridge meta-learner, 10-fold CV  

**Base model 1 — LightGBM (LGBM-optuna-v2 params):**
```
n_estimators:              3000
early_stopping_rounds:     50
learning_rate:             0.024864985683759746
num_leaves:                78
min_child_samples:         86
min_sum_hessian_in_leaf:   0.013837807248223857
subsample:                 0.9940870655183781
subsample_freq:            1
colsample_bytree:          0.6580004324956996
feature_fraction_bynode:   0.9019956230179453
reg_alpha:                 0.29733122973301546
reg_lambda:                0.006579169405573432
path_smooth:               5.094559948404132
```

**Base model 2 — XGBoost (XGBoost-optuna params):**
```
n_estimators:       500
learning_rate:      0.03809720744535876
max_depth:          9
min_child_weight:   90.28669329772264
gamma:              0.016744637853508566
subsample:          0.9335838518925651
colsample_bytree:   0.9389806288470189
colsample_bylevel:  0.8337241352904957
reg_alpha:          0.00035666207579412883
reg_lambda:         1.8707902692338567
```

**Meta-model — Ridge:**
```
alpha: 1.0
```

**Results:**
```
MSE:   6523.09
RMSE:  80.76s
R²:    0.3290
```

---

## Summary Table

| Experiment | RMSE (s) | R² |
|---|---|---|
| XGBoost-optuna | 81.85 | 0.3107 |
| CatBoost-optuna | 81.52 | 0.3163 |
| LGBM-HistMean | 81.30 | 0.3199 |
| LGBM-optuna-v1 | 81.10 | 0.3233 |
| LGBM-Optuna-Full (v3) | 81.02 | 0.3246 |
| LGBM-optuna-v2 | 80.94 | 0.3259 |
| **Stacking2** | **80.76** | **0.3290** |

**Performance ceiling note:** All models plateau around R²≈0.32–0.33. This is a data constraint: features are limited to weather + temporal patterns at a single stop. Adding previous-stop departure delays or route-level features would be necessary to push beyond this ceiling.
