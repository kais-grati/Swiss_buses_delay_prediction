# Tutorial — Swiss Bus Delay Prediction

This guide walks through using the project: from loading pre-trained models for inference to running your own experiments and training new models.

## Prerequisites

```bash
git clone https://github.com/<user>/swiss-buses-delay-prediction.git
cd swiss-buses-delay-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You'll need the parquet dataset file(s). The 705 subset (8MB, 483K rows) is the recommended starting point. Place it at `data/705_bus_2025_weather_traffic.parquet`.

---

## 1. Inference with Saved Models

Pre-trained models are in `saved_models/`. They load in seconds and predict instantly.

### Predict delay duration (regression)

```python
from ml.pipeline import MLPipeline
from ml.data import DataLoader

# Load data (any parquet with matching schema)
loader = DataLoader(
    path="data/705_bus_2025_weather_traffic.parquet",
    target="arrival_delay_s",
    drop_cols=["stop_name", "departure_delay_s", "trip_id"],
)
_, X_test, _, y_test = loader.load()

# Load model and predict
pipe = MLPipeline.load("saved_models/regression_catboost_705")
predictions = pipe.predict(X_test.head(100))

print(f"Predicted delays: {predictions[:5]} seconds")
print(f"Actual delays:    {y_test.head(5).tolist()} seconds")
```

Available regression models:

| Model | Path | R² | RMSE | Load Time |
|-------|------|-----|------|-----------|
| CatBoost (fast) | `regression_catboost_705` | 0.8800 | 32.06s | ~2s |
| Stack-CB-Ridge (best) | `regression_stack_cb_ridge_705` | 0.8801 | 32.05s | ~5s |

### Classify delay severity (4-class)

```python
import joblib
from ml.pipeline import MLPipeline

pipe = MLPipeline.load("saved_models/classification_catboost_705_4cls")
binner = joblib.load("saved_models/classification_catboost_705_4cls/binner.joblib")

class_ids = pipe.predict(X_test.head(100))
labels = [binner.class_names[i] for i in class_ids]

# → ['≤60s', '60–120s', '120–300s', '>300s']
print(f"Class names: {binner.class_names}")
print(f"Predictions: {labels[:10]}")
```

The 4 classes represent operationally meaningful delay levels:

| Class | Range | Meaning |
|-------|-------|---------|
| 0 | ≤60s | On-time |
| 1 | 60–120s | Slight delay |
| 2 | 120–300s | Moderate delay |
| 3 | >300s | Severe delay |

![](charts/data_distribution.png)

---

## 2. Run Existing Experiments

The experiment files are self-contained Python modules. Just import and run:

```python
from experiments_regression import run_regression
results = run_regression()
# Prints a rich-formatted table with MSE, RMSE, R² for all 22 models
```

```python
from experiments_classification import run_classification
results = run_classification()
# Prints per-class metrics, confusion matrices, and a summary table
```

### Run a single experiment

You can also instantiate and run individual experiments:

```python
from config import loader_lag, evaluator
from ml.pipeline import MLPipeline
from ml.preprocessors.temporal import TemporalFeatureExtractor
from ml.preprocessors.wind_merger import WindMerger
from ml.preprocessors.string_encoder import StringEncoder
from ml.models.catboost_model import CatBoostModel
from ml.experiment import Experiment

exp = Experiment(
    loader=loader_lag,
    pipeline=MLPipeline(
        preprocessors=[
            TemporalFeatureExtractor(),
            WindMerger(),
            StringEncoder(cols=["operator", "line"]),
        ],
        model=CatBoostModel(
            n_estimators=1410,
            learning_rate=0.05149,
            depth=10,
            early_stopping_rounds=50,
        ),
    ),
    evaluator=evaluator,
)
metrics = exp.run()
print(metrics)  # {"mse": ..., "rmse": ..., "r2": ...}
```

---

## 3. Hyperparameter Optimization

The project uses Optuna with TPE sampling for Bayesian hyperparameter search.

### Classification optimization

```bash
# Optimize all 3 models on 705 with 4-class bins, 50 trials each
python optimize_classifiers.py --dataset 705 --bins 60 120 300 --models LGBM,XGBoost,CatBoost --trials 50
```

The script uses a two-phase approach:
1. **Phase 1** — 50K stratified sample for fast trials (~3s/trial for binary, ~30s/trial for 4-class)
2. **Phase 2** — Evaluate best hyperparameters on the full dataset

### Regression optimization

Import optimizer classes directly from `ml.optimizer`:

```python
from ml.optimizer import LGBMRegressorOptimizer, CatBoostRegressorOptimizer
from config import loader_lag

opt = CatBoostRegressorOptimizer(
    loader=loader_lag,
    n_trials=50,
    n_estimators=2000,
)
study = opt.optimize()
print(f"Best RMSE: {study.best_value:.2f}s")
print(f"Best params: {study.best_params}")
```

---

## 4. Train a Custom Model

```python
from ml.data import DataLoader
from ml.pipeline import MLPipeline
from ml.preprocessors.temporal import TemporalFeatureExtractor
from ml.preprocessors.wind_merger import WindMerger
from ml.preprocessors.string_encoder import StringEncoder
from ml.preprocessors.scaler import FeatureScaler
from ml.models.ridge import RidgeModel
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data (trip_stop_index is now kept — see RESULTS_6.md)
loader = DataLoader(
    path="data/705_bus_2025_weather_traffic.parquet",
    target="arrival_delay_s",
    drop_cols=["stop_name", "departure_delay_s", "trip_id"],
)
X_train, X_test, y_train, y_test = loader.load()

# Build pipeline
pipe = MLPipeline(
    preprocessors=[
        TemporalFeatureExtractor(),
        WindMerger(),
        StringEncoder(cols=["operator", "line"]),
        FeatureScaler(cols=[
            "temperature", "precipitation", "sunshine", "humidity",
            "wind", "pressure", "snow_depth",
            "hour", "dow", "month",
            "prev_stop_delay", "dist_to_prev_stop",
        ]),
    ],
    model=RidgeModel(alpha=19.18),
)

# Train and evaluate
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print(f"RMSE: {rmse:.2f}s  R²: {r2:.4f}")
```

---

## 5. Build the Dataset from Scratch

The full data pipeline is documented in [PIPELINE.md](../PIPELINE.md). Quick start:

```bash
# 1. Build dataset from raw CSV ZIP archives
python scripts/build_dataset.py --workers 4

# 2. Fetch hourly weather
python scripts/fetch_weather_hourly.py

# 3. Join weather to dataset
python scripts/add_weather.py

# 4. Remove rows with missing weather
python scripts/drop_missing_weather.py

# 5. Precompute lag delay feature
python scripts/add_lag_delay.py

# 6. (Optional) Add traffic features
python scripts/add_traffic_features.py

# 7. (Optional) Add stop distance
python scripts/add_stop_distance.py

# 8. (Optional) Add trip stop index
python scripts/add_trip_stop_index.py
```

### Create a regional subset

```bash
python scripts/filter_lausanne.py
# → data/lausanne_bus_2025_weather_traffic.parquet
```

### Create a stratified sample

```bash
python scripts/sample_dataset.py --input data/swiss_bus_2025_weather_traffic.parquet --n 50000 --stratify-on arrival_delay_s
# → data/sampled_50000.parquet
```

---

## 6. Datasets Quick Reference

| Dataset | Rows | Size | Use |
|---------|------|------|-----|
| `705_bus_2025_weather_traffic.parquet` | 492K | 8 MB | Primary dev (fast iteration) |
| `lausanne50k_bus_2025_weather_traffic.parquet` | 50K | 4 MB | Secondary benchmark |
| `swiss_bus_2025_weather_traffic.parquet` | 509M | 16 GB | Full training & production |

Full schema and details: [DATASETS.md](DATASETS.md). Build instructions: [PIPELINE.md](../PIPELINE.md).

![](charts/dataset_comparison.png)

---

## 7. Experiment Log

All experiment runs append to `results/experiment_log.jsonl`:

```python
import json
with open("results/experiment_log.jsonl") as f:
    for line in f:
        run = json.loads(line)
        print(f"{run['timestamp']} {run['name']}: R²={run.get('r2', 'N/A')}")
```

---

## Next Steps

- Read [RESULTS.md](RESULTS.md) for the full performance analysis
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for ML design details
- Read [PIPELINE.md](../PIPELINE.md) for the data pipeline
- Check `notebooks/misc.ipynb` for exploratory analysis
