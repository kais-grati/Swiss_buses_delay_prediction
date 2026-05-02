from ml.data import DataLoader
from ml.pipeline import MLPipeline
from ml.preprocessors.scaler import FeatureScaler
from ml.models.lgbm import LightGBMModel
from ml.models.xgboost_model import XGBoostModel
from ml.evaluation import Evaluator
from ml.experiment import Experiment

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET = "data/dataset_705_echandens.parquet"
TARGET  = "arrival_delay_s"

DROP_COLS = [
    "timestamp", "stop_id", "stop_name", "operator", "line",
    "departure_delay_s",
]

NUMERIC_COLS = [
    "temperature", "precipitation", "sunshine", "humidity",
    "wind_speed", "wind_gust", "pressure", "snow_depth",
]

# ── Experiments ────────────────────────────────────────────────────────────────

loader = DataLoader(path=DATASET, target=TARGET, drop_cols=DROP_COLS)
evaluator = Evaluator()

experiments = {
    "LightGBM": Experiment(
        loader=loader,
        pipeline=MLPipeline(
            preprocessors=[FeatureScaler(cols=NUMERIC_COLS)],
            model=LightGBMModel(n_estimators=500, learning_rate=0.05, num_leaves=31),
        ),
        evaluator=evaluator,
    ),
    "LightGBM-tuned": Experiment(
        loader=loader,
        pipeline=MLPipeline(
            preprocessors=[FeatureScaler(cols=NUMERIC_COLS)],
            model=LightGBMModel(
                n_estimators=3000,
                learning_rate=0.01,
                num_leaves=50,
                min_child_samples=50,
                max_bin=512,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_gain_to_split=0.01,
                early_stopping_rounds=50,
                log_every=100,
            ),
        ),
        evaluator=evaluator,
    ),
    "XGBoost": Experiment(
        loader=loader,
        pipeline=MLPipeline(
            preprocessors=[FeatureScaler(cols=NUMERIC_COLS)],
            model=XGBoostModel(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
            ),
        ),
        evaluator=evaluator,
    ),
}

# ── Run ────────────────────────────────────────────────────────────────────────

results = {}
for name, exp in experiments.items():
    print(f"\n{'─' * 50}")
    print(f"Running: {name}")
    print(f"{'─' * 50}")
    results[name] = exp.run()

# ── Summary ────────────────────────────────────────────────────────────────────

print(f"\n{'═' * 50}")
print(f"{'Model':<12} {'MSE':>10} {'RMSE':>10} {'R²':>8}")
print(f"{'─' * 50}")
for name, m in results.items():
    print(f"{name:<12} {m['mse']:>10.2f} {m['rmse']:>9.2f}s {m['r2']:>8.4f}")
print(f"{'═' * 50}")
