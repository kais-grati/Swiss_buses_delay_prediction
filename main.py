from ml.data import DataLoader
from ml.pipeline import MLPipeline
from ml.preprocessors.scaler import FeatureScaler
from ml.preprocessors.polynomial import PolynomialExpander
from ml.models.ridge import RidgeModel
from ml.models.lgbm import LightGBMModel
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
    "temperature", "precipitation", "humidity",
    "wind_speed", "wind_gust", "pressure", "snow_depth",
]

WIND_CYCLICAL = ["wind_dir_sin", "wind_dir_cos"]

# ── Experiments ────────────────────────────────────────────────────────────────

loader = DataLoader(path=DATASET, target=TARGET, drop_cols=DROP_COLS)
evaluator = Evaluator()

experiments = {
    "Ridge": Experiment(
        loader=loader,
        pipeline=MLPipeline(
            preprocessors=[
                FeatureScaler(cols=NUMERIC_COLS),
                PolynomialExpander(cols=NUMERIC_COLS + WIND_CYCLICAL, degree=2),
            ],
            model=RidgeModel(alpha=1.0),
        ),
        evaluator=evaluator,
    ),
    "LightGBM": Experiment(
        loader=loader,
        pipeline=MLPipeline(
            preprocessors=[
                FeatureScaler(cols=NUMERIC_COLS),
            ],
            model=LightGBMModel(n_estimators=500, learning_rate=0.05, num_leaves=31),
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
