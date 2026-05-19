from pathlib import Path
from ml.data import DataLoader
from ml.evaluation import Evaluator
from ml.preprocessors.delay_binner import DelayBinner
from ml.logger import ExperimentLogger

from rich.console import Console

console = Console()

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET = "data/705_bus_2025_weather_traffic.parquet"
TARGET  = "arrival_delay_s"

DROP_COLS = [
    "timestamp", "stop_id", "stop_name", "operator", "line",
    "departure_delay_s", "trip_id",
    # "trip_stop_index",  ← KEPT: +0.031 R², +0.036 F1 (see RESULTS_6.md)
]

DROP_COLS_KEEP_TS = [
    "stop_name", "departure_delay_s", "trip_id",
    # "trip_stop_index",  ← KEPT
]

NUMERIC_COLS = [
    "temperature", "precipitation", "sunshine", "humidity",
    "wind_speed", "wind_gust", "pressure", "snow_depth",
]

# Temporal integers extracted by TemporalFeatureExtractor — must be scaled for linear models
TEMPORAL_COLS = ["hour", "dow", "month"]

# Traffic features (precomputed by add_traffic_features.py via GPKG spatial join)
TRAFFIC_COLS = [
    "traffic_dtv",
    "traffic_peak",
    "traffic_heavy_share",
    "traffic_peak_ratio",
]

# Lag delay feature (precomputed by build_dataset.py, no runtime LagDelayEncoder needed)
LAG_COLS = ["prev_stop_delay", "dist_to_prev_stop"]

# Full numeric set for linear models: weather + temporal integers + engineered features
NUMERIC_COLS_LOGREG = NUMERIC_COLS + TEMPORAL_COLS + TRAFFIC_COLS
NUMERIC_COLS_LOGREG_FULL = NUMERIC_COLS + TEMPORAL_COLS + ["wind_chill", "hist_mean_delay"] + TRAFFIC_COLS

# Enhanced set for tree models (trees don't need temporal scaling but benefit from engineered features)
NUMERIC_COLS_ENHANCED = NUMERIC_COLS + ["wind_chill"] + TRAFFIC_COLS

# Full numeric set with lag features + traffic
NUMERIC_COLS_LAG = NUMERIC_COLS + LAG_COLS + TRAFFIC_COLS

# Scaled numeric set for logistic regression after WindMerger + TemporalFeatureExtractor
LOGREG_NUMERIC = [
    "temperature", "precipitation", "sunshine", "humidity",
    "wind", "pressure", "snow_depth", "hour", "dow", "month",
]

# ── Shared instances ───────────────────────────────────────────────────────────

loader = DataLoader(path=DATASET, target=TARGET, drop_cols=DROP_COLS)
loader_enhanced = DataLoader(path=DATASET, target=TARGET, drop_cols=DROP_COLS_KEEP_TS)
loader_lag = DataLoader(path=DATASET, target=TARGET, drop_cols=["stop_name", "departure_delay_s", "trip_id"])

# Lausanne 50k dataset loader — regional subset for separate evaluation
DATASET_LAUSANNE_50K = "data/lausanne50k_bus_2025_weather_traffic.parquet"
loader_lausanne_50k = DataLoader(
    path=DATASET_LAUSANNE_50K, target=TARGET,
    drop_cols=["stop_name", "departure_delay_s", "trip_id"],
)

evaluator = Evaluator()
binner = DelayBinner()
logger = ExperimentLogger()
