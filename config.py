from pathlib import Path
from ml.data import DataLoader
from ml.evaluation import Evaluator
from ml.preprocessors.delay_binner import DelayBinner
from ml.logger import ExperimentLogger

from rich.console import Console

console = Console()

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET = "data/dataset_lausanne_50k.parquet"
TARGET  = "arrival_delay_s"

DROP_COLS = [
    "timestamp", "stop_id", "stop_name", "operator", "line",
    "departure_delay_s",
]

DROP_COLS_KEEP_TS = [
    "stop_name", "departure_delay_s",
]

NUMERIC_COLS = [
    "temperature", "precipitation", "sunshine", "humidity",
    "wind_speed", "wind_gust", "pressure", "snow_depth",
]

# Temporal integers extracted by TemporalFeatureExtractor — must be scaled for linear models
TEMPORAL_COLS = ["hour", "dow", "month"]

# Full numeric set for linear models: weather + temporal integers + engineered features
NUMERIC_COLS_LOGREG = NUMERIC_COLS + TEMPORAL_COLS
NUMERIC_COLS_LOGREG_FULL = NUMERIC_COLS + TEMPORAL_COLS + ["wind_chill", "hist_mean_delay"]

# Enhanced set for tree models (trees don't need temporal scaling but benefit from engineered features)
NUMERIC_COLS_ENHANCED = NUMERIC_COLS + ["wind_chill"]

# Scaled numeric set for logistic regression after WindMerger + TemporalFeatureExtractor
LOGREG_NUMERIC = [
    "temperature", "precipitation", "sunshine", "humidity",
    "wind", "pressure", "snow_depth", "hour", "dow", "month",
]

# ── Shared instances ───────────────────────────────────────────────────────────

loader = DataLoader(path=DATASET, target=TARGET, drop_cols=DROP_COLS)
loader_enhanced = DataLoader(path=DATASET, target=TARGET, drop_cols=DROP_COLS_KEEP_TS)
evaluator = Evaluator()
binner = DelayBinner()
logger = ExperimentLogger()
