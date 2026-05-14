from ml.optimizer import (
    LGBMRegressorOptimizer, XGBoostRegressorOptimizer, CatBoostRegressorOptimizer,
    LGBMClassifierOptimizer, OrdinalLGBMClassifierOptimizer,
    OrdinalXGBoostClassifierOptimizer,
    XGBoostClassifierOptimizer, CatBoostClassifierOptimizer,
    RandomForestClassifierOptimizer, MLPClassifierOptimizer,
)
from ml.preprocessors.scaler import FeatureScaler
from ml.preprocessors.string_encoder import StringEncoder
from ml.preprocessors.temporal import TemporalFeatureExtractor
from ml.preprocessors.wind_merger import WindMerger

from rich.panel import Panel

from config import console, loader_enhanced, loader_lag, binner


def run_optimization():
    console.print("\n", Panel("[bold yellow]CLASSIFIER OPTUNA OPTIMIZATION[/bold yellow]", expand=False))

    # ── Ordinal LGBM Optimizer ────────────────────────────────────────────────────
    """
      n_estimators: 683
      scale_pos_weight: 0.6658178271472204
      learning_rate: 0.012067638787755042
      num_leaves: 184
      min_child_samples: 83
      min_sum_hessian_in_leaf: 0.03792040702863287
      subsample: 0.6382348323819672
      colsample_bytree: 0.6580390699089372
      feature_fraction_bynode: 0.7730795062891931
      reg_alpha: 0.002551973673478356
      reg_lambda: 0.01849403951716357
    """
    ordinal_optimizer = OrdinalLGBMClassifierOptimizer(
        loader=loader_enhanced,
        binner=binner,
        n_trials=100,
        n_estimators=800,
        preprocessors= [
            TemporalFeatureExtractor(),
            WindMerger(),
            StringEncoder(cols=["operator", "line"]),
            FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth",]),
        ]
    )
    # ordinal_study = ordinal_optimizer.optimize()

    # ── Ordinal XGBoost Optimizer ──────────────────────────────────────────────────
    """
      n_estimators: 573
      scale_pos_weight: 0.6569387115853094
      learning_rate: 0.022365990585442086
      max_depth: 8
      min_child_weight: 2.594662538392548
      gamma: 0.00043870628771772006
      subsample: 0.5893496666560825
      colsample_bytree: 0.756233084869846
      colsample_bylevel: 0.7301284243764258
      reg_alpha: 0.001212947904518468
      reg_lambda: 0.23978457224831312
    """
    ordinal_xgb_optimizer = OrdinalXGBoostClassifierOptimizer(
        loader=loader_enhanced,
        binner=binner,
        n_trials=100,
        n_estimators=2000,
        preprocessors= [
            TemporalFeatureExtractor(),
            WindMerger(),
            StringEncoder(cols=["operator", "line"]),
            FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth",]),
        ]
    )
    # ordinal_xgb_study = ordinal_xgb_optimizer.optimize()

    """
      n_estimators: 1387
      scale_pos_weight: 0.6986522731068953
      learning_rate: 0.08504465534445489
      num_leaves: 138
      min_child_samples: 95
      min_sum_hessian_in_leaf: 0.017830620376047334
      subsample: 0.9603385505500266
      colsample_bytree: 0.6506589536610681
      feature_fraction_bynode: 0.43693599956361384
      reg_alpha: 0.06257402445676312
      reg_lambda: 0.008566280525044545
    """
    optimizer = LGBMClassifierOptimizer(
        loader=loader_enhanced,
        n_trials=60,
        preprocessors= [
            TemporalFeatureExtractor(),
            WindMerger(),
            StringEncoder(cols=["operator", "line"]),
            FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth",]),
        ]
    )
    # study = optimizer.optimize()

    """
      n_estimators: 1975
      scale_pos_weight: 0.5878398014833414
      learning_rate: 0.033623211512864536
      max_depth: 10
      min_child_weight: 21.77477570888255
      gamma: 0.05106025524429455
      subsample: 0.8194299288251159
      colsample_bytree: 0.8395862195507668
      colsample_bylevel: 0.9707654648613212
      reg_alpha: 1.2217369358360592
      reg_lambda: 0.4404197461177921
    """

    optimizer = XGBoostClassifierOptimizer(
        loader=loader_enhanced,
        binner=binner,
        n_trials=60,
        n_estimators=2000,
        preprocessors= [
            TemporalFeatureExtractor(),
            WindMerger(),
            StringEncoder(cols=["operator", "line"]),
            FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth",]),
        ]
    )
    # study = optimizer.optimize()

    """
      n_estimators: 440
      auto_class_weights: SqrtBalanced
      learning_rate: 0.06879543869404516
      depth: 7
      l2_leaf_reg: 0.4735532917047472
      random_strength: 0.07528620336596245
      bagging_temperature: 0.2196657341097641
      min_data_in_leaf: 73
    """

    optimizer = CatBoostClassifierOptimizer(
        loader=loader_enhanced,
        binner=binner,
        n_trials=40,
        preprocessors= [
            TemporalFeatureExtractor(),
            WindMerger(),
            StringEncoder(cols=["operator", "line"]),
            FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth",]),
        ]
    )
    # study = optimizer.optimize()

    """
      n_estimators: 357
      max_depth: 23
      min_samples_split: 15
      min_samples_leaf: 7
      max_features: sqrt
      class_weight: balanced
    """

    optimizer = RandomForestClassifierOptimizer(
        loader=loader_enhanced,
        binner=binner,
        n_trials=40,
        preprocessors= [
            TemporalFeatureExtractor(),
            WindMerger(),
            StringEncoder(cols=["operator", "line"]),
            FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth",]),
        ]
    )
    # study = optimizer.optimize()

    # ── MLP-Lag Optimizer ──────────────────────────────────────────────────────
    optimizer = MLPClassifierOptimizer(
        loader=loader_lag,
        binner=binner,
        n_trials=80,
        max_iter=500,
        preprocessors= [
            TemporalFeatureExtractor(),
            WindMerger(),
            StringEncoder(cols=["operator", "line"]),
            FeatureScaler([
                "temperature", "precipitation", "sunshine", "humidity",
                "wind", "pressure", "snow_depth", "prev_stop_delay",
                "hour", "dow", "month",
            ]),
        ]
    )
    # study = optimizer.optimize()
