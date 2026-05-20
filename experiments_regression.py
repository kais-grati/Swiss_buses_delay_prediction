from ml.pipeline import MLPipeline
from ml.preprocessors.scaler import FeatureScaler
from ml.preprocessors.temporal import TemporalFeatureExtractor
from ml.preprocessors.string_encoder import StringEncoder
from ml.preprocessors.wind_merger import WindMerger
from ml.preprocessors.target_encoder import HistoricalMeanEncoder
from ml.preprocessors.weather_engineer import WeatherFeatureEngineer
from ml.preprocessors.nystroem import NystroemExpander
from ml.preprocessors.polynomial import PolynomialExpander
from ml.preprocessors.poly_trig import PolyTrigExpander
from ml.preprocessors.pca import PCAReducer
from ml.models.lgbm import LightGBMModel
from ml.models.xgboost_model import XGBoostModel
from ml.models.catboost_model import CatBoostModel
from ml.models.ridge import RidgeModel
from ml.models.random_forest_regressor import RandomForestRegressorModel
from ml.models.stacking import StackingModel
from ml.models.residual_stacking import ResidualStackingModel
from ml.models.log_target import LogTargetModel
from ml.models.hierarchical import HierarchicalRegressor
from ml.models.ordinal_regressor import OrdinalRegressorModel
from ml.models.lgbm_classifier import LightGBMClassifierModel
from ml.models.xgboost_classifier import XGBoostClassifierModel
from ml.preprocessors.delay_binner import DelayBinner
from ml.experiment import Experiment

from rich.table import Table
from rich.panel import Panel

from config import (
    console, loader, loader_enhanced, loader_lag, evaluator, logger,
)

experiments = {
    # ═══════════════════════════════════════════════════════════════════════════════
    # BEST MODEL — Stack CB+Ridge
    #   705: MSE=1219.14  RMSE=32.05s  R²=0.8801  (beats CatBoost 1222.12)
    #   Lausanne 50k: MSE=2262.04  RMSE=47.56s  R²=0.8491  (beats Ridge 2321.78)
    "Stack-CB-Ridge": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler(cols=[
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                    "hour", "dow", "month",
                    "prev_stop_delay",
                    "dist_to_prev_stop",
                ]),
            ],
            model=StackingModel(
                base_models=[
                    CatBoostModel(
                        n_estimators=1410,
                        learning_rate=0.05149,
                        depth=10,
                        l2_leaf_reg=1.968,
                        random_strength=0.03778,
                        bagging_temperature=0.7664,
                        min_data_in_leaf=72,
                        early_stopping_rounds=50,
                    ),
                    RidgeModel(alpha=19.1791),
                ],
                meta_model=RidgeModel(alpha=1.0),
                n_folds=5,
            ),
        ),
        evaluator=evaluator,
    ),
    "LightGBM": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
            ],
            model=LightGBMModel(
                n_estimators=1062,
                learning_rate=0.01103,
                num_leaves=65,
                min_child_samples=76,
                min_sum_hessian_in_leaf=0.1572,
                subsample=0.9314,
                subsample_freq=1,
                colsample_bytree=0.5783,
                feature_fraction_bynode=0.8145,
                reg_alpha=0.001796,
                reg_lambda=0.2327,
                path_smooth=1.373,
                early_stopping_rounds=50,
            ),
        ),
        evaluator=evaluator,
    ),
    "XGBoost": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
            ],
            model=XGBoostModel(
                n_estimators=1102,
                learning_rate=0.02089,
                max_depth=7,
                min_child_weight=6.068,
                gamma=0.000983,
                subsample=0.9602,
                colsample_bytree=0.6652,
                colsample_bylevel=0.9709,
                reg_alpha=0.7871,
                reg_lambda=5.793,
                early_stopping_rounds=50,
            ),
        ),
        evaluator=evaluator,
    ),
    # Previously best single model on 705 — MSE=1222.12  RMSE=34.96s  R²=0.8573
    # Now superseded by Stack-CB-Ridge above
    "CatBoost": Experiment(
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
                l2_leaf_reg=1.968,
                random_strength=0.03778,
                bagging_temperature=0.7664,
                min_data_in_leaf=72,
                early_stopping_rounds=50,
            ),
        ),
        evaluator=evaluator,
    ),
    "RandomForest": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
            ],
            model=RandomForestRegressorModel(
                n_estimators=393,
                max_depth=26,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features=0.5,
            ),
        ),
        evaluator=evaluator,
    ),
    # BEST on Lausanne 50k — MSE=2321.78  R²=0.8452  α=19.18
    "Ridge": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler(cols=[
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                    "hour", "dow", "month",
                    "prev_stop_delay",
                    "dist_to_prev_stop",
                ]),
            ],
            model=RidgeModel(alpha=19.1791),
        ),
        evaluator=evaluator,
    ),
    "Ridge-PCA": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler(cols=[
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                    "hour", "dow", "month",
                    "prev_stop_delay",
                    "dist_to_prev_stop",
                ]),
                PCAReducer(variance_threshold=0.99),
            ],
            model=RidgeModel(alpha=2.882),
        ),
        evaluator=evaluator,
    ),
    "Ridge-Poly": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler(cols=[
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                    "hour", "dow", "month",
                    "prev_stop_delay",
                    "dist_to_prev_stop",
                ]),
                PolynomialExpander(cols=[
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "hour", "dow",
                ], degree=2),
            ],
            model=RidgeModel(alpha=2.882),
        ),
        evaluator=evaluator,
    ),
    "Ridge-Poly2": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler(cols=[
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                    "hour", "dow", "month",
                    "prev_stop_delay",
                    "dist_to_prev_stop",
                ]),
                PolyTrigExpander(cols=[
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "hour", "dow",
                ])
                
            ],
            model=RidgeModel(alpha=2.882),
        ),
        evaluator=evaluator,
    ),
    "Ridge-HistMean": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                HistoricalMeanEncoder(group_cols=["operator", "line"]),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler(cols=[
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                    "hour", "dow", "month",
                    "prev_stop_delay",
                    "dist_to_prev_stop",
                ]),
            ],
            model=RidgeModel(alpha=2.882),
        ),
        evaluator=evaluator,
    ),
    "Ridge-Nystroem": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler(cols=[
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                    "hour", "dow", "month",
                    "prev_stop_delay",
                    "dist_to_prev_stop",
                ]),
                NystroemExpander(n_components=100, kernel="rbf"),
            ],
            model=RidgeModel(alpha=2.882),
        ),
        evaluator=evaluator,
    ),
    "Ordinal-LGBM": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler([
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                ]),
            ],
            model=OrdinalRegressorModel(
                base_model=LightGBMClassifierModel(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=63,
                    min_child_samples=100,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                ),
                binner=DelayBinner(bins=[60, 120, 300]),
            ),
        ),
        evaluator=evaluator,
    ),
    "Ordinal-LGBM-fine": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler([
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                ]),
            ],
            model=OrdinalRegressorModel(
                base_model=LightGBMClassifierModel(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=63,
                    min_child_samples=100,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                ),
                binner=DelayBinner(bins=[30, 60, 120, 300, 600]),
            ),
        ),
        evaluator=evaluator,
    ),
    "Ordinal-XGB": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler([
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                ]),
            ],
            model=OrdinalRegressorModel(
                base_model=XGBoostClassifierModel(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                ),
                binner=DelayBinner(bins=[60, 120, 300]),
            ),
        ),
        evaluator=evaluator,
    ),
    "Residual-LGBM-XGB": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
            ],
            model=ResidualStackingModel(
                stage1_model=LightGBMModel(
                    n_estimators=1062,
                    learning_rate=0.01103,
                    num_leaves=65,
                    min_child_samples=76,
                    min_sum_hessian_in_leaf=0.1572,
                    subsample=0.9314,
                    subsample_freq=1,
                    colsample_bytree=0.5783,
                    feature_fraction_bynode=0.8145,
                    reg_alpha=0.001796,
                    reg_lambda=0.2327,
                    path_smooth=1.373,
                    early_stopping_rounds=50,
                ),
                stage2_model=XGBoostModel(
                    n_estimators=1102,
                    learning_rate=0.02089,
                    max_depth=7,
                    min_child_weight=6.068,
                    gamma=0.000983,
                    subsample=0.9602,
                    colsample_bytree=0.6652,
                    colsample_bylevel=0.9709,
                    reg_alpha=0.7871,
                    reg_lambda=5.793,
                    early_stopping_rounds=50,
                ),
                n_folds=5,
            ),
        ),
        evaluator=evaluator,
    ),
    "Residual-LGBM-Ridge": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler(cols=[
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                    "hour", "dow", "month",
                    "prev_stop_delay",
                    "dist_to_prev_stop",
                ]),
            ],
            model=ResidualStackingModel(
                stage1_model=LightGBMModel(
                    n_estimators=1062,
                    learning_rate=0.01103,
                    num_leaves=65,
                    min_child_samples=76,
                    min_sum_hessian_in_leaf=0.1572,
                    subsample=0.9314,
                    subsample_freq=1,
                    colsample_bytree=0.5783,
                    feature_fraction_bynode=0.8145,
                    reg_alpha=0.001796,
                    reg_lambda=0.2327,
                    path_smooth=1.373,
                    early_stopping_rounds=50,
                ),
                stage2_model=RidgeModel(alpha=2.882),
                n_folds=5,
            ),
        ),
        evaluator=evaluator,
    ),
    "Log-LGBM": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
            ],
            model=LogTargetModel(
                model=LightGBMModel(
                    n_estimators=1062,
                    learning_rate=0.01103,
                    num_leaves=65,
                    min_child_samples=76,
                    min_sum_hessian_in_leaf=0.1572,
                    subsample=0.9314,
                    subsample_freq=1,
                    colsample_bytree=0.5783,
                    feature_fraction_bynode=0.8145,
                    reg_alpha=0.001796,
                    reg_lambda=0.2327,
                    path_smooth=1.373,
                    early_stopping_rounds=50,
                ),
            ),
        ),
        evaluator=evaluator,
    ),
    "Log-Ridge": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler(cols=[
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                    "hour", "dow", "month",
                    "prev_stop_delay",
                    "dist_to_prev_stop",
                ]),
            ],
            model=LogTargetModel(
                model=RidgeModel(alpha=2.882),
            ),
        ),
        evaluator=evaluator,
    ),
    "Log-XGB": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
            ],
            model=LogTargetModel(
                model=XGBoostModel(
                    n_estimators=1102,
                    learning_rate=0.02089,
                    max_depth=7,
                    min_child_weight=6.068,
                    gamma=0.000983,
                    subsample=0.9602,
                    colsample_bytree=0.6652,
                    colsample_bylevel=0.9709,
                    reg_alpha=0.7871,
                    reg_lambda=5.793,
                    early_stopping_rounds=50,
                ),
            ),
        ),
        evaluator=evaluator,
    ),
    "Hier-LGBM-LGBM": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler([
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                ]),
            ],
            model=HierarchicalRegressor(
                classifier=LightGBMClassifierModel(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=63,
                    min_child_samples=100,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                ),
                regressor=LightGBMModel(
                    n_estimators=1062,
                    learning_rate=0.01103,
                    num_leaves=65,
                    min_child_samples=76,
                    min_sum_hessian_in_leaf=0.1572,
                    subsample=0.9314,
                    subsample_freq=1,
                    colsample_bytree=0.5783,
                    feature_fraction_bynode=0.8145,
                    reg_alpha=0.001796,
                    reg_lambda=0.2327,
                    path_smooth=1.373,
                    early_stopping_rounds=50,
                ),
                binner=DelayBinner(bins=[60, 120, 300]),
            ),
        ),
        evaluator=evaluator,
    ),
    "Hier-LGBM-Ridge": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler([
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                    "hour", "dow", "month",
                    "prev_stop_delay",
                    "dist_to_prev_stop",
                ]),
            ],
            model=HierarchicalRegressor(
                classifier=LightGBMClassifierModel(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=63,
                    min_child_samples=100,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                ),
                regressor=RidgeModel(alpha=2.882),
                binner=DelayBinner(bins=[60, 120, 300]),
            ),
        ),
        evaluator=evaluator,
    ),
    "Hier-LGBM-XGB": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
                FeatureScaler([
                    "temperature", "precipitation", "sunshine", "humidity",
                    "wind", "pressure", "snow_depth",
                ]),
            ],
            model=HierarchicalRegressor(
                classifier=LightGBMClassifierModel(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=63,
                    min_child_samples=100,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                ),
                regressor=XGBoostModel(
                    n_estimators=1102,
                    learning_rate=0.02089,
                    max_depth=7,
                    min_child_weight=6.068,
                    gamma=0.000983,
                    subsample=0.9602,
                    colsample_bytree=0.6652,
                    colsample_bylevel=0.9709,
                    reg_alpha=0.7871,
                    reg_lambda=5.793,
                    early_stopping_rounds=50,
                ),
                binner=DelayBinner(bins=[60, 120, 300]),
            ),
        ),
        evaluator=evaluator,
    ),
    "Stack-LGBM-XGB-CB": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
            ],
            model=StackingModel(
                base_models=[
                    LightGBMModel(
                        n_estimators=500,
                        learning_rate=0.05,
                        num_leaves=63,
                        min_child_samples=100,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                    ),
                    XGBoostModel(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                    ),
                    CatBoostModel(
                        n_estimators=300,
                        learning_rate=0.05,
                        depth=6,
                    ),
                ],
                meta_model=RidgeModel(alpha=1.0),
                n_folds=5,
            ),
        ),
        evaluator=evaluator,
    ),
    "Stack-LGBM-CB": Experiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
            ],
            model=StackingModel(
                base_models=[
                    LightGBMModel(
                        n_estimators=500,
                        learning_rate=0.05,
                        num_leaves=63,
                        min_child_samples=100,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                    ),
                    CatBoostModel(
                        n_estimators=300,
                        learning_rate=0.05,
                        depth=6,
                    ),
                ],
                meta_model=RidgeModel(alpha=1.0),
                n_folds=5,
            ),
        ),
        evaluator=evaluator,
    ),
}


def run_regression():
    console.print(Panel("[bold blue]REGRESSION EXPERIMENTS[/bold blue]", expand=False))

    results = {}
    for name, exp in experiments.items():
        console.print(Panel(f"[bold blue]Running Regression: {name}[/bold blue]"))
        results[name] = exp.run()
        logger.log(name, results[name], kind="regression")

    reg_table = Table(title="Regression Performance Summary", show_header=True, header_style="bold magenta")
    reg_table.add_column("Model", style="dim", width=20)
    reg_table.add_column("MSE", justify="right")
    reg_table.add_column("RMSE", justify="right")
    reg_table.add_column("R²", justify="right")

    for name, m in results.items():
        reg_table.add_row(name, f"{m['mse']:>10.2f}", f"{m['rmse']:>9.2f}s", f"{m['r2']:>8.4f}")

    console.print("\n", reg_table)
    return results
