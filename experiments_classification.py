from ml.pipeline import MLPipeline
from ml.preprocessors.scaler import FeatureScaler
from ml.preprocessors.string_encoder import StringEncoder
from ml.preprocessors.temporal import TemporalFeatureExtractor
from ml.preprocessors.wind_merger import WindMerger
from ml.preprocessors.polynomial import PolynomialExpander
from ml.preprocessors.pca import PCAReducer
from ml.preprocessors.poly_trig import PolyTrigExpander
from ml.preprocessors.nystroem import NystroemExpander
from ml.models.lgbm_classifier import LightGBMClassifierModel
from ml.models.xgboost_classifier import XGBoostClassifierModel
from ml.models.catboost_classifier import CatBoostClassifierModel
from ml.models.random_forest_classifier import RandomForestClassifierModel
from ml.models.ordinal_classifier import OrdinalClassifierModel
from ml.models.logistic_regression import LogisticRegressionModel
from ml.models.classification_stacking import ClassificationStackingModel
from ml.models.pipelined_classifier import PipelinedClassifierModel
from ml.models.mlp_classifier import MLPClassifierModel
from ml.experiment import ClassificationExperiment

from pathlib import Path

from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from config import (
    console, loader_enhanced, loader_lag, evaluator, binner, logger,
    LOGREG_NUMERIC,
)

class_experiments = {
    # "LogReg-Best": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(cols=LOGREG_NUMERIC),
    #             PolynomialExpander(cols=LOGREG_NUMERIC, degree=2),
    #             PCAReducer(variance_threshold=0.99),
    #         ],
    #         model=LogisticRegressionModel(C=1.0, l1_ratio=0.5, max_iter=5000),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # "LogReg-PolyTrig": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(cols=LOGREG_NUMERIC),
    #             PolyTrigExpander(cols=LOGREG_NUMERIC, degree=2),
    #             PCAReducer(variance_threshold=0.99),
    #         ],
    #         model=LogisticRegressionModel(C=1.0, l1_ratio=0.5, max_iter=5000),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # "LogReg-RBF": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(cols=LOGREG_NUMERIC),
    #             NystroemExpander(),
    #             # PCAReducer(variance_threshold=0.99),
    #         ],
    #         model=LogisticRegressionModel(C=1, l1_ratio=0.5, max_iter=5000),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # "LogReg-PCA": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(cols=LOGREG_NUMERIC),
    #             PCAReducer(variance_threshold=0.99),
    #         ],
    #         model=LogisticRegressionModel(C=1.0, l1_ratio=0.5, max_iter=5000),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),

    # "LGBM": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth",]),
    #         ],
    #         model=LightGBMClassifierModel(
    #             early_stopping_rounds=50,
    #             n_estimators = 1387,
    #             scale_pos_weight = 0.6986522731068953,
    #             learning_rate = 0.08504465534445489,
    #             num_leaves = 138,
    #             min_child_samples = 95,
    #             min_sum_hessian_in_leaf = 0.017830620376047334,
    #             subsample = 0.9603385505500266,
    #             colsample_bytree = 0.6506589536610681,
    #             feature_fraction_bynode = 0.43693599956361384,
    #             reg_alpha = 0.06257402445676312,
    #             reg_lambda = 0.008566280525044545
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # "XGBoost": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth",]),
    #         ],
    #         model=XGBoostClassifierModel(
    #             n_estimators = 1975,
    #             scale_pos_weight = 0.5878398014833414,
    #             learning_rate = 0.033623211512864536,
    #             max_depth = 10,
    #             min_child_weight = 21.77477570888255,
    #             gamma = 0.05106025524429455,
    #             subsample = 0.8194299288251159,
    #             colsample_bytree = 0.8395862195507668,
    #             colsample_bylevel = 0.9707654648613212,
    #             reg_alpha = 1.2217369358360592,
    #             reg_lambda = 0.4404197461177921
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # "CatBoost": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth",]),
    #         ],
    #         model=CatBoostClassifierModel(
    #             n_estimators = 440,
    #             auto_class_weights = "SqrtBalanced",
    #             learning_rate = 0.06879543869404516,
    #             depth = 7,
    #             l2_leaf_reg = 0.4735532917047472,
    #             random_strength = 0.07528620336596245,
    #             bagging_temperature = 0.2196657341097641,
    #             min_data_in_leaf = 73
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # "RandomForest": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth",]),
    #         ],
    #         model=RandomForestClassifierModel(
    #             n_estimators = 357,
    #             max_depth = 23,
    #             min_samples_split = 15,
    #             min_samples_leaf = 7,
    #             max_features = "sqrt",
    #             class_weight = "balanced"
    #             ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # "Ordinal-LGBM": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth",]),
    #         ],
    #         model=OrdinalClassifierModel(
    #             base_model=LightGBMClassifierModel(
    #                 n_estimators = 683,
    #                 scale_pos_weight = 0.6658178271472204,
    #                 learning_rate = 0.012067638787755042,
    #                 num_leaves = 184,
    #                 min_child_samples = 83,
    #                 min_sum_hessian_in_leaf = 0.03792040702863287,
    #                 subsample = 0.6382348323819672,
    #                 colsample_bytree = 0.6580390699089372,
    #                 feature_fraction_bynode = 0.7730795062891931,
    #                 reg_alpha = 0.002551973673478356,
    #                 reg_lambda = 0.01849403951716357
    #             ),
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # "Ordinal-XGBoost": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth",]),
    #         ],
    #         model=OrdinalClassifierModel(
    #             base_model=XGBoostClassifierModel(
    #                 n_estimators = 573,
    #                 scale_pos_weight = 0.6569387115853094,
    #                 learning_rate = 0.022365990585442086,
    #                 max_depth = 8,
    #                 min_child_weight = 2.594662538392548,
    #                 gamma = 0.00043870628771772006,
    #                 subsample = 0.5893496666560825,
    #                 colsample_bytree = 0.756233084869846,
    #                 colsample_bylevel = 0.7301284243764258,
    #                 reg_alpha = 0.001212947904518468,
    #                 reg_lambda = 0.23978457224831312
    #             ),
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),

    "LGBM-Lag": ClassificationExperiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                WindMerger(),
                StringEncoder(cols=["operator", "line"]),
            ],
            model=LightGBMClassifierModel(
                early_stopping_rounds=50,
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=127,
                min_child_samples=100,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=0.1,
            ),
        ),
        evaluator=evaluator,
        encoder=binner,
    ),



    # "Stacking-1": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth"]),
    #         ],
    #         model=ClassificationStackingModel(
    #             base_models=[
    #                 LightGBMClassifierModel(
    #                     early_stopping_rounds=50,
    #                     n_estimators = 1387,
    #                     scale_pos_weight = 0.6986522731068953,
    #                     learning_rate = 0.08504465534445489,
    #                     num_leaves = 138,
    #                     min_child_samples = 95,
    #                     min_sum_hessian_in_leaf = 0.017830620376047334,
    #                     subsample = 0.9603385505500266,
    #                     colsample_bytree = 0.6506589536610681,
    #                     feature_fraction_bynode = 0.43693599956361384,
    #                     reg_alpha = 0.06257402445676312,
    #                     reg_lambda = 0.008566280525044545
    #                 ),
    #                 RandomForestClassifierModel(
    #                     n_estimators = 357,
    #                     max_depth = 23,
    #                     min_samples_split = 15,
    #                     min_samples_leaf = 7,
    #                     max_features = "sqrt",
    #                     class_weight = "balanced"
    #                 ),
    #             ],
    #             meta_model=LogisticRegressionModel(C=1.0, max_iter=2000),
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),

    # "Stacking-2": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth"]),
    #         ],
    #         model=ClassificationStackingModel(
    #             base_models=[
    #                 OrdinalClassifierModel(
    #                     base_model=LightGBMClassifierModel(
    #                         n_estimators = 683,
    #                         scale_pos_weight = 0.6658178271472204,
    #                         learning_rate = 0.012067638787755042,
    #                         num_leaves = 184,
    #                         min_child_samples = 83,
    #                         min_sum_hessian_in_leaf = 0.03792040702863287,
    #                         subsample = 0.6382348323819672,
    #                         colsample_bytree = 0.6580390699089372,
    #                         feature_fraction_bynode = 0.7730795062891931,
    #                         reg_alpha = 0.002551973673478356,
    #                         reg_lambda = 0.01849403951716357
    #                     ),
    #                 ),
    #                 RandomForestClassifierModel(
    #                     n_estimators = 357,
    #                     max_depth = 23,
    #                     min_samples_split = 15,
    #                     min_samples_leaf = 7,
    #                     max_features = "sqrt",
    #                     class_weight = "balanced"
    #                 ),
    #             ],
    #             meta_model=LogisticRegressionModel(C=1.0, max_iter=2000),
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),

    # "Stacking-3": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth"]),
    #         ],
    #         model=ClassificationStackingModel(
    #             base_models=[
    #                 LightGBMClassifierModel(
    #                     early_stopping_rounds=50,
    #                     n_estimators = 1387,
    #                     scale_pos_weight = 0.6986522731068953,
    #                     learning_rate = 0.08504465534445489,
    #                     num_leaves = 138,
    #                     min_child_samples = 95,
    #                     min_sum_hessian_in_leaf = 0.017830620376047334,
    #                     subsample = 0.9603385505500266,
    #                     colsample_bytree = 0.6506589536610681,
    #                     feature_fraction_bynode = 0.43693599956361384,
    #                     reg_alpha = 0.06257402445676312,
    #                     reg_lambda = 0.008566280525044545
    #                 ),
    #                 RandomForestClassifierModel(
    #                     n_estimators = 357,
    #                     max_depth = 23,
    #                     min_samples_split = 15,
    #                     min_samples_leaf = 7,
    #                     max_features = "sqrt",
    #                     class_weight = "balanced"
    #                 ),
    #                 PipelinedClassifierModel(
    #                     preprocessors=[
    #                         StringEncoder(cols=["operator", "line"]),
    #                         NystroemExpander()
    #                     ],
    #                     classifier=LogisticRegressionModel(C=1.0, l1_ratio=0.5, max_iter=5000),
    #                 ),
    #             ],
    #             meta_model=LogisticRegressionModel(C=1.0, max_iter=2000),
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),

    # "Stacking-4": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WindMerger(),
    #             StringEncoder(cols=["operator", "line"]),
    #             FeatureScaler(["temperature", "precipitation", "sunshine", "humidity", "wind", "pressure", "snow_depth"]),
    #         ],
    #         model=ClassificationStackingModel(
    #             base_models=[
    #                 LightGBMClassifierModel(
    #                     early_stopping_rounds=50,
    #                     n_estimators = 1387,
    #                     scale_pos_weight = 0.6986522731068953,
    #                     learning_rate = 0.08504465534445489,
    #                     num_leaves = 138,
    #                     min_child_samples = 95,
    #                     min_sum_hessian_in_leaf = 0.017830620376047334,
    #                     subsample = 0.9603385505500266,
    #                     colsample_bytree = 0.6506589536610681,
    #                     feature_fraction_bynode = 0.43693599956361384,
    #                     reg_alpha = 0.06257402445676312,
    #                     reg_lambda = 0.008566280525044545
    #                 ),
    #                 RandomForestClassifierModel(
    #                     n_estimators = 357,
    #                     max_depth = 23,
    #                     min_samples_split = 15,
    #                     min_samples_leaf = 7,
    #                     max_features = "sqrt",
    #                     class_weight = "balanced"
    #                 ),
    #             ],
    #             meta_model=LightGBMClassifierModel(
    #                 n_estimators=300,
    #                 learning_rate=0.05,
    #                 num_leaves=15,
    #                 min_child_samples=20,
    #                 reg_alpha=0.5,
    #                 reg_lambda=0.5,
    #             ),
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),

}


def run_classification(save: bool = False):
    console.print("\n", Panel("[bold green]CLASSIFICATION EXPERIMENTS[/bold green]", expand=False))

    results = {}
    for name, exp in class_experiments.items():
        console.print(Panel(f"[bold green]Running Classification: {name}[/bold green]"))
        results[name] = exp.run()
        logger.log(name, results[name], kind="classification")

    clf_table = Table(title="Classification Performance Summary", show_header=True, header_style="bold cyan")
    clf_table.add_column("Model", style="dim", width=30)
    clf_table.add_column("Macro-F1", justify="right")

    for name, m in results.items():
        clf_table.add_row(name, f"{m['f1']:>15.4f}")

    console.print("\n", clf_table)

    if save:
        SAVE_DIR = "saved_models"
        console.print(Panel(f"[bold]Saving models to {SAVE_DIR}/[/bold]"))
        for name, exp in class_experiments.items():
            path = Path(SAVE_DIR) / name
            exp.pipeline.save(str(path))
            rprint(f"  [green]✓[/green] {name} → {path}")

    return results
