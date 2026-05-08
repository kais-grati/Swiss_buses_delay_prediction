from ml.data import DataLoader
from ml.pipeline import MLPipeline
from ml.preprocessors.scaler import FeatureScaler
from ml.preprocessors.temporal import TemporalFeatureExtractor
from ml.preprocessors.target_encoder import HistoricalMeanEncoder
from ml.preprocessors.weather_engineer import WeatherFeatureEngineer
from ml.models.lgbm import LightGBMModel
from ml.models.lgbm_classifier import LightGBMClassifierModel
from ml.models.xgboost_model import XGBoostModel
from ml.models.xgboost_classifier import XGBoostClassifierModel
from ml.models.catboost_model import CatBoostModel
from ml.models.ridge import RidgeModel
from ml.models.stacking import StackingModel
from ml.models.classification_stacking import ClassificationStackingModel
from ml.models.pipelined_classifier import PipelinedClassifierModel
from ml.models.logistic_regression import LogisticRegressionModel
from ml.models.catboost_classifier import CatBoostClassifierModel
from ml.models.ordinal_classifier import OrdinalClassifierModel
from ml.models.random_forest_classifier import RandomForestClassifierModel
from ml.evaluation import Evaluator
from ml.experiment import Experiment, ClassificationExperiment
from ml.preprocessors.delay_binner import DelayBinner
from ml.preprocessors.polynomial import PolynomialExpander
from ml.preprocessors.pca import PCAReducer
from ml.preprocessors.poly_trig import PolyTrigExpander
from ml.preprocessors.nystroem import NystroemExpander
from ml.logger import ExperimentLogger

from rich.console import Console
from rich.table import Table                                                                                                                         
from rich.panel import Panel                                                                                                                         
from rich import print as rprint

console = Console()

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET = "data/dataset_705_echandens.parquet"
TARGET  = "arrival_delay_s"

DROP_COLS = [
    "timestamp", "stop_id", "stop_name", "operator", "line",
    "departure_delay_s",
]

DROP_COLS_KEEP_TS = [
    "stop_id", "stop_name", "operator", "line",
    "departure_delay_s",
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

# ── Experiments ────────────────────────────────────────────────────────────────

loader = DataLoader(path=DATASET, target=TARGET, drop_cols=DROP_COLS)
loader_enhanced = DataLoader(path=DATASET, target=TARGET, drop_cols=DROP_COLS_KEEP_TS)
evaluator = Evaluator()

experiments = {
    # "LightGBM": Experiment(
    #     loader=loader,
    #     pipeline=MLPipeline(
    #         preprocessors=[FeatureScaler(cols=NUMERIC_COLS)],
    #         model=LightGBMModel(n_estimators=500, learning_rate=0.05, num_leaves=31),
    #     ),
    #     evaluator=evaluator,
    # ),
    # "LightGBM-tuned": Experiment(
    #     loader=loader,
    #     pipeline=MLPipeline(
    #         preprocessors=[FeatureScaler(cols=NUMERIC_COLS)],
    #         model=LightGBMModel(
    #             n_estimators=4000,
    #             learning_rate=0.01,
    #             num_leaves=50,
    #             min_child_samples=50,
    #             max_bin=512,
    #             subsample_freq=1,
    #             subsample=0.8,
    #             colsample_bytree=0.8,
    #             reg_alpha=0.1,
    #             reg_lambda=0.1,
    #             min_gain_to_split=0.01,
    #             early_stopping_rounds=50,
    #             log_every=100,
    #         ),
    #     ),
    #     evaluator=evaluator,
    # ),
    # "LightGBM-optuna": Experiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[FeatureScaler(cols=NUMERIC_COLS), 
    #                        TemporalFeatureExtractor()],
    #         model=LightGBMModel(
    #             n_estimators=3000,
    #             early_stopping_rounds=50,
    #             learning_rate=0.011164245970961571,
    #             num_leaves=116,
    #             min_child_samples=43,
    #             min_sum_hessian_in_leaf=0.9756353974751673,
    #             subsample=0.9363929416897576,
    #             subsample_freq=1,
    #             colsample_bytree=0.9988094388849174,
    #             feature_fraction_bynode=0.9450155202549205,
    #             reg_alpha=0.27245190090589816,
    #             reg_lambda=0.78391786930204,
    #             path_smooth=4.405998450654837,
    #         ),
    #     ),
    #     evaluator=evaluator,
    # ),    
    # "LightGBM-optuna2": Experiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[FeatureScaler(cols=NUMERIC_COLS), 
    #                        TemporalFeatureExtractor()],
    #         model=LightGBMModel(
    #             n_estimators=3000,
    #             early_stopping_rounds=50,
    #             learning_rate=0.024864985683759746,
    #             num_leaves=78,
    #             min_child_samples=86,
    #             min_sum_hessian_in_leaf=0.013837807248223857,
    #             subsample=0.9940870655183781,
    #             subsample_freq=1,
    #             colsample_bytree=0.6580004324956996,
    #             feature_fraction_bynode=0.9019956230179453,
    #             reg_alpha=0.29733122973301546,
    #             reg_lambda=0.006579169405573432,
    #             path_smooth=5.094559948404132,
    #         ),
    #     ),
    #     evaluator=evaluator,
    # ),    
    # "LightGBM-optuna3": Experiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #                         WeatherFeatureEngineer(),
    #                         FeatureScaler(cols=NUMERIC_COLS), 
    #                         TemporalFeatureExtractor()
    #                         ],
    #         model=LightGBMModel(
    #             n_estimators=3000,
    #             early_stopping_rounds=50,
    #             learning_rate=0.007126947042377407,
    #             num_leaves=168,
    #             min_child_samples=40,
    #             min_sum_hessian_in_leaf=0.0899176805301358,
    #             subsample=0.9998756827358574,
    #             subsample_freq=1,
    #             colsample_bytree=0.5757330744549699,
    #             feature_fraction_bynode=0.9734799081519098,
    #             reg_alpha=1.7021369826715338,
    #             reg_lambda=7.970747211586931,
    #             path_smooth=4.671171717084908,
    #         ),
    #     ),
    #     evaluator=evaluator,
    # ),
    # "LightGBM-dart": Experiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(cols=NUMERIC_COLS),
    #         ],
    #         model=LightGBMModel(
    #             boosting_type="dart",
    #             n_estimators=1000,
    #             learning_rate=0.05,
    #             num_leaves=116,
    #             min_child_samples=43,
    #             subsample=0.9363929416897576,
    #             subsample_freq=1,
    #             colsample_bytree=0.9988094388849174,
    #             reg_alpha=0.27245190090589816,
    #             reg_lambda=0.78391786930204,
    #             drop_rate=0.1,
    #             skip_drop=0.5,
    #         ),
    #     ),
    #     evaluator=evaluator,
    # ),
    # "Stacking": Experiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(cols=NUMERIC_COLS),
    #         ],
    #         model=StackingModel(
    #             base_models=[
    #                 LightGBMModel(
    #                     n_estimators=3000,
    #                     early_stopping_rounds=50,
    #                     learning_rate=0.011164245970961571,
    #                     num_leaves=116,
    #                     min_child_samples=43,
    #                     min_sum_hessian_in_leaf=0.9756353974751673,
    #                     subsample=0.9363929416897576,
    #                     subsample_freq=1,
    #                     colsample_bytree=0.9988094388849174,
    #                     feature_fraction_bynode=0.9450155202549205,
    #                     reg_alpha=0.27245190090589816,
    #                     reg_lambda=0.78391786930204,
    #                     path_smooth=4.405998450654837,
    #                 ),
    #                 XGBoostModel(
    #                     n_estimators=500,
    #                     learning_rate=0.03809720744535876,
    #                     min_child_weight=90.28669329772264,
    #                     gamma=0.016744637853508566,
    #                     max_depth=9,
    #                     subsample=0.9335838518925651,
    #                     colsample_bytree=0.9389806288470189,
    #                     colsample_bylevel=0.8337241352904957,
    #                     reg_alpha=0.00035666207579412883,
    #                     reg_lambda=1.8707902692338567
    #                 ),
    #             ],
    #             meta_model=RidgeModel(alpha=1.0),
    #             n_folds=5,
    #         ),
    #     ),
    #     evaluator=evaluator,
    # ),
    # "Stacking2": Experiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(cols=NUMERIC_COLS),
    #         ],
    #         model=StackingModel(
    #             base_models=[
    #                 LightGBMModel(
    #                     n_estimators=3000,
    #                     early_stopping_rounds=50,
    #                     learning_rate=0.024864985683759746,
    #                     num_leaves=78,
    #                     min_child_samples=86,
    #                     min_sum_hessian_in_leaf=0.013837807248223857,
    #                     subsample=0.9940870655183781,
    #                     subsample_freq=1,
    #                     colsample_bytree=0.6580004324956996,
    #                     feature_fraction_bynode=0.9019956230179453,
    #                     reg_alpha=0.29733122973301546,
    #                     reg_lambda=0.006579169405573432,
    #                     path_smooth=5.094559948404132,
    #                 ),
    #                 XGBoostModel(
    #                     n_estimators=500,
    #                     learning_rate=0.03809720744535876,
    #                     min_child_weight=90.28669329772264,
    #                     gamma=0.016744637853508566,
    #                     max_depth=9,
    #                     subsample=0.9335838518925651,
    #                     colsample_bytree=0.9389806288470189,
    #                     colsample_bylevel=0.8337241352904957,
    #                     reg_alpha=0.00035666207579412883,
    #                     reg_lambda=1.8707902692338567
    #                 ),
    #                 CatBoostModel(
    #                     n_estimators=2000,
    #                     learning_rate=0.05543418338273171,
    #                     depth=5,
    #                     early_stopping_rounds=50,
    #                     l2_leaf_reg=0.540507962054327,
    #                     random_strength=0.16061584861239986,
    #                     bagging_temperature=0.49356279590945856,
    #                     min_data_in_leaf=62
    #         ),
    #             ],
    #             meta_model=RidgeModel(alpha=1),
    #             n_folds=5,
    #         ),
    #     ),
    #     evaluator=evaluator,
    # ),
    # "CatBoost": Experiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(cols=NUMERIC_COLS),
    #         ],
    #         model=CatBoostModel(
    #             n_estimators=200,
    #             learning_rate=0.05543418338273171,
    #             depth=5,
    #             early_stopping_rounds=50,
    #             l2_leaf_reg=0.540507962054327,
    #             random_strength=0.16061584861239986,
    #             bagging_temperature=0.49356279590945856,
    #             min_data_in_leaf=62
    #         ),
    #     ),
    #     evaluator=evaluator,
    # )
    # "LightGBM-enhanced": Experiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             HistoricalMeanEncoder(group_cols=["hour", "dow"], output_col="hist_mean_delay"),
    #             WeatherFeatureEngineer(),
    #             FeatureScaler(cols=NUMERIC_COLS_ENHANCED),
    #         ],
    #         model=LightGBMModel(
    #             n_estimators=3000,
    #             early_stopping_rounds=50,
    #             learning_rate=0.011164245970961571,
    #             num_leaves=116,
    #             min_child_samples=43,
    #             min_sum_hessian_in_leaf=0.9756353974751673,
    #             subsample=0.9363929416897576,
    #             subsample_freq=1,
    #             colsample_bytree=0.9988094388849174,
    #             feature_fraction_bynode=0.9450155202549205,
    #             reg_alpha=0.27245190090589816,
    #             reg_lambda=0.78391786930204,
    #             path_smooth=4.405998450654837,
    #         ),
    #     ),
    #     evaluator=evaluator,
    # ),
    # "XGBoost": Experiment(
    #     loader=loader,
    #     pipeline=MLPipeline(
    #         preprocessors=[FeatureScaler(cols=NUMERIC_COLS)],
    #         model=XGBoostModel(
    #             n_estimators=500,
    #             learning_rate=0.03809720744535876,
    #             min_child_weight=90.28669329772264,
    #             gamma=0.016744637853508566,
    #             max_depth=9,
    #             subsample=0.9335838518925651,
    #             colsample_bytree=0.9389806288470189,
    #             colsample_bylevel=0.8337241352904957,
    #             reg_alpha=0.00035666207579412883,
    #             reg_lambda=1.8707902692338567,
    #         ),
    #     ),
    #     evaluator=evaluator,
    # ),
    # ── LightGBM: best optuna params + full feature engineering ──────────────────
    # "LGBM-Optuna-Full": Experiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WeatherFeatureEngineer(),
    #             FeatureScaler(cols=NUMERIC_COLS_ENHANCED),
    #         ],
    #         model=LightGBMModel(
    #             n_estimators=3000,
    #             early_stopping_rounds=50,
    #             learning_rate=0.007126947042377407,
    #             num_leaves=168,
    #             min_child_samples=40,
    #             min_sum_hessian_in_leaf=0.0899176805301358,
    #             subsample=0.9998756827358574,
    #             subsample_freq=1,
    #             colsample_bytree=0.5757330744549699,
    #             feature_fraction_bynode=0.9734799081519098,
    #             reg_alpha=1.7021369826715338,
    #             reg_lambda=7.970747211586931,
    #             path_smooth=4.671171717084908,
    #         ),
    #     ),
    #     evaluator=evaluator,
    # ),
    # ── LightGBM: + HistoricalMeanEncoder (never tried for regression) ───────────
    # "LGBM-HistMean": Experiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WeatherFeatureEngineer(),
    #             HistoricalMeanEncoder(group_cols=["hour", "dow"], output_col="hist_mean_delay"),
    #             FeatureScaler(cols=NUMERIC_COLS_ENHANCED + ["hist_mean_delay"]),
    #         ],
    #         model=LightGBMModel(
    #             n_estimators=3000,
    #             early_stopping_rounds=50,
    #             learning_rate=0.007126947042377407,
    #             num_leaves=168,
    #             min_child_samples=40,
    #             min_sum_hessian_in_leaf=0.0899176805301358,
    #             subsample=0.9998756827358574,
    #             subsample_freq=1,
    #             colsample_bytree=0.5757330744549699,
    #             feature_fraction_bynode=0.9734799081519098,
    #             reg_alpha=1.7021369826715338,
    #             reg_lambda=7.970747211586931,
    #             path_smooth=4.671171717084908,
    #         ),
    #     ),
    #     evaluator=evaluator,
    # ),
}


# ── Run Regression ────────────────────────────────────────────────────────────────      
                                                           
logger = ExperimentLogger()

results_reg = {}                                                                                                                                     
for name, exp in experiments.items():                                                                                                                
    console.print(Panel(f"[bold blue]Running Regression: {name}[/bold blue]"))                                                                       
    results_reg[name] = exp.run()
    logger.log(name, results_reg[name], kind="regression")                                                                                                                    
                                                                                                                                                    
# ── Summary Regression ──────────────────────────────────────────────────────────────                                                               
reg_table = Table(title="Regression Performance Summary", show_header=True, header_style="bold magenta")                                             
reg_table.add_column("Model", style="dim", width=20)                                                                                                 
reg_table.add_column("MSE", justify="right")                                                                                                         
reg_table.add_column("RMSE", justify="right")                                                                                                        
reg_table.add_column("R²", justify="right")                                                                                                          
                                                                                                                                                    
for name, m in results_reg.items():                                                                                                                  
    reg_table.add_row(name, f"{m['mse']:>10.2f}", f"{m['rmse']:>9.2f}s", f"{m['r2']:>8.4f}")                                                         
                                                                                                                                                    
console.print("\n", reg_table)                                                                                                                       
                                                                                                                                                    
# ── Classification Experiments ──────────────────────────────────────────────────────

console.print("\n", Panel("[bold green]CLASSIFICATION EXPERIMENTS[/bold green]", expand=False))

binner = DelayBinner()

class_experiments = {
    # ── LogReg best (ElasticNet + full features) — linear ceiling reference ──────
    # "LogReg-ElasticNet": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WeatherFeatureEngineer(),
    #             HistoricalMeanEncoder(group_cols=["hour", "dow"], output_col="hist_mean_delay"),
    #             FeatureScaler(cols=NUMERIC_COLS_LOGREG_FULL),
    #         ],
    #         model=LogisticRegressionModel(C=1.0, l1_ratio=0.5, max_iter=5000),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # ── LGBM ───────────────────────────────────────────────
    # "LGBM-v1-1k-rebalanced_classes": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(NUMERIC_COLS)
    #         ],
    #         model=LightGBMClassifierModel(
    #             n_estimators=1000,
    #             learning_rate=0.07464436769258928,
    #             num_leaves=80,
    #             min_child_samples=30,
    #             min_sum_hessian_in_leaf=0.007859583418210319,
    #             subsample=0.9651336534916378,
    #             subsample_freq=1,
    #             colsample_bytree=0.8570097400335258,
    #             feature_fraction_bynode=0.733164261500722,
    #             reg_alpha=0.003622284016704569,
    #             reg_lambda=0.6238096481891778,
    #             early_stopping_rounds=50,
    #             class_weight="balanced",
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # ── LGBM (default) ──────────────────────────────────────────────────────
    "LGBM-v1": ClassificationExperiment(
        loader=loader_enhanced,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                # HistoricalMeanEncoder(group_cols=["hour", "dow"], output_col="hist_mean_delay"),
                FeatureScaler(NUMERIC_COLS ),
            ],
            model=LightGBMClassifierModel(
                early_stopping_rounds=50,
                n_estimators = 871,
                learning_rate = 0.012640427507525883,
                num_leaves = 76,
                min_child_samples = 72,
                min_sum_hessian_in_leaf = 0.000774294856017507,
                subsample = 0.9742887241908837,
                colsample_bytree = 0.4024274899752304,
                feature_fraction_bynode = 0.8649700229855798,
                reg_alpha = 0.01239086706698562,
                reg_lambda = 0.00019579208402214886
            ),
        ),
        evaluator=evaluator,
        encoder=binner,
    ),
    # ── XGBoost (default) ────────────────────────────────────────────────────
    "XGBoost-v1": ClassificationExperiment(
        loader=loader_enhanced,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                # HistoricalMeanEncoder(group_cols=["hour", "dow"], output_col="hist_mean_delay"),
                FeatureScaler(NUMERIC_COLS ),
            ],
            model=XGBoostClassifierModel(
                n_estimators = 1701,
                learning_rate = 0.06237053859955787,
                max_depth = 3,
                min_child_weight = 13.415832894977182,
                gamma = 0.03576555396806631,
                subsample = 0.9489968100219347,
                colsample_bytree = 0.44933374667783377,
                colsample_bylevel = 0.967285333439918,
                reg_alpha = 0.0017661547027850485,
                reg_lambda = 0.7839210110834374
            ),
        ),
        evaluator=evaluator,
        encoder=binner,
    ),
    # ── CatBoost (default) ───────────────────────────────────────────────────
    "CatBoost-v1": ClassificationExperiment(
        loader=loader_enhanced,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                # HistoricalMeanEncoder(group_cols=["hour", "dow"], output_col="hist_mean_delay"),
                FeatureScaler(NUMERIC_COLS ),
            ],
            model=CatBoostClassifierModel(
                early_stopping_rounds=50,
            ),
        ),
        evaluator=evaluator,
        encoder=binner,
    ),
    # ── Random Forest (default) ──────────────────────────────────────────────
    "RandomForest-v1": ClassificationExperiment(
        loader=loader_enhanced,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                # HistoricalMeanEncoder(group_cols=["hour", "dow"], output_col="hist_mean_delay"),
                FeatureScaler(NUMERIC_COLS ),
            ],
            model=RandomForestClassifierModel(
                n_estimators = 275,
                max_depth = 29,
                min_samples_split = 12,
                min_samples_leaf = 7,
                max_features = None
            ),
        ),
        evaluator=evaluator,
        encoder=binner,
    ),
    # ── Ordinal LGBM ────────────────────────────────────────────────────────
    "Ordinal-LGBM-v1": ClassificationExperiment(
        loader=loader_enhanced,
        pipeline=MLPipeline(
            preprocessors=[
                TemporalFeatureExtractor(),
                # HistoricalMeanEncoder(group_cols=["hour", "dow"], output_col="hist_meadimsn_delay"),
                FeatureScaler(NUMERIC_COLS ),
            ],
            model=OrdinalClassifierModel(
                base_model=LightGBMClassifierModel(
                    n_estimators = 166,
                    learning_rate = 0.006018966874652439,
                    num_leaves = 147,
                    min_child_samples = 12,
                    min_sum_hessian_in_leaf = 0.0001537852534699058,
                    subsample = 0.7923097407871328,
                    colsample_bytree = 0.49577036575939126,
                    feature_fraction_bynode = 0.9992193261712309,
                    reg_alpha = 0.00021930676830366464,
                    reg_lambda = 0.017232126786473668
                ),
            ),
        ),
        evaluator=evaluator,
        encoder=binner,
    ),
    # ── LGBM: lower min_child_samples to help minority class 3 ──────────────────
    # "LGBM-v2-LowChild": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WeatherFeatureEngineer(),
    #             HistoricalMeanEncoder(group_cols=["hour", "dow"], output_col="hist_mean_delay"),
    #         ],
    #         model=LightGBMClassifierModel(
    #             n_estimators=2000,
    #             learning_rate=0.03,
    #             num_leaves=63,
    #             min_child_samples=1,
    #             min_sum_hessian_in_leaf=1e-5,
    #             subsample=0.8,
    #             subsample_freq=1,
    #             colsample_bytree=0.8,
    #             reg_alpha=0.1,
    #             reg_lambda=0.1,
    #             early_stopping_rounds=50,
    #             class_weight="balanced",
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # ── LGBM: deeper trees + richer hist features ────────────────────────────────
    # "LGBM-v3-DeepHist": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WeatherFeatureEngineer(),
    #             HistoricalMeanEncoder(group_cols=["hour", "dow"], output_col="hist_mean_delay"),
    #             HistoricalMeanEncoder(group_cols=["hour"], output_col="hist_mean_hour"),
    #             HistoricalMeanEncoder(group_cols=["dow"], output_col="hist_mean_dow"),
    #         ],
    #         model=LightGBMClassifierModel(
    #             n_estimators=2000,
    #             learning_rate=0.03,
    #             num_leaves=127,
    #             min_child_samples=5,
    #             subsample=0.8,
    #             subsample_freq=1,
    #             colsample_bytree=0.8,
    #             feature_fraction_bynode=0.9,
    #             reg_alpha=0.05,
    #             reg_lambda=0.05,
    #             early_stopping_rounds=50,
    #             class_weight="balanced",
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # ── XGBoost ───────────────────────────────────────────────────────────────────
    # "XGBoost-Classifier": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(NUMERIC_COLS)
    #         ],
    #         model=XGBoostClassifierModel(
    #             n_estimators=1000,
    #             learning_rate=0.05,
    #             max_depth=6,
    #             min_child_weight=5.0,
    #             subsample=0.8,
    #             colsample_bytree=0.8,
    #             reg_alpha=0.1,
    #             reg_lambda=1.0,
    #             early_stopping_rounds=50,
    #             class_weight="balanced",
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # ── LGBM: manual class weights — heavily boost rare class 3 ──────────────────
    # "LGBM-v4-HeavyClass3": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             WeatherFeatureEngineer(),
    #             HistoricalMeanEncoder(group_cols=["hour", "dow"], output_col="hist_mean_delay"),
    #         ],
    #         model=LightGBMClassifierModel(
    #             n_estimators=1000,
    #             learning_rate=0.05,
    #             num_leaves=63,
    #             min_child_samples=1,
    #             min_sum_hessian_in_leaf=1e-5,
    #             subsample=0.8,
    #             subsample_freq=1,
    #             colsample_bytree=0.8,
    #             reg_alpha=0.1,
    #             reg_lambda=0.1,
    #             early_stopping_rounds=50,
    #             # manual weights: inversely proportional to class frequency, but 10x boost on class 3
    #             class_weight={0: 1.0, 1: 1.08, 2: 3.32, 3: 100.0},
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # ── CatBoost (fixed: Accuracy eval metric) ────────────────────────────────────
    # ── CatBoost: no class weighting — natural class distribution ────────────────
    # "CatBoost-NoWeights": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(NUMERIC_COLS)
    #         ],
    #         model=CatBoostClassifierModel(
    #             n_estimators=1000,
    #             learning_rate=0.05,
    #             depth=6,
    #             l2_leaf_reg=3.0,
    #             min_data_in_leaf=5,
    #             auto_class_weights=None,
    #             early_stopping_rounds=50,
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # ── Stacking: LGBM + XGBoost + CatBoost + LogReg → LogisticRegression meta-learner ──
    # "Stacking-LGBM-XGB-CatBoost-LogReg": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     evaluator=evaluator,
    #     encoder=binner,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(cols=NUMERIC_COLS_LOGREG),
    #         ],
    #         model=ClassificationStackingModel(
    #             base_models=[
    #                 LightGBMClassifierModel(
    #                     n_estimators=1000,
    #                     learning_rate=0.05,
    #                     num_leaves=63,
    #                     min_child_samples=10,
    #                     subsample=0.8,
    #                     subsample_freq=1,
    #                     colsample_bytree=0.8,
    #                     reg_alpha=0.1,
    #                     reg_lambda=0.1,
    #                     early_stopping_rounds=50,
    #                     class_weight="balanced",
    #                 ),
    #                 XGBoostClassifierModel(
    #                     n_estimators=1000,
    #                     learning_rate=0.05,
    #                     max_depth=6,
    #                     min_child_weight=5.0,
    #                     subsample=0.8,
    #                     colsample_bytree=0.8,
    #                     reg_alpha=0.1,
    #                     reg_lambda=1.0,
    #                     early_stopping_rounds=50,
    #                     class_weight="balanced",
    #                 ),
    #                 CatBoostClassifierModel(
    #                     n_estimators=1000,
    #                     learning_rate=0.05,
    #                     depth=6,
    #                     l2_leaf_reg=3.0,
    #                     min_data_in_leaf=5,
    #                     early_stopping_rounds=50,
    #                 ),
    #                 # Nyström RBF LogReg: input is already scaled by the pipeline,
    #                 # so only the kernel expansion is applied internally.
    #                 PipelinedClassifierModel(
    #                     preprocessors=[NystroemExpander(n_components=100, kernel="rbf")],
    #                     classifier=LogisticRegressionModel(C=1.0, max_iter=5000),
    #                 ),
    #             ],
    #             meta_model=LogisticRegressionModel(C=1.0, max_iter=2000),
    #         ),
    #     ),
    # ),

    # "Stacking-LGBM-LogReg": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(cols=NUMERIC_COLS_LOGREG),
    #         ],
    #         model=ClassificationStackingModel(
    #             base_models=[
    #                 LightGBMClassifierModel(
    #                     n_estimators=1000,
    #                     learning_rate=0.05,
    #                     num_leaves=63,
    #                     min_child_samples=10,
    #                     subsample=0.8,
    #                     subsample_freq=1,
    #                     colsample_bytree=0.8,
    #                     reg_alpha=0.1,
    #                     reg_lambda=0.1,
    #                     early_stopping_rounds=50,
    #                     class_weight="balanced",
    #                 ),
                    
    #                 # Nyström RBF LogReg: input is already scaled by the pipeline,
    #                 # so only the kernel expansion is applied internally.
    #                 PipelinedClassifierModel(
    #                     preprocessors=[NystroemExpander(n_components=100, kernel="rbf")],
    #                     classifier=LogisticRegressionModel(C=1.0, max_iter=5000),
    #                 ),
    #             ],
    #             meta_model=LogisticRegressionModel(C=1.0, max_iter=2000),
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # "Preproc_Stacking-LGBM-LogReg": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(cols=NUMERIC_COLS_LOGREG),
    #         ],
    #         model=ClassificationStackingModel(
    #             base_models=[
    #                 LightGBMClassifierModel(
    #                     n_estimators=1000,
    #                     learning_rate=0.05,
    #                     num_leaves=63,
    #                     min_child_samples=10,
    #                     subsample=0.8,
    #                     subsample_freq=1,
    #                     colsample_bytree=0.8,
    #                     reg_alpha=0.1,
    #                     reg_lambda=0.1,
    #                     early_stopping_rounds=50,
    #                     class_weight="balanced",
    #                 ),
    #                 XGBoostClassifierModel(
    #                     n_estimators=1000,
    #                     learning_rate=0.05,
    #                     max_depth=6,
    #                     min_child_weight=5.0,
    #                     subsample=0.8,
    #                     colsample_bytree=0.8,
    #                     reg_alpha=0.1,
    #                     reg_lambda=1.0,
    #                     early_stopping_rounds=50,
    #                     class_weight="balanced",
    #                 ),
                    
    #                 # Nyström RBF LogReg: input is already scaled by the pipeline,
    #                 # so only the kernel expansion is applied internally.
    #                 PipelinedClassifierModel(
    #                     preprocessors=[NystroemExpander(n_components=100, kernel="rbf")],
    #                     classifier=LogisticRegressionModel(C=1.0, max_iter=5000),
    #                 ),
    #             ],
    #             meta_model=PipelinedClassifierModel(
    #                 preprocessors=[NystroemExpander(n_components=20, kernel="rbf")],
    #                 classifier=LogisticRegressionModel(C=1.0, max_iter=2000),
    #             ),
    #         ),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),

    # ── LogReg + degree-2 polynomial expansion of scaled numeric features ────────
    # "LogReg-Poly2": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(cols=NUMERIC_COLS_LOGREG),
    #         ],
    #         model=LogisticRegressionModel(C=1.0, max_iter=5000),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # ── LogReg + Poly2 + PCA: decorrelate the ~77 polynomial features ─────────────
    # "LogReg-Poly2-PCA": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(cols=NUMERIC_COLS_LOGREG),
    #             # PolynomialExpander(cols=NUMERIC_COLS_LOGREG, degree=2),
    #             PCAReducer(variance_threshold=0.9),
    #         ],
    #         model=LogisticRegressionModel(C=1.0, max_iter=5000),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # ── LogReg + polynomial + trig (sin/cos) expansions ──────────────────────────
    # "LogReg-PolyTrig": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(cols=NUMERIC_COLS_LOGREG),
    #             PolyTrigExpander(cols=NUMERIC_COLS_LOGREG, degree=2, n_trig=1),
    #         ],
    #         model=LogisticRegressionModel(C=1.0, max_iter=5000),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # ── Same but PCA-compressed to handle the ~99-feature expansion ──────────────
    # "LogReg-PolyTrig-PCA": ClassificationExperiment(
    #     loader=loader_enhanced,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             TemporalFeatureExtractor(),
    #             FeatureScaler(cols=NUMERIC_COLS_LOGREG),
    #             PolyTrigExpander(cols=NUMERIC_COLS_LOGREG, degree=2, n_trig=1),
    #             PCAReducer(variance_threshold=0.95),
    #         ],
    #         model=LogisticRegressionModel(C=1.0, max_iter=5000),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
    # ── LogReg + Nyström RBF kernel approximation ─────────────────────────────────
    # "LogReg-Nystroem": ClassificationExperiment(
    #     loader=loader,
    #     pipeline=MLPipeline(
    #         preprocessors=[
    #             # TemporalFeatureExtractor(),
    #             # FeatureScaler(cols=NUMERIC_COLS_LOGREG),
    #             NystroemExpander(n_components=100, kernel="rbf"),
    #         ],
    #         model=LogisticRegressionModel(C=1.0, max_iter=5000),
    #     ),
    #     evaluator=evaluator,
    #     encoder=binner,
    # ),
}

results_clf = {}
for name, exp in class_experiments.items():
    console.print(Panel(f"[bold green]Running Classification: {name}[/bold green]"))
    results_clf[name] = exp.run()
    logger.log(name, results_clf[name], kind="classification")

# ── Summary Classification ───────────────────────────────────────────────────────
clf_table = Table(title="Classification Performance Summary", show_header=True, header_style="bold cyan")
clf_table.add_column("Model", style="dim", width=30)
clf_table.add_column("Macro-F1", justify="right")

for name, m in results_clf.items():
    clf_table.add_row(name, f"{m['f1']:>15.4f}")

console.print("\n", clf_table)

# ── Hyperparameter Optimization ────────────────────────────────────────────────

from ml.optimizer import LGBMRegressorOptimizer, XGBoostRegressorOptimizer, CatBoostRegressorOptimizer, LGBMClassifierOptimizer, OrdinalLGBMClassifierOptimizer, XGBoostClassifierOptimizer, CatBoostClassifierOptimizer, RandomForestClassifierOptimizer

console.print("\n", Panel("[bold yellow]CLASSIFIER OPTUNA OPTIMIZATION[/bold yellow]", expand=False))

# ── Ordinal LGBM Optimizer ────────────────────────────────────────────────────
"""
                    n_estimators = 166,
                    learning_rate = 0.006018966874652439,
                    num_leaves = 147,
                    min_child_samples = 12,
                    min_sum_hessian_in_leaf = 0.0001537852534699058,
                    subsample = 0.7923097407871328,
                    colsample_bytree = 0.49577036575939126,
                    feature_fraction_bynode = 0.9992193261712309,
                    reg_alpha = 0.00021930676830366464,
                    reg_lambda = 0.017232126786473668
"""
# ordinal_optimizer = OrdinalLGBMClassifierOptimizer(
#     loader=loader_enhanced,
#     binner=binner,
#     n_trials=75,
#     n_estimators=800,
# )
# ordinal_study = ordinal_optimizer.optimize()

"""
                    n_estimators = 871
                    learning_rate = 0.012640427507525883
                    num_leaves = 76
                    min_child_samples = 72
                    min_sum_hessian_in_leaf = 0.000774294856017507
                    subsample = 0.9742887241908837
                    colsample_bytree = 0.4024274899752304
                    feature_fraction_bynode = 0.8649700229855798
                    reg_alpha = 0.01239086706698562
                    reg_lambda = 0.00019579208402214886
"""
# optimizer = LGBMClassifierOptimizer(
#     loader=loader_enhanced,
#     n_trials=60,
# )
# study = optimizer.optimize()

"""
                n_estimators = 1701
                learning_rate = 0.06237053859955787
                max_depth = 3
                min_child_weight = 13.415832894977182
                gamma = 0.03576555396806631
                subsample = 0.9489968100219347
                colsample_bytree = 0.44933374667783377
                colsample_bylevel = 0.967285333439918
                reg_alpha = 0.0017661547027850485
                reg_lambda = 0.7839210110834374
"""

# optimizer = XGBoostClassifierOptimizer(
#     loader=loader_enhanced,
#     binner=binner,
#     n_trials=60,
#     n_estimators=2000,
# )
# study = optimizer.optimize()

"""

"""

optimizer = CatBoostClassifierOptimizer(
    loader=loader_enhanced,
    binner=binner,
    n_trials=60,
)
study = optimizer.optimize()

"""
            n_estimators = 275
            max_depth = 29
            min_samples_split = 12
            min_samples_leaf = 7
            max_features = None
"""

# optimizer = RandomForestClassifierOptimizer(
#     loader=loader_enhanced,
#     binner=binner,
#     n_trials=40,
# )
# study = optimizer.optimize()