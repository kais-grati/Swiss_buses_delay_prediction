from ml.data import DataLoader
from ml.pipeline import MLPipeline
from ml.preprocessors.scaler import FeatureScaler
from ml.preprocessors.temporal import TemporalFeatureExtractor
from ml.preprocessors.target_encoder import HistoricalMeanEncoder
from ml.preprocessors.weather_engineer import WeatherFeatureEngineer
from ml.preprocessors.weather_rush_hour import WeatherRushHourPreprocessor
from ml.models.lgbm import LightGBMModel
from ml.evaluation import Evaluator
from ml.experiment import Experiment, ClassificationExperiment
from ml.preprocessors.delay_binner import DelayBinner

from ml.models.xgboost_model import XGBoostModel
from ml.models.catboost_model import CatBoostModel
from ml.models.ridge import RidgeModel
from ml.models.stacking import StackingModel
from ml.models.logistic_regression import LogisticRegressionModel

from ml.models.logistic_regression import LogisticRegressionModel                                                                                    
from ml.experiment import Experiment, ClassificationExperiment                                                                                       
from ml.preprocessors.delay_binner import DelayBinner

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
    "XGBoost": Experiment(
        loader=loader,
        pipeline=MLPipeline(
            preprocessors=[FeatureScaler(cols=NUMERIC_COLS)],
            model=XGBoostModel(
                n_estimators=500,
                learning_rate=0.03809720744535876,
                min_child_weight=90.28669329772264,
                gamma=0.016744637853508566,
                max_depth=9,
                subsample=0.9335838518925651,
                colsample_bytree=0.9389806288470189,
                colsample_bylevel=0.8337241352904957,
                reg_alpha=0.00035666207579412883,
                reg_lambda=1.8707902692338567

            ),
        ),
        evaluator=evaluator,
    ),
}


# ── Run Regression ────────────────────────────────────────────────────────────────      
                                                           
results_reg = {}                                                                                                                                     
for name, exp in experiments.items():                                                                                                                
    console.print(Panel(f"[bold blue]Running Regression: {name}[/bold blue]"))                                                                       
    results_reg[name] = exp.run()                                                                                                                    
                                                                                                                                                    
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
                
binner = DelayBinner(bins=[120, 600])                                                                                                                
                
class_experiments = {                                                                                                                                
    "LogReg-Balanced": ClassificationExperiment(
        loader=loader_enhanced,                                                                                                                      
        pipeline=MLPipeline(                                                                                                                         
            preprocessors=[                                                                                                                          
                TemporalFeatureExtractor(),                                                                                                          
                FeatureScaler(cols=NUMERIC_COLS),                                                                                                    
            ],                                                                                                                                       
            model=LogisticRegressionModel(class_weight="balanced"),                                                                                  
        ),                                                                                                                                           
        evaluator=evaluator,                                                                                                                         
        encoder=binner,                                                                                                                              
    ),                                                                                                                                               
}                                                                                                                                                    
                                                                                                                                                    
results_clf = {}                                                                                                                                     
for name, exp in class_experiments.items():                                                                                                          
    console.print(Panel(f"[bold green]Running Classification: {name}[/bold green]"))                                                                 
    results_clf[name] = exp.run()                                                                                                                    
                                                                                                                                                    
# ── Summary Classification ──────────────────────────────────────────────────────────                                                               
clf_table = Table(title="Classification Performance Summary", show_header=True, header_style="bold cyan")                                            
clf_table.add_column("Model", style="dim", width=20)                                                                                                 
clf_table.add_column("Macro-F1", justify="right")                                                                                                    
                                                                                                                                                    
for name, m in results_clf.items():                                                                                                                  
    clf_table.add_row(name, f"{m['f1']:>15.4f}")                                                                                                     
                                                                                                                                                    
console.print("\n", clf_table)

# ── Hyperparameter Optimization ────────────────────────────────────────────────

from ml.optimizer import LGBMOptimizer, XGBoostOptimizer, CatBoostOptimizer

# optimizer = LGBMOptimizer(
#     loader=loader_enhanced,
#     numeric_cols=NUMERIC_COLS_ENHANCED,
#     n_trials=100,
# )
# study = optimizer.optimize()

# optimizer = XGBoostOptimizer(
#     loader=loader_enhanced,
#     numeric_cols=NUMERIC_COLS,
#     n_trials=100,
# )
# study = optimizer.optimize()

# optimizer = CatBoostOptimizer(
#     loader=loader_enhanced,
#     numeric_cols=NUMERIC_COLS,
#     n_trials=100,
# )
# study = optimizer.optimize()