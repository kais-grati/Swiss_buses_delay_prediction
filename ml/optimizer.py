from typing import List, Optional
import optuna
from ml.data import DataLoader
from ml.models.lgbm import LightGBMModel
from ml.models.xgboost_model import XGBoostModel
from ml.pipeline import MLPipeline
from ml.preprocessors.scaler import FeatureScaler
from ml.preprocessors.temporal import TemporalFeatureExtractor
from ml.preprocessors.weather_engineer import WeatherFeatureEngineer

optuna.logging.set_verbosity(optuna.logging.WARNING)


class LGBMOptimizer:
    def __init__(
        self,
        loader: DataLoader,
        numeric_cols: List[str],
        n_trials: int = 100,
        n_estimators: int = 3000,
        early_stopping_rounds: int = 50,
        val_fraction: float = 0.1,
        seed: Optional[int] = 42,
    ):
        self.loader = loader
        self.numeric_cols = numeric_cols
        self.n_trials = n_trials
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.seed = seed
        self._data = None

    def _load_once(self):
        if self._data is None:
            self._data = self.loader.load()
        return self._data

    def _objective(self, trial: optuna.Trial) -> float:
        X_train, _, y_train, _ = self._load_once()

        model = LightGBMModel(
            n_estimators=self.n_estimators,
            early_stopping_rounds=self.early_stopping_rounds,
            val_fraction=self.val_fraction,
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 20, 300),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
            min_sum_hessian_in_leaf=trial.suggest_float("min_sum_hessian_in_leaf", 1e-4, 1.0, log=True),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            feature_fraction_bynode=trial.suggest_float("feature_fraction_bynode", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            path_smooth=trial.suggest_float("path_smooth", 0.0, 10.0),
        )
        pipeline = MLPipeline(
            preprocessors=[WeatherFeatureEngineer(), FeatureScaler(cols=self.numeric_cols), TemporalFeatureExtractor()],
            model=model,
        )
        pipeline.fit(X_train, y_train)
        return model._model.best_score_["valid_0"]["rmse"]

    def optimize(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        print(f"\nBest RMSE: {study.best_value:.4f}s")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return study


class XGBoostOptimizer:
    def __init__(
        self,
        loader: DataLoader,
        numeric_cols: List[str],
        n_trials: int = 100,
        n_estimators: int = 2000,
        early_stopping_rounds: int = 50,
        val_fraction: float = 0.1,
        seed: Optional[int] = 42,
    ):
        self.loader = loader
        self.numeric_cols = numeric_cols
        self.n_trials = n_trials
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.seed = seed
        self._data = None

    def _load_once(self):
        if self._data is None:
            self._data = self.loader.load()
        return self._data

    def _objective(self, trial: optuna.Trial) -> float:
        X_train, _, y_train, _ = self._load_once()

        model = XGBoostModel(
            n_estimators=self.n_estimators,
            early_stopping_rounds=self.early_stopping_rounds,
            val_fraction=self.val_fraction,
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        )
        pipeline = MLPipeline(
            preprocessors=[FeatureScaler(cols=self.numeric_cols), TemporalFeatureExtractor()],
            model=model,
        )
        pipeline.fit(X_train, y_train)
        return model._model.best_score

    def optimize(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        print(f"\nBest RMSE: {study.best_value:.4f}s")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return study
