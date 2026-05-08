from typing import List, Optional
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ml.data import DataLoader
from ml.models.lgbm import LightGBMModel
from ml.models.lgbm_classifier import LightGBMClassifierModel
from ml.models.xgboost_model import XGBoostModel
from ml.models.xgboost_classifier import XGBoostClassifierModel
from ml.models.catboost_model import CatBoostModel
from ml.models.catboost_classifier import CatBoostClassifierModel
from ml.models.random_forest_classifier import RandomForestClassifierModel
from ml.pipeline import MLPipeline
from ml.preprocessors.scaler import FeatureScaler
from ml.preprocessors.temporal import TemporalFeatureExtractor
from ml.preprocessors.weather_engineer import WeatherFeatureEngineer
from ml.preprocessors.target_encoder import HistoricalMeanEncoder
from ml.preprocessors.delay_binner import DelayBinner

optuna.logging.set_verbosity(optuna.logging.WARNING)


class LGBMRegressorOptimizer:
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
            n_estimators=trial.suggest_int("n_estimators", 100, self.n_estimators),
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
        try:
            study.optimize(
                self._objective,
                n_trials=self.n_trials,
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Printing best results found so far...")

        print(f"\nBest RMSE: {study.best_value:.4f}s")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return study


class LGBMClassifierOptimizer:
    """Bayesian hyperparameter search for LightGBMClassifier, maximising macro-F1.

    Uses a manual inner train/val split (stratified) so the binner can be
    applied to both splits without touching the held-out test set.
    """

    def __init__(
        self,
        loader: DataLoader,
        binner: DelayBinner | None = None,
        n_trials: int = 100,
        n_estimators: int = 2000,
        early_stopping_rounds: int = 50,
        val_fraction: float = 0.15,
        seed: Optional[int] = 42,
    ):
        self.loader = loader
        self.binner = binner or DelayBinner()
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

    def _build_preprocessors(self):
        return [
            TemporalFeatureExtractor(),
            # WeatherFeatureEngineer(),
            # HistoricalMeanEncoder(group_cols=["hour", "dow"], output_col="hist_mean_delay"),
        ]

    def _objective(self, trial: optuna.Trial) -> float:
        X_train_full, _, y_train_full, _ = self._load_once()

        # Inner split (stratified on encoded labels to preserve class ratios)
        y_enc_full = self.binner.encode(y_train_full)
        X_tr, X_val, y_tr_raw, y_val_raw = train_test_split(
            X_train_full, y_train_full,
            test_size=self.val_fraction, random_state=trial.number, stratify=y_enc_full,
        )
        y_tr = self.binner.encode(y_tr_raw)
        y_val = self.binner.encode(y_val_raw)

        # Build and fit pipeline on inner train split
        preprocessors = self._build_preprocessors()
        X_tr_proc = X_tr.copy()
        for p in preprocessors:
            X_tr_proc = p.fit_transform(X_tr_proc, y_tr_raw)
        X_val_proc = X_val.copy()
        for p in preprocessors:
            X_val_proc = p.transform(X_val_proc)

        model = LightGBMClassifierModel(
            n_estimators=trial.suggest_int("n_estimators", 100, self.n_estimators),
            early_stopping_rounds=self.early_stopping_rounds,
            val_fraction=self.val_fraction,
            class_weight="balanced",
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 200),
            min_child_samples=trial.suggest_int("min_child_samples", 1, 100),
            min_sum_hessian_in_leaf=trial.suggest_float("min_sum_hessian_in_leaf", 1e-5, 1.0, log=True),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            feature_fraction_bynode=trial.suggest_float("feature_fraction_bynode", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        )
        model.fit(X_tr_proc, y_tr)
        preds = model.predict(X_val_proc)
        return f1_score(y_val, preds, average="macro")

    def optimize(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        print(f"\nBest macro-F1: {study.best_value:.4f}")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return study


class XGBoostRegressorOptimizer:
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
            n_estimators=trial.suggest_int("n_estimators", 100, self.n_estimators),
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
        try:
            study.optimize(
                self._objective,
                n_trials=self.n_trials,
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Printing best results found so far...")

        print(f"\nBest RMSE: {study.best_value:.4f}s")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return study


class OrdinalLGBMClassifierOptimizer:
    """Bayesian hyperparameter search for OrdinalClassifierModel(LightGBM), maximising macro-F1.

    Each trial fits K-1 binary classifiers, so early stopping is disabled on
    the inner models to avoid triple-nested splits. n_estimators is kept
    moderate (default 800) so trials remain fast enough for Optuna.
    """

    def __init__(
        self,
        loader: DataLoader,
        binner: DelayBinner | None = None,
        n_trials: int = 75,
        n_estimators: int = 800,
        val_fraction: float = 0.15,
        seed: Optional[int] = 42,
    ):
        self.loader = loader
        self.binner = binner or DelayBinner()
        self.n_trials = n_trials
        self.n_estimators = n_estimators
        self.val_fraction = val_fraction
        self.seed = seed
        self._data = None

    def _load_once(self):
        if self._data is None:
            self._data = self.loader.load()
        return self._data

    def _build_preprocessors(self):
        return [TemporalFeatureExtractor()]

    def _objective(self, trial: optuna.Trial) -> float:
        from ml.models.ordinal_classifier import OrdinalClassifierModel

        X_train_full, _, y_train_full, _ = self._load_once()

        y_enc_full = self.binner.encode(y_train_full)
        X_tr, X_val, y_tr_raw, y_val_raw = train_test_split(
            X_train_full, y_train_full,
            test_size=self.val_fraction, random_state=trial.number, stratify=y_enc_full,
        )
        y_tr = self.binner.encode(y_tr_raw)
        y_val = self.binner.encode(y_val_raw)

        preprocessors = self._build_preprocessors()
        X_tr_proc = X_tr.copy()
        for p in preprocessors:
            X_tr_proc = p.fit_transform(X_tr_proc, y_tr_raw)
        X_val_proc = X_val.copy()
        for p in preprocessors:
            X_val_proc = p.transform(X_val_proc)

        base = LightGBMClassifierModel(
            n_estimators=trial.suggest_int("n_estimators", 100, self.n_estimators),
            early_stopping_rounds=0,           # no inner split per binary clf
            class_weight="balanced",
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 200),
            min_child_samples=trial.suggest_int("min_child_samples", 1, 100),
            min_sum_hessian_in_leaf=trial.suggest_float("min_sum_hessian_in_leaf", 1e-5, 1.0, log=True),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            feature_fraction_bynode=trial.suggest_float("feature_fraction_bynode", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        )
        model = OrdinalClassifierModel(base_model=base)
        model.fit(X_tr_proc, y_tr)
        preds = model.predict(X_val_proc)
        return f1_score(y_val, preds, average="macro")

    def optimize(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        try:
            study.optimize(
                self._objective,
                n_trials=self.n_trials,
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Printing best results found so far...")

        print(f"\nBest macro-F1: {study.best_value:.4f}")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return study


class CatBoostRegressorOptimizer:
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

        model = CatBoostModel(
            n_estimators=trial.suggest_int("n_estimators", 100, self.n_estimators),
            early_stopping_rounds=self.early_stopping_rounds,
            val_fraction=self.val_fraction,
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            depth=trial.suggest_int("depth", 4, 12),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            random_strength=trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 100),
        )
        pipeline = MLPipeline(
            preprocessors=[FeatureScaler(cols=self.numeric_cols), TemporalFeatureExtractor()],
            model=model,
        )
        pipeline.fit(X_train, y_train)
        return model._model.get_best_score()["validation"]["RMSE"]

    def optimize(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        try:
            study.optimize(
                self._objective,
                n_trials=self.n_trials,
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Printing best results found so far...")

        print(f"\nBest RMSE: {study.best_value:.4f}s")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return study


class XGBoostClassifierOptimizer:
    """Bayesian hyperparameter search for XGBoostClassifierModel, maximising macro-F1."""

    def __init__(
        self,
        loader: DataLoader,
        binner: DelayBinner | None = None,
        n_trials: int = 100,
        n_estimators: int = 2000,
        early_stopping_rounds: int = 50,
        val_fraction: float = 0.15,
        seed: Optional[int] = 42,
    ):
        self.loader = loader
        self.binner = binner or DelayBinner()
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

    def _build_preprocessors(self):
        return [TemporalFeatureExtractor()]

    def _objective(self, trial: optuna.Trial) -> float:
        X_train_full, _, y_train_full, _ = self._load_once()

        y_enc_full = self.binner.encode(y_train_full)
        X_tr, X_val, y_tr_raw, y_val_raw = train_test_split(
            X_train_full, y_train_full,
            test_size=self.val_fraction, random_state=trial.number, stratify=y_enc_full,
        )
        y_tr = self.binner.encode(y_tr_raw)
        y_val = self.binner.encode(y_val_raw)

        preprocessors = self._build_preprocessors()
        X_tr_proc = X_tr.copy()
        for p in preprocessors:
            X_tr_proc = p.fit_transform(X_tr_proc, y_tr_raw)
        X_val_proc = X_val.copy()
        for p in preprocessors:
            X_val_proc = p.transform(X_val_proc)

        model = XGBoostClassifierModel(
            n_estimators=trial.suggest_int("n_estimators", 100, self.n_estimators),
            early_stopping_rounds=self.early_stopping_rounds,
            val_fraction=self.val_fraction,
            class_weight="balanced",
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
        model.fit(X_tr_proc, y_tr)
        preds = model.predict(X_val_proc)
        return f1_score(y_val, preds, average="macro")

    def optimize(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        print(f"\nBest macro-F1: {study.best_value:.4f}")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return study


class CatBoostClassifierOptimizer:
    """Bayesian hyperparameter search for CatBoostClassifierModel, maximising macro-F1."""

    def __init__(
        self,
        loader: DataLoader,
        binner: DelayBinner | None = None,
        n_trials: int = 100,
        n_estimators: int = 2000,
        early_stopping_rounds: int = 50,
        val_fraction: float = 0.15,
        seed: Optional[int] = 42,
    ):
        self.loader = loader
        self.binner = binner or DelayBinner()
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

    def _build_preprocessors(self):
        return [TemporalFeatureExtractor()]

    def _objective(self, trial: optuna.Trial) -> float:
        X_train_full, _, y_train_full, _ = self._load_once()

        y_enc_full = self.binner.encode(y_train_full)
        X_tr, X_val, y_tr_raw, y_val_raw = train_test_split(
            X_train_full, y_train_full,
            test_size=self.val_fraction, random_state=trial.number, stratify=y_enc_full,
        )
        y_tr = self.binner.encode(y_tr_raw)
        y_val = self.binner.encode(y_val_raw)

        preprocessors = self._build_preprocessors()
        X_tr_proc = X_tr.copy()
        for p in preprocessors:
            X_tr_proc = p.fit_transform(X_tr_proc, y_tr_raw)
        X_val_proc = X_val.copy()
        for p in preprocessors:
            X_val_proc = p.transform(X_val_proc)

        model = CatBoostClassifierModel(
            n_estimators=trial.suggest_int("n_estimators", 100, self.n_estimators),
            early_stopping_rounds=self.early_stopping_rounds,
            val_fraction=self.val_fraction,
            auto_class_weights="Balanced",
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            depth=trial.suggest_int("depth", 4, 12),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            random_strength=trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 100),
        )
        model.fit(X_tr_proc, y_tr)
        preds = model.predict(X_val_proc)
        return f1_score(y_val, preds, average="macro")

    def optimize(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        print(f"\nBest macro-F1: {study.best_value:.4f}")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return study


class RandomForestClassifierOptimizer:
    """Bayesian hyperparameter search for RandomForestClassifierModel, maximising macro-F1.

    Random Forest does not benefit from early stopping, so n_estimators is tuned
    directly without a validation split overhead per iteration. The inner split is
    used only for final F1 evaluation.
    """

    def __init__(
        self,
        loader: DataLoader,
        binner: DelayBinner | None = None,
        n_trials: int = 100,
        n_estimators: int = 1000,
        val_fraction: float = 0.15,
        seed: Optional[int] = 42,
    ):
        self.loader = loader
        self.binner = binner or DelayBinner()
        self.n_trials = n_trials
        self.n_estimators = n_estimators
        self.val_fraction = val_fraction
        self.seed = seed
        self._data = None

    def _load_once(self):
        if self._data is None:
            self._data = self.loader.load()
        return self._data

    def _build_preprocessors(self):
        return [TemporalFeatureExtractor()]

    def _objective(self, trial: optuna.Trial) -> float:
        X_train_full, _, y_train_full, _ = self._load_once()

        y_enc_full = self.binner.encode(y_train_full)
        X_tr, X_val, y_tr_raw, y_val_raw = train_test_split(
            X_train_full, y_train_full,
            test_size=self.val_fraction, random_state=trial.number, stratify=y_enc_full,
        )
        y_tr = self.binner.encode(y_tr_raw)
        y_val = self.binner.encode(y_val_raw)

        preprocessors = self._build_preprocessors()
        X_tr_proc = X_tr.copy()
        for p in preprocessors:
            X_tr_proc = p.fit_transform(X_tr_proc, y_tr_raw)
        X_val_proc = X_val.copy()
        for p in preprocessors:
            X_val_proc = p.transform(X_val_proc)

        model = RandomForestClassifierModel(
            n_estimators=trial.suggest_int("n_estimators", 100, self.n_estimators),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight="balanced",
        )
        model.fit(X_tr_proc, y_tr)
        preds = model.predict(X_val_proc)
        return f1_score(y_val, preds, average="macro")

    def optimize(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        print(f"\nBest macro-F1: {study.best_value:.4f}")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return study
