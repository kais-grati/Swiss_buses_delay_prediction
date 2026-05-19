"""Two-phase classifier optimization: sample search → full-data evaluation.

Usage:
    python optimize_classifiers.py --dataset 705 --bins 60
    python optimize_classifiers.py --dataset lausanne_50k --bins 60 --models LGBM,XGBoost
    python optimize_classifiers.py --dataset 705 --bins 60 180  # 3-class

Phase 1: Optuna search on a 50k stratified sample (fast trials).
Phase 2: Evaluate best params on the full dataset and report test F1.
"""

import argparse
import json
import time
import numpy as np
from sklearn.metrics import f1_score

from ml.data import DataLoader
from ml.pipeline import MLPipeline
from ml.preprocessors.temporal import TemporalFeatureExtractor
from ml.preprocessors.wind_merger import WindMerger
from ml.preprocessors.string_encoder import StringEncoder
from ml.preprocessors.delay_binner import DelayBinner
from ml.models.lgbm_classifier import LightGBMClassifierModel
from ml.models.xgboost_classifier import XGBoostClassifierModel
from ml.models.catboost_classifier import CatBoostClassifierModel
from ml.optimizer import (
    LGBMClassifierOptimizer,
    XGBoostClassifierOptimizer,
    CatBoostClassifierOptimizer,
)

DATASETS = {
    "705": "data/705_bus_2025_weather_traffic.parquet",
    "lausanne_50k": "data/lausanne50k_bus_2025_weather_traffic.parquet",
}

DEFAULT_PREPROCESSORS = [
    TemporalFeatureExtractor(),
    WindMerger(),
    StringEncoder(cols=["operator", "line"]),
]

OPTIMIZERS = {
    "LGBM": (LGBMClassifierOptimizer, LightGBMClassifierModel),
    "XGBoost": (XGBoostClassifierOptimizer, XGBoostClassifierModel),
    "CatBoost": (CatBoostClassifierOptimizer, CatBoostClassifierModel),
}

SAMPLE_N = 50000
DROP_COLS = ["stop_name", "departure_delay_s", "trip_id"]  # trip_stop_index kept per RESULTS_6


def phase1_optimize(model_name, loader_sample, binner, n_trials):
    """Run Optuna on the 50k sample."""
    opt_cls, _ = OPTIMIZERS[model_name]
    print(f"\n{'='*60}")
    print(f"PHASE 1: Optimizing {model_name} | bins={binner.bins} | trials={n_trials}")
    print(f"{'='*60}")
    opt = opt_cls(
        loader=loader_sample,
        binner=binner,
        preprocessors=DEFAULT_PREPROCESSORS,
        n_trials=n_trials,
        n_estimators=2000,
    )
    study = opt.optimize()
    print(f"  Best val F1: {study.best_value:.4f}")
    return study


def phase2_evaluate(model_name, best_params, loader_full, binner):
    """Train with best params on full data and evaluate on test set."""
    print(f"\n{'='*60}")
    print(f"PHASE 2: Evaluating {model_name} on full dataset | bins={binner.bins}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = loader_full.load()
    y_train_enc = binner.encode(y_train)
    y_test_enc = binner.encode(y_test)

    pipeline = MLPipeline(
        preprocessors=DEFAULT_PREPROCESSORS,
        model=_build_model(model_name, best_params),
    )

    t0 = time.time()
    pipeline.fit(X_train, y_train_enc)
    preds = pipeline.predict(X_test)
    elapsed = time.time() - t0

    f1 = f1_score(y_test_enc, preds, average="macro")
    print(f"  Test F1: {f1:.4f}  |  Fit time: {elapsed:.0f}s")
    return f1


def _build_model(model_name, params):
    """Instantiate a classifier model from Optuna params."""
    if model_name == "LGBM":
        return LightGBMClassifierModel(
            n_estimators=params.get("n_estimators", 1000),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 127),
            min_child_samples=params.get("min_child_samples", 100),
            min_sum_hessian_in_leaf=params.get("min_sum_hessian_in_leaf", 0.1),
            subsample=params.get("subsample", 0.8),
            subsample_freq=1,
            colsample_bytree=params.get("colsample_bytree", 0.7),
            feature_fraction_bynode=params.get("feature_fraction_bynode", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 0.1),
            scale_pos_weight=params.get("scale_pos_weight", 1.0),
            early_stopping_rounds=50,
        )
    elif model_name == "XGBoost":
        return XGBoostClassifierModel(
            n_estimators=params.get("n_estimators", 1000),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 6),
            min_child_weight=params.get("min_child_weight", 1.0),
            gamma=params.get("gamma", 0.0),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            colsample_bylevel=params.get("colsample_bylevel", 1.0),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            scale_pos_weight=params.get("scale_pos_weight", 1.0),
            early_stopping_rounds=50,
        )
    elif model_name == "CatBoost":
        return CatBoostClassifierModel(
            n_estimators=params.get("n_estimators", 1000),
            learning_rate=params.get("learning_rate", 0.05),
            depth=params.get("depth", 6),
            l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
            random_strength=params.get("random_strength", 1.0),
            bagging_temperature=params.get("bagging_temperature", 1.0),
            min_data_in_leaf=params.get("min_data_in_leaf", 1),
            auto_class_weights=params.get("auto_class_weights", "Balanced"),
            early_stopping_rounds=50,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="705", choices=["705", "lausanne_50k"])
    parser.add_argument("--bins", type=int, nargs="+", default=[60])
    parser.add_argument("--models", default="LGBM,XGBoost,CatBoost")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip Optuna, load params from --params-file")
    parser.add_argument("--params-file", default="",
                        help="JSON file with {model_name: {params}} dict")
    parser.add_argument("--output", default="",
                        help="Save best params to JSON file")
    args = parser.parse_args()

    dataset_path = DATASETS[args.dataset]
    binner = DelayBinner(bins=list(args.bins))

    # Phase 1 loader: 50k stratified sample
    loader_sample = DataLoader(
        path=dataset_path, target="arrival_delay_s",
        drop_cols=DROP_COLS, sample_n=SAMPLE_N,
    )

    # Phase 2 loader: full dataset
    loader_full = DataLoader(
        path=dataset_path, target="arrival_delay_s",
        drop_cols=DROP_COLS,
    )

    print(f"Dataset: {args.dataset} ({dataset_path})")
    print(f"Bins: {binner.bins} → {binner.class_names} ({len(binner.class_names)} classes)")
    print(f"Models: {args.models}")
    print(f"Trials per model: {args.trials}")
    print(f"Sample size (Phase 1): {SAMPLE_N:,}")

    models = [m.strip() for m in args.models.split(",")]

    if args.skip_phase1 and args.params_file:
        with open(args.params_file) as f:
            best_params = json.load(f)
        print(f"Loaded params from {args.params_file}")
    else:
        best_params = {}
        for model_name in models:
            study = phase1_optimize(model_name, loader_sample, binner, args.trials)
            best_params[model_name] = {
                "val_f1": study.best_value,
                "params": study.best_params,
            }

    # Phase 2: Evaluate each model on full dataset
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — {args.dataset} bins={list(args.bins)}")
    print(f"{'='*60}")

    final_results = {}
    for model_name in models:
        params = best_params[model_name]["params"]
        test_f1 = phase2_evaluate(model_name, params, loader_full, binner)
        final_results[model_name] = {
            "val_f1": best_params[model_name].get("val_f1"),
            "test_f1": test_f1,
            "params": params,
        }
        val_f1 = best_params[model_name].get('val_f1', None)
        val_str = f"{val_f1:.4f}" if val_f1 else "N/A"
        print(f"  {model_name}: val_F1={val_str}  test_F1={test_f1:.4f}")

    if args.output:
        out = {
            "dataset": args.dataset,
            "bins": list(args.bins),
            "results": final_results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved results to {args.output}")

    # Print best params for copy-paste into experiments file
    print(f"\n{'='*60}")
    print("BEST PARAMETERS (for experiments file)")
    print(f"{'='*60}")
    for model_name in models:
        print(f"\n# {model_name} — test F1={final_results[model_name]['test_f1']:.4f}")
        for k, v in final_results[model_name]["params"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
