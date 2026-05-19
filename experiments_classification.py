from ml.pipeline import MLPipeline
from ml.preprocessors.string_encoder import StringEncoder
from ml.preprocessors.temporal import TemporalFeatureExtractor
from ml.preprocessors.wind_merger import WindMerger
from ml.models.lgbm_classifier import LightGBMClassifierModel
from ml.models.xgboost_classifier import XGBoostClassifierModel
from ml.models.catboost_classifier import CatBoostClassifierModel
from ml.experiment import ClassificationExperiment
from ml.preprocessors.delay_binner import DelayBinner

from rich.table import Table
from rich.panel import Panel

from config import (
    console, loader_lag, loader_lausanne_50k, evaluator, logger,
)

# ═══════════════════════════════════════════════════════════════════════════════════
# 4-class bins: on-time (≤60s), slight (60-120s), moderate (120-300s), severe (>300s)
# ═══════════════════════════════════════════════════════════════════════════════════

BINS_4CLS = [60, 120, 300]
PREPROC = [
    TemporalFeatureExtractor(),
    WindMerger(),
    StringEncoder(cols=["operator", "line"]),
]

class_experiments = {

    # ── 705 Dataset — Optimized models ──────────────────────────────────────────

    "CB-705-4cls": ClassificationExperiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=PREPROC,
            model=CatBoostClassifierModel(
                n_estimators=1263, learning_rate=0.01960, depth=11,
                l2_leaf_reg=0.03972, random_strength=0.3489,
                bagging_temperature=0.5924, min_data_in_leaf=5,
                auto_class_weights=None, early_stopping_rounds=50,
            ),
        ),
        evaluator=evaluator,
        encoder=DelayBinner(bins=BINS_4CLS),
    ),

    "LGBM-705-4cls": ClassificationExperiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=PREPROC,
            model=LightGBMClassifierModel(
                n_estimators=655, learning_rate=0.01960, num_leaves=161,
                min_child_samples=20, min_sum_hessian_in_leaf=0.00373,
                subsample=0.7554, colsample_bytree=0.4279,
                feature_fraction_bynode=0.7645,
                reg_alpha=0.000712, reg_lambda=0.000211,
                scale_pos_weight=0.3863, early_stopping_rounds=50,
            ),
        ),
        evaluator=evaluator,
        encoder=DelayBinner(bins=BINS_4CLS),
    ),

    "XGB-705-4cls": ClassificationExperiment(
        loader=loader_lag,
        pipeline=MLPipeline(
            preprocessors=PREPROC,
            model=XGBoostClassifierModel(
                n_estimators=655, learning_rate=0.01960, max_depth=10,
                min_child_weight=2.509, gamma=0.02608,
                subsample=0.7554, colsample_bytree=0.4279,
                colsample_bylevel=0.7645,
                reg_alpha=0.000712, reg_lambda=0.000211,
                scale_pos_weight=0.3863, early_stopping_rounds=50,
            ),
        ),
        evaluator=evaluator,
        encoder=DelayBinner(bins=BINS_4CLS),
    ),

    # ── Lausanne 50k Dataset — 705-optimized params transferred ─────────────────

    "CB-L50k-4cls": ClassificationExperiment(
        loader=loader_lausanne_50k,
        pipeline=MLPipeline(
            preprocessors=PREPROC,
            model=CatBoostClassifierModel(
                n_estimators=1263, learning_rate=0.01960, depth=11,
                l2_leaf_reg=0.03972, random_strength=0.3489,
                bagging_temperature=0.5924, min_data_in_leaf=5,
                auto_class_weights=None, early_stopping_rounds=50,
            ),
        ),
        evaluator=evaluator,
        encoder=DelayBinner(bins=BINS_4CLS),
    ),

    "LGBM-L50k-4cls": ClassificationExperiment(
        loader=loader_lausanne_50k,
        pipeline=MLPipeline(
            preprocessors=PREPROC,
            model=LightGBMClassifierModel(
                n_estimators=655, learning_rate=0.01960, num_leaves=161,
                min_child_samples=20, min_sum_hessian_in_leaf=0.00373,
                subsample=0.7554, colsample_bytree=0.4279,
                feature_fraction_bynode=0.7645,
                reg_alpha=0.000712, reg_lambda=0.000211,
                scale_pos_weight=0.3863, early_stopping_rounds=50,
            ),
        ),
        evaluator=evaluator,
        encoder=DelayBinner(bins=BINS_4CLS),
    ),

}


def run_classification(save: bool = False):
    console.print("\n", Panel("[bold green]CLASSIFICATION EXPERIMENTS[/bold green]", expand=False))

    results = {}
    for name, exp in class_experiments.items():
        console.print(Panel(f"[bold green]Running Classification: {name}[/bold green]"))
        results[name] = exp.run()
        logger.log(name, results[name], kind="classification")

    clf_table = Table(title="Classification Performance Summary", show_header=True, header_style="bold cyan")
    clf_table.add_column("Model", style="dim", width=35)
    clf_table.add_column("Macro-F1", justify="right")
    clf_table.add_column("Accuracy", justify="right")

    for name, m in results.items():
        acc = m.get("report", {}).get("accuracy", 0)
        clf_table.add_row(name, f"{m['f1']:>15.4f}", f"{acc:>15.4f}")

    console.print("\n", clf_table)
    return results
