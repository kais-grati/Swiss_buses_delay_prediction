import argparse
import sys
from pathlib import Path

import numpy as np

from ml.pipeline import MLPipeline
from ml.evaluation import Evaluator
from ml.models.base import ClassifierModel
from ml.preprocessors.delay_binner import DelayBinner
from ml.data import DataLoader
from rich.console import Console
from rich.panel import Panel

console = Console()

DROP_COLS = [
    "timestamp", "stop_id", "stop_name", "operator", "line",
    "departure_delay_s",
]

DROP_COLS_KEEP_TS = [
    "stop_id", "stop_name", "operator", "line",
    "departure_delay_s",
]

BINNER_BOUNDARIES = {
    "v1": [90],
}


def main():
    parser = argparse.ArgumentParser(
        description="Load a saved pipeline and run inference on a dataset."
    )
    parser.add_argument(
        "-m", "--model", required=True,
        help="Path to the saved model directory (e.g. saved_models/LGBM-v1).",
    )
    parser.add_argument(
        "-d", "--dataset", required=True,
        help="Path to the parquet dataset.",
    )
    parser.add_argument(
        "-t", "--target", default="arrival_delay_s",
        help="Target column name (default: arrival_delay_s).",
    )
    parser.add_argument(
        "--drop-cols", nargs="*",
        help="Columns to drop.  Defaults: timeseries-aware set if --ts is set, else standard set.",
    )
    parser.add_argument(
        "--ts", action="store_true",
        help="Keep the timestamp column (for models trained with TemporalFeatureExtractor).",
    )
    parser.add_argument(
        "--binner", default="v1",
        choices=["v1"],
        help="Delay binner boundaries to use for classification (default: v1 = [90]).",
    )
    parser.add_argument(
        "--no-class-names", action="store_true",
        help="If set, don't show class names in confusion matrix (use numeric labels).",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluation (use when dataset has no target column).",
    )
    parser.add_argument(
        "-o", "--output",
        help="Save predictions to file (.parquet or .csv).",
    )
    args = parser.parse_args()
    

    # ── Load pipeline ──────────────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.is_dir():
        console.print(f"[red]Error:[/red] model directory not found: {model_path}")
        sys.exit(1)

    console.print(Panel(f"[bold]Loading pipeline from [cyan]{model_path}[/cyan][/bold]"))
    pipeline = MLPipeline.load(str(model_path))
    console.print(f"  Model: [green]{type(pipeline.model).__name__}[/green]")
    console.print(f"  Preprocessors: {len(pipeline.preprocessors)}")

    is_classification = isinstance(pipeline.model, ClassifierModel)

    # ── Load data ──────────────────────────────────────────────────────────────
    drop_cols = args.drop_cols if args.drop_cols is not None else (
        DROP_COLS_KEEP_TS if args.ts else DROP_COLS
    )

    loader = DataLoader(
        path=args.dataset,
        target=args.target,
        drop_cols=drop_cols,
        test_size=0,
    )

    # ── Chunked prediction ─────────────────────────────────────────────────────
    console.print(Panel("[bold]Running inference (chunked)...[/bold]"))
    y_pred_parts = []
    y_true_parts = [] if not args.no_eval else None
    total = 0

    for X_chunk, y_chunk in loader.stream():
        y_pred_parts.append(pipeline.predict(X_chunk))
        if y_true_parts is not None:
            y_true_parts.append(y_chunk.to_numpy())
        total += len(X_chunk)
        console.print(f"  [dim]Processed {total:,} rows...[/dim]")

    y_pred = np.concatenate(y_pred_parts)
    console.print(f"  Features: {X_chunk.shape[1]}")
    console.print(f"[green]Predictions complete: {len(y_pred):,} rows[/green]")

    # ── Save predictions ───────────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        import pandas as pd
        result = pd.DataFrame({"prediction": y_pred})
        if out.suffix == ".csv":
            result.to_csv(out, index=False)
        else:
            result.to_parquet(out, index=False)
        console.print(f"[green]Saved to {out}[/green]")

    if args.no_eval:
        return

    # ── Evaluate ───────────────────────────────────────────────────────────────
    if y_true_parts:
        y = np.concatenate(y_true_parts)
    else:
        y = None

    if y is None:
        console.print("[yellow]No target column found — skipping evaluation.[/yellow]")
        return

    if is_classification:
        binner = DelayBinner(bins=BINNER_BOUNDARIES[args.binner])
        y = binner.encode(y)

    evaluator = Evaluator()

    if is_classification:
        class_names = None if args.no_class_names else binner.class_names
        return evaluator.evaluate(
            y, y_pred,
            model_name=f"{type(pipeline.model).__name__} ({model_path.name})",
            is_classification=True,
            class_names=class_names,
        )
    else:
        return evaluator.evaluate(
            y, y_pred,
            model_name=f"{type(pipeline.model).__name__} ({model_path.name})",
        )


if __name__ == "__main__":
    main()
