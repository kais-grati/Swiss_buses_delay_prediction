import json
import os
from datetime import datetime
from typing import Any, Dict


class ExperimentLogger:
    def __init__(self, path: str = "results/experiment_log.jsonl"):
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, name: str, results: Dict[str, Any], kind: str) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "name": name,
            "kind": kind,
        }

        if kind == "regression":
            entry["mse"] = round(results["mse"], 4)
            entry["rmse"] = round(results["rmse"], 4)
            entry["r2"] = round(results["r2"], 4)

        elif kind == "classification":
            entry["macro_f1"] = round(results["f1"], 4)
            report = results.get("report", {})
            for label, metrics in report.items():
                if label in ("accuracy", "macro avg", "weighted avg"):
                    continue
                if isinstance(metrics, dict):
                    entry[f"f1_class_{label}"] = round(metrics["f1-score"], 4)

        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")
