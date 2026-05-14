import copy
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from ml.models.base import BaseModel, ClassifierModel, _MODEL_REGISTRY, _lookup
from ml.preprocessors.delay_binner import DelayBinner


class HierarchicalRegressor(BaseModel):
    """Two-stage hierarchical regressor: classify the delay bin, then regress within it.

    Stage 1 — a classifier predicts which delay bin the sample falls into.
    Stage 2 — per-bin regression models, each trained only on samples from that bin.

    At prediction time, the classifier routes each sample to the appropriate
    per-bin regressor.
    """

    def __init__(
        self,
        classifier: ClassifierModel,
        regressor: BaseModel,
        binner: DelayBinner | None = None,
    ):
        self._classifier = classifier
        self._regressor_template = regressor
        self._binner = binner or DelayBinner()
        self._bins: list = []
        self._bin_regressors: list[BaseModel] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HierarchicalRegressor":
        y_binned = self._binner.encode(y)
        self._bins = sorted(y_binned.unique())

        # Stage 1: train bin classifier
        self._classifier.fit(X, y_binned)

        # Stage 2: train per-bin regressors
        X_arr = X.reset_index(drop=True)
        y_arr = y.reset_index(drop=True)
        y_binned_arr = y_binned.reset_index(drop=True)

        self._bin_regressors = []
        for b in self._bins:
            mask = (y_binned_arr == b).values
            reg = copy.deepcopy(self._regressor_template)
            if mask.sum() >= 10:
                reg.fit(X_arr[mask], y_arr[mask])
            else:
                # Fall back to global model for tiny bins
                reg.fit(X_arr, y_arr)
            self._bin_regressors.append(reg)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        bin_preds = self._classifier.predict(X)
        preds = np.zeros(len(X))
        for i, b in enumerate(self._bins):
            mask = bin_preds == b
            if mask.sum() == 0:
                continue
            preds[mask] = self._bin_regressors[i].predict(X[mask])
        return preds

    def save(self, path):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "type": "HierarchicalRegressor",
            "classifier_type": type(self._classifier).__name__,
            "regressor_type": type(self._regressor_template).__name__,
            "bins": self._binner.bins,
            "n_bins": len(self._bins),
        }
        if type(self._classifier).__name__ in _MODEL_REGISTRY:
            self._classifier.save(root / "classifier")
        else:
            joblib.dump(self._classifier, root / "classifier.joblib")
        for i, reg in enumerate(self._bin_regressors):
            sub = root / f"regressor_{i}"
            if type(reg).__name__ in _MODEL_REGISTRY:
                reg.save(sub)
            else:
                sub.mkdir(exist_ok=True)
                joblib.dump(reg, sub / "model.joblib")
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path):
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        clf_name = manifest["classifier_type"]
        reg_name = manifest["regressor_type"]
        classifier = (
            _lookup(clf_name).load(root / "classifier")
            if clf_name in _MODEL_REGISTRY
            else joblib.load(root / "classifier.joblib")
        )
        # Load the first bin regressor to use as template, then load the rest
        first = (
            _lookup(reg_name).load(root / "regressor_0")
            if reg_name in _MODEL_REGISTRY
            else joblib.load(root / "regressor_0" / "model.joblib")
        )
        instance = cls(
            classifier=classifier,
            regressor=first,
            binner=DelayBinner(bins=manifest["bins"]),
        )
        instance._bins = list(range(manifest["n_bins"]))
        instance._bin_regressors = [first]
        for i in range(1, manifest["n_bins"]):
            sub = root / f"regressor_{i}"
            instance._bin_regressors.append(
                _lookup(reg_name).load(sub)
                if reg_name in _MODEL_REGISTRY
                else joblib.load(sub / "model.joblib")
            )
        return instance


from ml.models.base import _register
_register("HierarchicalRegressor", HierarchicalRegressor)
