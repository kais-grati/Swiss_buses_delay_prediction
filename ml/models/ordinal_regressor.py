import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ml.models.base import BaseModel, ClassifierModel, _MODEL_REGISTRY, _lookup
from ml.preprocessors.delay_binner import DelayBinner


class OrdinalRegressorModel(BaseModel):
    """Ordinal regression via K-1 binary threshold classifiers (Frank & Hall, 2001).

    Discretises the continuous target into K ordered classes via DelayBinner,
    trains K-1 binary classifiers ("is y >= threshold_k?"), and produces
    continuous predictions as the expected value over the class probability
    distribution:  ŷ = Σ P(class_k) · midpoint_k
    """

    def __init__(
        self,
        base_model: ClassifierModel,
        binner: DelayBinner | None = None,
    ):
        self._base_model = base_model
        self._binner = binner or DelayBinner()
        self._classifiers: list[ClassifierModel] = []
        self._midpoints: np.ndarray | None = None

    @property
    def K(self) -> int:
        return len(self._classifiers) + 1

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OrdinalRegressorModel":
        y_enc = self._binner.encode(y)
        classes = sorted(y_enc.unique())
        self._classifiers = []

        for k in range(len(classes) - 1):
            clf = copy.deepcopy(self._base_model)
            binary_y = (y_enc >= classes[k + 1]).astype(int)
            clf.fit(X, pd.Series(binary_y, index=y.index))
            self._classifiers.append(clf)

        # Precompute midpoints for soft decoding
        edges = [-float("inf")] + list(self._binner.bins) + [float("inf")]
        midpoints = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if lo == -float("inf"):
                midpoints.append(hi - 30)
            elif hi == float("inf"):
                midpoints.append(lo + 50)
            else:
                midpoints.append((lo + hi) / 2)
        self._midpoints = np.array(midpoints)

        return self

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns (n_samples, K) class probability matrix."""
        # cumulative[:, k] = P(y_enc >= class[k+1])
        cumulative = np.stack(
            [clf.predict_proba(X)[:, 1] for clf in self._classifiers],
            axis=1,
        )
        # Enforce monotone decreasing
        for k in range(1, cumulative.shape[1]):
            cumulative[:, k] = np.minimum(cumulative[:, k], cumulative[:, k - 1])

        n_samples = len(X)
        K = self.K
        proba = np.zeros((n_samples, K))
        proba[:, 0] = 1.0 - cumulative[:, 0]
        for i in range(1, K - 1):
            proba[:, i] = cumulative[:, i - 1] - cumulative[:, i]
        proba[:, -1] = cumulative[:, -1]

        proba = np.clip(proba, 0.0, None)
        row_sums = proba.sum(axis=1, keepdims=True)
        return proba / row_sums

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self._predict_proba(X)
        return proba @ self._midpoints

    def predict_class(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard class predictions (argmax)."""
        proba = self._predict_proba(X)
        return np.argmax(proba, axis=1)

    def save(self, path):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "type": "OrdinalRegressorModel",
            "base_model_type": type(self._base_model).__name__,
            "n_classifiers": len(self._classifiers),
            "bins": self._binner.bins,
        }
        for i, clf in enumerate(self._classifiers):
            clf.save(root / f"classifier_{i}")
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path):
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        base_template = _lookup(manifest["base_model_type"])()
        instance = cls(
            base_model=base_template,
            binner=DelayBinner(bins=manifest["bins"]),
        )
        instance._classifiers = [
            _lookup(manifest["base_model_type"]).load(root / f"classifier_{i}")
            for i in range(manifest["n_classifiers"])
        ]
        # Recompute midpoints
        edges = [-float("inf")] + list(manifest["bins"]) + [float("inf")]
        midpoints = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if lo == -float("inf"):
                midpoints.append(hi - 30)
            elif hi == float("inf"):
                midpoints.append(lo + 50)
            else:
                midpoints.append((lo + hi) / 2)
        instance._midpoints = np.array(midpoints)
        return instance


from ml.models.base import _register
_register("OrdinalRegressorModel", OrdinalRegressorModel)
