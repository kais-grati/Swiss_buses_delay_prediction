import copy
import numpy as np
import pandas as pd
from ml.models.base import ClassifierModel


class OrdinalClassifierModel(ClassifierModel):
    """
    Ordinal classifier via K-1 binary threshold models (Frank & Hall, 2001).

    For K ordered classes, trains K-1 binary classifiers:
        classifier k: "is y >= class[k+1]?"

    Combines them into class probabilities using the telescoping identity:
        P(y=0)   = 1 - P(y>=1)
        P(y=k)   = P(y>=k) - P(y>=k+1)    for 0 < k < K-1
        P(y=K-1) = P(y>=K-1)

    Monotonicity P(y>=k) >= P(y>=k+1) is enforced by clipping, since
    the K-1 independent classifiers have no built-in ordering constraint.
    """

    def __init__(self, base_model: ClassifierModel):
        self._base_model = base_model
        self._classifiers: list[ClassifierModel] = []
        self._classes: list = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OrdinalClassifierModel":
        self._classes = sorted(y.unique())
        self._classifiers = []

        for k in range(len(self._classes) - 1):
            clf = copy.deepcopy(self._base_model)
            # Binary target: 1 if y is above the k-th threshold
            binary_y = (y >= self._classes[k + 1]).astype(int)
            clf.fit(X, pd.Series(binary_y, index=y.index))
            self._classifiers.append(clf)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # cumulative_probs[:, k] = P(y >= class[k+1])
        cumulative = np.stack(
            [clf.predict_proba(X)[:, 1] for clf in self._classifiers],
            axis=1,
        )

        # Enforce monotone decreasing: P(y>=k) >= P(y>=k+1)
        for k in range(1, cumulative.shape[1]):
            cumulative[:, k] = np.minimum(cumulative[:, k], cumulative[:, k - 1])

        n_samples = len(X)
        k = len(self._classes)
        proba = np.zeros((n_samples, k))
        proba[:, 0] = 1.0 - cumulative[:, 0]
        for i in range(1, k - 1):
            proba[:, i] = cumulative[:, i - 1] - cumulative[:, i]
        proba[:, -1] = cumulative[:, -1]

        # Clip floating-point negatives, renormalize
        proba = np.clip(proba, 0.0, None)
        row_sums = proba.sum(axis=1, keepdims=True)
        return proba / row_sums

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(self._classes)[np.argmax(self.predict_proba(X), axis=1)]
