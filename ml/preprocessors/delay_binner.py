import pandas as pd
import numpy as np
from ml.preprocessors.class_encoder import ClassEncoder

class DelayBinner(ClassEncoder):
    """
    Converts continuous delay seconds into discrete classes.

    Boundaries chosen from data exploration on dataset_705.parquet (n=502,890):
      distribution peaks at 60-90s; <3% of rows exceed 300s, so coarse upper bins waste capacity.

    Notes:
        Balanced distribution bins: 10 60 100 160
    """
    def __init__(self, bins=[90]):
        self.bins = bins

    def encode(self, y: pd.Series) -> pd.Series:
        n_classes = len(self.bins) + 1
        return pd.cut(
            y,
            bins=[-float('inf')] + list(self.bins) + [float('inf')],
            labels=list(range(n_classes)),
        ).astype(int)

    def decode(self, y_encoded: np.ndarray) -> np.ndarray:
        midpoints = []
        edges = [-float('inf')] + list(self.bins) + [float('inf')]
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if lo == -float('inf'):
                midpoints.append(hi - 30)
            elif hi == float('inf'):
                midpoints.append(lo + 50)
            else:
                midpoints.append((lo + hi) / 2)
        return np.array([midpoints[i] for i in y_encoded])

    @property
    def class_names(self) -> list[str]:
        names = [f"≤{self.bins[0]}s"]
        for i in range(1, len(self.bins)):
            names.append(f"{self.bins[i-1]}–{self.bins[i]}s")
        names.append(f">{self.bins[-1]}s")
        return names
