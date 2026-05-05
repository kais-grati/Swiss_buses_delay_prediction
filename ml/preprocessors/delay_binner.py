import pandas as pd
import numpy as np
from ml.preprocessors.class_encoder import ClassEncoder

class DelayBinner(ClassEncoder):
    """
    Converts continuous delay seconds into discrete classes.

    Classes:
    0: On-Time       (delay <= 60s,  includes early arrivals)
    1: Slight Delay  (60s  < delay <= 180s, within SBB 3-min tolerance)
    2: Moderate Delay(180s < delay <= 600s, likely missed connections)
    3: Severe Delay  (delay > 600s)
    """
    def __init__(self, bins=[60, 180, 600]):
        self.bins = bins

    def encode(self, y: pd.Series) -> pd.Series:
        return pd.cut(
            y,
            bins=[-float('inf')] + self.bins + [float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)

    def decode(self, y_encoded: np.ndarray) -> np.ndarray:
        # Returns the midpoint of each bin as a representative value
        midpoints = [0, 120, 390, 900]
        return np.array([midpoints[i] for i in y_encoded])

    @property
    def class_names(self) -> list[str]:
        names = [f"≤{self.bins[0]}s"]
        for i in range(1, len(self.bins)):
            names.append(f"{self.bins[i-1]}–{self.bins[i]}s")
        names.append(f">{self.bins[-1]}s")
        return names
