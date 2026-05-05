import pandas as pd
import numpy as np
from ml.preprocessors.class_encoder import ClassEncoder

class DelayBinner(ClassEncoder):
    """
    Converts continuous delay seconds into discrete classes.

    Classes:
    0: On-Time (delay <= 120s)
    1: Minor Delay (120s < delay <= 600s)
    2: Severe Delay (delay > 600s)
    """
    def __init__(self, bins=[120, 600]):
        self.bins = bins

    def encode(self, y: pd.Series) -> pd.Series:
        return pd.cut(
            y,
            bins=[-float('inf')] + self.bins + [float('inf')],
            labels=[0, 1, 2]
        ).astype(int)

    def decode(self, y_encoded: np.ndarray) -> np.ndarray:
        # Returns the midpoint of the bin as a representative value
        midpoints = [0, (self.bins[0] + self.bins[1])/2, self.bins[1] + 300]
        return np.array([midpoints[i] for i in y_encoded])
