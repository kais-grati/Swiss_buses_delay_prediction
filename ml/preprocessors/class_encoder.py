from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class ClassEncoder(ABC):
    """
    Abstract base class for transforming continuous target variables
    into discrete classes.
    """
    @abstractmethod
    def encode(self, y: pd.Series) -> pd.Series:
        ...

    @abstractmethod
    def decode(self, y_encoded: np.ndarray) -> np.ndarray:
        """Optional: Convert classes back to approximate continuous values."""
        ...
