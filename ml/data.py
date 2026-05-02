from typing import List, Optional, Tuple
import duckdb
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(
        self,
        path: str,
        target: str,
        drop_cols: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
        sample_n: Optional[int] = None,
    ):
        self.path = path
        self.target = target
        self.drop_cols = drop_cols
        self.test_size = test_size
        self.random_state = random_state
        self.sample_n = sample_n

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.sample_n is not None:
            df = duckdb.query(
                f"SELECT * FROM read_parquet('{self.path}') "
                f"USING SAMPLE {self.sample_n} ROWS (reservoir, {self.random_state})"
            ).df()
        else:
            df = pq.read_table(self.path).to_pandas()
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns])
        X = df.drop(columns=[self.target])
        y = df[self.target]
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
