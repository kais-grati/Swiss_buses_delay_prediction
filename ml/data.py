from typing import Generator, List, Optional, Tuple
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
        if self.test_size <= 0:
            return None, X, None, y
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def stream(self) -> Generator[Tuple[pd.DataFrame, pd.Series], None, None]:
        """Yield (X_chunk, y_chunk) by reading row groups one at a time."""
        pf = pq.ParquetFile(self.path)
        for i in range(pf.metadata.num_row_groups):
            chunk = pf.read_row_group(i).to_pandas()
            chunk = chunk.drop(columns=[c for c in self.drop_cols if c in chunk.columns])
            X = chunk.drop(columns=[self.target])
            y = chunk[self.target]
            yield X, y
