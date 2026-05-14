import pandas as pd
from ml.preprocessors.base import BasePreprocessor


class LagDelayEncoder(BasePreprocessor):
    """Provides prev_stop_delay — either precomputed or computed at runtime.

    Fast path: if prev_stop_delay is already in X (added by
    scripts/add_lag_delay.py or build_dataset.py), just fills NaN with the
    median default and drops trip_id / arrival_delay_s.  No groupby needed.

    Slow path: if prev_stop_delay is missing, computes it via
    groupby(_date, trip_id).shift(arrival_delay_s) — requires timestamp,
    trip_id, and arrival_delay_s in X (keep_target_in_X=True).

    Must be placed after TemporalFeatureExtractor in the pipeline (needs
    timestamp in the slow path).
    """

    def __init__(self):
        super().__init__()
        self._default_lag = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "LagDelayEncoder":
        if y is not None:
            self._default_lag = float(y.median())
        elif "arrival_delay_s" in X.columns:
            self._default_lag = float(X["arrival_delay_s"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if "arrival_delay_s" not in X.columns:
            # Nothing to lag — target isn't in X
            X = X.copy()
            return X

        drop_cols = ["trip_id", "arrival_delay_s"]

        if "prev_stop_delay" in X.columns:
            # Precomputed path — just fill NaNs, no groupby
            X = X.copy()
            X["prev_stop_delay"] = (
                X["prev_stop_delay"].fillna(self._default_lag).astype("float32")
            )
            X.drop(columns=[c for c in drop_cols if c in X.columns], inplace=True)
            return X

        # Fallback: compute lag at runtime
        X = X.copy()

        if "trip_id" not in X.columns:
            X.drop(columns=["arrival_delay_s"], inplace=True)
            X["prev_stop_delay"] = self._default_lag
            return X

        X["_date"] = pd.to_datetime(X["timestamp"]).dt.date

        X["prev_stop_delay"] = (
            X.groupby(["_date", "trip_id"], sort=False)["arrival_delay_s"]
            .shift(1)
            .fillna(self._default_lag)
            .astype("float32")
        )

        X.drop(columns=["_date"] + drop_cols, inplace=True)

        return X
