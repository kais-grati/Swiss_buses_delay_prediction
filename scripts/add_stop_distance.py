"""
Add dist_to_prev_stop (metres) to a bus-delay dataset.

Computes Euclidean distance between consecutive stops of the same trip
using LV95 coordinates from station_data.parquet, then writes the
augmented dataset in-place.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parent.parent / "data"
STATION_DATA = DATA / "station_data.parquet"


def build_stop_coords() -> dict[int, tuple[float, float]]:
    """Return {stop_id: (lv95east, lv95north)} from station_data."""
    t = pq.read_table(
        STATION_DATA, columns=["number", "lv95east", "lv95north"]
    ).to_pandas().dropna(subset=["lv95east", "lv95north"])
    t["number"] = t["number"].astype(int)
    return dict(zip(t["number"], zip(t["lv95east"], t["lv95north"])))


def euclidean_lv95(e1: float, n1: float, e2: float, n2: float) -> float:
    """Euclidean distance in LV95 metres."""
    return float(np.sqrt((e2 - e1) ** 2 + (n2 - n1) ** 2))


def add_stop_distance(input_path: Path) -> None:
    coords = build_stop_coords()
    print(f"Loaded {len(coords)} stop coordinates")

    table = pq.read_table(str(input_path))
    df = table.to_pandas()
    required = {"trip_id", "stop_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: dataset missing columns: {missing}")
        sys.exit(1)

    # Sort by trip then time so prev row is the previous stop
    df = df.sort_values(["trip_id", "timestamp"])

    # Compute distance to previous stop in same trip
    distances = np.full(len(df), np.nan, dtype="float32")
    prev_trip = None
    prev_east = prev_north = None

    for i in range(len(df)):
        trip = df["trip_id"].iloc[i]
        sid = df["stop_id"].iloc[i]
        if trip != prev_trip:
            # First stop of a trip — no previous stop
            distances[i] = np.nan
        else:
            east, north = coords.get(sid, (np.nan, np.nan))
            distances[i] = euclidean_lv95(prev_east, prev_north, east, north)
        prev_trip = trip
        prev_east, prev_north = coords.get(sid, (np.nan, np.nan))

    df["dist_to_prev_stop"] = distances

    n_before = len(df)
    # Drop first-stop rows (NaN distance) and zero-distance rows
    df = df[
        df["dist_to_prev_stop"].notna()
        & (df["dist_to_prev_stop"] > 0.0)
    ].reset_index(drop=True)

    # Write back with the new column appended
    new_table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(new_table, str(input_path))
    dropped = n_before - len(df)
    print(f"Wrote {input_path} with dist_to_prev_stop ({len(df)} rows, dropped {dropped})")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        add_stop_distance(Path(arg))
    if len(sys.argv) == 1:
        print("Usage: python add_stop_distance.py <dataset.parquet> [dataset2.parquet ...]")
