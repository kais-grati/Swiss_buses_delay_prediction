"""
add_holidays.py — adds is_public_holiday to dataset_with_weather.parquet in-place.

Reads the existing parquet, joins canton-specific Swiss public holidays, writes
to a temp file, verifies row count, then atomically replaces the original.
Uses DuckDB streaming — the full file is never loaded into memory.

Requirements: pip install holidays duckdb pyarrow pandas
"""

import sys
import threading
import time
from multiprocessing import cpu_count
from pathlib import Path

import duckdb
import holidays
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

_DATA        = Path(__file__).resolve().parent.parent / "data"
DATASET      = _DATA / "dataset_with_weather.parquet"
STATION_DATA = _DATA / "station_data.parquet"
STOP_MAP_TMP = _DATA / "_stop_map_tmp.parquet"
HOLIDAYS_TMP = _DATA / "_holidays_tmp.parquet"


def build_stop_canton_map() -> None:
    stops = pq.read_table(
        STATION_DATA, columns=["number", "cantonabbreviation"]
    ).to_pandas()
    stops["stop_id"] = pd.to_numeric(stops["number"], errors="coerce").astype("Int64")
    stops = stops.dropna(subset=["stop_id"])
    result = pd.DataFrame({
        "stop_id": stops["stop_id"].astype("int64"),
        "canton":  stops["cantonabbreviation"],
    })
    pq.write_table(pa.Table.from_pandas(result, preserve_index=False), STOP_MAP_TMP)
    print(f"  Mapped {len(result):,} stops to cantons")


def build_holidays_table() -> None:
    con = duckdb.connect()
    years = [
        r[0] for r in con.execute(
            f"SELECT DISTINCT YEAR(timestamp) FROM read_parquet('{DATASET}')"
        ).fetchall()
    ]
    rows = []
    for canton in holidays.Switzerland.subdivisions:
        for date in holidays.Switzerland(subdiv=canton, years=years):
            rows.append({"date": date, "canton": canton, "is_public_holiday": True})
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), HOLIDAYS_TMP)
    print(f"  {len(df):,} public holiday records for years {years}")


def add_column() -> None:
    tmp = DATASET.with_suffix(".tmp.parquet")
    threads = min(cpu_count(), 8)

    con = duckdb.connect()
    con.execute(f"SET threads = {threads}")
    con.execute("SET memory_limit = '8GB'")

    n_before = con.execute(f"SELECT COUNT(*) FROM read_parquet('{DATASET}')").fetchone()[0]
    print(f"  Rows in dataset: {n_before:,}")

    query = f"""
        COPY (
            SELECT
                d.*,
                COALESCE(h.is_public_holiday, FALSE) AS is_public_holiday
            FROM read_parquet('{DATASET}') AS d
            LEFT JOIN read_parquet('{STOP_MAP_TMP}') AS sm
                ON d.stop_id = sm.stop_id
            LEFT JOIN read_parquet('{HOLIDAYS_TMP}') AS h
                ON sm.canton = h.canton
               AND CAST(d.timestamp AS DATE) = h.date
        ) TO '{tmp}' (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE 500000)
    """

    exc: list[Exception] = []

    def _run() -> None:
        try:
            con.execute(query)
        except Exception as e:
            exc.append(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    input_size = DATASET.stat().st_size
    with tqdm(total=input_size, unit="B", unit_scale=True, unit_divisor=1024,
              desc="  writing", dynamic_ncols=True) as bar:
        prev = 0
        while thread.is_alive():
            time.sleep(0.5)
            cur = tmp.stat().st_size if tmp.exists() else 0
            bar.update(cur - prev)
            prev = cur
        bar.update(input_size - prev)

    thread.join()
    if exc:
        tmp.unlink(missing_ok=True)
        raise exc[0]

    n_after = con.execute(f"SELECT COUNT(*) FROM read_parquet('{tmp}')").fetchone()[0]
    if n_after != n_before:
        tmp.unlink(missing_ok=True)
        sys.exit(f"Row count mismatch: expected {n_before:,}, got {n_after:,} — aborting.")

    tmp.replace(DATASET)
    size_gb = DATASET.stat().st_size / 1e9
    print(f"Done → {DATASET} ({n_after:,} rows, {size_gb:.2f} GB)")


def main() -> None:
    if not DATASET.exists():
        sys.exit(f"File not found: {DATASET}")

    print("Building stop → canton map...")
    build_stop_canton_map()
    print("Building holiday table...")
    build_holidays_table()
    print("Writing updated dataset...")
    try:
        add_column()
    finally:
        STOP_MAP_TMP.unlink(missing_ok=True)
        HOLIDAYS_TMP.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
