"""
Add prev_stop_delay column to an existing parquet dataset.

Computes prev_stop_delay = LAG(arrival_delay_s, 1) OVER (
    PARTITION BY CAST(timestamp AS DATE), trip_id
    ORDER BY timestamp
)

Since LAG is partitioned by (date, trip_id), each date is independent.
Processing date-by-date keeps memory bounded regardless of dataset size.
On a 7.6GB RAM machine with 499M rows, this peaks at ~300MB instead of 100GB+.

Usage:
    python scripts/add_lag_delay.py [--path data/dataset_with_weather.parquet]
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent


def add_lag_delay(input_path: str) -> None:
    input_path = str(ROOT / input_path) if not os.path.isabs(input_path) else input_path
    tmp_path = input_path + ".lag_tmp"

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA threads=2")

    # Check if prev_stop_delay already exists
    cols = con.execute(
        f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{input_path}'))"
    ).fetchall()
    col_names = {c[0] for c in cols}

    if "prev_stop_delay" in col_names:
        print(f"prev_stop_delay already present in {input_path}, nothing to do.")
        con.close()
        return

    # Get distinct dates — we process one date at a time
    print("Discovering distinct dates...")
    dates = con.execute(f"""
        SELECT DISTINCT CAST(timestamp AS DATE) AS d
        FROM read_parquet('{input_path}')
        ORDER BY d
    """).fetchall()
    con.close()
    print(f"Found {len(dates)} distinct dates.")

    # Work in a temp directory for intermediate per-date parquet files
    tmp_dir = tempfile.mkdtemp(prefix="lag_delay_", dir=ROOT / "data")
    date_files = []

    try:
        for i, (date,) in enumerate(dates):
            date_str = str(date)
            out_file = os.path.join(tmp_dir, f"{date_str}.parquet")
            date_files.append(out_file)

            print(f"  [{i+1}/{len(dates)}] {date_str}", end="", flush=True)

            con = duckdb.connect()
            con.execute("PRAGMA memory_limit='3GB'")
            con.execute("PRAGMA threads=2")

            con.execute(f"""
                COPY (
                    SELECT *,
                        LAG(arrival_delay_s, 1) OVER (
                            PARTITION BY trip_id
                            ORDER BY timestamp
                        ) AS prev_stop_delay
                    FROM read_parquet('{input_path}')
                    WHERE CAST(timestamp AS DATE) = DATE '{date_str}'
                ) TO '{out_file}'
                (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE 500000)
            """)
            con.close()
            print(" ✓")

        # Merge all date files into final output, preserving original column order
        print(f"\nMerging {len(date_files)} date files into final output...")

        con = duckdb.connect()
        con.execute("PRAGMA memory_limit='4GB'")
        con.execute("PRAGMA threads=2")

        # Get original column order from the first date file
        col_order = con.execute(f"""
            SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{date_files[0]}'))
        """).fetchall()
        col_list = ", ".join(f'"{c[0]}"' for c in col_order)

        # No ORDER BY needed — files are named by date (YYYY-MM-DD.parquet),
        # glob reads them in chronological order, and each file is internally
        # sorted by timestamp from the per-date window-function query.
        con.execute(f"""
            COPY (
                SELECT {col_list}
                FROM read_parquet('{tmp_dir}/*.parquet')
            ) TO '{tmp_path}'
            (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE 500000)
        """)
        con.close()

        # Validate the output
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(tmp_path)
        new_cols = pf.schema_arrow.names
        assert "prev_stop_delay" in new_cols, "prev_stop_delay missing from output schema"
        print(f"  Output: {pf.metadata.num_rows:,} rows, "
              f"{pf.metadata.num_row_groups} row groups, "
              f"{os.path.getsize(tmp_path) / 1e9:.2f} GB")

        # Atomic replace
        os.replace(tmp_path, input_path)
        print(f"  Replaced {input_path}")

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add prev_stop_delay column to an existing parquet dataset"
    )
    parser.add_argument(
        "--path", default="data/dataset_with_weather.parquet",
        help="Path to the parquet file (relative to project root)"
    )
    args = parser.parse_args()
    add_lag_delay(args.path)
