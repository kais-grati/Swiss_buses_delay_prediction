"""
Add trip_stop_index column to an existing parquet dataset.

Computes trip_stop_index = ordinal position (1-based) of each stop within its
(trip_id, date) group, ordered by arrival timestamp.  First-stop rows have
already been filtered from the dataset, so ROW_NUMBER() naturally yields the
correct original position: the second stop in a trip gets index 2, etc.

Since each date is independent, we process date-by-date.  Memory stays bounded
regardless of dataset size (~300MB peak for 500M rows).

Usage:
    python scripts/add_trip_stop_index.py [--path data/swiss_bus_2025_weather_traffic.parquet]
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent


def add_trip_stop_index(input_path: str) -> None:
    input_path = str(ROOT / input_path) if not os.path.isabs(input_path) else input_path
    tmp_path = input_path + ".tsi_tmp"

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA threads=2")

    # Check if trip_stop_index already exists
    cols = con.execute(
        f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{input_path}'))"
    ).fetchall()
    col_names = {c[0] for c in cols}

    if "trip_stop_index" in col_names:
        print(f"trip_stop_index already present in {input_path}, nothing to do.")
        con.close()
        return

    # Discover distinct dates
    print("Discovering distinct dates...")
    dates = con.execute(f"""
        SELECT DISTINCT CAST(timestamp AS DATE) AS d
        FROM read_parquet('{input_path}')
        ORDER BY d
    """).fetchall()
    con.close()
    print(f"Found {len(dates)} distinct dates.")

    tmp_dir = tempfile.mkdtemp(prefix="tsi_", dir=ROOT / "data")
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

            # ROW_NUMBER() gives 1-based positions.  Since first-stop rows
            # (position 0) were already dropped, this naturally yields the
            # correct original stop index (1, 2, 3, …).
            con.execute(f"""
                COPY (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY trip_id
                            ORDER BY timestamp
                        )::SMALLINT AS trip_stop_index
                    FROM read_parquet('{input_path}')
                    WHERE CAST(timestamp AS DATE) = DATE '{date_str}'
                ) TO '{out_file}'
                (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE 500000)
            """)
            con.close()
            print(" ✓")

        # Merge all date files into final output
        print(f"\nMerging {len(date_files)} date files into final output...")

        con = duckdb.connect()
        con.execute("PRAGMA memory_limit='4GB'")
        con.execute("PRAGMA threads=2")

        col_order = con.execute(f"""
            SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{date_files[0]}'))
        """).fetchall()
        col_list = ", ".join(f'"{c[0]}"' for c in col_order)

        con.execute(f"""
            COPY (
                SELECT {col_list}
                FROM read_parquet('{tmp_dir}/*.parquet')
            ) TO '{tmp_path}'
            (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE 500000)
        """)
        con.close()

        # Validate
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(tmp_path)
        assert "trip_stop_index" in pf.schema_arrow.names, \
            "trip_stop_index missing from output schema"
        print(f"  Output: {pf.metadata.num_rows:,} rows, "
              f"{pf.metadata.num_row_groups} row groups, "
              f"{os.path.getsize(tmp_path) / 1e9:.2f} GB")

        # Atomic replace
        os.replace(tmp_path, input_path)
        print(f"  Replaced {input_path}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add trip_stop_index column to an existing parquet dataset"
    )
    parser.add_argument(
        "--path", default="data/swiss_bus_2025_weather_traffic.parquet",
        help="Path to the parquet file (relative to project root)"
    )
    args = parser.parse_args()
    add_trip_stop_index(args.path)
