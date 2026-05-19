"""
Add dist_to_prev_stop (metres) to a bus-delay dataset.

Computes Euclidean distance between consecutive stops of the same trip
using LV95 coordinates from station_data.parquet, then writes the
augmented dataset in-place.

Uses DuckDB for out-of-core processing — sorts and window functions
spill to disk, so 9 GB+ datasets won't exhaust RAM.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import duckdb

DATA = Path(__file__).resolve().parent.parent / "data"
STATION_DATA = DATA / "station_data.parquet"

MEMORY_LIMIT = "8GB"
TEMP_DIR = "/tmp/duckdb"


def add_stop_distance(input_path: Path) -> None:
    input_abs = str(input_path.absolute())
    station_abs = str(STATION_DATA.absolute())
    output_tmp = str(input_path.parent / (input_path.stem + "_with_dist.parquet"))

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory='{TEMP_DIR}'")

    # Validate required columns exist
    col_names = {
        c[0] for c in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{input_abs}')"
        ).fetchall()
    }
    missing = {"trip_id", "stop_id", "timestamp"} - col_names
    if missing:
        print(f"ERROR: dataset missing columns: {missing}")
        sys.exit(1)

    t_start = time.perf_counter()

    n_before = con.execute(
        f"SELECT count(*) FROM read_parquet('{input_abs}')"
    ).fetchone()[0]
    print(f"  Input: {input_path.name} ({n_before:,} rows)")

    # 1. Load station LV95 coordinates
    t0 = time.perf_counter()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE station AS
        SELECT
            TRY_CAST(number AS INTEGER) AS stop_id,
            lv95east,
            lv95north,
        FROM read_parquet('{station_abs}')
        WHERE lv95east IS NOT NULL AND lv95north IS NOT NULL
    """)
    n_st = con.execute("SELECT count(*) FROM station").fetchone()[0]
    print(f"  [1/4] Station coords loaded ({n_st:,} stops)  "
          f"({time.perf_counter() - t0:.1f}s)")

    # 2. Sort input by (trip_id, timestamp) — DuckDB external sort spills to disk
    t0 = time.perf_counter()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sorted AS
        SELECT *
        FROM read_parquet('{input_abs}')
        ORDER BY trip_id, timestamp
    """)
    n_sorted = con.execute("SELECT count(*) FROM sorted").fetchone()[0]
    print(f"  [2/4] Sorted by (trip_id, timestamp) ({n_sorted:,} rows)  "
          f"({time.perf_counter() - t0:.1f}s)")

    # 3. Join coords + LAG to get prev-stop positions
    t0 = time.perf_counter()
    con.execute("""
        CREATE OR REPLACE TEMP TABLE with_coords AS
        SELECT
            s.*,
            st.lv95east,
            st.lv95north,
            LAG(st.lv95east)  OVER () AS prev_east,
            LAG(st.lv95north) OVER () AS prev_north,
            LAG(s.trip_id)    OVER () AS prev_trip
        FROM sorted s
        LEFT JOIN station st USING (stop_id)
    """)
    n_joined = con.execute("SELECT count(*) FROM with_coords").fetchone()[0]
    print(f"  [3/4] Joined coords + LAG ({n_joined:,} rows)  "
          f"({time.perf_counter() - t0:.1f}s)")

    # 4. Compute dist_to_prev_stop, filter NULL/zero, write output
    t0 = time.perf_counter()
    con.execute(f"""
        COPY (
            WITH with_dist AS (
                SELECT
                    * EXCLUDE (lv95east, lv95north, prev_east, prev_north, prev_trip),
                    CASE
                        WHEN prev_trip IS NULL OR prev_trip != trip_id
                            THEN NULL
                        WHEN prev_east IS NULL OR prev_north IS NULL
                          OR lv95east IS NULL OR lv95north IS NULL
                            THEN NULL
                        ELSE SQRT(POW(lv95east - prev_east, 2)
                                + POW(lv95north - prev_north, 2))
                    END AS dist_to_prev_stop
                FROM with_coords
            )
            SELECT *
            FROM with_dist
            WHERE dist_to_prev_stop IS NOT NULL AND dist_to_prev_stop > 0
        ) TO '{output_tmp}' (FORMAT PARQUET, COMPRESSION 'ZSTD')
    """)
    print(f"  [4/4] Distance computed + written  "
          f"({time.perf_counter() - t0:.1f}s)")

    n_after = duckdb.connect().execute(
        f"SELECT count(*) FROM read_parquet('{output_tmp}')"
    ).fetchone()[0]

    # Replace original with augmented version
    input_path.unlink()
    Path(output_tmp).rename(input_path)

    con.close()
    total_t = time.perf_counter() - t_start
    print(f"  -> {input_path.name}: {n_after:,} rows, "
          f"dropped {n_before - n_after:,}  ({total_t:.1f}s total)")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        add_stop_distance(Path(arg))
    if len(sys.argv) == 1:
        print("Usage: python add_stop_distance.py <dataset.parquet> [dataset2.parquet ...]")
