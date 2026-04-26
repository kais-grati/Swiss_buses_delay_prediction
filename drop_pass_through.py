"""
Remove pass_through rows and column from dataset_with_weather.parquet in-place.
Uses DuckDB streaming — the full file is never loaded into memory.
"""

import shutil
import sys
from pathlib import Path

import duckdb

TARGET = Path("dataset_with_weather.parquet")
TMP    = TARGET.with_suffix(".tmp.parquet")


def main() -> None:
    if not TARGET.exists():
        sys.exit(f"File not found: {TARGET}")

    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET threads = 4")

    # Count pass-through rows before removing them
    total, n_pass = con.execute("""
        SELECT
            COUNT(*),
            COUNT(*) FILTER (WHERE pass_through)
        FROM read_parquet(?)
    """, [str(TARGET)]).fetchone()

    print(f"Total rows        : {total:>12,}")
    print(f"Pass-through rows : {n_pass:>12,}  ({100 * n_pass / total:.2f}%)")
    print(f"Rows kept         : {total - n_pass:>12,}")

    if n_pass == 0:
        print("Nothing to remove.")
        return

    print(f"\nWriting filtered file to {TMP} …")

    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (pass_through)
            FROM read_parquet('{TARGET}')
            WHERE NOT pass_through
        ) TO '{TMP}' (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE 500000)
    """)

    # Verify row count
    kept = con.execute(f"SELECT COUNT(*) FROM read_parquet('{TMP}')").fetchone()[0]
    if kept != total - n_pass:
        TMP.unlink(missing_ok=True)
        sys.exit(f"Row count mismatch: expected {total - n_pass:,}, got {kept:,} — aborting.")

    TMP.replace(TARGET)
    print(f"Done. {TARGET} updated ({kept:,} rows, pass_through column removed).")


if __name__ == "__main__":
    main()
