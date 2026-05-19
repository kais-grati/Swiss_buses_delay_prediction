"""
Add traffic load features from the Swiss road traffic GeoPackage to the dataset.

For each bus stop, finds the nearest road segment from the national traffic
dataset and extracts congestion-related features:
  - traffic_dtv:          average daily total vehicles on nearest road
  - traffic_dwv:          weekday average total vehicles
  - traffic_msp / asp:    morning / evening peak hour vehicles
  - traffic_heavy_share:  fraction of heavy vehicles (trucks + articulated)
  - traffic_peak_ratio:   how much worse peak hours get vs daily average
  - traffic_peak:         traffic_peak that matches the trip's time of day
                          (MSP for AM stops, ASP for PM stops)

Uses DuckDB for streaming join — memory-efficient on the 12 GB dataset.
"""

import sqlite3
import sys
import time
import threading
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATASET_IN = ROOT / "data" / "swiss_bus_2026_weather.parquet"
DATASET_OUT = ROOT / "data" / "swiss_bus_2026_weather_traffic.parquet"
GPKG_PATH = ROOT / "data" / "traffic" / "belastung-personenverkehr-strasse_2056.gpkg"
STATION_DATA = ROOT / "data" / "station_data.parquet"
TRAFFIC_MAP_TMP = ROOT / "data" / "_traffic_map_tmp.parquet"

TRAFFIC_MAP_SCHEMA = pa.schema([
    ("stop_id",             pa.int32()),
    ("traffic_dtv",         pa.float32()),
    ("traffic_dwv",         pa.float32()),
    ("traffic_msp",         pa.float32()),
    ("traffic_asp",         pa.float32()),
    ("traffic_pw",          pa.float32()),
    ("traffic_lw",          pa.float32()),
    ("traffic_lz",          pa.float32()),
    ("traffic_li",          pa.float32()),
    ("traffic_heavy_share", pa.float32()),
    ("traffic_peak_ratio",  pa.float32()),
])


def _build_stop_coords():
    """Return {stop_id: (lv95east, lv95north)} from station_data.parquet."""
    if not STATION_DATA.exists():
        return {}
    stops = pq.read_table(
        STATION_DATA, columns=["number", "lv95east", "lv95north"]
    ).to_pandas().dropna(subset=["lv95east", "lv95north"])
    stops["stop_id"] = pd.to_numeric(stops["number"], errors="coerce").astype("Int64")
    stops = stops.dropna(subset=["stop_id"])
    return dict(zip(
        stops["stop_id"].astype("int32"),
        zip(stops["lv95east"].astype("float64"), stops["lv95north"].astype("float64")),
    ))


def build_traffic_map():
    """Build {stop_id → traffic features} via rtree nearest-neighbour on GPKG."""
    print("Building stop → road traffic map...")
    t0 = time.time()

    coords = _build_stop_coords()
    print(f"  Loaded {len(coords):,} stop coordinates")

    conn = sqlite3.connect(str(GPKG_PATH))
    cur = conn.cursor()

    traffic_cols_api = [
        "DTV_FZG", "DWV_FZG", "MSP_FZG", "ASP_FZG",
        "DTV_PW", "DTV_LW", "DTV_LZ", "DTV_LI",
    ]

    records = []
    n = len(coords)
    no_match = 0
    search_radii = [500.0, 1000.0, 2000.0, 5000.0]

    for i, (stop_id, (east, north)) in enumerate(coords.items()):
        nearest_traffic = None

        for radius in search_radii:
            cur.execute(f"""
                SELECT r.id,
                       MAX(r.minx - {east}, 0.0, {east} - r.maxx) AS dx,
                       MAX(r.miny - {north}, 0.0, {north} - r.maxy) AS dy
                FROM rtree_Personen_Gueterverkehr_Strasse_geom AS r
                WHERE r.minx <= {east} + {radius}
                  AND r.maxx >= {east} - {radius}
                  AND r.miny <= {north} + {radius}
                  AND r.maxy >= {north} - {radius}
                ORDER BY dx * dx + dy * dy
                LIMIT 1
            """)
            row = cur.fetchone()
            if row is None:
                continue

            rtree_id = row[0]
            cur.execute(f"""
                SELECT {", ".join(traffic_cols_api)}
                FROM Personen_Gueterverkehr_Strasse
                WHERE id = {rtree_id}
            """)
            traffic = cur.fetchone()
            if traffic is not None:
                nearest_traffic = traffic
                break

        if nearest_traffic is None:
            no_match += 1
            continue

        records.append({
            "stop_id": int(stop_id),
            "traffic_dtv": float(nearest_traffic[0]),
            "traffic_dwv": float(nearest_traffic[1]),
            "traffic_msp": float(nearest_traffic[2]),
            "traffic_asp": float(nearest_traffic[3]),
            "traffic_pw":  float(nearest_traffic[4]),
            "traffic_lw":  float(nearest_traffic[5]),
            "traffic_lz":  float(nearest_traffic[6]),
            "traffic_li":  float(nearest_traffic[7]),
        })

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  ... {i+1}/{n} stops mapped ({len(records):,} matched, "
                  f"{rate:.0f} stops/s)")

    conn.close()

    elapsed = time.time() - t0
    print(f"  Matched {len(records):,}/{n:,} stops ({no_match} no-match) "
          f"in {elapsed:.1f}s ({n/elapsed:.0f} stops/s)")

    if not records:
        raise RuntimeError("No stops matched to road segments — check GPKG data")

    df = pd.DataFrame(records)

    # Derived features — use np.divide with where= to avoid zero-division warnings
    dtv = df["traffic_dtv"].values.astype("float64")
    heavy = (df["traffic_lw"].values + df["traffic_lz"].values).astype("float64")
    asp = df["traffic_asp"].values.astype("float64")

    df["traffic_heavy_share"] = np.divide(
        heavy, dtv, out=np.zeros_like(heavy, dtype="float32"), where=dtv > 0
    )
    df["traffic_peak_ratio"] = np.divide(
        asp, dtv, out=np.zeros_like(asp, dtype="float32"), where=dtv > 0
    )

    # Ensure int32 for stop_id (pandas may promote to int64)
    df["stop_id"] = df["stop_id"].astype("int32")

    table = pa.Table.from_pandas(df, schema=TRAFFIC_MAP_SCHEMA, preserve_index=False)
    pq.write_table(table, TRAFFIC_MAP_TMP, compression="snappy")
    print(f"  Saved {TRAFFIC_MAP_TMP} ({TRAFFIC_MAP_TMP.stat().st_size / 1e6:.1f} MB)")


def join_traffic():
    """DuckDB join: dataset + traffic → dataset_with_traffic.parquet."""
    print("\nJoining traffic features with dataset (DuckDB, single pass)...")

    tmp = DATASET_OUT.with_suffix(".tmp.parquet")

    con = duckdb.connect()
    con.execute("SET threads = 2")
    con.execute("SET memory_limit = '8GB'")

    # traffic_peak picks MSP for AM, ASP for PM (Europe/Zurich local hour)
    query = f"""
        COPY (
            SELECT
                d.*,
                t.traffic_dtv,
                t.traffic_dwv,
                t.traffic_pw,
                t.traffic_lw,
                t.traffic_lz,
                t.traffic_li,
                t.traffic_heavy_share,
                t.traffic_peak_ratio,
                CASE
                    WHEN EXTRACT(HOUR FROM d.timestamp) < 12
                    THEN t.traffic_msp
                    ELSE t.traffic_asp
                END AS traffic_peak
            FROM read_parquet('{DATASET_IN}') AS d
            INNER JOIN read_parquet('{TRAFFIC_MAP_TMP}') AS t
                ON d.stop_id = t.stop_id
        ) TO '{tmp}' (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE 500000)
    """

    exc: list[Exception] = []

    def _run():
        try:
            con.execute(query)
        except Exception as e:
            exc.append(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    prev = 0
    while thread.is_alive():
        time.sleep(0.5)
        cur = tmp.stat().st_size if tmp.exists() else 0
        if cur != prev:
            print(f"  writing... {cur / 1e9:.1f} GB", end="\r")
            prev = cur

    thread.join()
    if exc:
        TRAFFIC_MAP_TMP.unlink(missing_ok=True)
        raise exc[0]

    size_gb = tmp.stat().st_size / 1e9
    print(f"  writing... {size_gb:.2f} GB")

    tmp.replace(DATASET_OUT)
    TRAFFIC_MAP_TMP.unlink(missing_ok=True)

    rows_before = pq.read_metadata(DATASET_IN).num_rows
    rows_after = pq.read_metadata(DATASET_OUT).num_rows
    print(f"Done → {DATASET_OUT} ({size_gb:.2f} GB)")
    print(f"  {rows_before:,} → {rows_after:,} rows "
          f"({abs(rows_after - rows_before):,} change)")

    # Null stats
    con2 = duckdb.connect()
    nulls = con2.execute(f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN traffic_dtv         IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN traffic_peak        IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN traffic_heavy_share IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN traffic_peak_ratio  IS NULL THEN 1 ELSE 0 END)
        FROM read_parquet('{DATASET_OUT}')
    """).fetchone()
    con2.close()

    print(f"  Nulls — DTV: {nulls[1]:,}  Peak: {nulls[2]:,}  "
          f"HeavyShare: {nulls[3]:,}  PeakRatio: {nulls[4]:,}")
    if nulls[1] > 0:
        pct = nulls[1] / nulls[0] * 100
        print(f"  ⚠ {nulls[1]:,} rows ({pct:.2f}%) missing traffic data "
              "(stops not matched to road segment)")


def main():
    if not DATASET_IN.exists():
        print(f"ERROR: {DATASET_IN} not found. Run build_dataset.py first.")
        sys.exit(1)

    if not GPKG_PATH.exists():
        print(f"WARNING: {GPKG_PATH} not found. Cannot add traffic features.")
        sys.exit(1)

    print(f"Input:  {DATASET_IN} ({DATASET_IN.stat().st_size / 1e9:.2f} GB)")
    print(f"GPKG:   {GPKG_PATH} ({GPKG_PATH.stat().st_size / 1e6:.1f} MB)")
    print()

    build_traffic_map()
    join_traffic()

    print(f"\nDone. Traffic features added to {DATASET_OUT}")


if __name__ == "__main__":
    main()
