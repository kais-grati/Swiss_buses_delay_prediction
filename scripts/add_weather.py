#!/usr/bin/env python3
"""
add_weather.py — enriches dataset.parquet with hourly weather features.

Pipeline:
  1. For each unique stop_id (BPUIC), find the nearest MeteoSwiss weather
     station using LV95 coordinates from station_data.parquet (converted to
     WGS84 via pyproj) and a KDTree over the 158 MeteoSwiss stations.
  2. Use DuckDB to join the full dataset with weather_hourly.parquet in one
     multi-threaded pass — no Python loops, no pandas chunks.
     Timestamp conversion (Swiss local → UTC, floor to hour) is done in SQL.
  3. Write result to dataset_with_weather.parquet (dataset.parquet unchanged).

Requirements:  pip install pyproj scipy duckdb tqdm
"""

import logging
import threading
import time
from multiprocessing import cpu_count
from pathlib import Path

import duckdb
import holidays
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer
from scipy.spatial import KDTree
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────

_DATA          = Path(__file__).resolve().parent.parent / "data"
DATASET_IN     = _DATA / "dataset.parquet"
DATASET_OUT    = _DATA / "dataset_with_weather.parquet"
STATION_DATA   = _DATA / "station_data.parquet"
STATION_META   = _DATA / "station_metadata.parquet"
WEATHER_HOURLY = _DATA / "weather_hourly.parquet"
STOP_MAP_TMP   = _DATA / "_stop_map_tmp.parquet"
HOLIDAYS_TMP   = _DATA / "_holidays_tmp.parquet"

WEATHER_COLS = [
    "temperature", "precipitation", "sunshine", "humidity",
    "wind_speed", "wind_gust", "wind_dir", "pressure", "snow_depth",
]

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Step 1: stop_id → nearest MeteoSwiss station ──────────────────────────────

def build_stop_station_map() -> None:
    """Builds {stop_id → weather_station} mapping and saves to STOP_MAP_TMP."""
    log.info("Building stop → MeteoSwiss station map...")

    stops = pq.read_table(
        STATION_DATA, columns=["number", "lv95east", "lv95north", "cantonabbreviation"]
    ).to_pandas().dropna(subset=["lv95east", "lv95north"])

    stops["stop_id"] = pd.to_numeric(stops["number"], errors="coerce").astype("Int64")
    stops = stops.dropna(subset=["stop_id"])

    lon, lat = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True).transform(
        stops["lv95east"].to_numpy(),
        stops["lv95north"].to_numpy(),
    )
    stops["lat"] = lat
    stops["lon"] = lon

    meta = pq.read_table(STATION_META, columns=["station_id", "lat", "lon"]).to_pandas().dropna()
    _, indices = KDTree(meta[["lat", "lon"]].to_numpy()).query(stops[["lat", "lon"]].to_numpy())
    stops["weather_station"] = meta["station_id"].iloc[indices].values

    result = pd.DataFrame({
        "stop_id":         stops["stop_id"].astype("int64"),
        "weather_station": stops["weather_station"],
        "canton":          stops["cantonabbreviation"],
    })
    pq.write_table(pa.Table.from_pandas(result, preserve_index=False), STOP_MAP_TMP)
    log.info(f"  Mapped {len(result):,} stops to MeteoSwiss stations")

# ── Step 2: public holiday lookup table ───────────────────────────────────────

def build_holidays_table() -> None:
    """Builds a (date, canton, is_public_holiday) parquet for all CH cantons."""
    log.info("Building public holiday table...")

    con = duckdb.connect()
    years = [
        r[0] for r in con.execute(
            f"SELECT DISTINCT YEAR(timestamp) FROM read_parquet('{DATASET_IN}')"
        ).fetchall()
    ]

    rows = []
    for canton in holidays.Switzerland.subdivisions:
        for date in holidays.Switzerland(subdiv=canton, years=years):
            rows.append({"date": date, "canton": canton, "is_public_holiday": True})

    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), HOLIDAYS_TMP)
    log.info(f"  {len(df):,} public holiday records across {len(holidays.Switzerland.subdivisions)} cantons")


# ── Step 3: single-pass DuckDB join ───────────────────────────────────────────

def join_weather() -> None:
    """Joins dataset with weather via a single DuckDB query."""
    log.info("Joining with DuckDB (single pass, multi-threaded)...")

    threads = min(cpu_count(), 8)
    tmp     = DATASET_OUT.with_suffix(".tmp.parquet")

    weather_cols_sql = ",\n                ".join(
        f"w.{c}::FLOAT AS {c}" for c in WEATHER_COLS
    )

    con = duckdb.connect()
    con.execute(f"SET threads = {threads}")
    con.execute("SET memory_limit = '8GB'")

    query = f"""
        COPY (
            SELECT
                d.timestamp,
                d.time_sin,  d.time_cos,
                d.dow_sin,   d.dow_cos,
                d.month_sin, d.month_cos,
                d.is_weekend,
                d.operator,  d.line,
                d.stop_id,   d.stop_name,
                d.additional_trip,
                d.arrival_delay_s, d.departure_delay_s,
                {weather_cols_sql},
                COALESCE(h.is_public_holiday, FALSE) AS is_public_holiday
            FROM read_parquet('{DATASET_IN}') AS d
            LEFT JOIN read_parquet('{STOP_MAP_TMP}') AS sm
                ON d.stop_id = sm.stop_id
            LEFT JOIN read_parquet('{WEATHER_HOURLY}') AS w
                ON sm.weather_station = w.station_id
               AND date_trunc('hour',
                       (d.timestamp AT TIME ZONE 'Europe/Zurich')::TIMESTAMP
                   ) = w.timestamp
            LEFT JOIN read_parquet('{HOLIDAYS_TMP}') AS h
                ON sm.canton = h.canton
               AND CAST(d.timestamp AS DATE) = h.date
            WHERE NOT d.pass_through
        ) TO '{tmp}' (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE 500000)
    """

    # d.timestamp is naive Swiss local time (Europe/Zurich).
    # "AT TIME ZONE 'Europe/Zurich'" interprets it as Zurich local → returns TIMESTAMPTZ (UTC).
    # Casting to ::TIMESTAMP strips the timezone → naive UTC.
    # date_trunc floors to the hour to match the hourly weather table.
    exc: list[Exception] = []

    def _run() -> None:
        try:
            con.execute(query)
        except Exception as e:
            exc.append(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    input_size = DATASET_IN.stat().st_size
    with tqdm(total=input_size, unit="B", unit_scale=True, unit_divisor=1024,
              desc="  writing", dynamic_ncols=True) as bar:
        prev = 0
        while thread.is_alive():
            time.sleep(0.5)
            cur = tmp.stat().st_size if tmp.exists() else 0
            bar.update(cur - prev)
            prev = cur
        bar.update(input_size - prev)  # fill to 100 % on completion

    thread.join()
    if exc:
        raise exc[0]

    tmp.replace(DATASET_OUT)
    size_gb = DATASET_OUT.stat().st_size / 1e9
    log.info(f"Done → {DATASET_OUT} ({size_gb:.2f} GB)")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    build_stop_station_map()
    build_holidays_table()
    try:
        join_weather()
    finally:
        STOP_MAP_TMP.unlink(missing_ok=True)
        HOLIDAYS_TMP.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
