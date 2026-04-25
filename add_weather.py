#!/usr/bin/env python3
"""
add_weather.py — enriches dataset.parquet with hourly weather features.

Pipeline:
  1. For each unique stop_id (BPUIC), find the nearest MeteoSwiss weather
     station using LV95 coordinates from station_data.parquet (converted to
     WGS84 via pyproj) and a KDTree over the 158 MeteoSwiss stations.
  2. For each dataset row, convert the scheduled timestamp from Swiss local
     time (Europe/Zurich) to UTC and floor to the hour.
  3. Left-join with weather_hourly.parquet on (meteoswiss_station_id, hour).
  4. Write result to dataset_with_weather.parquet (dataset.parquet unchanged).

Requirements:  pip install pyproj scipy
"""

import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from scipy.spatial import KDTree
from pyproj import Transformer

# ── Paths ──────────────────────────────────────────────────────────────────────

DATASET_IN      = Path("dataset.parquet")
DATASET_OUT     = Path("dataset_with_weather.parquet")
STATION_DATA    = Path("station_data.parquet")       # bus stop coords (LV95)
STATION_META    = Path("station_metadata.parquet")   # MeteoSwiss station coords
WEATHER_HOURLY  = Path("weather_hourly.parquet")

CHUNK_ROWS = 2_000_000   # rows per processing batch

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

# ── Step 1: build stop_id → nearest MeteoSwiss station map ────────────────────

def build_stop_station_map() -> dict[int, str]:
    """Returns {bpuic_int: meteoswiss_station_id} for all stops with coords."""

    log.info("Building stop → MeteoSwiss station map...")

    # Bus stop coordinates in LV95 (EPSG:2056)
    stops = pq.read_table(
        STATION_DATA, columns=["number", "lv95east", "lv95north"]
    ).to_pandas()
    stops = stops.dropna(subset=["lv95east", "lv95north"])
    stops["stop_id"] = pd.to_numeric(stops["number"], errors="coerce").astype("Int64")
    stops = stops.dropna(subset=["stop_id"])

    # Convert LV95 → WGS84
    lv95_to_wgs84 = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = lv95_to_wgs84.transform(
        stops["lv95east"].to_numpy(),
        stops["lv95north"].to_numpy(),
    )
    stops["lat"] = lat
    stops["lon"] = lon

    # MeteoSwiss stations in WGS84
    meta = pq.read_table(
        STATION_META, columns=["station_id", "lat", "lon"]
    ).to_pandas().dropna()

    # KDTree over MeteoSwiss station coordinates
    station_coords = meta[["lat", "lon"]].to_numpy()
    tree = KDTree(station_coords)

    # Query nearest station for every bus stop
    _, indices = tree.query(stops[["lat", "lon"]].to_numpy())
    stops["weather_station"] = meta["station_id"].iloc[indices].values

    mapping = dict(zip(stops["stop_id"].astype(int), stops["weather_station"]))
    log.info(f"  Mapped {len(mapping):,} unique bus stops to MeteoSwiss stations")
    return mapping

# ── Step 2: load weather into a fast lookup structure ─────────────────────────

def build_weather_lookup() -> pd.DataFrame:
    """
    Returns weather_hourly as a DataFrame indexed by (station_id, timestamp)
    where timestamp is already floored to the hour in UTC.
    """
    log.info("Loading weather_hourly.parquet...")
    weather = pq.read_table(WEATHER_HOURLY).to_pandas()

    # Floor to hour as naive UTC (timezone stripped for merge compatibility)
    weather["timestamp"] = (
        pd.to_datetime(weather["timestamp"], utc=True)
          .dt.floor("h")
          .dt.tz_localize(None)
    )
    weather = weather.set_index(["station_id", "timestamp"])
    log.info(f"  {len(weather):,} hourly observations across {weather.index.get_level_values(0).nunique()} stations")
    return weather

# ── Step 3: process dataset in chunks ─────────────────────────────────────────

def add_weather(stop_map: dict, weather: pd.DataFrame) -> None:
    ds_file  = pq.ParquetFile(DATASET_IN)
    existing = pq.read_schema(DATASET_IN)

    # Output schema = existing + weather columns (float32)
    extra_fields = [pa.field(c, pa.float32()) for c in WEATHER_COLS]
    out_schema   = pa.schema(list(existing) + extra_fields)

    total_rows = ds_file.metadata.num_rows
    log.info(f"Processing {total_rows:,} dataset rows in chunks of {CHUNK_ROWS:,}...")

    tmp = DATASET_OUT.with_suffix(".tmp.parquet")
    written = 0

    with pq.ParquetWriter(tmp, out_schema, compression="snappy") as writer:
        for batch in ds_file.iter_batches(batch_size=CHUNK_ROWS):
            df = batch.to_pandas()

            # Map stop_id → MeteoSwiss station_id
            df["_station"] = df["stop_id"].map(stop_map)

            # Convert scheduled timestamp: Swiss local → UTC → floor to hour (naive)
            ts = pd.to_datetime(df["timestamp"])
            df["_ts_utc"] = (
                ts.dt.tz_localize("Europe/Zurich", ambiguous="NaT", nonexistent="NaT")
                  .dt.tz_convert("UTC")
                  .dt.floor("h")
                  .dt.tz_localize(None)
            )
            joined = df.merge(
                weather.reset_index().rename(columns={"station_id": "_station", "timestamp": "_ts_utc"}),
                on=["_station", "_ts_utc"],
                how="left",
            )
            df = joined.drop(columns=["_station", "_ts_utc"])

            writer.write_table(
                pa.Table.from_pandas(df, schema=out_schema, preserve_index=False)
            )
            written += len(df)
            log.info(f"  {written:,} / {total_rows:,} rows written")

    tmp.replace(DATASET_OUT)
    size_gb = DATASET_OUT.stat().st_size / 1e9
    log.info(f"Done → {DATASET_OUT} ({size_gb:.2f} GB)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    stop_map = build_stop_station_map()
    weather  = build_weather_lookup()
    add_weather(stop_map, weather)


if __name__ == "__main__":
    main()
