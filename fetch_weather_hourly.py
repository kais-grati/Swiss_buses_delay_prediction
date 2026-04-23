#!/usr/bin/env python3
"""
Fetch hourly weather from Open-Meteo historical API for all MeteoSwiss station
locations in 2025.

Source:  https://open-meteo.com (free, no API key)
Input:   station_metadata.parquet  (lat/lon per MeteoSwiss station)
Output:  weather_hourly.parquet    (station_id, timestamp[UTC], weather vars)
Cache:   weather_hourly_cache/     (one parquet per station, resume-safe)

Columns per row (hourly):
  temperature   — air temperature 2m (°C)
  precipitation — rainfall + snowfall (mm)
  sunshine      — sunshine duration (seconds per hour, 0–3600)
  humidity      — relative humidity 2m (%)
  wind_speed    — wind speed at 10m (km/h)
  wind_gust     — wind gust at 10m (km/h)
  wind_dir      — wind direction at 10m (°)
  pressure      — surface pressure (hPa)
  snow_depth    — snow depth (cm)
"""

import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE   = "https://archive-api.open-meteo.com/v1/archive"
META_INPUT = Path("station_metadata.parquet")
CACHE_DIR  = Path("weather_hourly_cache")
OUTPUT     = Path("weather_hourly.parquet")
YEAR       = 2025
MAX_WORKERS  = 1    # free tier rate-limits concurrent requests
TIMEOUT      = 60
REQUEST_GAP  = 1.0  # seconds between requests

HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "sunshine_duration",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "surface_pressure",
    "snow_depth",
]

VAR_RENAME = {
    "temperature_2m":       "temperature",
    "precipitation":        "precipitation",
    "sunshine_duration":    "sunshine",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m":       "wind_speed",
    "wind_gusts_10m":       "wind_gust",
    "wind_direction_10m":   "wind_dir",
    "surface_pressure":     "pressure",
    "snow_depth":           "snow_depth",
}

OUTPUT_SCHEMA = pa.schema([
    ("station_id",    pa.string()),
    ("timestamp",     pa.timestamp("s")),
    ("temperature",   pa.float32()),
    ("precipitation", pa.float32()),
    ("sunshine",      pa.float32()),
    ("humidity",      pa.float32()),
    ("wind_speed",    pa.float32()),
    ("wind_gust",     pa.float32()),
    ("wind_dir",      pa.float32()),
    ("pressure",      pa.float32()),
    ("snow_depth",    pa.float32()),
])

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "swiss-bus-delay-ml/1.0 (research)"

# ── Per-station fetch ──────────────────────────────────────────────────────────

def fetch_station(station: dict) -> tuple[str, int, str | None]:
    sid     = station["station_id"]
    pq_path = CACHE_DIR / f"{sid}.parquet"

    if pq_path.exists():
        return sid, -1, None

    try:
        params = {
            "latitude":   round(station["lat"], 6),
            "longitude":  round(station["lon"], 6),
            "start_date": f"{YEAR}-01-01",
            "end_date":   f"{YEAR}-12-31",
            "hourly":     ",".join(HOURLY_VARS),
            "timezone":   "UTC",
        }

        for attempt in range(6):
            try:
                r = SESSION.get(API_BASE, params=params, timeout=TIMEOUT)
                r.raise_for_status()
                data = r.json()
                break
            except requests.HTTPError as e:
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 30)) + 5
                    log.warning(f"  [{sid}] rate-limited, waiting {wait}s...")
                    time.sleep(wait)
                elif attempt == 5:
                    raise
                else:
                    wait = 2 ** attempt
                    log.warning(f"  [{sid}] retry {attempt+1}/6 ({wait}s): {e}")
                    time.sleep(wait)
            except Exception as e:
                if attempt == 5:
                    raise
                wait = 2 ** attempt
                log.warning(f"  [{sid}] retry {attempt+1}/6 ({wait}s): {e}")
                time.sleep(wait)

        time.sleep(REQUEST_GAP)

        hourly     = data["hourly"]
        timestamps = pd.to_datetime(hourly["time"])

        df = pd.DataFrame({"timestamp": timestamps})
        df["station_id"] = sid

        for var in HOURLY_VARS:
            col  = VAR_RENAME[var]
            vals = hourly.get(var, [None] * len(timestamps))
            df[col] = pd.array(vals, dtype="Float64").astype("float32")

        # Open-Meteo returns snow_depth in metres → convert to cm
        df["snow_depth"] = df["snow_depth"] * 100

        ordered_cols = ["station_id", "timestamp"] + list(VAR_RENAME.values())
        table = pa.Table.from_pandas(df[ordered_cols], schema=OUTPUT_SCHEMA, preserve_index=False)
        pq.write_table(table, pq_path, compression="snappy")
        return sid, len(df), None

    except Exception as e:
        if pq_path.exists():
            pq_path.unlink()
        return sid, -1, str(e)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    CACHE_DIR.mkdir(exist_ok=True)

    meta     = pq.read_table(META_INPUT).to_pandas()
    stations = meta[["station_id", "lat", "lon"]].dropna().to_dict("records")
    log.info(f"Fetching hourly weather for {len(stations)} stations")

    todo  = [s for s in stations if not (CACHE_DIR / f"{s['station_id']}.parquet").exists()]
    n_cached = len(stations) - len(todo)
    if n_cached:
        log.info(f"Resuming: {n_cached} already cached, {len(todo)} remaining")

    failed = []
    done   = n_cached

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_station, s): s for s in todo}
        for future in as_completed(futures):
            sid, rows, err = future.result()
            done += 1
            tag = f"[{done}/{len(stations)}]"
            if err:
                failed.append((sid, err))
                log.error(f"{tag} FAIL  {sid}: {err}")
            else:
                log.info(f"{tag} OK    {sid}: {rows:,} rows")

    if failed:
        log.warning(f"\n{len(failed)} station(s) failed: {[s for s, _ in failed]}")

    # Merge all cached parquets into one file
    pq_files = sorted(CACHE_DIR.glob("*.parquet"))
    if not pq_files:
        log.error("No cached parquets to merge.")
        return

    log.info(f"\nMerging {len(pq_files)} station files → {OUTPUT}")
    tmp = OUTPUT.with_suffix(".tmp.parquet")
    with pq.ParquetWriter(tmp, OUTPUT_SCHEMA, compression="snappy") as writer:
        for path in pq_files:
            writer.write_table(pq.read_table(path, schema=OUTPUT_SCHEMA))
    tmp.replace(OUTPUT)

    size_mb = OUTPUT.stat().st_size / 1e6
    log.info(f"Done. {OUTPUT}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
