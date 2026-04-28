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
  sunshine      — sunshine fraction of the hour (0–1)
  humidity      — relative humidity 2m (%)
  wind_speed    — wind speed at 10m (m/s)
  wind_gust     — wind gust at 10m (m/s)
  wind_dir      — wind direction at 10m (°)
  pressure      — surface pressure (hPa)
  snow_depth    — snow depth (m)
"""

import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

STAC_BASE  = "https://data.geo.admin.ch/api/stac/v0.9"
COLLECTION = "ch.meteoschweiz.ogd-smn"

# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE   = "https://archive-api.open-meteo.com/v1/archive"
_DATA      = Path(__file__).resolve().parent.parent / "data"
META_INPUT = _DATA / "station_metadata.parquet"
CACHE_DIR  = _DATA / "weather_hourly_cache"
OUTPUT     = _DATA / "weather_hourly.parquet"
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

        # Unit normalisations
        df["sunshine"]   = df["sunshine"]   / 3600.0   # seconds/hour → fraction 0-1
        df["wind_speed"] = df["wind_speed"] / 3.6      # km/h → m/s
        df["wind_gust"]  = df["wind_gust"]  / 3.6      # km/h → m/s
        # snow_depth: Open-Meteo returns metres — keep as metres (no conversion)

        ordered_cols = ["station_id", "timestamp"] + list(VAR_RENAME.values())
        table = pa.Table.from_pandas(df[ordered_cols], schema=OUTPUT_SCHEMA, preserve_index=False)
        pq.write_table(table, pq_path, compression="snappy")
        return sid, len(df), None

    except Exception as e:
        if pq_path.exists():
            pq_path.unlink()
        return sid, -1, str(e)

# ── Station discovery (generates station_metadata.parquet if missing) ──────────

def get_stations() -> list[dict]:
    if META_INPUT.exists():
        meta = pq.read_table(META_INPUT).to_pandas()
        log.info(f"Loaded {len(meta)} stations from {META_INPUT}")
        return meta[["station_id", "lat", "lon"]].dropna().to_dict("records")

    log.info("station_metadata.parquet not found — fetching from MeteoSwiss STAC API...")
    stations, url, params = [], f"{STAC_BASE}/collections/{COLLECTION}/items", {"limit": 100}
    while url:
        r = SESSION.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        for feature in data.get("features", []):
            coords = feature.get("geometry", {}).get("coordinates", [])
            props  = feature.get("properties", {})
            if len(coords) >= 2:
                stations.append({
                    "station_id": feature["id"],
                    "name":       props.get("title", feature["id"]),
                    "lon":        coords[0],
                    "lat":        coords[1],
                    "alt":        coords[2] if len(coords) > 2 else None,
                })
        url    = next((l["href"] for l in data.get("links", []) if l.get("rel") == "next"), None)
        params = {}

    meta_df = pd.DataFrame(stations)
    pq.write_table(pa.Table.from_pandas(meta_df, preserve_index=False), META_INPUT, compression="snappy")
    log.info(f"Saved {META_INPUT} ({len(stations)} stations)")
    return meta_df[["station_id", "lat", "lon"]].dropna().to_dict("records")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    CACHE_DIR.mkdir(exist_ok=True)

    stations = get_stations()
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
