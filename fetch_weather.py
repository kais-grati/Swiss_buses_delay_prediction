#!/usr/bin/env python3
"""
Fetch MeteoSwiss 10-minute SwissMetNet data for all Swiss automatic stations, 2025.

Source:  MeteoSwiss Open Government Data
API:     https://data.geo.admin.ch/api/stac/v0.9/collections/ch.meteoschweiz.ogd-smn
Output:  weather_2025.parquet  +  station_metadata.parquet

Variables per station (10-min observations):
  temperature   — air temperature 2m (°C)
  precipitation — mm per 10 min
  sunshine      — sunshine duration (min per 10 min)
  humidity      — relative humidity 2m (%)
  wind_speed    — scalar mean at 10m (m/s)
  wind_gust     — peak gust at 10m (m/s)
  wind_dir      — vector mean direction at 10m (°)
  pressure      — barometric pressure at station level (hPa)
  snow_depth    — total snow depth (cm)

Timestamps are in UTC.
"""

import io
import re
import sys
import time
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Configuration ─────────────────────────────────────────────────────────────

STAC_BASE   = "https://data.geo.admin.ch/api/stac/v0.9"
COLLECTION  = "ch.meteoschweiz.ogd-smn"
CACHE_DIR   = Path("weather_cache")       # raw CSVs + per-station parquets
OUTPUT      = Path("weather_2025.parquet")
META_OUTPUT = Path("station_metadata.parquet")
YEAR        = 2025
MAX_WORKERS = 8
TIMEOUT     = 120   # seconds per HTTP request

# MeteoSwiss parameter codes → friendly column names
# Reference: https://www.meteoswiss.admin.ch (IDENT parameter list)
VAR_MAP = {
    "tre200d0": "temperature",    # Daily mean air temperature 2m (°C)
    "rre150d0": "precipitation",  # Daily precipitation (mm)
    "sre000d0": "sunshine",       # Daily sunshine duration (min)
    "ure200d0": "humidity",       # Daily mean relative humidity 2m (%)
    "fu3010d0": "wind_speed",     # Daily mean wind speed at 10m (m/s)
    "fu3010d1": "wind_gust",      # Daily max wind gust at 10m (m/s)
    "dkl010d0": "wind_dir",       # Daily mean wind direction at 10m (°)
    "prestad0":  "pressure",      # Daily mean pressure at station level (hPa)
    "htoautd0": "snow_depth",     # Daily snow depth (cm)
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

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── HTTP helpers ──────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "swiss-bus-delay-ml/1.0 (research)"


def get_json(url: str, params: dict = None, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            log.warning(f"  Retry {attempt+1}/{retries} ({wait}s) for {url}: {e}")
            time.sleep(wait)


def get_bytes(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=TIMEOUT, stream=True)
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            log.warning(f"  Retry {attempt+1}/{retries} ({wait}s) for {url}: {e}")
            time.sleep(wait)

# ── Station discovery ─────────────────────────────────────────────────────────

def get_all_stations() -> list[dict]:
    """
    Walk the STAC API and return a list of station dicts:
      {station_id, name, lat, lon, alt, asset_url}
    """
    stations = []
    url    = f"{STAC_BASE}/collections/{COLLECTION}/items"
    params = {"limit": 100}

    while url:
        data = get_json(url, params)

        for feature in data.get("features", []):
            station_id = feature["id"]
            props      = feature.get("properties", {})
            coords     = feature.get("geometry", {}).get("coordinates", [])

            # Find the best asset: prefer "historical", then any CSV href
            assets    = feature.get("assets", {})
            asset_url = None
            for key in ("historical", "smn", "data", "csv"):
                if key in assets:
                    asset_url = assets[key].get("href")
                    break
            if asset_url is None:
                for v in assets.values():
                    href = v.get("href", "")
                    if href.lower().endswith(".csv"):
                        asset_url = href
                        break

            if asset_url:
                stations.append({
                    "station_id": station_id,
                    "name":       props.get("title", station_id),
                    "lon":        coords[0] if len(coords) > 0 else None,
                    "lat":        coords[1] if len(coords) > 1 else None,
                    "alt":        coords[2] if len(coords) > 2 else None,
                    "asset_url":  asset_url,
                })
            else:
                log.warning(f"No CSV asset found for station {station_id} — skipping")

        # Follow STAC pagination
        next_link = next(
            (lnk["href"] for lnk in data.get("links", []) if lnk.get("rel") == "next"),
            None,
        )
        url    = next_link
        params = {}

    log.info(f"Discovered {len(stations)} stations")
    return stations

# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_smn_csv(raw: bytes, station_id: str) -> pd.DataFrame:
    """
    Parse a MeteoSwiss SMN CSV file.

    Expected format (semicolon-delimited, first row is header):
      stn;time;tre200s0;rre150z0;...
      KLO;202501010000;3.5;0.0;...

    Timestamp formats handled:
      - YYYYMMDDhhmm     (12 digits)
      - YYYY-MM-DDTHH:MM (ISO 8601)
      - DD.MM.YYYY HH:MM (Swiss)

    Missing values ("-", "–", empty) become NaN.
    Rows outside YEAR are dropped.
    """
    text = raw.decode("utf-8", errors="replace")

    df = pd.read_csv(
        io.StringIO(text),
        sep=";",
        dtype=str,
        na_values=["-", "–", "", "NA", "na"],
        keep_default_na=False,
    )
    df.columns = [c.strip().lower() for c in df.columns]

    # Locate the date/time column
    date_col = next(
        (c for c in df.columns if c in ("time", "date", "datum", "datetime", "reference_timestamp")),
        None,
    )
    if date_col is None:
        # Fall back: first column that looks like a timestamp
        for c in df.columns:
            sample = df[c].dropna().head(1)
            if not sample.empty and re.match(r"^\d{8,}", str(sample.iloc[0])):
                date_col = c
                break
    if date_col is None:
        raise ValueError(f"[{station_id}] Cannot find timestamp column. Columns: {list(df.columns)}")

    # Parse timestamp — detect format from first non-null value
    sample = df[date_col].dropna().iloc[0] if not df[date_col].dropna().empty else ""
    if re.match(r"^\d{12}$", sample):
        df["timestamp"] = pd.to_datetime(df[date_col], format="%Y%m%d%H%M", errors="coerce")
    elif re.match(r"^\d{10}$", sample):
        df["timestamp"] = pd.to_datetime(df[date_col], format="%Y%m%d%H%M", errors="coerce")
    elif re.match(r"^\d{2}\.\d{2}\.\d{4}", sample):
        df["timestamp"] = pd.to_datetime(df[date_col], format="%d.%m.%Y %H:%M", errors="coerce")
    elif re.match(r"^\d{4}-\d{2}-\d{2}T", sample):
        df["timestamp"] = pd.to_datetime(df[date_col], format="ISO8601", errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce")

    # Filter to target year and reset index so all subsequent assignments align
    df = df[df["timestamp"].dt.year == YEAR].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    # Build output DataFrame
    out = pd.DataFrame()
    out["timestamp"]  = df["timestamp"]
    out["station_id"] = station_id

    for code, col_name in VAR_MAP.items():
        if code in df.columns:
            out[col_name] = pd.to_numeric(df[code], errors="coerce").astype("float32")
        else:
            out[col_name] = pd.array([pd.NA] * len(df), dtype="Float32").astype("float32")

    return out.reset_index(drop=True)

# ── Per-station fetch ─────────────────────────────────────────────────────────

def fetch_station(station: dict) -> tuple[str, int, str | None]:
    """
    Download, parse, and cache one station.
    Returns (station_id, row_count, error_or_None).
    """
    sid        = station["station_id"]
    csv_path   = CACHE_DIR / f"{sid}.csv"
    pq_path    = CACHE_DIR / f"{sid}.parquet"

    try:
        # Load from disk cache if already downloaded
        if csv_path.exists():
            raw = csv_path.read_bytes()
        else:
            raw = get_bytes(station["asset_url"])
            csv_path.write_bytes(raw)

        df = parse_smn_csv(raw, sid)

        if df.empty:
            return sid, 0, None

        table = pa.Table.from_pandas(df, schema=OUTPUT_SCHEMA, preserve_index=False)
        pq.write_table(table, pq_path, compression="snappy")
        return sid, len(df), None

    except Exception as e:
        # Remove partial files so resume works cleanly
        for p in (csv_path, pq_path):
            if p.exists():
                p.unlink()
        return sid, -1, str(e)

# ── Main ──────────────────────────────────────────────────────────────────────

def main(test_mode: bool = False):
    CACHE_DIR.mkdir(exist_ok=True)

    # 1. Discover stations
    stations = get_all_stations()
    if not stations:
        log.error("No stations found. Check STAC API or collection name.")
        sys.exit(1)

    if test_mode:
        stations = stations[:3]
        log.info("TEST MODE: processing first 3 stations only")

    # 2. Save station metadata (coordinates, altitude)
    meta_df = pd.DataFrame(stations).drop(columns=["asset_url"])
    pq.write_table(
        pa.Table.from_pandas(meta_df, preserve_index=False),
        META_OUTPUT,
        compression="snappy",
    )
    log.info(f"Saved {META_OUTPUT} ({len(meta_df)} stations)")

    # 3. Skip already-converted stations (resume support)
    todo = [s for s in stations if not (CACHE_DIR / f"{s['station_id']}.parquet").exists()]
    n_done = len(stations) - len(todo)
    if n_done:
        log.info(f"Resuming: {n_done} already done, {len(todo)} remaining")

    # 4. Parallel download & parse
    failed    = []
    completed = n_done
    total     = len(stations)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_station, s): s for s in todo}
        for future in as_completed(futures):
            sid, rows, error = future.result()
            completed += 1
            tag = f"[{completed}/{total}]"
            if error:
                failed.append((sid, error))
                log.error(f"{tag} FAIL  {sid}: {error}")
            elif rows == 0:
                log.info(f"{tag} EMPTY {sid}: no {YEAR} data in file")
            else:
                log.info(f"{tag} OK    {sid}: {rows:,} rows")

    if failed:
        log.warning(f"\n{len(failed)} station(s) failed:")
        for sid, err in failed:
            log.warning(f"  {sid}: {err}")

    # 5. Merge all per-station parquets into one file
    pq_files = sorted(CACHE_DIR.glob("*.parquet"))
    if not pq_files:
        log.error("No station parquet files to merge.")
        sys.exit(1)

    tmp = OUTPUT.with_suffix(".tmp.parquet")
    log.info(f"\nMerging {len(pq_files)} station files → {OUTPUT}...")

    with pq.ParquetWriter(tmp, OUTPUT_SCHEMA, compression="snappy") as writer:
        for i, path in enumerate(pq_files, 1):
            writer.write_table(pq.read_table(path, schema=OUTPUT_SCHEMA))
            if i % 20 == 0 or i == len(pq_files):
                log.info(f"  Merged {i}/{len(pq_files)}")

    tmp.replace(OUTPUT)

    size_mb = OUTPUT.stat().st_size / 1e6
    log.info(f"\nDone. {OUTPUT}: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch MeteoSwiss 10-min weather data for Switzerland")
    parser.add_argument("--test", action="store_true", help="Run on first 3 stations only")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"Parallel workers (default: {MAX_WORKERS})")
    args = parser.parse_args()

    MAX_WORKERS = args.workers
    main(test_mode=args.test)
