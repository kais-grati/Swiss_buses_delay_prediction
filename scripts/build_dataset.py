"""
Build dataset.parquet and dataset_with_weather.parquet in a single run.

Merges the full 5-step data pipeline into one script:
  1. Read daily CSVs from ZIP (no disk extraction)
  2. Translate German headers → English
  3. Filter to bus rows only (PRODUCT_ID == "Bus")
  4. Keep only relevant columns
  5. Drop cancelled / pass-through / additional-trip rows
  6. Compute delays + cyclical time features
  7. Drop outlier delays (> 30 min or < -2 min)
  8. Add is_public_holiday (canton-aware Swiss holidays)
  9. Compute prev_stop_delay (lag within trip) + dist_to_prev_stop (LV95 distance)
  10. Write dataset.parquet (snappy-compressed)
  11. Fetch hourly weather from Open-Meteo (if not already cached)
  12. Join weather to dataset + drop unmatched rows → dataset_with_weather.parquet

RAM-efficient: one CSV in memory per worker at a time.
Resume-safe: skips already-converted CSVs and cached weather stations.
"""

import os
import sys
import io
import shutil
import sqlite3
import time
import threading
import argparse
import subprocess
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import duckdb
import holidays
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import requests
from multiprocessing import Pool
from pyproj import Transformer
from scipy.spatial import KDTree

ROOT         = Path(__file__).resolve().parent.parent
ZIP_DIR      = ROOT / "data" / "compressed_data"
TMP_DIR      = ROOT / "data" / "build_dataset_tmp"
OUTPUT       = ROOT / "data" / "swiss_bus_2025.parquet"
TMP_OUTPUT   = Path(str(OUTPUT) + ".tmp")
STATION_DATA = ROOT / "data" / "station_data.parquet"
DATASET_IN   = OUTPUT  # used by weather-join phase

TAU = 2 * np.pi

# Outlier delay bounds — rows outside these thresholds are dropped
DELAY_MIN = -120   # 2 min early (buses are prohibited from departing early)
DELAY_MAX = 1800   # 30 min late (beyond this, likely data corruption)

# Rows where the arrival/departure was actually observed (not estimated)
VALID_STATUSES = {"REAL"}

# ── Stop → canton map + holiday dates (computed once at import) ──

def _build_stop_canton_map():
    """Return dict {stop_id: canton_abbreviation} from station_data.parquet."""
    station_data = ROOT / "data" / "station_data.parquet"
    if not station_data.exists():
        print("WARNING: station_data.parquet not found — holidays will be False")
        return {}
    stops = pq.read_table(station_data, columns=["number", "cantonabbreviation"]).to_pandas()
    stops["stop_id"] = pd.to_numeric(stops["number"], errors="coerce").astype("Int64")
    stops = stops.dropna(subset=["stop_id"])
    return dict(zip(stops["stop_id"].astype("int32"), stops["cantonabbreviation"]))


def _build_holiday_dates():
    """Return dict {canton: set(dates)} of Swiss public holidays for 2025."""
    holiday_dates = {}
    for canton in holidays.Switzerland.subdivisions:
        holiday_dates[canton] = set(
            holidays.Switzerland(subdiv=canton, years=[2025])
        )
    return holiday_dates


def _build_stop_coords():
    """Return {stop_id: (lv95east, lv95north)} from station_data.parquet."""
    station_data = ROOT / "data" / "station_data.parquet"
    if not station_data.exists():
        return {}
    stops = pq.read_table(
        station_data, columns=["number", "lv95east", "lv95north"]
    ).to_pandas().dropna(subset=["lv95east", "lv95north"])
    stops["stop_id"] = pd.to_numeric(stops["number"], errors="coerce").astype("Int64")
    stops = stops.dropna(subset=["stop_id"])
    return dict(zip(
        stops["stop_id"].astype("int32"),
        zip(stops["lv95east"].astype("float64"), stops["lv95north"].astype("float64")),
    ))


GPKG_PATH = ROOT / "data" / "traffic" / "belastung-personenverkehr-strasse_2056.gpkg"
TRAFFIC_MAP_CACHE = ROOT / "data" / "_traffic_map_cache.parquet"


def _build_stop_traffic_map():
    """Return {stop_id: {traffic_dtv, traffic_msp, traffic_asp,
                         traffic_heavy_share, traffic_peak_ratio}}
    by spatial-joining each stop to the nearest road segment in the Swiss
    road-traffic GPKG (LV95 rtree index).  Cached to a parquet file so it
    only runs once.
    """
    if TRAFFIC_MAP_CACHE.exists():
        cached = pq.read_table(TRAFFIC_MAP_CACHE).to_pandas()
        out = {}
        for row in cached.itertuples(index=False):
            out[row.stop_id] = {
                "traffic_dtv":         row.traffic_dtv,
                "traffic_msp":         row.traffic_msp,
                "traffic_asp":         row.traffic_asp,
                "traffic_heavy_share": row.traffic_heavy_share,
                "traffic_peak_ratio":  row.traffic_peak_ratio,
            }
        print(f"Loaded {len(out):,} traffic-mapped stops from cache")
        return out

    if not GPKG_PATH.exists():
        print("WARNING: traffic GPKG not found — traffic features will be NaN")
        return {}

    coords = _build_stop_coords()
    if not coords:
        return {}

    print(f"Building stop → road traffic map ({len(coords):,} stops)...")
    t0 = time.time()

    conn = sqlite3.connect(str(GPKG_PATH))
    cur = conn.cursor()

    search_radii = [500.0, 1000.0, 2000.0, 5000.0]
    records = []
    n = len(coords)
    no_match = 0

    for i, (stop_id, (east, north)) in enumerate(coords.items()):
        nearest_dtv = nearest_msp = nearest_asp = nearest_lw = nearest_lz = None
        found = False

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

            cur.execute(f"""
                SELECT DTV_FZG, MSP_FZG, ASP_FZG, DTV_LW, DTV_LZ
                FROM Personen_Gueterverkehr_Strasse
                WHERE id = {row[0]}
            """)
            traffic = cur.fetchone()
            if traffic is not None:
                nearest_dtv, nearest_msp, nearest_asp, nearest_lw, nearest_lz = traffic
                found = True
                break

        if not found:
            no_match += 1
            continue

        heavy_share = (nearest_lw + nearest_lz) / nearest_dtv if nearest_dtv > 0 else 0.0
        peak_ratio  = nearest_asp / nearest_dtv if nearest_dtv > 0 else 0.0

        records.append({
            "stop_id":            int(stop_id),
            "traffic_dtv":        float(nearest_dtv),
            "traffic_msp":        float(nearest_msp),
            "traffic_asp":        float(nearest_asp),
            "traffic_heavy_share": float(heavy_share),
            "traffic_peak_ratio":  float(peak_ratio),
        })

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  ... {i+1}/{n} stops ({(i+1)*100/n:.0f}%, "
                  f"{rate:.0f} stops/s)")

    conn.close()

    df = pd.DataFrame(records)
    # Ensure int32 for join compatibility
    df["stop_id"] = df["stop_id"].astype("int32")
    for col in ["traffic_dtv", "traffic_msp", "traffic_asp",
                "traffic_heavy_share", "traffic_peak_ratio"]:
        df[col] = df[col].astype("float32")

    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        TRAFFIC_MAP_CACHE,
        compression="snappy",
    )

    elapsed = time.time() - t0
    print(f"  Matched {len(df):,}/{n:,} stops ({no_match} no-match) "
          f"in {elapsed:.1f}s ({n/elapsed:.0f} stops/s)")

    out = {}
    for row in df.itertuples(index=False):
        out[row.stop_id] = {
            "traffic_dtv":         row.traffic_dtv,
            "traffic_msp":         row.traffic_msp,
            "traffic_asp":         row.traffic_asp,
            "traffic_heavy_share": row.traffic_heavy_share,
            "traffic_peak_ratio":  row.traffic_peak_ratio,
        }
    return out


STOP_CANTON   = _build_stop_canton_map()
STOP_COORDS   = _build_stop_coords()
STOP_TRAFFIC  = _build_stop_traffic_map()
HOLIDAY_DATES = _build_holiday_dates()

# German → English column mapping (full set, 21 columns)
COLUMN_MAPPING = {
    "BETRIEBSTAG":          "DATE",
    "FAHRT_BEZEICHNER":     "TRIP_ID",
    "BETREIBER_ID":         "OPERATOR_ID",
    "BETREIBER_ABK":        "OPERATOR_ABB",
    "BETREIBER_NAME":       "OPERATOR_NAME",
    "PRODUKT_ID":           "PRODUCT_ID",
    "LINIEN_ID":            "LINE_ID",
    "LINIEN_TEXT":          "LINE_NAME",
    "UMLAUF_ID":            "CIRCULATION_ID",
    "VERKEHRSMITTEL_TEXT":  "TRANSPORT_TYPE",
    "ZUSATZFAHRT_TF":       "ADDITIONAL_TRIP",
    "FAELLT_AUS_TF":        "CANCELLED",
    "BPUIC":                "BPUIC",
    "HALTESTELLEN_NAME":    "STOP_NAME",
    "ANKUNFTSZEIT":         "ARRIVAL_TIME",
    "AN_PROGNOSE":          "ARRIVAL_FORECAST",
    "AN_PROGNOSE_STATUS":   "ARRIVAL_FORECAST_STATUS",
    "ABFAHRTSZEIT":         "DEPARTURE_TIME",
    "AB_PROGNOSE":          "DEPARTURE_FORECAST",
    "AB_PROGNOSE_STATUS":   "DEPARTURE_FORECAST_STATUS",
    "DURCHFAHRT_TF":        "PASS_THROUGH",
}

# Columns to keep after translation + bus filter
KEEP_COLS = [
    "DATE",
    "TRIP_ID",
    "OPERATOR_ABB",
    "LINE_NAME",
    "BPUIC",
    "STOP_NAME",
    "ARRIVAL_TIME",
    "ARRIVAL_FORECAST",
    "ARRIVAL_FORECAST_STATUS",
    "DEPARTURE_TIME",
    "DEPARTURE_FORECAST",
    "DEPARTURE_FORECAST_STATUS",
    "ADDITIONAL_TRIP",
    "CANCELLED",
    "PASS_THROUGH",
]

# PyArrow read options — semicolon-delimited CSV
PARSE_OPTIONS   = pa_csv.ParseOptions(delimiter=";")
READ_OPTIONS    = pa_csv.ReadOptions(block_size=64 * 1024 * 1024)
CONVERT_OPTIONS = pa_csv.ConvertOptions(
    true_values=["true", "True"],
    false_values=["false", "False"],
    null_values=["", "NA"],
)

# Parquet only supports ms/us/ns timestamp resolution — "s" is auto-converted
# to "ms" on write.  We declare "ms" here so the schema check passes.
EXPECTED_SCHEMA = pa.schema([
    ("timestamp",         pa.timestamp("ms")),
    ("time_sin",          pa.float32()),
    ("time_cos",          pa.float32()),
    ("dow_sin",           pa.float32()),
    ("dow_cos",           pa.float32()),
    ("month_sin",         pa.float32()),
    ("month_cos",         pa.float32()),
    ("is_weekend",        pa.bool_()),
    ("operator",          pa.large_string()),
    ("line",              pa.large_string()),
    ("stop_id",           pa.int32()),
    ("stop_name",         pa.large_string()),
    ("additional_trip",   pa.bool_()),
    ("pass_through",      pa.bool_()),
    ("arrival_delay_s",    pa.int32()),
    ("departure_delay_s",  pa.int32()),
    ("is_public_holiday",  pa.bool_()),
    ("trip_id",            pa.large_string()),
    ("trip_stop_index",     pa.int16()),
    ("prev_stop_delay",      pa.int32()),
    ("dist_to_prev_stop",   pa.float32()),
    ("traffic_dtv",         pa.float32()),
    ("traffic_heavy_share", pa.float32()),
    ("traffic_peak_ratio",  pa.float32()),
    ("traffic_peak",        pa.float32()),
])


def coerce_bool(col):
    """Convert a column to bool regardless of source type."""
    if pd.api.types.is_bool_dtype(col):
        return col.fillna(False)
    if pd.api.types.is_numeric_dtype(col):
        return col.fillna(0).astype(bool)
    return col.fillna("").astype(str).str.lower().isin({"true", "1"})


def build_job_list():
    """Return list of (zip_path, csv_name) tuples for all CSVs across all ZIPs."""
    jobs = []
    zip_files = sorted(
        f for f in os.listdir(ZIP_DIR) if f.endswith(".zip")
    )
    for zf_name in zip_files:
        zp = os.path.join(ZIP_DIR, zf_name)
        try:
            with ZipFile(zp) as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith(".csv"):
                        jobs.append((zp, name))
        except BadZipFile:
            print(f"WARNING: skipping corrupt ZIP: {zf_name}", file=sys.stderr)
    return jobs


def read_csv_from_zip(zip_path, csv_name):
    """Read a single CSV from inside a ZIP archive into a pandas DataFrame.

    Uses PyArrow's CSV parser (11× faster than pandas read_csv) by reading
    the raw bytes into a BufferReader.  Falls back to pandas if PyArrow
    cannot handle the file (e.g. malformed rows).

    Falls back to the system `unzip` CLI for Deflate64 entries (compress_type 9),
    which Python's zipfile module cannot decompress.
    """
    with ZipFile(zip_path) as zf:
        try:
            raw_bytes = zf.read(csv_name)
        except NotImplementedError:
            # Deflate64 compression — Python's zipfile can't handle it
            result = subprocess.run(
                ["unzip", "-p", zip_path, csv_name],
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"unzip failed for {csv_name}: {result.stderr.decode()}"
                )
            raw_bytes = result.stdout

    reader = pa.BufferReader(raw_bytes)
    try:
        table = pa_csv.read_csv(
            reader,
            read_options=READ_OPTIONS,
            parse_options=PARSE_OPTIONS,
            convert_options=CONVERT_OPTIONS,
        )
        df = table.to_pandas()
        del table
    except Exception:
        # PyArrow is strict — fall back to pandas for malformed CSVs
        df = pd.read_csv(
            io.BytesIO(raw_bytes), sep=";", dtype=str, encoding="utf-8",
        )
    del raw_bytes
    return df


def process_csv(args):
    """Process a single CSV from a ZIP archive → temp parquet file.

    Returns (csv_name, rows, size_bytes, error_string_or_None).
    """
    zip_path, csv_name = args
    dst = os.path.join(TMP_DIR, csv_name.replace(".csv", ".parquet"))

    try:
        df = read_csv_from_zip(zip_path, csv_name)

        # Translate German → English headers (only columns that exist)
        rename_map = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
        df.rename(columns=rename_map, inplace=True)

        # Filter to bus rows only
        if "PRODUCT_ID" not in df.columns:
            return csv_name, 0, 0, "missing PRODUCT_ID column"
        df = df[df["PRODUCT_ID"].str.lower() == "bus"]

        if len(df) == 0:
            return csv_name, 0, 0, None

        # Keep only the 15 relevant columns (only those present)
        present_cols = [c for c in KEEP_COLS if c in df.columns]
        missing = set(KEEP_COLS) - set(present_cols)
        if missing:
            return csv_name, 0, 0, f"missing columns: {missing}"
        df = df[present_cols].reset_index(drop=True)

        # ── Feature computation ──

        # Drop cancelled, pass-through, and additional trips
        df = df[~coerce_bool(df["CANCELLED"])].reset_index(drop=True)
        df = df[~coerce_bool(df["PASS_THROUGH"])].reset_index(drop=True)
        df = df[~coerce_bool(df["ADDITIONAL_TRIP"])].reset_index(drop=True)

        if len(df) == 0:
            return csv_name, 0, 0, None

        # Keep only REAL observations
        arr_real = df["ARRIVAL_FORECAST_STATUS"].isin(VALID_STATUSES)
        dep_real = df["DEPARTURE_FORECAST_STATUS"].isin(VALID_STATUSES)

        arr_sched  = pd.to_datetime(df["ARRIVAL_TIME"],           format="%d.%m.%Y %H:%M",    errors="coerce")
        dep_sched  = pd.to_datetime(df["DEPARTURE_TIME"],         format="%d.%m.%Y %H:%M",    errors="coerce")
        arr_actual = pd.to_datetime(df["ARRIVAL_FORECAST"].where(arr_real),  format="%d.%m.%Y %H:%M:%S", errors="coerce")
        dep_actual = pd.to_datetime(df["DEPARTURE_FORECAST"].where(dep_real), format="%d.%m.%Y %H:%M:%S", errors="coerce")

        # Time-of-day delay in seconds, wrapped to (−12h, +12h]
        def time_delay_s(sched, actual):
            s = sched.dt.hour * 3600 + sched.dt.minute * 60
            a = actual.dt.hour * 3600 + actual.dt.minute * 60 + actual.dt.second
            delta = a - s
            delta = delta.where(delta > -43200, delta + 86400)
            delta = delta.where(delta <= 43200, delta - 86400)
            return delta

        arrival_delay_s   = time_delay_s(arr_sched, arr_actual)
        departure_delay_s = time_delay_s(dep_sched, dep_actual)

        # Keep rows with at least one valid delay AND within outlier bounds.
        # Null delays pass the bounds check (only one delay needs to be valid).
        has_delay    = arrival_delay_s.notna() | departure_delay_s.notna()
        arr_in_range = arrival_delay_s.isna()   | arrival_delay_s.between(DELAY_MIN, DELAY_MAX)
        dep_in_range = departure_delay_s.isna() | departure_delay_s.between(DELAY_MIN, DELAY_MAX)
        mask         = has_delay & arr_in_range & dep_in_range

        if not mask.any():
            return csv_name, 0, 0, None

        df                = df[mask].reset_index(drop=True)
        arrival_delay_s   = arrival_delay_s[mask].reset_index(drop=True)
        departure_delay_s = departure_delay_s[mask].reset_index(drop=True)
        arr_sched         = arr_sched[mask].reset_index(drop=True)

        # Time-of-day features (string slicing avoids full datetime parse)
        time_str = df["ARRIVAL_TIME"].fillna(df["DEPARTURE_TIME"])
        hour     = pd.to_numeric(time_str.str[11:13], errors="coerce").fillna(0).astype("int16")
        minute   = pd.to_numeric(time_str.str[14:16], errors="coerce").fillna(0).astype("int16")
        time_min = hour * 60 + minute

        # Date features
        month = pd.to_numeric(df["DATE"].str[3:5], errors="coerce").fillna(1).astype("int16")
        date  = pd.to_datetime(df["DATE"], format="%d.%m.%Y", errors="coerce")
        dow   = date.dt.dayofweek.fillna(0).astype("int16")

        # Build timestamp column — convert to Unix ms so Arrow gets timestamp[ms]
        ts_col = arr_sched.fillna(
            pd.to_datetime(df["DEPARTURE_TIME"], format="%d.%m.%Y %H:%M", errors="coerce")
        )
        ts_col = ts_col.astype("int64") // 1_000  # μs → ms

        # ── Public holidays ──
        is_holiday = pd.Series(False, index=df.index)
        if STOP_CANTON:
            stop_ids = df["BPUIC"].astype("int32")
            canton_s = stop_ids.map(STOP_CANTON)
            date_only = date.dt.date
            for canton, h_dates in HOLIDAY_DATES.items():
                cmask = canton_s == canton
                if cmask.any():
                    is_holiday[cmask] = date_only[cmask].isin(h_dates)

        result = pd.DataFrame({
            "timestamp":         ts_col,
            "time_sin":          np.sin(TAU * time_min / 1440).astype("float32"),
            "time_cos":          np.cos(TAU * time_min / 1440).astype("float32"),
            "dow_sin":           np.sin(TAU * dow / 7).astype("float32"),
            "dow_cos":           np.cos(TAU * dow / 7).astype("float32"),
            "month_sin":         np.sin(TAU * month / 12).astype("float32"),
            "month_cos":         np.cos(TAU * month / 12).astype("float32"),
            "is_weekend":        dow >= 5,
            "operator":          df["OPERATOR_ABB"],
            "line":              df["LINE_NAME"],
            "stop_id":           df["BPUIC"].astype("int32"),
            "stop_name":         df["STOP_NAME"],
            "additional_trip":   coerce_bool(df["ADDITIONAL_TRIP"]),
            "pass_through":      coerce_bool(df["PASS_THROUGH"]),
            "arrival_delay_s":   arrival_delay_s.astype("Int32"),
            "departure_delay_s": departure_delay_s.astype("Int32"),
            "is_public_holiday": is_holiday,
            "trip_id":           df["TRIP_ID"],
        })

        # prev_stop_delay: lag of arrival_delay_s within each (date, trip_id)
        # group, sorted by timestamp.  First stop of each trip = NaN (Int32).
        result = result.sort_values("timestamp")
        ts_dates = pd.to_datetime(result["timestamp"], unit="ms").dt.date
        result["prev_stop_delay"] = (
            result.groupby([ts_dates, "trip_id"], sort=False)["arrival_delay_s"]
            .shift(1)
            .astype("Int32")
        )

        # Position of each stop within its trip (0 = first stop)
        result["trip_stop_index"] = result.groupby(
            [ts_dates, "trip_id"], sort=False
        ).cumcount().astype("int16")

        # dist_to_prev_stop: LV95 Euclidean distance (m) to previous stop in same trip
        if STOP_COORDS:
            coords_east, coords_north = zip(*[
                STOP_COORDS.get(sid, (np.nan, np.nan))
                for sid in result["stop_id"]
            ])
            coords_east = np.array(coords_east, dtype="float64")
            coords_north = np.array(coords_north, dtype="float64")

            prev_trip = None
            prev_east = prev_north = np.nan
            distances = np.full(len(result), np.nan, dtype="float32")

            for i in range(len(result)):
                trip = result["trip_id"].iloc[i]
                if trip != prev_trip:
                    distances[i] = np.nan
                else:
                    de = coords_east[i] - prev_east
                    dn = coords_north[i] - prev_north
                    distances[i] = np.sqrt(de * de + dn * dn) if not (
                        np.isnan(de) or np.isnan(dn)
                    ) else np.nan
                prev_trip = trip
                prev_east = coords_east[i]
                prev_north = coords_north[i]

            result["dist_to_prev_stop"] = distances
        else:
            result["dist_to_prev_stop"] = np.nan

        # Drop first-stop-of-trip rows and zero-distance rows (duplicate stop)
        result = result[
            result["prev_stop_delay"].notna()
            & ~(result["dist_to_prev_stop"].fillna(-1) == 0.0)
        ].reset_index(drop=True)

        if len(result) == 0:
            return csv_name, 0, 0, None

        # ── Traffic features (nearest-road spatial join on GPKG) ──
        if STOP_TRAFFIC:
            # Drop rows for stops with no matching road segment
            traffic_ids = {int(sid) for sid in STOP_TRAFFIC}
            keep_mask = [int(sid) in traffic_ids for sid in result["stop_id"]]
            keep_mask_arr = np.array(keep_mask)
            result = {
                col: values[keep_mask_arr]
                for col, values in result.items()
            }

            traffic_rows = [STOP_TRAFFIC[int(sid)] for sid in result["stop_id"]]
            result["traffic_dtv"] = np.array(
                [t["traffic_dtv"] for t in traffic_rows], dtype="float32"
            )
            result["traffic_heavy_share"] = np.array(
                [t["traffic_heavy_share"] for t in traffic_rows], dtype="float32"
            )
            result["traffic_peak_ratio"] = np.array(
                [t["traffic_peak_ratio"] for t in traffic_rows], dtype="float32"
            )
            # traffic_peak: morning peak for AM, evening peak for PM
            # timestamp is ms since epoch (Swiss-local-time-as-UTC)
            msp_arr = np.array(
                [t["traffic_msp"] for t in traffic_rows], dtype="float32"
            )
            asp_arr = np.array(
                [t["traffic_asp"] for t in traffic_rows], dtype="float32"
            )
            ts_hour = pd.to_datetime(result["timestamp"], unit="ms").dt.hour
            result["traffic_peak"] = np.where(
                ts_hour.values < 12, msp_arr, asp_arr
            ).astype("float32")
        else:
            for col in ["traffic_dtv", "traffic_peak",
                         "traffic_heavy_share", "traffic_peak_ratio"]:
                result[col] = np.nan

        n_rows = len(result)

        out_table = pa.Table.from_pandas(result, preserve_index=False)
        # Cast to expected schema — handles int64→timestamp[ms] etc.
        out_table = out_table.cast(EXPECTED_SCHEMA)
        del result, df
        pq.write_table(out_table, dst, compression="snappy")
        size = os.path.getsize(dst)
        del out_table

        return csv_name, n_rows, size, None

    except Exception as e:
        if os.path.exists(dst):
            os.remove(dst)
        return csv_name, -1, 0, str(e)


def run_tests():
    """Smoke-test on the first CSV from the first ZIP."""
    print("=== Running test on first available CSV ===\n")

    jobs = build_job_list()
    if not jobs:
        print("No ZIP files found.")
        return False

    os.makedirs(TMP_DIR, exist_ok=True)
    zip_path, csv_name = jobs[0]
    print(f"Testing: {csv_name} from {os.path.basename(zip_path)}")

    name, rows, size, error = process_csv((zip_path, csv_name))

    if error:
        print(f"  FAIL  process_csv raised: {error}")
        shutil.rmtree(TMP_DIR)
        return False

    dst = os.path.join(TMP_DIR, name.replace(".csv", ".parquet"))
    df_out = pq.read_table(dst).to_pandas()
    actual_schema = pq.read_schema(dst)

    all_ok = True

    # Schema check
    schema_ok = actual_schema.equals(EXPECTED_SCHEMA)
    print(f"  {'PASS' if schema_ok else 'FAIL'}  Schema correct")
    if not schema_ok:
        print(f"       Expected: {EXPECTED_SCHEMA}")
        print(f"       Got:      {actual_schema}")
        all_ok = False

    # Row count
    count_ok = rows > 0
    print(f"  {'PASS' if count_ok else 'FAIL'}  Row count {rows}")
    if not count_ok:
        all_ok = False

    # Cyclical features in [-1, 1]
    for col in ["time_sin", "time_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        mn, mx = df_out[col].min(), df_out[col].max()
        ok = -1.0 <= mn and mx <= 1.0
        print(f"  {'PASS' if ok else 'FAIL'}  {col} in [-1, 1]  (min={mn:.4f} max={mx:.4f})")
        if not ok:
            all_ok = False

    # No NaN in feature columns
    feature_cols = ["time_sin", "time_cos", "dow_sin", "dow_cos",
                    "month_sin", "month_cos", "is_weekend", "is_public_holiday",
                    "operator", "line", "stop_id", "additional_trip", "pass_through"]
    for col in feature_cols:
        nans = df_out[col].isna().sum()
        if nans > 0:
            print(f"  FAIL  {col} has {nans} NaN values")
            all_ok = False
    print(f"  PASS  No NaN in feature columns")

    # Delay range — within outlier bounds [DELAY_MIN, DELAY_MAX]
    for col in ["arrival_delay_s", "departure_delay_s"]:
        valid = df_out[col].dropna()
        if len(valid):
            ok = valid.between(DELAY_MIN, DELAY_MAX).all()
            print(f"  {'PASS' if ok else 'FAIL'}  {col} in [{DELAY_MIN}, {DELAY_MAX}]  "
                  f"(min={valid.min():.0f}s  max={valid.max():.0f}s)")
            if not ok:
                all_ok = False

    # At least one delay per row
    both_null = df_out["arrival_delay_s"].isna() & df_out["departure_delay_s"].isna()
    ok = both_null.sum() == 0
    print(f"  {'PASS' if ok else 'FAIL'}  Every row has at least one delay")
    if not ok:
        all_ok = False

    # prev_stop_delay — no NaN allowed (first-stop rows are filtered out)
    lags = df_out["prev_stop_delay"]
    lag_nans = lags.isna().sum()
    lag_valid = lags.dropna()
    ok = lag_nans == 0 and len(lag_valid) > 0
    print(f"  {'PASS' if ok else 'FAIL'}  prev_stop_delay has {len(lag_valid):,} non-null values "
          f"({lag_nans:,} NaN)")
    if not ok:
        all_ok = False
    if len(lag_valid):
        ok = lag_valid.between(DELAY_MIN, DELAY_MAX).all()
        print(f"  {'PASS' if ok else 'FAIL'}  prev_stop_delay in [{DELAY_MIN}, {DELAY_MAX}]  "
              f"(min={lag_valid.min():.0f}s  max={lag_valid.max():.0f}s)")
        if not ok:
            all_ok = False

    # trip_stop_index — no NaN, all >= 0, min should be >= 1 (first stops are dropped)
    tsi = df_out["trip_stop_index"]
    tsi_nans = tsi.isna().sum()
    ok = tsi_nans == 0 and len(tsi) > 0
    print(f"  {'PASS' if ok else 'FAIL'}  trip_stop_index has {len(tsi):,} non-null values "
          f"({tsi_nans:,} NaN)")
    if not ok:
        all_ok = False
    if len(tsi):
        tsi_min, tsi_max = tsi.min(), tsi.max()
        ok_range = tsi_min >= 0 and tsi_max < 500
        print(f"  {'PASS' if ok_range else 'FAIL'}  trip_stop_index in [0, 500)  "
              f"(min={tsi_min} max={tsi_max})")
        if not ok_range:
            all_ok = False

    # dist_to_prev_stop — no NaN, no zeros, all >= 1m
    dists = df_out["dist_to_prev_stop"]
    dist_nans = dists.isna().sum()
    dist_zeros = (dists == 0.0).sum()
    ok = dist_nans == 0 and dist_zeros == 0 and len(dists) > 0
    print(f"  {'PASS' if ok else 'FAIL'}  dist_to_prev_stop has {len(dists):,} values "
          f"(NaN={dist_nans}, zero={dist_zeros})  "
          f"min={dists.min():.0f}m  max={dists.max():.0f}m")
    if not ok:
        all_ok = False

    # Traffic features — float32, allow small NaN rate (< 5%)
    if STOP_TRAFFIC:
        for col in ["traffic_dtv", "traffic_peak",
                     "traffic_heavy_share", "traffic_peak_ratio"]:
            col_nans = df_out[col].isna().sum()
            col_valid = df_out[col].dropna()
            nan_rate = col_nans / max(len(df_out), 1)
            ok = nan_rate < 0.05 and len(col_valid) > 0
            print(f"  {'PASS' if ok else 'FAIL'}  {col} has {len(col_valid):,} non-null "
                  f"({col_nans:,} NaN, {nan_rate:.2%})")
            if not ok:
                all_ok = False
            elif len(col_valid) > 0:
                if col in ("traffic_heavy_share", "traffic_peak_ratio"):
                    in_range = col_valid.between(0, 10).all()
                    print(f"  {'PASS' if in_range else 'FAIL'}  {col} in [0, 10]  "
                          f"(min={col_valid.min():.4f} max={col_valid.max():.4f})")
                    if not in_range:
                        all_ok = False
                elif col == "traffic_dtv":
                    ok_nonneg = (col_valid >= 0).all()
                    print(f"  {'PASS' if ok_nonneg else 'FAIL'}  {col} >= 0  "
                          f"(min={col_valid.min():.0f} max={col_valid.max():.0f})")
                    if not ok_nonneg:
                        all_ok = False

    # Public holiday is a boolean column with both values
    holiday_vals = df_out["is_public_holiday"].unique()
    is_bool = set(holiday_vals).issubset({True, False})
    print(f"  {'PASS' if is_bool else 'FAIL'}  is_public_holiday is bool "
          f"(values={sorted(holiday_vals)})")
    if not is_bool:
        all_ok = False

    # Valid parquet
    try:
        pq.read_metadata(dst)
        print(f"  PASS  Parquet file is valid")
    except Exception as e:
        print(f"  FAIL  Parquet file invalid: {e}")
        all_ok = False

    print(f"       {rows} rows | {size/1e6:.1f} MB parquet\n")
    shutil.rmtree(TMP_DIR)
    print("All tests passed." if all_ok else "SOME TESTS FAILED.")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Fetch hourly weather from Open-Meteo (if not already cached)
# ═══════════════════════════════════════════════════════════════════════════════

WEATHER_HOURLY = ROOT / "data" / "weather_hourly.parquet"
STATION_META   = ROOT / "data" / "station_metadata.parquet"
WEATHER_CACHE  = ROOT / "data" / "weather_hourly_cache"

STAC_BASE    = "https://data.geo.admin.ch/api/stac/v0.9"
COLLECTION   = "ch.meteoschweiz.ogd-smn"
API_BASE     = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m", "precipitation", "sunshine_duration",
    "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m",
    "wind_direction_10m", "surface_pressure", "snow_depth",
]

VAR_RENAME = {
    "temperature_2m": "temperature", "precipitation": "precipitation",
    "sunshine_duration": "sunshine", "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed", "wind_gusts_10m": "wind_gust",
    "wind_direction_10m": "wind_dir", "surface_pressure": "pressure",
    "snow_depth": "snow_depth",
}

WEATHER_SCHEMA = pa.schema([
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


def _get_weather_session():
    s = requests.Session()
    s.headers["User-Agent"] = "swiss-bus-delay-ml/1.0 (research)"
    return s


def _discover_weather_stations():
    """Return list of {station_id, lat, lon} dicts, generating station_metadata
    from the MeteoSwiss STAC API if not already cached."""
    if STATION_META.exists():
        meta = pq.read_table(STATION_META).to_pandas()
        print(f"Loaded {len(meta)} stations from {STATION_META}")
        return meta[["station_id", "lat", "lon"]].dropna().to_dict("records")

    print("station_metadata.parquet not found — fetching from MeteoSwiss STAC API...")
    session = _get_weather_session()
    stations, url, params = [], f"{STAC_BASE}/collections/{COLLECTION}/items", {"limit": 100}
    while url:
        r = session.get(url, params=params, timeout=60)
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
    pq.write_table(pa.Table.from_pandas(meta_df, preserve_index=False), STATION_META,
                   compression="snappy")
    print(f"Saved {STATION_META} ({len(stations)} stations)")
    return meta_df[["station_id", "lat", "lon"]].dropna().to_dict("records")


def _fetch_one_station(session, station, cache_dir):
    """Fetch hourly weather for one station → cache parquet. Returns (station_id, rows, error)."""
    sid      = station["station_id"]
    pq_path  = cache_dir / f"{sid}.parquet"

    if pq_path.exists():
        return sid, -1, None

    try:
        params = {
            "latitude":   round(station["lat"], 6),
            "longitude":  round(station["lon"], 6),
            "start_date": "2025-01-01",
            "end_date":   "2025-12-31",
            "hourly":     ",".join(HOURLY_VARS),
            "timezone":   "UTC",
        }

        for attempt in range(6):
            try:
                r = session.get(API_BASE, params=params, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except requests.HTTPError as e:
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 30)) + 5
                    print(f"  [{sid}] rate-limited, waiting {wait}s...")
                    time.sleep(wait)
                elif attempt == 5:
                    raise
                else:
                    wait = 2 ** attempt
                    print(f"  [{sid}] retry {attempt+1}/6 ({wait}s): {e}")
                    time.sleep(wait)
            except Exception as e:
                if attempt == 5:
                    raise
                wait = 2 ** attempt
                print(f"  [{sid}] retry {attempt+1}/6 ({wait}s): {e}")
                time.sleep(wait)

        time.sleep(1.0)  # rate-limit gap

        hourly     = data["hourly"]
        timestamps = pd.to_datetime(hourly["time"])

        df = pd.DataFrame({"timestamp": timestamps})
        df["station_id"] = sid

        for var in HOURLY_VARS:
            col  = VAR_RENAME[var]
            vals = hourly.get(var, [None] * len(timestamps))
            df[col] = pd.array(vals, dtype="Float64").astype("float32")

        df["sunshine"]   = df["sunshine"]   / 3600.0   # s/h → fraction 0–1
        df["wind_speed"] = df["wind_speed"] / 3.6      # km/h → m/s
        df["wind_gust"]  = df["wind_gust"]  / 3.6      # km/h → m/s

        ordered_cols = ["station_id", "timestamp"] + list(VAR_RENAME.values())
        table = pa.Table.from_pandas(df[ordered_cols], schema=WEATHER_SCHEMA, preserve_index=False)
        pq.write_table(table, pq_path, compression="snappy")
        return sid, len(df), None

    except Exception as e:
        if pq_path.exists():
            pq_path.unlink()
        return sid, -1, str(e)


def fetch_weather_if_needed():
    """Fetch hourly weather from Open-Meteo for all MeteoSwiss stations if
    weather_hourly.parquet doesn't already exist.  Resume-safe via per-station cache."""
    if WEATHER_HOURLY.exists():
        print(f"Weather already cached → {WEATHER_HOURLY}  ({WEATHER_HOURLY.stat().st_size / 1e6:.0f} MB)")
        return

    WEATHER_CACHE.mkdir(exist_ok=True)
    stations = _discover_weather_stations()
    cache_dir = WEATHER_CACHE

    todo = [s for s in stations if not (cache_dir / f"{s['station_id']}.parquet").exists()]
    n_cached = len(stations) - len(todo)
    if n_cached:
        print(f"Resuming weather fetch: {n_cached} cached, {len(todo)} remaining")

    session = _get_weather_session()
    done = n_cached
    failed = []

    for station in todo:
        sid, rows, err = _fetch_one_station(session, station, cache_dir)
        done += 1
        tag = f"[{done}/{len(stations)}]"
        if err:
            failed.append((sid, err))
            print(f"{tag} FAIL  {sid}: {err}")
        else:
            print(f"{tag} OK    {sid}: {rows:,} rows")

    if failed:
        print(f"\n{len(failed)} station(s) failed: {[s for s, _ in failed]}")

    # Merge per-station parquets into one file
    pq_files = sorted(cache_dir.glob("*.parquet"))
    if not pq_files:
        raise RuntimeError("No weather cache files to merge.")

    print(f"\nMerging {len(pq_files)} station files → {WEATHER_HOURLY}")
    tmp = WEATHER_HOURLY.with_suffix(".tmp.parquet")
    with pq.ParquetWriter(tmp, WEATHER_SCHEMA, compression="snappy") as writer:
        for path in pq_files:
            writer.write_table(pq.read_table(path, schema=WEATHER_SCHEMA))
    tmp.replace(WEATHER_HOURLY)
    print(f"Done. {WEATHER_HOURLY}: {WEATHER_HOURLY.stat().st_size / 1e6:.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3+4 — Join weather + drop missing → dataset_with_weather.parquet
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_WITH_WEATHER = ROOT / "data" / "swiss_bus_2026_weather.parquet"
STOP_MAP_TMP  = ROOT / "data" / "_stop_map_tmp.parquet"

WEATHER_COLS = [
    "temperature", "precipitation", "sunshine", "humidity",
    "wind_speed", "wind_gust", "wind_dir", "pressure", "snow_depth",
]


def build_stop_station_map():
    """Build {stop_id → weather_station_id} mapping via LV95→WGS84 + KDTree."""
    print("\nBuilding stop → MeteoSwiss station map...")

    stops = pq.read_table(
        STATION_DATA, columns=["number", "lv95east", "lv95north"]
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
    })
    pq.write_table(pa.Table.from_pandas(result, preserve_index=False), STOP_MAP_TMP)
    print(f"  Mapped {len(result):,} stops to MeteoSwiss stations")


def join_weather_and_clean():
    """DuckDB join: dataset.parquet + weather → dataset_with_weather.parquet.
    Preserves prev_stop_delay and dist_to_prev_stop.  Drops rows where the
    weather join did not match (the LEFT JOIN produces NULL weather columns)."""
    if not STATION_DATA.exists():
        raise FileNotFoundError(f"{STATION_DATA} required for weather join")

    build_stop_station_map()

    print("\nJoining dataset with weather (DuckDB, single pass)...")
    tmp = DATASET_WITH_WEATHER.with_suffix(".tmp.parquet")

    weather_cols_sql = ",\n                ".join(
        f"w.{c}::FLOAT AS {c}" for c in WEATHER_COLS
    )

    con = duckdb.connect()
    con.execute("SET threads = 2")
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
                d.trip_id,
                d.trip_stop_index,
                d.prev_stop_delay,
                d.dist_to_prev_stop,
                d.traffic_dtv,
                d.traffic_peak,
                d.traffic_heavy_share,
                d.traffic_peak_ratio,
                d.is_public_holiday,
                {weather_cols_sql}
            FROM read_parquet('{DATASET_IN}') AS d
            LEFT JOIN read_parquet('{STOP_MAP_TMP}') AS sm
                ON d.stop_id = sm.stop_id
            LEFT JOIN read_parquet('{WEATHER_HOURLY}') AS w
                ON sm.weather_station = w.station_id
               AND date_trunc('hour',
                       (d.timestamp AT TIME ZONE 'Europe/Zurich')::TIMESTAMP
                   ) = w.timestamp
            WHERE NOT d.pass_through
              AND d.prev_stop_delay IS NOT NULL
              AND w.temperature IS NOT NULL
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
        STOP_MAP_TMP.unlink(missing_ok=True)
        raise exc[0]

    size_gb = tmp.stat().st_size / 1e9
    print(f"  writing... {size_gb:.2f} GB")

    tmp.replace(DATASET_WITH_WEATHER)
    STOP_MAP_TMP.unlink(missing_ok=True)

    # Print stats
    rows_before = pq.read_metadata(DATASET_IN).num_rows
    rows_after  = pq.read_metadata(DATASET_WITH_WEATHER).num_rows
    dropped = rows_before - rows_after
    print(f"Done → {DATASET_WITH_WEATHER} ({size_gb:.2f} GB)")
    print(f"  {rows_before:,} → {rows_after:,} rows "
          f"({dropped:,} dropped, {dropped * 100 / rows_before:.2f}%)")


def main(workers):
    os.makedirs(TMP_DIR, exist_ok=True)

    # Clean up corrupt leftover files from a previous crashed run
    for f in os.listdir(TMP_DIR):
        path = os.path.join(TMP_DIR, f)
        if f.endswith(".parquet"):
            try:
                pq.read_metadata(path)
            except Exception:
                os.remove(path)
                print(f"Removed corrupt leftover: {f}")

    # Resume: skip CSVs already converted
    already_done = {
        f.replace(".parquet", ".csv")
        for f in os.listdir(TMP_DIR)
        if f.endswith(".parquet")
    }

    all_jobs = build_job_list()
    jobs = [(zp, cn) for zp, cn in all_jobs if cn not in already_done]

    if already_done:
        print(f"Resuming: {len(already_done)} files already done, "
              f"{len(jobs)} remaining\n")

    if not jobs:
        print("Nothing to do — all CSVs already converted.")
        return

    total     = len(all_jobs)
    completed = len(already_done)
    failed    = []

    print(f"Processing {len(jobs)} CSVs from {len(set(zp for zp, _ in all_jobs))} "
          f"ZIP archives with {workers} workers...\n")

    with Pool(workers) as pool:
        for csv_name, rows, size, error in pool.imap_unordered(process_csv, jobs):
            completed += 1
            if error:
                failed.append((csv_name, error))
                print(f"[{completed}/{total}] FAILED  {csv_name}: {error}")
            elif rows == 0:
                print(f"[{completed}/{total}] EMPTY   {csv_name}: no valid rows")
            else:
                print(f"[{completed}/{total}] OK      {csv_name}: "
                      f"{rows} rows → {size/1e6:.1f} MB")

    if failed:
        print(f"\n{len(failed)} file(s) failed:")
        for f, e in failed:
            print(f"  {f}: {e}")
        sys.exit(1)

    # Merge all temp parquets into final dataset.parquet
    parquet_files = sorted(
        os.path.join(TMP_DIR, f)
        for f in os.listdir(TMP_DIR)
        if f.endswith(".parquet")
    )

    if not parquet_files:
        print("No parquet files to merge.")
        sys.exit(1)

    print(f"\nMerging {len(parquet_files)} temp files into {OUTPUT}...")

    with pq.ParquetWriter(TMP_OUTPUT, EXPECTED_SCHEMA, compression="snappy") as writer:
        for i, path in enumerate(parquet_files, 1):
            writer.write_table(pq.read_table(path, schema=EXPECTED_SCHEMA))
            if i % 50 == 0 or i == len(parquet_files):
                print(f"  Merged {i}/{len(parquet_files)}...")

    os.replace(TMP_OUTPUT, OUTPUT)
    shutil.rmtree(TMP_DIR)

    size_gb = os.path.getsize(OUTPUT) / 1e9
    print(f"\nDone. {OUTPUT}: {size_gb:.2f} GB")

    # ═════════════════════════════════════════════════════════════════════════
    # Phase 2 — Fetch hourly weather
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\nPhase 2 — Fetch hourly weather\n{'='*60}")
    fetch_weather_if_needed()

    # ═════════════════════════════════════════════════════════════════════════
    # Phase 3+4 — Join weather + drop unmatched rows
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\nPhase 3+4 — Join weather + clean\n{'='*60}")
    join_weather_and_clean()

    print(f"\n{'='*60}")
    print(f"All phases complete.")
    print(f"  {OUTPUT}: {size_gb:.2f} GB")
    weather_size = os.path.getsize(DATASET_WITH_WEATHER) / 1e9
    print(f"  {DATASET_WITH_WEATHER}: {weather_size:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build dataset.parquet + dataset_with_weather.parquet "
                    "from monthly ZIP archives"
    )
    parser.add_argument("--test",    action="store_true",
                        help="Run smoke test on first CSV then exit")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for CSV processing (default: 4)")
    args = parser.parse_args()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)
    else:
        main(args.workers)
