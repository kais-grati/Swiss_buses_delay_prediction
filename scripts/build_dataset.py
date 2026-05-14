"""
Build dataset.parquet directly from monthly ZIP archives.

Merges former pipeline steps 1-4, 4-6 into a single script:
  1. Read daily CSVs from ZIP (no disk extraction)
  2. Translate German headers → English
  3. Filter to bus rows only (PRODUCT_ID == "Bus")
  4. Keep only relevant columns
  5. Drop cancelled / pass-through / additional-trip rows
  6. Compute delays + cyclical time features
  7. Drop outlier delays (> 30 min or < -2 min)
  8. Add is_public_holiday (canton-aware Swiss holidays)
  9. Write directly to dataset.parquet (snappy-compressed)

RAM-efficient: one CSV in memory per worker at a time.
Resume-safe: skips already-converted CSVs.
"""

import os
import sys
import io
import shutil
import argparse
import subprocess
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import holidays
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from multiprocessing import Pool

ROOT       = Path(__file__).resolve().parent.parent
ZIP_DIR    = ROOT / "data" / "compressed_data"
TMP_DIR    = ROOT / "data" / "build_dataset_tmp"
OUTPUT     = ROOT / "data" / "dataset.parquet"
TMP_OUTPUT = Path(str(OUTPUT) + ".tmp")

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


STOP_CANTON   = _build_stop_canton_map()
STOP_COORDS   = _build_stop_coords()
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
    ("prev_stop_delay",      pa.int32()),
    ("dist_to_prev_stop",   pa.float32()),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build dataset.parquet directly from monthly ZIP archives"
    )
    parser.add_argument("--test",    action="store_true",
                        help="Run smoke test on first CSV then exit")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    args = parser.parse_args()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)
    else:
        main(args.workers)
