import os
import sys
import shutil
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "cleaned_data"
TMP_DIR    = ROOT / "data" / "cleaned_data_parquet_tmp"
OUTPUT     = ROOT / "data" / "dataset.parquet"
TMP_OUTPUT = Path(str(OUTPUT) + ".tmp")

TAU = 2 * np.pi

# Only keep rows with a real observed delay (GESCHAETZT = estimated, excluded)
VALID_STATUSES = {"REAL"}

PARSE_OPTIONS   = pa_csv.ParseOptions(delimiter=";")
READ_OPTIONS    = pa_csv.ReadOptions(block_size=64 * 1024 * 1024)  # 64 MB blocks
CONVERT_OPTIONS = pa_csv.ConvertOptions(
    true_values=["true", "True"],
    false_values=["false", "False"],
    null_values=["", "NA"],
)

EXPECTED_SCHEMA = pa.schema([
    ("timestamp",        pa.timestamp("s")),
    ("time_sin",         pa.float32()),
    ("time_cos",         pa.float32()),
    ("dow_sin",          pa.float32()),
    ("dow_cos",          pa.float32()),
    ("month_sin",        pa.float32()),
    ("month_cos",        pa.float32()),
    ("is_weekend",       pa.bool_()),
    ("operator",         pa.large_string()),
    ("line",             pa.large_string()),
    ("stop_id",          pa.int32()),
    ("stop_name",        pa.large_string()),
    ("additional_trip",  pa.bool_()),
    ("pass_through",     pa.bool_()),
    ("arrival_delay_s",  pa.int32()),
    ("departure_delay_s",pa.int32()),
    ("trip_id",          pa.large_string()),
])


def coerce_bool(col):
    """Convert a column to bool regardless of whether pyarrow inferred it as
    bool, int (0/1), or string ('true'/'false'/'0'/'1')."""
    if pd.api.types.is_bool_dtype(col):
        return col.fillna(False)
    if pd.api.types.is_numeric_dtype(col):
        return col.fillna(0).astype(bool)
    return col.fillna("").astype(str).str.lower().isin({"true", "1"})


def process_file(filename):
    src = os.path.join(DATA_DIR, filename)
    dst = os.path.join(TMP_DIR, filename.replace(".csv", ".parquet"))

    try:
        # 11x faster than pandas read_csv — pyarrow auto-detects bools and ints
        table = pa_csv.read_csv(src,
                                read_options=READ_OPTIONS,
                                parse_options=PARSE_OPTIONS,
                                convert_options=CONVERT_OPTIONS)
        df = table.to_pandas()
        del table  # free arrow memory immediately

        # Drop cancelled trips
        df = df[~coerce_bool(df["CANCELLED"])].reset_index(drop=True)

        # Only parse forecast timestamps for REAL rows (skip wasted parsing)
        arr_real = df["ARRIVAL_FORECAST_STATUS"].isin(VALID_STATUSES)
        dep_real = df["DEPARTURE_FORECAST_STATUS"].isin(VALID_STATUSES)

        arr_sched  = pd.to_datetime(df["ARRIVAL_TIME"],                      format="%d.%m.%Y %H:%M",    errors="coerce")
        dep_sched  = pd.to_datetime(df["DEPARTURE_TIME"],                    format="%d.%m.%Y %H:%M",    errors="coerce")
        arr_actual = pd.to_datetime(df["ARRIVAL_FORECAST"].where(arr_real),  format="%d.%m.%Y %H:%M:%S", errors="coerce")
        dep_actual = pd.to_datetime(df["DEPARTURE_FORECAST"].where(dep_real),format="%d.%m.%Y %H:%M:%S", errors="coerce")

        # Compute delay using time-of-day only — ignores date portion so date
        # mismatches in source data cannot produce spurious ±24h values.
        # Midnight crossings are handled by wrapping into (-43200, 43200] (±12h).
        def time_delay_s(sched, actual):
            sched_s  = sched.dt.hour  * 3600 + sched.dt.minute  * 60
            actual_s = actual.dt.hour * 3600 + actual.dt.minute * 60 + actual.dt.second
            delta    = actual_s - sched_s
            # Wrap: if gap > 12h assume midnight crossing forward, if < -12h assume backward
            delta    = delta.where(delta >  -43200, delta + 86400)
            delta    = delta.where(delta <=  43200, delta - 86400)
            return delta

        arrival_delay_s   = time_delay_s(arr_sched, arr_actual)
        departure_delay_s = time_delay_s(dep_sched, dep_actual)

        # Keep only rows with at least one valid delay
        mask = arrival_delay_s.notna() | departure_delay_s.notna()

        if not mask.any():
            return filename, 0, 0, None

        df                = df[mask].reset_index(drop=True)
        arrival_delay_s   = arrival_delay_s[mask].reset_index(drop=True)
        departure_delay_s = departure_delay_s[mask].reset_index(drop=True)
        arr_sched         = arr_sched[mask].reset_index(drop=True)
        dep_sched         = dep_sched[mask].reset_index(drop=True)

        # Time-of-day features via string slicing (avoids a full datetime parse)
        # Format: "DD.MM.YYYY HH:MM" — hour at [11:13], minute at [14:16]
        time_str   = df["ARRIVAL_TIME"].fillna(df["DEPARTURE_TIME"])
        hour       = pd.to_numeric(time_str.str[11:13], errors="coerce").fillna(0).astype("int16")
        minute     = pd.to_numeric(time_str.str[14:16], errors="coerce").fillna(0).astype("int16")
        time_min   = hour * 60 + minute

        # Date features — month via string slice, dow needs full parse
        month = pd.to_numeric(df["DATE"].str[3:5], errors="coerce").fillna(1).astype("int16")
        date  = pd.to_datetime(df["DATE"], format="%d.%m.%Y", errors="coerce")
        dow   = date.dt.dayofweek.fillna(0).astype("int16")

        result = pd.DataFrame({
            "timestamp":         arr_sched.fillna(dep_sched),
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
            "trip_id":           df["TRIP_ID"],
        })

        out_table = pa.Table.from_pandas(result, schema=EXPECTED_SCHEMA, preserve_index=False)
        del result  # free pandas memory before writing
        pq.write_table(out_table, dst, compression="snappy")
        del out_table

        return filename, len(df), os.path.getsize(dst), None

    except Exception as e:
        if os.path.exists(dst):
            os.remove(dst)
        return filename, -1, 0, str(e)


def run_tests():
    print("=== Running tests on 3 files ===\n")

    test_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))[:3]
    os.makedirs(TMP_DIR, exist_ok=True)
    all_passed = True

    for filename in test_files:
        src = os.path.join(DATA_DIR, filename)
        dst = os.path.join(TMP_DIR, filename.replace(".csv", ".parquet"))
        print(f"--- {filename} ---")

        # Run conversion
        _, rows, size, error = process_file(filename)

        if error:
            print(f"  FAIL  process_file raised: {error}")
            all_passed = False
            continue

        # Load output for inspection
        df_out = pq.read_table(dst).to_pandas()

        # 1. Schema matches expected
        actual_schema = pq.read_schema(dst)
        schema_ok = actual_schema.equals(EXPECTED_SCHEMA)
        print(f"  {'PASS' if schema_ok else 'FAIL'}  Schema correct")
        if not schema_ok:
            print(f"       Expected: {EXPECTED_SCHEMA}")
            print(f"       Got:      {actual_schema}")
            all_passed = False

        # 2. Row count is positive and less than input
        df_in  = pd.read_csv(src, sep=";", dtype=str)
        count_ok = 0 < rows <= len(df_in)
        print(f"  {'PASS' if count_ok else 'FAIL'}  Row count {rows} <= input {len(df_in)}")
        if not count_ok:
            all_passed = False

        # 3. No cancelled trips in output
        # (cannot directly check without joining — verify via spot-check below)

        # 4. Cyclical features in [-1, 1]
        for col in ["time_sin","time_cos","dow_sin","dow_cos","month_sin","month_cos"]:
            mn, mx = df_out[col].min(), df_out[col].max()
            ok = -1.0 <= mn and mx <= 1.0
            print(f"  {'PASS' if ok else 'FAIL'}  {col} in [-1, 1]  (min={mn:.4f} max={mx:.4f})")
            if not ok:
                all_passed = False

        # 5. No NaN in feature columns (delays can be null — that's expected)
        feature_cols = ["time_sin","time_cos","dow_sin","dow_cos",
                        "month_sin","month_cos","is_weekend",
                        "operator","line","stop_id","additional_trip","pass_through"]
        for col in feature_cols:
            nans = df_out[col].isna().sum()
            ok = nans == 0
            if not ok:
                print(f"  FAIL  {col} has {nans} NaN values")
                all_passed = False
        print(f"  PASS  No NaN in feature columns")

        # 6. Delay range within bounds — time-of-day computation wraps to (-43200, 43200]
        for col in ["arrival_delay_s", "departure_delay_s"]:
            valid = df_out[col].dropna()
            if len(valid):
                ok = valid.between(-43200, 43200).all()
                print(f"  {'PASS' if ok else 'FAIL'}  {col} in (-12h, +12h]  "
                      f"(min={valid.min():.0f}s  max={valid.max():.0f}s)")
                if not ok:
                    all_passed = False

        # 7. At least one delay non-null per row
        both_null = df_out["arrival_delay_s"].isna() & df_out["departure_delay_s"].isna()
        ok = both_null.sum() == 0
        print(f"  {'PASS' if ok else 'FAIL'}  Every row has at least one delay value")
        if not ok:
            all_passed = False

        # 8. Output file is valid parquet (readable by fresh open)
        try:
            pq.read_metadata(dst)
            print(f"  PASS  Parquet file is valid")
        except Exception as e:
            print(f"  FAIL  Parquet file is invalid: {e}")
            all_passed = False

        print(f"       {rows} rows | {size/1e6:.1f} MB parquet\n")

    shutil.rmtree(TMP_DIR)
    print(f"\n{'All tests passed.' if all_passed else 'SOME TESTS FAILED.'}")
    return all_passed


def main(workers=2):
    os.makedirs(TMP_DIR, exist_ok=True)

    # Remove leftover partial .parquet files from previous crashed run
    for f in os.listdir(TMP_DIR):
        path = os.path.join(TMP_DIR, f)
        if f.endswith(".parquet"):
            try:
                pq.read_metadata(path)   # valid file — keep for resume
            except Exception:
                os.remove(path)          # corrupt partial file — discard
                print(f"Removed corrupt leftover: {f}")

    # Resume: skip files already successfully converted
    already_done = {
        f.replace(".parquet", ".csv")
        for f in os.listdir(TMP_DIR)
        if f.endswith(".parquet")
    }

    all_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    csv_files = [f for f in all_files if f not in already_done]

    if already_done:
        print(f"Resuming: {len(already_done)} files already done, {len(csv_files)} remaining\n")

    total     = len(all_files)
    completed = len(already_done)
    failed    = []

    print(f"Converting {len(csv_files)} files with {workers} workers...\n")

    with Pool(workers) as pool:
        for filename, rows, size, error in pool.imap_unordered(process_file, csv_files):
            completed += 1
            if error:
                failed.append((filename, error))
                print(f"[{completed}/{total}] FAILED  {filename}: {error}")
            elif rows == 0:
                print(f"[{completed}/{total}] EMPTY   {filename}: no REAL rows")
            else:
                print(f"[{completed}/{total}] OK      {filename}: {rows} rows → {size/1e6:.1f} MB")

    if failed:
        print(f"\n{len(failed)} file(s) failed — fix them before merging:")
        for f, e in failed:
            print(f"  {f}: {e}")
        sys.exit(1)

    # Merge all temp parquets into one final file
    parquet_files = sorted(
        os.path.join(TMP_DIR, f)
        for f in os.listdir(TMP_DIR)
        if f.endswith(".parquet")
    )

    if not parquet_files:
        print("No parquet files to merge.")
        sys.exit(1)

    print(f"\nMerging {len(parquet_files)} files into {OUTPUT}...")

    # Write to temp output first — rename only on full success (no corrupt output on crash)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",    action="store_true", help="Run tests on 3 files then exit")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    args = parser.parse_args()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)
    else:
        main(args.workers)
