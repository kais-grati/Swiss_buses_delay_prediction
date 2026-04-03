import os
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count

DATA_DIR = "cleaned_data"
TMP_DIR = "cleaned_data_parquet_tmp"
OUTPUT = "dataset.parquet"

os.makedirs(TMP_DIR, exist_ok=True)

TAU = 2 * np.pi

DATE_FMT     = "%d.%m.%Y"
TIME_FMT     = "%d.%m.%Y %H:%M"
FORECAST_FMT = "%d.%m.%Y %H:%M:%S"


def process_file(filename):
    src = os.path.join(DATA_DIR, filename)
    dst = os.path.join(TMP_DIR, filename.replace(".csv", ".parquet"))

    df = pd.read_csv(src, sep=";", dtype=str).reset_index(drop=True)

    # Drop cancelled trips
    df = df[df["CANCELLED"].str.lower() != "true"].reset_index(drop=True)

    # Parse datetimes
    date        = pd.to_datetime(df["DATE"],               format=DATE_FMT,     errors="coerce")
    arr_sched   = pd.to_datetime(df["ARRIVAL_TIME"],       format=TIME_FMT,     errors="coerce")
    dep_sched   = pd.to_datetime(df["DEPARTURE_TIME"],     format=TIME_FMT,     errors="coerce")
    arr_actual  = pd.to_datetime(df["ARRIVAL_FORECAST"],   format=FORECAST_FMT, errors="coerce")
    dep_actual  = pd.to_datetime(df["DEPARTURE_FORECAST"], format=FORECAST_FMT, errors="coerce")

    # Delays in seconds — null if status is not REAL
    arr_real = df["ARRIVAL_FORECAST_STATUS"]   == "REAL"
    dep_real = df["DEPARTURE_FORECAST_STATUS"] == "REAL"

    arrival_delay_s   = (arr_actual - arr_sched).dt.total_seconds().where(arr_real)
    departure_delay_s = (dep_actual - dep_sched).dt.total_seconds().where(dep_real)

    # Keep only rows where at least one real delay exists
    mask = arrival_delay_s.notna() | departure_delay_s.notna()
    df                = df[mask].reset_index(drop=True)
    date              = date[mask].reset_index(drop=True)
    arr_sched         = arr_sched[mask].reset_index(drop=True)
    dep_sched         = dep_sched[mask].reset_index(drop=True)
    arrival_delay_s   = arrival_delay_s[mask].reset_index(drop=True)
    departure_delay_s = departure_delay_s[mask].reset_index(drop=True)

    # Scheduled time for time-of-day features (prefer arrival, fall back to departure)
    sched_time   = arr_sched.fillna(dep_sched)
    time_minutes = sched_time.dt.hour * 60 + sched_time.dt.minute

    result = pd.DataFrame({
        # Cyclical time-of-day (period = 1440 min)
        "time_sin":  np.sin(TAU * time_minutes / 1440).astype("float32"),
        "time_cos":  np.cos(TAU * time_minutes / 1440).astype("float32"),
        # Cyclical day-of-week (period = 7)
        "dow_sin":   np.sin(TAU * date.dt.dayofweek / 7).astype("float32"),
        "dow_cos":   np.cos(TAU * date.dt.dayofweek / 7).astype("float32"),
        # Cyclical month (period = 12)
        "month_sin": np.sin(TAU * date.dt.month / 12).astype("float32"),
        "month_cos": np.cos(TAU * date.dt.month / 12).astype("float32"),
        # Calendar flags
        "is_weekend":     (date.dt.dayofweek >= 5),
        # Identifiers
        "operator":       df["OPERATOR_ABB"],
        "line":           df["LINE_NAME"],
        "stop_id":        pd.to_numeric(df["BPUIC"], errors="coerce").astype("Int32"),
        "stop_name":      df["STOP_NAME"],
        # Trip flags
        "additional_trip": df["ADDITIONAL_TRIP"].str.lower() == "true",
        "pass_through":    df["PASS_THROUGH"].str.lower()    == "true",
        # Targets
        "arrival_delay_s":   arrival_delay_s.astype("Int32"),
        "departure_delay_s": departure_delay_s.astype("Int32"),
    })

    table = pa.Table.from_pandas(result, preserve_index=False)
    pq.write_table(table, dst, compression="snappy")

    return filename, len(result), os.path.getsize(dst)


if __name__ == "__main__":
    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

    # Step 1: convert each CSV → parquet in parallel
    workers = cpu_count()
    print(f"Converting {len(csv_files)} files with {workers} workers...\n")

    with Pool(workers) as pool:
        for filename, rows, size in pool.imap_unordered(process_file, csv_files):
            print(f"{filename}: {rows} rows → {size / 1e6:.1f} MB")

    # Step 2: stream-merge all parquets into one file
    print(f"\nMerging into {OUTPUT}...")
    parquet_files = sorted(
        os.path.join(TMP_DIR, f)
        for f in os.listdir(TMP_DIR)
        if f.endswith(".parquet")
    )

    schema = pq.read_schema(parquet_files[0])
    with pq.ParquetWriter(OUTPUT, schema, compression="snappy") as writer:
        for path in parquet_files:
            writer.write_table(pq.read_table(path))

    shutil.rmtree(TMP_DIR)

    size_gb = os.path.getsize(OUTPUT) / 1e9
    print(f"\nDone. {OUTPUT}: {size_gb:.2f} GB")
