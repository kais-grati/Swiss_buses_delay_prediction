# Dataset Pipeline

This document describes every step required to go from the raw SBB open data
CSV files to the final `dataset_with_weather.parquet` used for model training.

---

## Overview

```
data/*.csv  (raw, German headers, all transport modes)
    │
    ├─ 1. translate_headers.py   → rename German columns to English
    ├─ 2. keep_only_buses.py     → filter to bus rows only  →  cleaned_data/*.csv
    ├─ 3. prepare_features.py    → drop unused columns      →  cleaned_data/*.csv (in-place)
    ├─ 4. to_parquet.py          → feature engineering      →  dataset.parquet
    │
    ├─ 5. fetch_weather_hourly.py→ hourly Open-Meteo weather→  weather_hourly.parquet
    │                                                           station_metadata.parquet
    ├─ 6. add_weather.py         → join weather to dataset  →  dataset_with_weather.parquet
    ├─ 7. add_holidays.py        → add is_public_holiday    →  dataset_with_weather.parquet (in-place)
    ├─ 8. drop_outlier_delays.py → remove delay > 30 min   →  dataset_with_weather.parquet (in-place)
    ├─ 9. drop_early_outliers.py  → remove delay < -2 min  →  dataset_with_weather.parquet (in-place)
    └─10. drop_missing_weather.py → remove missing weather →  dataset_with_weather.parquet (in-place)
```

---

## Data Sources

| Source | URL | Content |
|--------|-----|---------|
| SBB Istdaten | https://data.opentransportdata.swiss/dataset/istdaten | One CSV per day with all public transport departures/arrivals in Switzerland and their actual vs scheduled times |
| MeteoSwiss OGD | https://data.geo.admin.ch/api/stac/v0.9/collections/ch.meteoschweiz.ogd-smn | Daily climate observations for 158 automatic weather stations |
| Open-Meteo Archive | https://archive-api.open-meteo.com/v1/archive | Hourly historical weather for any lat/lon (free, no key) |
| Swiss Stop Coordinates | `station_data.parquet` (pre-downloaded) | 28,982 Swiss public transport stops with LV95 coordinates and BPUIC identifiers |

---

## Step 1 — Translate Headers

**Script:** `translate_headers.py`  
**Input:** `data/*.csv` (365 daily files, German column names)  
**Output:** same files, header rewritten in English  
**Reads into memory:** one file at a time

Renames German SBB column names to English equivalents, e.g.:
- `BETRIEBSTAG` → `DATE`
- `HALTESTELLEN_NAME` → `STOP_NAME`
- `AN_PROGNOSE` → `ARRIVAL_FORECAST`

Run:
```bash
python translate_headers.py
```

---

## Step 2 — Keep Only Buses

**Script:** `keep_only_buses.py`  
**Input:** `data/*.csv`  
**Output:** `cleaned_data/*.csv` (new files, bus rows only)

Filters each daily CSV to rows where `PRODUCT_ID == "Bus"`, discarding trains,
trams, boats, etc. Processes files in parallel with all available CPU cores.

Run:
```bash
python keep_only_buses.py
```

---

## Step 3 — Drop Unused Columns

**Script:** `prepare_features.py`  
**Input:** `cleaned_data/*.csv`  
**Output:** same files, rewritten in-place with only the needed columns

Keeps only these 14 columns, dropping everything else:

| Column | Description |
|--------|-------------|
| `DATE` | Service date (DD.MM.YYYY) |
| `OPERATOR_ABB` | Operator abbreviation |
| `LINE_NAME` | Line identifier |
| `BPUIC` | Stop identifier (numeric) |
| `STOP_NAME` | Stop name |
| `ARRIVAL_TIME` | Scheduled arrival (DD.MM.YYYY HH:MM) |
| `ARRIVAL_FORECAST` | Actual arrival (DD.MM.YYYY HH:MM:SS) |
| `ARRIVAL_FORECAST_STATUS` | `REAL` / `GESCHAETZT` / etc. |
| `DEPARTURE_TIME` | Scheduled departure |
| `DEPARTURE_FORECAST` | Actual departure |
| `DEPARTURE_FORECAST_STATUS` | `REAL` / `GESCHAETZT` / etc. |
| `ADDITIONAL_TRIP` | Extra unscheduled trip flag |
| `CANCELLED` | Trip cancelled flag |
| `PASS_THROUGH` | Bus passed without stopping |

Skips files already processed (idempotent). Writes atomically via `.tmp` files.

Run:
```bash
python prepare_features.py
```

---

## Step 4 — Feature Engineering → dataset.parquet

**Script:** `to_parquet.py`  
**Input:** `cleaned_data/*.csv`  
**Output:** `dataset.parquet` (~3.5 GB, snappy-compressed)

This is the main processing step. For each daily CSV:

1. **Drop cancelled trips** (`CANCELLED == true`)
2. **Keep only REAL observations** — rows where at least one of
   `ARRIVAL_FORECAST_STATUS` or `DEPARTURE_FORECAST_STATUS` is `"REAL"`.
   Estimated (`GESCHAETZT`) forecasts are excluded.
3. **Compute delays** — `(actual_time − scheduled_time)` in seconds, using
   time-of-day only to avoid ±24h errors from date mismatches. Wrapped to
   (−12h, +12h].
4. **Encode cyclical time features:**
   - `time_sin / time_cos` — minute of day mapped onto unit circle
   - `dow_sin / dow_cos` — day of week (0=Mon … 6=Sun)
   - `month_sin / month_cos` — month (1–12)
   - `is_weekend` — bool
5. **Preserve raw timestamp** — scheduled arrival (falling back to scheduled
   departure) kept as `timestamp` for weather joining.

Output schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | timestamp[s] | Scheduled arrival/departure (Swiss local, naive) |
| `time_sin` / `time_cos` | float32 | Cyclical time of day |
| `dow_sin` / `dow_cos` | float32 | Cyclical day of week |
| `month_sin` / `month_cos` | float32 | Cyclical month |
| `is_weekend` | bool | Saturday or Sunday |
| `operator` | string | Operator abbreviation |
| `line` | string | Line name |
| `stop_id` | int32 | BPUIC stop number |
| `stop_name` | string | Stop name |
| `additional_trip` | bool | Unscheduled extra trip |
| `pass_through` | bool | No stop made (filtered out in Step 6) |
| `arrival_delay_s` | int32 | Arrival delay in seconds (nullable) |
| `departure_delay_s` | int32 | Departure delay in seconds (nullable) |

Processes files in parallel, writes per-file parquets to `cleaned_data_parquet_tmp/`,
then merges into `dataset.parquet`. Supports resume (skips already-converted files).

Run:
```bash
python to_parquet.py --workers 4
# or test on 3 files first:
python to_parquet.py --test
```

---

## Step 5 — Hourly Open-Meteo Weather

**Script:** `fetch_weather_hourly.py`  
**Input:** Open-Meteo Archive API, MeteoSwiss STAC API (station discovery)  
**Output:**
- `weather_hourly.parquet` — hourly observations for 158 station locations
- `station_metadata.parquet` — MeteoSwiss station coordinates (generated once if missing)
- `weather_hourly_cache/` — per-station parquets (resume cache)

Queries Open-Meteo's free historical archive API for each of the 158
MeteoSwiss station lat/lon coordinates. Returns 8,760 rows per station
(one per hour of 2025). Timestamps are in UTC.

Weather variables (hourly):

| Column | Description | Unit |
|--------|-------------|------|
| `temperature` | Air temperature 2m | °C |
| `precipitation` | Rainfall + snowfall | mm |
| `sunshine` | Fraction of the hour with sunshine | 0–1 |
| `humidity` | Relative humidity 2m | % |
| `wind_speed` | Wind speed at 10m | m/s |
| `wind_gust` | Wind gust at 10m | m/s |
| `wind_dir` | Wind direction at 10m | ° |
| `pressure` | Surface pressure | hPa |
| `snow_depth` | Snow depth | m |

Runs sequentially (1 worker) with a 1-second gap between requests to respect
the free-tier rate limit. ~3 minutes total for all 158 stations.

Run:
```bash
python fetch_weather_hourly.py
```

---

## Step 6 — Join Weather → dataset_with_weather.parquet

**Script:** `add_weather.py`  
**Input:** `dataset.parquet`, `weather_hourly.parquet`, `station_data.parquet`, `station_metadata.parquet`  
**Output:** `dataset_with_weather.parquet`

Three sub-steps:

**6a. Stop → nearest MeteoSwiss station mapping**

Loads all 28,982 bus stop coordinates from `station_data.parquet` (Swiss LV95
grid, EPSG:2056), converts them to WGS84 (lat/lon) using `pyproj`, then uses
a `scipy` KDTree to find the nearest of the 158 MeteoSwiss stations for each
stop. Result: a `{bpuic → station_id}` dictionary held in memory.

**6b. Weather lookup table**

Loads `weather_hourly.parquet` (~80 MB) fully into RAM, indexed by
`(station_id, timestamp)` where timestamp is UTC-naive, floored to the hour.

**6c. Filtered join**

Reads `dataset.parquet` via DuckDB streaming:
1. **Excludes pass-through rows** (`WHERE NOT pass_through`) — stops where the
   bus never opens its doors have no meaningful delay to predict.
2. **Drops the `pass_through` column** from the output (all remaining rows are
   `pass_through = FALSE`).
3. Maps `stop_id` → `meteoswiss_station_id` via the dictionary from 6a.
4. Converts `timestamp` from Swiss local time (`Europe/Zurich`) to UTC, floors to hour.
5. Left-joins with the weather table on `(station_id, utc_hour)`.

Adds these columns to the dataset (all float32):
`temperature`, `precipitation`, `sunshine`, `humidity`,
`wind_speed`, `wind_gust`, `wind_dir`, `pressure`, `snow_depth`

Run:
```bash
pip install pyproj scipy   # first time only
python add_weather.py
```

---

## Step 7 — Add Public Holidays → dataset_with_weather.parquet

**Script:** `add_holidays.py`  
**Input:** `dataset_with_weather.parquet`, `station_data.parquet`  
**Output:** `dataset_with_weather.parquet` (in-place, adds `is_public_holiday` column)

Three sub-steps:

1. **Stop → canton mapping** — loads `station_data.parquet`, maps each `stop_id` (BPUIC) to its Swiss canton abbreviation (e.g. `VD`, `ZH`). Writes a temporary `_stop_map_tmp.parquet`.
2. **Holiday table** — uses the `holidays` library to generate all canton-specific Swiss public holidays for each year present in the dataset. Writes `_holidays_tmp.parquet`.
3. **Join** — DuckDB streaming join: `dataset ⟕ stop_map ⟕ holidays` on `(stop_id → canton, DATE(timestamp) = holiday_date)`. Sets `is_public_holiday = TRUE` for matching rows, `FALSE` otherwise. Verifies row count before atomically replacing the original file.

Temporary files are cleaned up on exit (even on failure).

Run:
```bash
pip install holidays   # first time only
python add_holidays.py
```

---

## Step 8 — Drop Outlier Delays → dataset_with_weather.parquet

**Script:** `drop_outlier_delays.py`  
**Input:** `dataset_with_weather.parquet`  
**Output:** `dataset_with_weather.parquet` (in-place, outlier rows removed)

Drops rows where `arrival_delay_s > 1800` **or** `departure_delay_s > 1800` (30-minute threshold). These represent ~15.6% of the dataset and are likely corrupt records rather than real delays.

Processes the file one PyArrow row group at a time — RAM usage stays bounded regardless of file size. Writes to `.tmp`, then atomically replaces the original.

Run:
```bash
python drop_outlier_delays.py
```

---

## Step 9 — Drop Implausibly Early Arrivals → dataset_with_weather.parquet

**Script:** `drop_early_outliers.py`  
**Input:** `dataset_with_weather.parquet`  
**Output:** `dataset_with_weather.parquet` (in-place, early outlier rows removed)

Drops rows where `arrival_delay_s < -120` **or** `departure_delay_s < -120` (2-minute early threshold). Swiss transit regulations prohibit early departures, so values beyond -120s are data artifacts rather than real early arrivals (~0.60% of rows). Dropping is preferred over capping to avoid creating an artificial spike at -120s in the delay distribution.

Same row-group streaming approach as Step 8.

Run:
```bash
python drop_early_outliers.py
```

---

## Step 10 — Drop Missing Weather → dataset_with_weather.parquet

**Script:** `drop_missing_weather.py`  
**Input:** `dataset_with_weather.parquet`  
**Output:** `dataset_with_weather.parquet` (in-place, incomplete rows removed)

Drops rows where any of the 9 weather columns (`temperature`, `precipitation`, `sunshine`, `humidity`, `wind_speed`, `wind_gust`, `wind_dir`, `pressure`, `snow_depth`) is NULL. These arise from stops that could not be matched to a MeteoSwiss station or timestamps outside the weather data coverage (~0.47% of rows).

Same row-group streaming approach as previous steps.

Run:
```bash
python drop_missing_weather.py
```

---

## Full Run Order

```bash
# 1–3: one-time data cleaning (already done if cleaned_data/ exists)
python translate_headers.py
python keep_only_buses.py
python prepare_features.py

# 4: build dataset (re-run if to_parquet.py was modified)
python to_parquet.py --workers 4

# 5: fetch hourly weather (run once, resume-safe)
python fetch_weather_hourly.py

# 6: join everything (pass-through rows excluded automatically)
python add_weather.py

# 7: add canton-aware public holiday flag
python add_holidays.py

# 8: drop outlier delays (> 30 min)
python drop_outlier_delays.py

# 9: drop implausibly early arrivals/departures (< -2 min — ~0.60% of rows)
python drop_early_outliers.py

# 10: drop rows with any missing weather field (~0.47% of rows)
python drop_missing_weather.py
```

### Utility: create a line/stop sub-dataset

To extract a filtered subset without touching the main file:

```bash
# Example: line 705, Echandens Chocolatière only
python filter_705_echandens.py
# → data/dataset_705_echandens.parquet (27,959 rows)
```

Uses DuckDB predicate pushdown — scans only relevant row groups.

### Utility: clean an existing dataset_with_weather.parquet

If `dataset_with_weather.parquet` was generated before the pass-through filter
was added to `add_weather.py`, run this once to remove those rows in-place:

```bash
python drop_pass_through.py
```

Uses DuckDB streaming — the full file is never loaded into RAM. Writes to a
`.tmp` file first, verifies the row count, then atomically replaces the original.

## Output Files

| File | Size | Description |
|------|------|-------------|
| `dataset.parquet` | ~3.5 GB | Bus trip features without weather |
| `station_metadata.parquet` | ~15 KB | MeteoSwiss station locations |
| `weather_hourly.parquet` | ~50–80 MB | Hourly weather per station |
| `station_data.parquet` | ~2.6 MB | Swiss bus stop coordinates |
| `dataset_with_weather.parquet` | ~5–6 GB | Final training dataset |
