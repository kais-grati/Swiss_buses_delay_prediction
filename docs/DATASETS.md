# Datasets — Swiss Bus Delay Prediction

## Overview

The project works with several parquet datasets at different scales. All share the same 38-column schema (the final processed format including weather, traffic, and lag features).

| Dataset | Rows | Size | Description |
|---------|------|------|-------------|
| `swiss_bus_2025_weather_traffic.parquet` | 509M | 16.2 GB | Full Switzerland, all lines, 2025 |
| `swiss_bus_2025.parquet` | 512M | 9.4 GB | Full Switzerland, no weather/traffic |
| `lausanne_bus_2025_weather_traffic.parquet` | 18.2M | 365 MB | Lausanne region, all lines |
| `lausanne50k_bus_2025_weather_traffic.parquet` | 50K | 4.1 MB | Lausanne region, 50K stratified sample |
| `705_bus_2025_weather_traffic.parquet` | 492K | 8.0 MB | Line 705 only |
| `station_data.parquet` | 29K | 2.6 MB | Swiss stop coordinates & metadata |
| `weather_hourly.parquet` | 1.38M | 22.6 MB | Hourly Open-Meteo weather for 158 stations |

---

## Primary ML Datasets

### 705 (`705_bus_2025_weather_traffic.parquet`)

The primary dataset for model development and optimization.

- **Rows:** 491,570 (483,345 after NaN target removal)
- **Scope:** Line 705 bus route (Lausanne–Echandens area), full year 2025
- **Features:** 38 columns including weather, traffic, lag delays, cyclical time encodings
- **Target:** `arrival_delay_s` — the delay in seconds at each stop (range: -120 to 1,767s, median 72s)
- **Train/test split:** 386K / 97K (80/20)
- **Why this dataset:** Large enough for reliable tree model training, small enough for fast iteration (8MB loads instantly). Representative of Swiss bus operations with diverse weather and traffic conditions.

### Lausanne 50k (`lausanne50k_bus_2025_weather_traffic.parquet`)

A stratified sample from the Lausanne regional subset, used as a secondary evaluation benchmark.

- **Rows:** 50,000
- **Scope:** Lausanne metropolitan area, sampled proportionally to delay distribution
- **Train/test split:** 40K / 10K (80/20)
- **Why this dataset:** Tests model generalization to a different geographic area with a smaller, more homogeneous dataset. The different leaderboard (Ridge #2 vs #5 on 705) reveals how model rankings depend on dataset size and diversity.

### Full Switzerland (`swiss_bus_2025_weather_traffic.parquet`)

The complete dataset covering all of Switzerland. Not used for routine experimentation due to size.

- **Rows:** 509,558,380
- **Size:** 16.2 GB
- **Scope:** Every bus stop in Switzerland, all lines, full year 2025
- **Use cases:** Final model training, production deployment, large-scale inference

---

## Feature Schema

All ML datasets share this 38-column schema:

### Identity & Temporal

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | TIMESTAMP | Scheduled arrival time (Swiss local, naive) |
| `trip_id` | VARCHAR | Unique trip identifier |
| `trip_stop_index` | SMALLINT | Ordinal position (1-based) of stop within trip |
| `stop_id` | INTEGER | BPUIC stop number |
| `stop_name` | VARCHAR | Human-readable stop name |
| `operator` | VARCHAR | Transport operator abbreviation |
| `line` | VARCHAR | Line name/number |
| `additional_trip` | BOOLEAN | Unscheduled extra trip |

### Target & Related

| Column | Type | Description |
|--------|------|-------------|
| `arrival_delay_s` | INTEGER | Arrival delay in seconds (target) |
| `departure_delay_s` | INTEGER | Departure delay in seconds (not used as feature) |

### Lag Features

| Column | Type | Description |
|--------|------|-------------|
| `prev_stop_delay` | INTEGER | Arrival delay at previous stop on same trip (NaN for first stop) |
| `dist_to_prev_stop` | FLOAT | Euclidean distance (meters) to previous stop (NaN for first stop) |

### Cyclical Time Encodings

| Column | Type | Description |
|--------|------|-------------|
| `time_sin`, `time_cos` | FLOAT | Time of day mapped to unit circle |
| `dow_sin`, `dow_cos` | FLOAT | Day of week (0=Mon…6=Sun) mapped to unit circle |
| `month_sin`, `month_cos` | FLOAT | Month (1–12) mapped to unit circle |
| `is_weekend` | BOOLEAN | Saturday or Sunday |
| `is_public_holiday` | BOOLEAN | Swiss public holiday for the stop's canton |

### Weather (hourly Open-Meteo via MeteoSwiss stations)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `temperature` | FLOAT | °C | Air temperature at 2m |
| `precipitation` | FLOAT | mm | Rainfall + snowfall |
| `sunshine` | FLOAT | 0–1 | Fraction of hour with sunshine |
| `humidity` | FLOAT | % | Relative humidity at 2m |
| `wind_speed` | FLOAT | m/s | Wind speed at 10m |
| `wind_gust` | FLOAT | m/s | Wind gust at 10m |
| `wind_dir` | FLOAT | ° | Wind direction at 10m |
| `pressure` | FLOAT | hPa | Surface pressure |
| `snow_depth` | FLOAT | m | Snow depth |

### Traffic (ASTRA road counts, nearest segment)

| Column | Type | Description |
|--------|------|-------------|
| `traffic_dtv` | FLOAT | Average daily traffic |
| `traffic_dwv` | FLOAT | Average weekday traffic |
| `traffic_pw` | FLOAT | Passenger car share |
| `traffic_lw` | FLOAT | Truck share |
| `traffic_lz` | FLOAT | Delivery vehicle share |
| `traffic_li` | FLOAT | Motorcycle share |
| `traffic_heavy_share` | FLOAT | Heavy vehicle fraction |
| `traffic_peak_ratio` | FLOAT | Peak/off-peak traffic ratio |
| `traffic_peak` | FLOAT | Peak hour traffic volume |

---

## Supporting Datasets

### Station Data (`station_data.parquet`)

Swiss public transport stops with coordinates and metadata.

- **Rows:** 28,982 stops
- **Key columns:** `number` (BPUIC stop ID), `lv95east`, `lv95north` (Swiss grid coordinates), `designation` (stop name), `canton`

Used for:
- Spatial joins (stop → nearest weather station, stop → nearest road segment)
- Stop distance computation (`dist_to_prev_stop`)
- Holiday detection (stop → canton → holiday calendar)

### Weather Hourly (`weather_hourly.parquet`)

Hourly weather observations from Open-Meteo for 158 MeteoSwiss station locations across 2025.

- **Rows:** 1,384,080 (158 stations × 8,760 hours)
- **Key columns:** `station_id`, `timestamp` (UTC), 9 weather variables

Used in `add_weather.py` for the spatial join: each bus stop is mapped to its nearest MeteoSwiss station, then weather is joined on the UTC hour.

---

## Data Pipeline

For details on how these datasets are built from raw SBB CSV data, see [PIPELINE.md](../PIPELINE.md).
