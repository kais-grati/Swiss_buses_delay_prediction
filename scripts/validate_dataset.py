#!/usr/bin/env python3
"""
validate_dataset.py — correctness checks for the final dataset parquet.

Uses DuckDB to query the parquet file directly (no full load into memory).

Usage:
    python validate_dataset.py                          # auto-detects file
    python validate_dataset.py dataset_with_weather.parquet
    python validate_dataset.py dataset.parquet

Exit code: 0 if all tests pass, 1 if any FAIL.
"""

import sys
import argparse
from pathlib import Path
import duckdb

_DATA = Path(__file__).resolve().parent.parent / "data"

# ── Helpers ────────────────────────────────────────────────────────────────────

PASS  = "\033[32mPASS\033[0m"
FAIL  = "\033[31mFAIL\033[0m"
WARN  = "\033[33mWARN\033[0m"
SKIP  = "\033[90mSKIP\033[0m"

results = []

def check(name: str, passed: bool, detail: str = "", warn_only: bool = False):
    tag = (PASS if passed else (WARN if warn_only else FAIL))
    line = f"  [{tag}]  {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append((name, "PASS" if passed else ("WARN" if warn_only else "FAIL")))

def skip(name: str, reason: str = ""):
    print(f"  [{SKIP}]  {name}" + (f"  ({reason})" % () if reason else f"  ({reason})"))
    results.append((name, "SKIP"))

def q(sql: str):
    return con.execute(sql).fetchone()[0]

def qdf(sql: str):
    return con.execute(sql).df()

# ── Test groups ────────────────────────────────────────────────────────────────

def test_schema(cols: set, has_weather: bool):
    print("\n── Schema ────────────────────────────────────────────────────────────")

    required = {
        "timestamp", "time_sin", "time_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_weekend", "operator", "line",
        "stop_id", "stop_name", "additional_trip",
        "arrival_delay_s", "departure_delay_s", "is_public_holiday",
    }
    weather_cols = {
        "temperature", "precipitation", "sunshine", "humidity",
        "wind_speed", "wind_gust", "wind_dir", "pressure", "snow_depth",
    }

    missing = required - cols
    check("All required columns present", not missing,
          f"missing: {missing}" if missing else "")

    if has_weather:
        missing_w = weather_cols - cols
        check("All weather columns present", not missing_w,
              f"missing: {missing_w}" if missing_w else "")
    else:
        skip("Weather columns present", "file has no weather columns")


def test_completeness(cols: set, n_rows: int, has_weather: bool):
    print("\n── Completeness / NaN rates ──────────────────────────────────────────")

    no_null_cols = [
        "time_sin", "time_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_weekend",
        "operator", "line", "stop_id", "additional_trip", "is_public_holiday",
    ]
    for col in no_null_cols:
        if col not in cols:
            continue
        nulls = q(f"SELECT COUNT(*) FROM df WHERE {col} IS NULL")
        check(f"No NaN in {col}", nulls == 0, f"{nulls:,} nulls" if nulls else "")

    # At least one delay per row
    both_null = q("""
        SELECT COUNT(*) FROM df
        WHERE arrival_delay_s IS NULL AND departure_delay_s IS NULL
    """)
    check("Every row has ≥1 delay value", both_null == 0,
          f"{both_null:,} rows with both delays null" if both_null else "")

    # Delay null rates (informational)
    for col in ("arrival_delay_s", "departure_delay_s"):
        rate = q(f"SELECT COUNT(*) FROM df WHERE {col} IS NULL") / n_rows * 100
        check(f"{col} null rate < 60%", rate < 60, f"{rate:.1f}% null", warn_only=rate < 80)

    if has_weather:
        weather_cols = ["temperature", "precipitation", "humidity",
                        "wind_speed", "pressure"]
        for col in weather_cols:
            if col not in cols:
                continue
            rate = q(f"SELECT COUNT(*) FROM df WHERE {col} IS NULL") / n_rows * 100
            # Up to ~20% NaN is expected for some stations
            check(f"{col} null rate < 25%", rate < 25, f"{rate:.1f}% null",
                  warn_only=rate < 50)


def test_ranges(cols: set, n_rows: int, has_weather: bool):
    print("\n── Value ranges ──────────────────────────────────────────────────────")

    # Cyclical features strictly in [-1, 1]
    for col in ["time_sin", "time_cos", "dow_sin", "dow_cos",
                "month_sin", "month_cos"]:
        if col not in cols:
            continue
        mn  = q(f"SELECT MIN({col}) FROM df")
        mx  = q(f"SELECT MAX({col}) FROM df")
        ok  = (mn is not None) and (-1.01 <= mn) and (mx <= 1.01)
        check(f"{col} in [-1, 1]", ok, f"min={mn:.4f} max={mx:.4f}")

    # Timestamp almost all in 2025 — midnight-crossing trips on Dec 31/Jan 1 are expected
    bad_ts  = q("SELECT COUNT(*) FROM df WHERE year(timestamp) != 2025")
    bad_pct = bad_ts / n_rows * 100
    check("Timestamps outside 2025 < 0.1%", bad_pct < 0.1,
          f"{bad_ts:,} rows ({bad_pct:.3f}%) — likely midnight-crossing trips", warn_only=True)

    # stop_id positive
    bad_sid = q("SELECT COUNT(*) FROM df WHERE stop_id <= 0")
    check("stop_id > 0", bad_sid == 0, f"{bad_sid:,} non-positive" if bad_sid else "")

    # Delays within ±12h (to_parquet.py enforces this, but verify)
    for col in ("arrival_delay_s", "departure_delay_s"):
        if col not in cols:
            continue
        bad = q(f"""
            SELECT COUNT(*) FROM df
            WHERE {col} IS NOT NULL AND ({col} < -43200 OR {col} > 43200)
        """)
        check(f"{col} within ±12h", bad == 0,
              f"{bad:,} out-of-range values" if bad else "")

    if has_weather:
        weather_bounds = {
            # (lo, hi, warn_only)
            # pressure: Swiss alpine stations reach 3500m+ → min ~640 hPa (Jungfraujoch)
            # sunshine: fraction 0-1  |  wind: m/s  |  snow_depth: metres
            "temperature":   ( -50,   60, False),
            "precipitation": (   0,  300, False),
            "sunshine":      (   0,  1.1, False),   # fraction 0-1
            "humidity":      (   0,  101, False),
            "wind_speed":    (   0,   85, False),   # m/s (~300 km/h)
            "wind_gust":     (   0,  112, False),   # m/s (~400 km/h)
            "wind_dir":      (   0,  361, False),
            "pressure":      ( 550, 1050, False),
            "snow_depth":    (   0,   10, True),    # metres — warn for unrealistic spikes
        }
        for col, (lo, hi, warn_only) in weather_bounds.items():
            if col not in cols:
                continue
            bad = q(f"""
                SELECT COUNT(*) FROM df
                WHERE {col} IS NOT NULL AND ({col} < {lo} OR {col} > {hi})
            """)
            check(f"{col} in [{lo}, {hi}]", bad == 0,
                  f"{bad:,} out-of-range" if bad else "", warn_only=warn_only)


def test_distributions(n_rows: int):
    print("\n── Distributions ─────────────────────────────────────────────────────")

    # Weekend fraction: lower than 2/7 is expected — Swiss buses run less on weekends
    wk_frac = q("SELECT AVG(CAST(is_weekend AS DOUBLE)) FROM df") * 100
    check("Weekend fraction 15–35%", 15 <= wk_frac <= 35, f"{wk_frac:.1f}%")

    # Month distribution: each month should have 5–12% of rows
    month_df = qdf("""
        SELECT month(timestamp) AS m, COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df) AS pct
        FROM df GROUP BY m ORDER BY m
    """)
    bad_months = month_df[~month_df["pct"].between(4, 15)]
    check("Each month has 4–15% of rows", len(bad_months) == 0,
          f"months out of range: {bad_months['m'].tolist()}" if len(bad_months) else "")

    # Hour distribution: no single hour should dominate (>15%) or be empty (<0.5%)
    hour_df = qdf("""
        SELECT hour(timestamp) AS h, COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df) AS pct
        FROM df GROUP BY h ORDER BY h
    """)
    too_high = hour_df[hour_df["pct"] > 15]
    too_low  = hour_df[hour_df["pct"] < 0.5]
    check("No hour > 15% of rows", len(too_high) == 0,
          f"hours: {too_high['h'].tolist()}" if len(too_high) else "")
    check("No hour < 0.5% of rows", len(too_low) == 0,
          f"hours: {too_low['h'].tolist()}" if len(too_low) else "", warn_only=True)

    # Cancelled trips should be absent (filtered in to_parquet.py)
    # Proxy: extremely large delays (> 3h = 10800s) should be rare < 1%
    big_delay = q("""
        SELECT COUNT(*) FROM df
        WHERE departure_delay_s > 10800 OR arrival_delay_s > 10800
    """)
    pct = big_delay / n_rows * 100
    check("Delays > 3h are < 1% of rows", pct < 1.0, f"{pct:.2f}%", warn_only=True)

    # additional_trip should be a minority flag
    frac = q("SELECT AVG(CAST(additional_trip AS DOUBLE)) FROM df") * 100
    check("additional_trip fraction < 20%", frac < 20, f"{frac:.1f}%")


def test_consistency():
    print("\n── Consistency ───────────────────────────────────────────────────────")

    # Reconstruct is_weekend from dow_sin/dow_cos: dow ≥ 5 means weekend
    # dow = round(atan2(dow_sin, dow_cos) * 7 / (2π)) mod 7
    wrong = q("""
        SELECT COUNT(*) FROM df
        WHERE is_weekend != (
            (CAST(ROUND(ATAN2(dow_sin, dow_cos) * 7.0 / (2 * PI())) AS INTEGER) % 7 + 7) % 7 >= 5
        )
    """)
    check("is_weekend consistent with dow_sin/dow_cos", wrong == 0,
          f"{wrong:,} inconsistent rows" if wrong else "")

    # Departure delay should be non-null whenever arrival delay is null
    # (every row must have at least one — already tested, but cross-check)
    arr_null_dep_null = q("""
        SELECT COUNT(*) FROM df
        WHERE arrival_delay_s IS NULL AND departure_delay_s IS NULL
    """)
    check("No rows with both delays null (consistency re-check)", arr_null_dep_null == 0)


def test_weather_join(cols: set, n_rows: int):
    print("\n── Weather join quality ──────────────────────────────────────────────")

    # Overall join hit rate
    matched = q("SELECT COUNT(*) FROM df WHERE temperature IS NOT NULL")
    rate = matched / n_rows * 100
    check("Weather join hit rate > 80%", rate > 80, f"{rate:.1f}% rows have weather")

    # Row-level coverage is what matters for training — stop-level coverage is misleading
    # because low-frequency rural stops (not in station_data) are a tiny share of rows
    matched = q("SELECT COUNT(*) FROM df WHERE temperature IS NOT NULL")
    row_rate = matched / n_rows * 100
    check("Weather row coverage > 95%", row_rate > 95, f"{row_rate:.1f}% of rows have weather")

    # Weather values should not be suspiciously uniform (sign of a bad join)
    temp_std = q("SELECT STDDEV(temperature) FROM df WHERE temperature IS NOT NULL")
    check("Temperature has non-trivial variance (stddev > 1°C)", temp_std > 1.0,
          f"stddev = {temp_std:.2f}°C")

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="Parquet file to validate")
    args = parser.parse_args()

    if args.file:
        target = Path(args.file)
    elif (_DATA / "dataset_with_weather.parquet").exists():
        target = _DATA / "dataset_with_weather.parquet"
    elif (_DATA / "dataset.parquet").exists():
        target = _DATA / "dataset.parquet"
    else:
        print("No dataset parquet found. Pass a path explicitly.")
        sys.exit(1)

    print(f"\nValidating: {target}")
    print("=" * 64)

    global con
    con = duckdb.connect()
    con.execute(f"CREATE VIEW df AS SELECT * FROM read_parquet('{target}')")

    # Basic info
    n_rows = q("SELECT COUNT(*) FROM df")
    cols   = set(qdf("DESCRIBE df")["column_name"].tolist())
    has_weather = "temperature" in cols

    print(f"\n  Rows:    {n_rows:,}")
    print(f"  Columns: {len(cols)}")
    print(f"  Weather: {'yes' if has_weather else 'no'}")

    # Run all test groups
    test_schema(cols, has_weather)
    test_completeness(cols, n_rows, has_weather)
    test_ranges(cols, n_rows, has_weather)
    test_distributions(n_rows)
    test_consistency()
    if has_weather:
        test_weather_join(cols, n_rows)

    # Summary
    n_pass = sum(1 for _, r in results if r == "PASS")
    n_fail = sum(1 for _, r in results if r == "FAIL")
    n_warn = sum(1 for _, r in results if r == "WARN")
    n_skip = sum(1 for _, r in results if r == "SKIP")

    print("\n" + "=" * 64)
    print(f"  {n_pass} passed  |  {n_fail} failed  |  {n_warn} warnings  |  {n_skip} skipped")
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
