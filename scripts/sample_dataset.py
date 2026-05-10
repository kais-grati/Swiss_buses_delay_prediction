"""Create a smaller representative sample of a parquet dataset.

Two sampling modes:
  - reservoir (default): Single-pass uniform random sample via DuckDB.
    Works on datasets of any size with bounded memory.
  - stratified: Samples proportionally within quantile bins of a column
    so the output preserves the distribution of that column. Reads the
    stratify column in full (one column, low memory) then reservoir-
    samples per stratum. Best when you need the sample to match the
    original distribution.

Examples:
  # 1M-row uniform sample
  python sample_dataset.py data/dataset.parquet -o data/sample.parquet -n 1_000_000

  # 10% stratified sample preserving arrival_delay_s distribution
  python sample_dataset.py data/dataset.parquet -o data/sample.parquet --frac 0.1 --stratify-on arrival_delay_s

  # Exact row count with stratification and custom seed
  python sample_dataset.py data/dataset.parquet -o data/sample.parquet -n 500000 --stratify-on arrival_delay_s --seed 123
"""

import argparse
import sys

import duckdb
import numpy as np
import pandas as pd


def reservoir_sample(path: str, output: str, n: int, seed: int) -> int:
    """Single-pass reservoir sampling. Returns row count written."""
    duckdb.query(f"""
        COPY (
            SELECT * FROM read_parquet('{path}')
            USING SAMPLE {n} ROWS (reservoir, {seed})
        ) TO '{output}' (FORMAT PARQUET)
    """)
    return n


def stratified_sample(
    path: str, output: str, n: int, stratify_on: str, seed: int, n_bins: int = 10
) -> int:
    """Sample proportionally within quantile bins of `stratify_on`.

    Reads only the stratify column to compute bin edges, then uses a single
    DuckDB query with row_number() per stratum. The full dataset is streamed,
    never fully loaded into memory.
    """
    # 1. Read just the stratify column to compute bin edges
    series = duckdb.query(
        f"SELECT {stratify_on} FROM read_parquet('{path}')"
    ).df()[stratify_on]

    total_rows = len(series)

    # 2. Bin into quantiles
    n_bins = min(n_bins, series.nunique())
    if n_bins < 2:
        print(f"  Column '{stratify_on}' has only {series.nunique()} unique value(s) — "
              f"falling back to reservoir sampling.")
        return reservoir_sample(path, output, n, seed)

    try:
        bins = pd.qcut(series, q=n_bins, duplicates="drop", retbins=True)[1]
    except ValueError:
        bins = pd.cut(series, bins=n_bins, retbins=True)[1]
    n_bins = len(bins) - 1
    del series

    edges = [float(b) for b in bins]
    print(f"  {n_bins} strata from {stratify_on}: edges = {[f'{e:.1f}' for e in edges]}")

    # 3. Count rows per stratum via DuckDB (consistent counting)
    when_clauses = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            cond = f"{stratify_on} <= {hi}"
        elif i == n_bins - 1:
            cond = f"{stratify_on} > {lo}"
        else:
            cond = f"{stratify_on} > {lo} AND {stratify_on} <= {hi}"
        when_clauses.append(f"WHEN {cond} THEN {i}")

    case_expr = f"CASE {' '.join(when_clauses)} END"
    stratum_counts = duckdb.query(
        f"SELECT {case_expr} AS stratum, count(*) AS cnt "
        f"FROM read_parquet('{path}') "
        f"GROUP BY stratum ORDER BY stratum"
    ).df()

    # 4. Proportional allocation (at least 1 row per stratum)
    counts = stratum_counts["cnt"].values
    props = counts / counts.sum()
    per_stratum = np.maximum(1, (props * n).round().astype(int))
    # Adjust to hit target n exactly
    diff = n - per_stratum.sum()
    while diff != 0:
        idx = per_stratum.argmax() if diff > 0 else per_stratum.argmin()
        per_stratum[idx] += 1 if diff > 0 else -1
        diff = n - per_stratum.sum()

    print(f"  Target: {n:,} rows across {n_bins} strata")
    for i, (count, target) in enumerate(zip(counts, per_stratum)):
        lo, hi = edges[i], edges[i + 1]
        print(f"    Stratum {i} ({lo:.0f}, {hi:.0f}]: {count:,} → {target:,}")

    # 5. Single-query stratified sample via row_number() per stratum
    duckdb.query(f"SELECT setseed({seed} * 0.371)")  # deterministic random() for reproducibility
    limits = ",".join(str(int(x)) for x in per_stratum)
    duckdb.query(f"""
        COPY (
            SELECT * EXCLUDE (_stratum, _rn) FROM (
                SELECT *,
                    {case_expr} AS _stratum,
                    row_number() OVER (
                        PARTITION BY _stratum ORDER BY random()
                    ) AS _rn
                FROM read_parquet('{path}')
            )
            WHERE _rn <= list_value({limits})[_stratum + 1]
        ) TO '{output}' (FORMAT PARQUET)
    """)

    return n


def main():
    parser = argparse.ArgumentParser(
        description="Create a smaller representative sample of a parquet dataset."
    )
    parser.add_argument("input", help="Path to the input parquet file.")
    parser.add_argument("-o", "--output", required=True, help="Path for the output parquet file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-n", "--rows", type=_parse_int, help="Number of rows to sample.")
    group.add_argument("--frac", type=float, help="Fraction of rows to sample (e.g. 0.1 for 10%%).")
    parser.add_argument(
        "--stratify-on",
        help="Column to stratify by (preserves its distribution via quantile bins).",
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Number of quantile bins for stratification (default: 10).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    # Resolve row count
    if args.frac is not None:
        if not 0 < args.frac < 1:
            print("Error: --frac must be between 0 and 1.")
            sys.exit(1)
        n_rows = duckdb.query(
            f"SELECT count(*) FROM read_parquet('{args.input}')"
        ).fetchone()[0]
        n = max(1, int(n_rows * args.frac))
        print(f"Total rows: {n_rows:,}  →  target: {n:,} ({args.frac:.1%})")
    else:
        n = args.rows

    mode = "stratified" if args.stratify_on else "reservoir"
    print(f"Sampling {n:,} rows from {args.input}  [{mode}]")

    if args.stratify_on:
        written = stratified_sample(args.input, args.output, n, args.stratify_on, args.seed, args.n_bins)
    else:
        written = reservoir_sample(args.input, args.output, n, args.seed)

    # Verify
    actual = duckdb.query(f"SELECT count(*) FROM read_parquet('{args.output}')").fetchone()[0]
    print(f"Wrote {actual:,} rows to {args.output}")


def _parse_int(value: str) -> int:
    """Accept underscores as thousands separators."""
    return int(value.replace("_", ""))


if __name__ == "__main__":
    main()
