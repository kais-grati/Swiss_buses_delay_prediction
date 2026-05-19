import argparse
import duckdb

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to input parquet file")
parser.add_argument("-o", "--output", default="data/705.parquet", help="Output parquet path (default: data/705.parquet)")
args = parser.parse_args()

print(f"Filtering line 705, operator MBC Auto from {args.input}...")

duckdb.query(f"""
    COPY (
        SELECT * FROM '{args.input}'
        WHERE line = '705'
          AND operator = 'MBC Auto'
    ) TO '{args.output}' (FORMAT PARQUET)
""")

count = duckdb.query(f"SELECT COUNT(*) FROM '{args.output}'").fetchone()[0]
print(f"Done. {count:,} rows written to {args.output}")
