import duckdb

print("Filtering line 705, operator MBC Auto (all stops)...")

duckdb.query("""
    COPY (
        SELECT * FROM 'data/dataset_with_weather.parquet'
        WHERE line = '705'
          AND operator = 'MBC Auto'
    ) TO 'data/dataset_705.parquet' (FORMAT PARQUET)
""")

count = duckdb.query("SELECT COUNT(*) FROM 'data/dataset_705.parquet'").fetchone()[0]
print(f"Done. {count:,} rows written to data/dataset_705.parquet")
