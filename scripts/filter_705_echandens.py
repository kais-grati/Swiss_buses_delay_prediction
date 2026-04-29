import duckdb

print("Filtering line 705, stop 'Echandens, Chocolatière'...")

duckdb.query("""
    COPY (
        SELECT * FROM 'data/dataset_with_weather.parquet'
        WHERE line = '705'
          AND stop_name = 'Echandens, Chocolatière'
    ) TO 'data/dataset_705_echandens.parquet' (FORMAT PARQUET)
""")

count = duckdb.query("SELECT COUNT(*) FROM 'data/dataset_705_echandens.parquet'").fetchone()[0]
print(f"Done. {count:,} rows written to data/dataset_705_echandens.parquet")
