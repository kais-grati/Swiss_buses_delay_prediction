import argparse
import os
import duckdb

MUNICIPALITIES = [
    "Bussigny",
    "Chavannes-près-Renens",
    "Crissier",
    "Ecublens (VD)",
    "Prilly",
    "Renens (VD)",
    "Saint-Sulpice (VD)",
    "Lausanne",
]

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to input parquet file")
parser.add_argument("-o", "--output", default="data/lausanne_region.parquet", help="Output parquet path (default: data/lausanne_region.parquet)")
parser.add_argument("--stations", default="data/station_data.parquet", help="Path to station_data parquet (default: data/station_data.parquet)")
args = parser.parse_args()

OUTPUT = args.output
SOURCE = args.input
STATIONS = args.stations

# Cap DuckDB at 2 GB RAM — it will spill to temp_directory instead of crashing
con = duckdb.connect()
con.execute("SET memory_limit = '2GB'")
os.makedirs("/tmp/duckdb_spill", exist_ok=True)
con.execute("SET temp_directory = '/tmp/duckdb_spill'")

print(f"Filtering {len(MUNICIPALITIES)} municipalities: {', '.join(MUNICIPALITIES)}")

# Step 1: get matching stop_ids from the tiny station_data file (~2.6 MB)
muni_list = ", ".join(f"'{m}'" for m in MUNICIPALITIES)
rows = con.execute(f"""
    SELECT DISTINCT TRY_CAST(s.number AS INTEGER) AS stop_id
    FROM read_parquet('{STATIONS}') s
    WHERE s.municipalityname IN ({muni_list})
""").fetchall()

stop_ids = [r[0] for r in rows if r[0] is not None]
print(f"Found {len(stop_ids)} unique stop IDs in target municipalities")

# Step 2: filter the large dataset with a WHERE IN clause.
# DuckDB pushes this down to the Parquet scanner — row groups without
# matching stop_ids are skipped, keeping memory usage low.
id_list = ", ".join(str(i) for i in stop_ids)
print("Extracting matching rows from dataset (this streams, not loads)...")
con.execute(f"""
    COPY (
        SELECT * FROM read_parquet('{SOURCE}')
        WHERE stop_id IN ({id_list})
    ) TO '{OUTPUT}' (FORMAT PARQUET)
""")

count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUTPUT}')").fetchone()[0]
print(f"Done. {count:,} rows written to {OUTPUT}")
