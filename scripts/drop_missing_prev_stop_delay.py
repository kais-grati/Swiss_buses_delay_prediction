"""
Drop rows with missing prev_stop_delay from dataset_with_weather.parquet.

Streams row groups to avoid loading the 8.5 GB file into memory.
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "data/swiss_bus_2026_weather.parquet"
TMP = "data/dataset_with_weather.parquet.tmp"

print("Dropping rows with missing prev_stop_delay...")

pf = pq.ParquetFile(SRC)
total_before = pf.metadata.num_rows
total_seen = 0
total_after = 0

writer = None
try:
    for batch in pf.iter_batches():
        table = pa.Table.from_batches([batch])
        filtered = table.filter(table["prev_stop_delay"].is_valid())
        total_seen += len(table)
        total_after += len(filtered)
        if writer is None:
            writer = pq.ParquetWriter(TMP, filtered.schema)
        writer.write_table(filtered)
        if total_seen % 5_000_000 == 0:
            print(f"  {total_seen:,}/{total_before:,} rows processed...")
finally:
    if writer:
        writer.close()

os.replace(TMP, SRC)

dropped = total_before - total_after
print(f"Dropped {dropped:,} rows ({dropped * 100 / total_before:.2f}%)")
print(f"Remaining: {total_after:,} rows")
