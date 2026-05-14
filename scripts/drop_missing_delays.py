import os
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DELAY_COLS = ["arrival_delay_s", "departure_delay_s"]

SRC = "data/dataset_with_weather.parquet"
TMP = "data/dataset_with_weather.parquet.tmp"

print("Dropping rows with missing arrival or departure delay...")

pf = pq.ParquetFile(SRC)
total_before = pf.metadata.num_rows
total_after = 0
total_seen = 0

writer = None
try:
    for batch in pf.iter_batches():
        table = pa.Table.from_batches([batch])
        masks = [table[col].is_valid() for col in DELAY_COLS]
        mask = masks[0]
        for m in masks[1:]:
            mask = pc.and_(mask, m)
        filtered = table.filter(mask)
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
print(f"Dropped {dropped:,} rows ({dropped * 100 / total_before:.3f}%)")
print(f"Remaining: {total_after:,} rows")
