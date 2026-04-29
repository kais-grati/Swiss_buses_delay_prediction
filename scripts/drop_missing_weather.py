import os
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

WEATHER_COLS = [
    "temperature", "precipitation", "sunshine", "humidity",
    "wind_speed", "wind_gust", "wind_dir", "pressure", "snow_depth",
]

SRC = "data/dataset_with_weather.parquet"
TMP = "data/dataset_with_weather.parquet.tmp"

print("Dropping rows with any missing weather field...")

pf = pq.ParquetFile(SRC)
total_before = pf.metadata.num_rows
total_after = 0

writer = None
try:
    for i, batch in enumerate(pf.iter_batches()):
        table = pa.Table.from_batches([batch])
        masks = [table[col].is_valid() for col in WEATHER_COLS]
        mask = masks[0]
        for m in masks[1:]:
            mask = pc.and_(mask, m)
        filtered = table.filter(mask)
        total_after += len(filtered)
        if writer is None:
            writer = pq.ParquetWriter(TMP, filtered.schema)
        writer.write_table(filtered)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{pf.num_row_groups} row groups processed...")
finally:
    if writer:
        writer.close()

os.replace(TMP, SRC)

dropped = total_before - total_after
print(f"Dropped {dropped:,} rows ({dropped * 100 / total_before:.3f}%)")
print(f"Remaining: {total_after:,} rows")
