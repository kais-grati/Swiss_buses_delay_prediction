import os
import pyarrow as pa
import pyarrow.parquet as pq

THRESHOLD_S = 1800  # 30 minutes
SRC = "data/dataset_with_weather.parquet"
TMP = "data/dataset_with_weather.parquet.tmp"

print(f"Dropping rows with arrival_delay_s or departure_delay_s > {THRESHOLD_S}s ({THRESHOLD_S // 60} min)...")

pf = pq.ParquetFile(SRC)
total_before = pf.metadata.num_rows
total_after = 0

writer = None
try:
    for i, batch in enumerate(pf.iter_batches()):
        table = pa.Table.from_batches([batch])
        mask = (
            (table["arrival_delay_s"].to_pylist() is not None) and True  # ensure column exists
        )
        # Filter using pyarrow compute
        import pyarrow.compute as pc
        filtered = table.filter(
            pc.and_(
                pc.less_equal(table["arrival_delay_s"], THRESHOLD_S),
                pc.less_equal(table["departure_delay_s"], THRESHOLD_S),
            )
        )
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
