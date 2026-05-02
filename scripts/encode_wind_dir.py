import os
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = "data/dataset_with_weather.parquet"
TMP = "data/dataset_with_weather.parquet.tmp"

print("Encoding wind_dir → wind_dir_sin / wind_dir_cos...")

pf = pq.ParquetFile(SRC)
n_groups = pf.metadata.num_row_groups
total_rows = pf.metadata.num_rows
writer = None
rows_written = 0

try:
    for i, batch in enumerate(pf.iter_batches()):
        table = pa.Table.from_batches([batch])
        radians = pc.multiply(table["wind_dir"], np.pi / 180.0)
        table = table.append_column("wind_dir_sin", pc.sin(radians))
        table = table.append_column("wind_dir_cos", pc.cos(radians))
        table = table.remove_column(table.schema.get_field_index("wind_dir"))
        if writer is None:
            writer = pq.ParquetWriter(TMP, table.schema, compression="snappy")
        writer.write_table(table)
        rows_written += len(table)
        print(f"  {i + 1}/{n_groups} row groups processed ({rows_written:,}/{total_rows:,} rows)...")
finally:
    if writer:
        writer.close()

os.replace(TMP, SRC)
size_gb = os.path.getsize(SRC) / 1e9
print(f"Done. {rows_written:,} rows written → {SRC} ({size_gb:.2f} GB)")
