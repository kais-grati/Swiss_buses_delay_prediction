import os
from multiprocessing import Pool, cpu_count

DATA_DIR = "cleaned_data"

KEEP_COLS = {
    "DATE",
    "OPERATOR_ABB",
    "LINE_NAME",
    "BPUIC",
    "STOP_NAME",
    "ARRIVAL_TIME",
    "ARRIVAL_FORECAST",
    "ARRIVAL_FORECAST_STATUS",
    "DEPARTURE_TIME",
    "DEPARTURE_FORECAST",
    "DEPARTURE_FORECAST_STATUS",
    "ADDITIONAL_TRIP",
    "CANCELLED",
    "PASS_THROUGH",
}

# Remove leftover .tmp files from a previous crashed run
for f in os.listdir(DATA_DIR):
    if f.endswith(".tmp"):
        os.remove(os.path.join(DATA_DIR, f))


def already_processed(filepath):
    """Return True if the file header already has only the kept columns."""
    with open(filepath, "rb") as f:
        cols = set(f.readline().decode().strip().split(";"))
    return cols == KEEP_COLS


def process_file(filename):
    src = os.path.join(DATA_DIR, filename)
    tmp = src + ".tmp"

    if already_processed(src):
        return filename, -1, []

    with open(src, "rb") as infile, open(tmp, "wb") as outfile:
        raw_header = infile.readline()
        cols = raw_header.decode().strip().split(";")
        keep_indices = [i for i, c in enumerate(cols) if c in KEEP_COLS]

        out_header = ";".join(cols[i] for i in keep_indices).encode() + b"\n"
        outfile.write(out_header)

        row_count = 0
        for line in infile:
            fields = line.rstrip(b"\r\n").split(b";")
            outfile.write(b";".join(fields[i] if i < len(fields) else b"" for i in keep_indices) + b"\n")
            row_count += 1

    os.replace(tmp, src)
    return filename, row_count, keep_indices


if __name__ == "__main__":
    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

    workers = cpu_count()
    print(f"Processing {len(csv_files)} files with {workers} workers...\n")

    with Pool(workers) as pool:
        for filename, rows, _ in pool.imap_unordered(process_file, csv_files):
            if rows == -1:
                print(f"{filename}: skipped (already processed)")
            else:
                print(f"{filename}: {rows} rows")

    print("\nDone.")
