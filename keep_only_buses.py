import os
from multiprocessing import Pool, cpu_count

DATA_DIR = "data"
CLEANED_DIR = "cleaned_data"

os.makedirs(CLEANED_DIR, exist_ok=True)


def process_file(filename):
    src = os.path.join(DATA_DIR, filename)
    dst = os.path.join(CLEANED_DIR, filename)

    kept = 0
    total = 0

    with open(src, "rb") as infile, open(dst, "wb") as outfile:
        header = infile.readline()
        outfile.write(header)
        cols = header.decode().strip().split(";")
        idx = cols.index("PRODUCT_ID")

        for line in infile:
            total += 1
            if line.split(b";")[idx].lower() == b"bus":
                outfile.write(line)
                kept += 1

    return filename, kept, total


if __name__ == "__main__":
    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

    workers = cpu_count()
    print(f"Processing {len(csv_files)} files with {workers} workers...\n")

    with Pool(workers) as pool:
        for filename, kept, total in pool.imap_unordered(process_file, csv_files):
            print(f"{filename}: {kept}/{total} rows kept")

    print("\nDone.")
