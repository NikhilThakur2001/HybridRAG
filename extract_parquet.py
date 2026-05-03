import pandas as pd
import time

ZIP_PATH = "C:\\Users\\nikhi\\Downloads\\complaints.csv.zip"  # update this path if zip is elsewhere

chunk_size = 50_000
chunks = []
total_rows = 0
matched_rows = 0
chunk_num = 0
start = time.time()

print(f"Reading {ZIP_PATH} ...")
print(f"Chunk size: {chunk_size:,} rows\n")

# --- DEBUG: inspect first chunk only ---
_debug = pd.read_csv(ZIP_PATH, compression="zip", nrows=500, low_memory=False)
print("COLUMNS:", _debug.columns.tolist())
print("\nProduct value counts (top 20):")
if "Product" in _debug.columns:
    print(_debug["Product"].value_counts().head(20))
else:
    print("WARNING: 'Product' column not found!")
print("\nNarrative column exists:", "Consumer complaint narrative" in _debug.columns)
print("---\n")
del _debug
# --- END DEBUG ---

for chunk in pd.read_csv(
    ZIP_PATH,
    compression="zip",
    chunksize=chunk_size,
    low_memory=False
):
    chunk_num += 1
    total_rows += len(chunk)

    filtered = chunk[
        chunk["Product"].str.contains(
            "credit card|money transfer|prepaid|virtual currency|checking|savings",
            case=False, na=False
        ) &
        chunk["Consumer complaint narrative"].notna()
    ]
    
    matched_rows += len(filtered)
    chunks.append(filtered)

    elapsed = time.time() - start
    rate = total_rows / elapsed
    print(
        f"Chunk {chunk_num:>4} | "
        f"Rows read: {total_rows:>9,} | "
        f"Matched so far: {matched_rows:>7,} | "
        f"Elapsed: {elapsed:>6.1f}s | "
        f"Rate: {rate:>8,.0f} rows/s"
    )

print(f"\nConcatenating {len(chunks)} chunks ...")
telecom = pd.concat(chunks, ignore_index=True)

print(f"Saving parquet ...")
telecom.to_parquet("cfpb_telecom.parquet")

print(f"\nDone.")
print(f"Total rows scanned : {total_rows:,}")
print(f"Telecom rows kept  : {matched_rows:,}")
print(f"Parquet shape      : {telecom.shape}")
print(f"Total time         : {time.time() - start:.1f}s")
