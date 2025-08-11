import os
import glob
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc

# Config
DATA_GLOB = "/mnt/bn/liuwenzhuo-hl-data/hf_cache/hub/datasets--lmms-lab--LLaVA-NeXT-Data/snapshots/c8aef391ce214167c4ebc7f06bc05a50dee2e75f/data/*.parquet"
OUT_DIR = "/mnt/bn/liuwenzhuo-hl-data/779k-shuffle"
CHUNK_SIZE = 10000
SEED = 20250811
COMPRESSION = "snappy"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all parquet files into one table
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No parquet files found: {DATA_GLOB}")
    table = ds.dataset(files, format="parquet").to_table(use_threads=True)
    print(f"Loaded {len(table):,} rows, {table.num_columns} columns.")

    # Shuffle
    if SEED is not None:
        np.random.seed(SEED)
    idx = pa.array(np.random.permutation(len(table)), type=pa.int64())
    shuffled = pc.take(table, idx)

    # Save in chunks
    for part, start in enumerate(range(0, len(shuffled), CHUNK_SIZE)):
        chunk = shuffled.slice(start, min(CHUNK_SIZE, len(shuffled) - start))
        pq.write_table(chunk, os.path.join(OUT_DIR, f"part-{part:05d}.parquet"),
                       compression=COMPRESSION)
        print(f"Saved part-{part:05d}.parquet ({len(chunk):,} rows)")

if __name__ == "__main__":
    main()
