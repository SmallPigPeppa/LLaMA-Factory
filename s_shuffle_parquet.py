import os
import glob
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc

# ---- optional: tqdm progress bar (graceful fallback) ----
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ================== Config ==================
DATA_GLOB = "/mnt/bn/liuwenzhuo-hl-data/hf_cache/hub/datasets--lmms-lab--LLaVA-NeXT-Data/snapshots/c8aef391ce214167c4ebc7f06bc05a50dee2e75f/data/*.parquet"
OUT_DIR = "/mnt/bn/liuwenzhuo-hl-data/779k-shuffle"
CHUNK_SIZE = 10000                # rows per output file
SEED = 20250811
COMPRESSION = "snappy"
USE_THREADS = True                # arrow parallel read
# ============================================


def _progress(iterable, **kwargs):
    """tqdm wrapper with graceful fallback."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def cast_to_large_types(table: pa.Table) -> pa.Table:
    """
    Cast string/binary to their Large* counterparts (64-bit offsets)
    to avoid 'offset overflow while concatenating arrays'.
    """
    fields = []
    for f in table.schema:
        t = f.type
        if pa.types.is_string(t):
            fields.append((f.name, pa.large_string()))
        elif pa.types.is_binary(t):
            fields.append((f.name, pa.large_binary()))
        elif pa.types.is_large_string(t) or pa.types.is_large_binary(t):
            fields.append((f.name, t))  # already large*
        else:
            fields.append((f.name, t))
    large_schema = pa.schema([(name, typ) for name, typ in fields])

    # Only cast if needed to avoid no-op cost
    need_cast = any(
        (pa.types.is_string(f.type) or pa.types.is_binary(f.type))
        for f in table.schema
    )
    return table.cast(large_schema) if need_cast else table


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Discover files ----
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No parquet files found: {DATA_GLOB}")

    # ---- Load dataset -> table ----
    # (Arrow will read lazily via dataset; .to_table() materializes.)
    if tqdm:
        print(f"Discovered {len(files)} parquet files. Loading…")
    table = ds.dataset(files, format="parquet").to_table(use_threads=USE_THREADS)
    n_rows = len(table)
    n_cols = table.num_columns
    print(f"Loaded {n_rows:,} rows, {n_cols} columns.")

    # ---- Cast to Large* to avoid 32-bit offset overflow ----
    table = cast_to_large_types(table)

    # ---- Prepare permutation indices (numpy) ----
    if SEED is not None:
        np.random.seed(SEED)
    perm = np.random.permutation(n_rows)  # numpy array of int64

    # ---- Chunked take + write with progress ----
    total_parts = (n_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    iterator = range(0, n_rows, CHUNK_SIZE)
    iterator = _progress(
        iterator,
        total=total_parts,
        desc="Shuffling + writing",
        unit="part",
        leave=True,
    )

    part = 0
    for start in iterator:
        end = min(start + CHUNK_SIZE, n_rows)
        idx_chunk = pa.array(perm[start:end], type=pa.int64())
        # take only this chunk (avoid building a giant shuffled table)
        chunk_tbl = pc.take(table, idx_chunk)

        out_path = os.path.join(OUT_DIR, f"part-{part:05d}.parquet")
        pq.write_table(
            chunk_tbl,
            out_path,
            compression=COMPRESSION
        )

        if tqdm is None:
            print(f"Saved part-{part:05d}.parquet ({len(chunk_tbl):,} rows)")

        part += 1

    print(f"Done. Wrote {part} files to: {OUT_DIR}")


if __name__ == "__main__":
    main()
