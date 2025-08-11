import os
import glob
import math
import uuid
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ================== Config ==================
DATA_GLOB = "/mnt/bn/liuwenzhuo-hl-data/hf_cache/hub/datasets--lmms-lab--LLaVA-NeXT-Data/snapshots/c8aef391ce214167c4ebc7f06bc05a50dee2e75f/data/*.parquet"
OUT_DIR = "/mnt/bn/liuwenzhuo-hl-data/779k-shuffle"
TMP_DIR = os.path.join(OUT_DIR, "_tmp_shuffle")  # 中间桶目录
CHUNK_SIZE = 10000
SEED = 20250811
COMPRESSION = "snappy"
USE_THREADS = True
BATCH_SIZE = 16384               # 读取批大小，按机器内存可调
NUM_BUCKETS = 64                 # 桶数量，越大单桶越小；64~256 都可
RAND_COL = "_rand_key"           # 临时随机键列名
# ============================================

def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)

def _cast_large_recursive(t: pa.DataType) -> pa.DataType:
    """
    递归把变长类型升级为 Large*（LargeString/LargeBinary/LargeList），
    避免任意层级的 32-bit offset。
    """
    if pa.types.is_string(t):
        return pa.large_string()
    if pa.types.is_binary(t):
        return pa.large_binary()
    if pa.types.is_large_string(t) or pa.types.is_large_binary(t):
        return t
    # 处理 list 类型
    if pa.types.is_list(t):
        return pa.large_list(_cast_large_recursive(t.value_type))
    if pa.types.is_large_list(t):
        return pa.large_list(_cast_large_recursive(t.value_type))
    # 处理 struct/map（递归其子字段）
    if pa.types.is_struct(t):
        new_fields = [pa.field(f.name, _cast_large_recursive(f.type), f.nullable, f.metadata)
                      for f in t]
        return pa.struct(new_fields)
    if pa.types.is_map(t):
        # map<k,v> 是 list<struct<key,value>> 的语义，递归键/值
        key_t = _cast_large_recursive(t.key_type)
        item_t = _cast_large_recursive(t.item_type)
        # pyarrow 没有 LargeMap；保持 map，但其内部列表会是 large_list
        return pa.map_(key_t, item_t)
    return t

def cast_table_large_all(table: pa.Table) -> pa.Table:
    new_fields = []
    need_cast = False
    for f in table.schema:
        new_t = _cast_large_recursive(f.type)
        new_fields.append(pa.field(f.name, new_t, f.nullable, f.metadata))
        if new_t != f.type:
            need_cast = True
    return table.cast(pa.schema(new_fields)) if need_cast else table

def ensure_clean_dir(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def pass1_bucketize(files):
    """
    第一遍：流式读取 → 升级 Large* → 生成随机键 → 分桶写 parquet 文件。
    """
    ensure_clean_dir(TMP_DIR)
    # 每个桶一个 ParquetWriter，避免频繁打开关闭
    writers = {}
    try:
        dataset = ds.dataset(files, format="parquet")
        # 估个总行数用于进度条（可能略慢；如果不想统计，可以去掉）
        try:
            n_rows = dataset.count_rows()
        except Exception:
            n_rows = None

        scanner = dataset.scan(use_threads=USE_THREADS, batch_size=BATCH_SIZE)
        it = scanner.to_batches()
        it = _progress(it, desc="Pass1: bucketizing", unit="batch")

        if SEED is not None:
            np.random.seed(SEED)

        total_rows = 0
        for batch in it:
            # 升级 Large*，并对 batch 级别做转换（更省内存）
            tbl = pa.Table.from_batches([batch])
            tbl = cast_table_large_all(tbl)

            # 生成随机键（float64）并追加到表
            r = np.random.random(len(tbl))
            rand_arr = pa.array(r, type=pa.float64())
            tbl = tbl.append_column(RAND_COL, rand_arr)

            # 计算桶 id
            # 用 Arrow 的 hash 不稳定版本很多，直接用 numpy 对随机键进行分桶
            bucket_ids = (np.floor(rand_arr.to_numpy() * NUM_BUCKETS)).astype(np.int64)

            # 按桶切分追加写入
            for b in np.unique(bucket_ids):
                mask = pa.array(bucket_ids == b)
                sub_tbl = tbl.filter(mask)

                # 初始化 writer（保持 schema 一致）
                if b not in writers:
                    out_path = os.path.join(TMP_DIR, f"bucket-{int(b):03d}.parquet")
                    writers[b] = pq.ParquetWriter(
                        out_path, sub_tbl.schema, compression=COMPRESSION
                    )
                writers[b].write_table(sub_tbl)

            total_rows += len(tbl)
            if tqdm is None and n_rows is not None and total_rows % (BATCH_SIZE * 10) == 0:
                print(f"Pass1… {total_rows}/{n_rows} rows")

    finally:
        for w in writers.values():
            w.close()

def pass2_sort_and_emit(out_dir: str):
    """
    第二遍：逐桶读取 → 以随机键排序 → 去掉随机键 → 按 CHUNK_SIZE 写出最终段。
    """
    os.makedirs(out_dir, exist_ok=True)
    bucket_files = sorted(glob.glob(os.path.join(TMP_DIR, "bucket-*.parquet")))
    if not bucket_files:
        raise RuntimeError("No bucket files produced in TMP_DIR; pass1 failed?")

    part = 0
    it = _progress(bucket_files, desc="Pass2: finalize", unit="bucket")
    for bf in it:
        tbl = pq.read_table(bf, use_threads=USE_THREADS)
        # 以随机键排序（稳定全局洗牌）
        order = pc.sort_indices(tbl, sort_keys=[(RAND_COL, "ascending")])
        tbl = pc.take(tbl, order)
        # 删除随机键列
        tbl = tbl.drop_columns([RAND_COL])

        # 输出分块
        n = len(tbl)
        for start in range(0, n, CHUNK_SIZE):
            chunk = tbl.slice(start, min(CHUNK_SIZE, n - start))
            out_path = os.path.join(out_dir, f"part-{part:05d}.parquet")
            pq.write_table(chunk, out_path, compression=COMPRESSION)
            if tqdm is None:
                print(f"Saved {os.path.basename(out_path)} ({len(chunk):,} rows)")
            part += 1

    if tqdm is None:
        print(f"Done. Wrote {part} files to: {out_dir}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No parquet files found: {DATA_GLOB}")

    print(f"Discovered {len(files)} parquet files. Two-pass external shuffle starting…")
    # Pass 1：分桶
    pass1_bucketize(files)
    # Pass 2：逐桶排序并输出
    pass2_sort_and_emit(OUT_DIR)
    # 清理中间文件（如果想保留可注释掉）
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("All done.")

if __name__ == "__main__":
    main()
