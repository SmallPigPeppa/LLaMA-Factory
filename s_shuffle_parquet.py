# -*- coding: utf-8 -*-
# 需求：
# 1) 一次性读取所有 parquet
# 2) 打乱全部行
# 3) 保存到 /mnt/bn/liuwenzhuo-hl-data/779k-shuffle
# 4) 每个 parquet 最多 100000 行

import os
import glob
import pandas as pd

# ======= 可按需修改的参数 =======
DATA_GLOB = "/mnt/bn/liuwenzhuo-hl-data/hf_cache/hub/datasets--lmms-lab--LLaVA-NeXT-Data/snapshots/c8aef391ce214167c4ebc7f06bc05a50dee2e75f/data/*.parquet"
OUT_DIR = "/mnt/bn/liuwenzhuo-hl-data/779k-shuffle"
CHUNK_SIZE = 10000
RANDOM_STATE = 42  # 想要可复现的随机顺序就固定种子
# =================================

os.makedirs(OUT_DIR, exist_ok=True)

# 1) 读取所有 parquet 文件
files = sorted(glob.glob(DATA_GLOB))
if not files:
    raise FileNotFoundError(f"未匹配到任何 parquet 文件，检查路径通配符是否正确：{DATA_GLOB}")

print(f"发现 {len(files)} 个文件，开始读取…")
dfs = []
for f in files:
    df_part = pd.read_parquet(f, engine="pyarrow")
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)
print(f"合并后总行数：{len(df):,}")

# 2) 全量 shuffle
df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

# 3) & 4) 切片保存为新的 parquet（每个最多 CHUNK_SIZE 行）
n_rows = len(df)
n_parts = (n_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"将写出 {n_parts} 个 parquet，每个最多 {CHUNK_SIZE:,} 行 → {OUT_DIR}")

for i in range(n_parts):
    start = i * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, n_rows)
    out_path = os.path.join(OUT_DIR, f"part-{i:05d}.parquet")
    df.iloc[start:end].to_parquet(out_path, engine="pyarrow", index=False)
    print(f"[{i+1}/{n_parts}] 写出 {out_path} 行数={end-start}")

print("完成 ✅")
