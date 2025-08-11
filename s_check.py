import glob
import pandas as pd
from tqdm import tqdm

# 文件路径模式
data_files = "/mnt/bn/liuwenzhuo-hl-data/hf_cache/hub/datasets--lmms-lab--LLaVA-NeXT-Data/snapshots/c8aef391ce214167c4ebc7f06bc05a50dee2e75f/data/*.parquet"

# 找到所有 parquet 文件
parquet_files = glob.glob(data_files)

if not parquet_files:
    print("没有找到任何 Parquet 文件，请检查路径是否正确。")
else:
    for file in tqdm(parquet_files, desc="Processing files"):
        try:
            # 只读取 conversations 列
            df = pd.read_parquet(file, columns=["conversations"])
        except Exception as e:
            print(f"读取 {file} 时出错: {e}")
            continue

        # 遍历并查找包含 <video> 的行
        for conv in df["conversations"]:
            if pd.isna(conv):
                continue
            if "<video>" in str(conv):
                print(conv)
