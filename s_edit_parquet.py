import pandas as pd
import glob
import os
from concurrent.futures import ThreadPoolExecutor

src_dir = "/mnt/bn/liuwenzhuo-hl-data/hf_cache/hub/datasets--lmms-lab--LLaVA-NeXT-Data/snapshots/c8aef391ce214167c4ebc7f06bc05a50dee2e75f/data"
dst_dir = "/mnt/bn/liuwenzhuo-hl-data/779k/data_renamed"
max_workers=64
os.makedirs(dst_dir, exist_ok=True)

files = glob.glob(os.path.join(src_dir, "*.parquet"))

rename_dict = {
    "id": "id",
    "conversations": "messages",
    "data_source": "data_source",
    "image": "images"
}

def process_file(file):
    try:
        df = pd.read_parquet(file)
        df = df.rename(columns=rename_dict)
        dst_file = os.path.join(dst_dir, os.path.basename(file))
        df.to_parquet(dst_file)
        print(f"Processed: {os.path.basename(file)}")
    except Exception as e:
        print(f"Error: {file}, {e}")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(process_file, files)
