import os
from huggingface_hub import snapshot_download

os.environ["HF_HOME"] = "/mnt/bn/liuwenzhuo-hl-data/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/bn/liuwenzhuo-hl-data/hf_cache/datasets"
os.environ["HF_METRICS_CACHE"] = "/mnt/bn/liuwenzhuo-hl-data/hf_cache/metrics"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/bn/liuwenzhuo-hl-data/hf_cache/transformers"

snapshot_download(
    repo_id="lmms-lab/LLaVA-NeXT-Data",
    repo_type="dataset",
    local_dir="/mnt/bn/liuwenzhuo-hl-data/hf_cache/LLaVA-NeXT-Data",
    max_workers=64,              # 线程数可调
    resume_download=True,
)
