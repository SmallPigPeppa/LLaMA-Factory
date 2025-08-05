from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lmms-lab/LLaVA-NeXT-Data",
    repo_type="dataset",
    cache_dir="/mnt/bn/liuwenzhuo-hl-data/hf_cache",
    max_workers=64,              # 线程数可调
    resume_download=True,
)
