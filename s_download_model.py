from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    repo_type="model",
    local_dir="/mnt/bn/liuwenzhuo-hl-data/Qwen/Qwen2.5-VL-3B-Instruct",
    max_workers=32,
    resume_download=True,
)
