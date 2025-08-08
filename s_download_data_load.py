from huggingface_hub import snapshot_download
# /home/tiger/hf_cache
# /mnt/bn/liuwenzhuo-hl-data/hf_cache
# snapshot_download(
#     repo_id="lmms-lab/LLaVA-NeXT-Data",
#     repo_type="dataset",
#     cache_dir="/home/tiger/hf_cache_new",
#     max_workers=128,              # 线程数可调
#     resume_download=True,
# )
from datasets import load_dataset

ds = load_dataset("lmms-lab/LLaVA-NeXT-Data",cache_dir="/home/tiger/hf_cache_tmp",num_proc=64)