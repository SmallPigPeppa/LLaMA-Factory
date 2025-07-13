from huggingface_hub import snapshot_download

# 填入你实际用的 model_id，比如 "Qwen/Qwen2.5-VL-7B-Instruct"
local_dir = "~/qwen2_5_vl_my"
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir=local_dir,
    trust_remote_code=True,   # 如果仓库里带有自定义 code
)
