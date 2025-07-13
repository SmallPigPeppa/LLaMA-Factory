from huggingface_hub import snapshot_download

# 填入你实际用的 model_id，比如 "Qwen/Qwen2.5-VL-7B-Instruct"
local_dir = "~/qwen2_5_vl_my"
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir=local_dir,
    # 如果你需要访问私有仓库，这里可以加 token：
    # use_auth_token="YOUR_HF_TOKEN",
)
