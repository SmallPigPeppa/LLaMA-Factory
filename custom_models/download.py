from huggingface_hub import snapshot_download

# local_dir = "/home/tiger/qwen2_5_vl_7b"
# snapshot_download(
#     repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
#     local_dir=local_dir,
#     # 如果你需要访问私有仓库，这里可以加 token：
#     # use_auth_token="YOUR_HF_TOKEN",
# )

local_dir = "/home/tiger/qwen2_5_vl_3b"
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    local_dir=local_dir,
    # 如果你需要访问私有仓库，这里可以加 token：
    # use_auth_token="YOUR_HF_TOKEN",
)
