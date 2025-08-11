# edit_qwen_safetensors.py
import os, json, math
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file

# ======== 配置（按需改） ========
CKPT_DIR = "/mnt/bn/liuwenzhuo-hl-data/Qwen/Qwen2.5-VL-3B-Instruct/"  # 目录，内含 index.json 与分片
OUT_PATH = os.path.join(CKPT_DIR, "model_modified.safetensors")


level_seq = [1, 2, 3]
original_patch_size = 24

# ======== 读分片到 state_dict ========
index_path = os.path.join(CKPT_DIR, "model.safetensors.index.json")
assert os.path.exists(index_path), f"index not found: {index_path}"

with open(index_path, "r") as f:
    index = json.load(f)
weight_map = index["weight_map"]

state_dict = {}
opened = {}
for key, shard_name in weight_map.items():
    shard_path = os.path.join(CKPT_DIR, shard_name)
    if shard_path not in opened:
        opened[shard_path] = safe_open(shard_path, framework="pt", device="cpu")
    state_dict[key] = opened[shard_path].get_tensor(key)
opened.clear()

import pdb;pdb.set_trace()

# ======== Step 1: 替换 patchifier 为多尺度 ========
base_w_key = "module.model.vision_model.preprocessor.patchifier.proj.weight"
base_b_key = "module.model.vision_model.preprocessor.patchifier.proj.bias"
norm_w_key = "module.model.vision_model.preprocessor.patchifier.norm.weight"

assert base_w_key in state_dict and norm_w_key in state_dict, "找不到 patchifier 的权重键，请核对键名"

base_weight = state_dict[base_w_key]
base_bias   = state_dict.get(base_b_key, None)
norm_weight = state_dict[norm_w_key]

for idx, ratio in enumerate(level_seq):
    patch_size = original_patch_size // ratio
    prefix = f"module.model.vision_model.preprocessor.patchifier.patch_embed_seq.{idx}"

    resized_weight = pi_resize(base_weight, target_size=patch_size)
    state_dict[f"{prefix}.proj.weight"] = resized_weight.clone()
    if base_bias is not None:
        state_dict[f"{prefix}.proj.bias"] = base_bias.clone()
    state_dict[f"{prefix}.norm.weight"] = norm_weight.clone()

# 删除旧的单尺度 patchifier 参数（可选但推荐）
for k in [base_w_key, base_b_key, norm_w_key]:
    state_dict.pop(k, None)

# ======== 保存为单文件 safetensors + 新 index ========
save_file(state_dict, OUT_PATH)
total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
new_index = {
    "metadata": {"total_size": str(total_size)},
    "weight_map": {k: os.path.basename(OUT_PATH) for k in state_dict.keys()}
}
with open(os.path.join(CKPT_DIR, "model_modified.safetensors.index.json"), "w") as f:
    json.dump(new_index, f, indent=2)

print(f"✅ Saved: {OUT_PATH}")
print(f"✅ Index: {os.path.join(CKPT_DIR, 'model_modified.safetensors.index.json')}")
