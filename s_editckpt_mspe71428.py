import os, json
from safetensors.torch import safe_open, save_file
import torch
from typing import List, Tuple, Union
import torch.nn.functional as F

def to_2tuple(x: Union[int, Tuple[int, int], List[int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, (list, tuple)):
        if len(x) == 2:
            return (int(x[0]), int(x[1]))
        raise ValueError(f"Input must be int or pair of ints, got {x}")
    raise TypeError(f"Unsupported type: {type(x)}")

def pi_resize3d(
        conv_weight: torch.Tensor,
        target_size: Union[int, Tuple[int, int], List[int]],
        interpolation: str = "bicubic",
        antialias: bool = True,
) -> torch.Tensor:
    """
    Pseudo-inverse resize for 3D conv weights along H and W only.

    Args:
        conv_weight: Tensor of shape [out_ch, in_ch, D, H, W]
        target_size: New (H, W) or single int
        interpolation: F.interpolate mode
        antialias: whether to apply antialiasing

    Returns:
        Resized weight of shape [out_ch, in_ch, D, H', W']
    """
    assert conv_weight.ndim == 5, "Expect 5D tensor [O, I, D, H, W]"
    out_ch, in_ch, depth, h, w = conv_weight.shape
    h_new, w_new = to_2tuple(target_size)
    device = conv_weight.device
    dtype = conv_weight.dtype
    float_dtype = torch.float
    conv_weight = conv_weight.to(dtype=float_dtype)

    # no-op if same spatial size
    if (h, w) == (h_new, w_new):
        return conv_weight.to(dtype=dtype)

    # resize a single 2D slice via F.interpolate
    def _resize2d(x: torch.Tensor) -> torch.Tensor:
        y = F.interpolate(
            x[None, None], (h_new, w_new),
            mode=interpolation,
            antialias=antialias
        )
        return y[0, 0]

    # build pseudo-inverse matrix mapping vec(HxW) -> vec(H'xW')
    def _make_pinv(old: Tuple[int, int], new: Tuple[int, int]) -> torch.Tensor:
        mats = []
        for i in range(old[0] * old[1]):
            basis = torch.zeros(old, device=device, dtype=float_dtype)
            basis.view(-1)[i] = 1.0
            mats.append(_resize2d(basis).view(-1))
        M = torch.stack(mats, dim=0)  # [H*W, H'*W']
        return torch.linalg.pinv(M)  # [H'*W', H*W]

    pinv = _make_pinv((h, w), (h_new, w_new))

    # flatten spatial dims and apply pinv to each slice
    flat = conv_weight.view(-1, h * w)  # [O*I*D, H*W]
    resized_flat = flat @ pinv.t()  # [O*I*D, H'*W']
    # reshape back to 5D
    return resized_flat.view(out_ch, in_ch, depth, h_new, w_new).to(dtype=dtype)



# ======== 配置（按需改） ========
CKPT_DIR = "/mnt/bn/liuwenzhuo-hl-data/Qwen/Qwen2.5-VL-3B-Instruct/"
CKPT_DIR_MODIFIED = "/mnt/bn/liuwenzhuo-hl-data/Qwen/Qwen2.5-VL-3B-Instruct-mspe71428/"
OUT_PATH = os.path.join(CKPT_DIR_MODIFIED, "model.safetensors")
PATCH_SIZE = 12

os.makedirs(CKPT_DIR_MODIFIED, exist_ok=True)

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




base_weight = state_dict["visual.patch_embed.proj.weight"]
# base_bias = state_dict["visual.patch_embed.proj.bias"]

patch_sizes = [7, 14, 28]
patch_sizes = sorted(patch_sizes)

for idx, patch_size in enumerate(patch_sizes):
    prefix = f"visual.patch_embed.patch_embed_seq.{idx}"

    # Resize conv weight
    resized_weight = pi_resize3d(base_weight, target_size=patch_size)
    state_dict[f"{prefix}.proj.weight"] = resized_weight.clone()

    # # Copy bias
    # if base_bias is not None:
    #     state_dict[f"{prefix}.proj.bias"] = base_bias.clone()


# 删除旧的 patchifier 参数（可选）
for key in [
    "visual.patch_embed.proj.weight",
    # "visual.patch_embed.proj.bias",
]:
    state_dict.pop(key, None)



# ======== 保存为单文件 safetensors + 新 index ========
save_file(state_dict, OUT_PATH)
total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
new_index = {
    "metadata": {"total_size": str(total_size)},
    "weight_map": {k: os.path.basename(OUT_PATH) for k in state_dict.keys()}
}

index_out_path = os.path.join(CKPT_DIR_MODIFIED, "model.safetensors.index.json")
with open(index_out_path, "w") as f:
    json.dump(new_index, f, indent=2)

print(f"✅ Saved: {OUT_PATH}")
print(f"✅ Index: {index_out_path}")
