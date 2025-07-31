import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


# ==============================
# 你的 patch 拆分 & 重构函数
# ==============================
def image_to_patches(img: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    C, H, W = img.shape
    p = patch_size
    if H % p or W % p:
        raise ValueError("H and W must be divisible by patch_size")
    n_h, n_w = H // p, W // p
    patches = img.view(C, n_h, p, n_w, p).permute(1, 3, 0, 2, 4).reshape(-1, C, p, p)
    return patches, (n_h, n_w)


def patches_to_image(patches: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
    if patches.ndim != 4:
        raise ValueError("patches must have shape (N, C, ps, ps)")
    n_h, n_w = grid_shape
    N, C, p_h, p_w = patches.shape
    if p_h != p_w or N != n_h * n_w:
        raise ValueError("Invalid patches shape or grid_shape")
    img = patches.view(n_h, n_w, C, p_h, p_w).permute(2, 0, 3, 1, 4).reshape(C, n_h * p_h, n_w * p_w)
    return img


# ==============================
# 可视化函数
# ==============================
def show_tensor_with_numbers(ax, array, title):
    ax.imshow(np.ones_like(array), cmap='gray', vmin=0, vmax=1)  # white background
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(j, i, str(int(array[i, j])), ha='center', va='center', fontsize=8)
    ax.set_title(title)
    ax.axis('off')


def visualize_patch_process_torch(H=12, W=12, patch_size=6):
    # 生成数字图像 (1, H, W)
    img = torch.arange(1, H * W + 1).reshape(1, H, W).float()

    # 应用你的函数
    patches, grid_shape = image_to_patches(img, patch_size)
    reconstructed = patches_to_image(patches, grid_shape)

    # 转换为 NumPy，方便绘图
    img_np = img.squeeze().numpy()
    recon_np = reconstructed.squeeze().numpy()
    patch_np = patches.squeeze(1).numpy()  # (N, H_p, W_p)

    # 绘图
    num_patches = patch_np.shape[0]
    total_items = 2 + num_patches  # 原图 + patch们 + 重建
    fig, axs = plt.subplots(1, total_items, figsize=(3 * total_items, 3))

    show_tensor_with_numbers(axs[0], img_np, "Original")
    for i in range(num_patches):
        show_tensor_with_numbers(axs[i + 1], patch_np[i], f"Patch {i}")
    show_tensor_with_numbers(axs[-1], recon_np, "Reconstructed")

    plt.tight_layout()
    plt.show()


# ✅ 运行例子
visualize_patch_process_torch(H=12, W=12, patch_size=6)
