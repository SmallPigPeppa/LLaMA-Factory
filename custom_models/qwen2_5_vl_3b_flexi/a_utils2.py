import torch
from typing import Tuple

def image_to_patches(img: torch.Tensor, p: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Split img (C, H, W) into patches of size p×p.
    Returns patches (N, C, p, p) and grid (n_h, n_w).
    """
    C, H, W = img.shape
    if H % p or W % p:
        raise ValueError("Image dimensions must be divisible by patch size")
    n_h, n_w = H // p, W // p
    patches = (
        img
        .view(C, n_h, p, n_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(-1, C, p, p)
    )
    return patches, (n_h, n_w)

def patches_to_image(patches: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
    """
    Reassemble patches (N, C, p, p) into img (C, H, W) using grid (n_h, n_w).
    """
    n_h, n_w = grid
    N, C, p, _ = patches.shape
    if N != n_h * n_w:
        raise ValueError("Patch count does not match grid")
    img = (
        patches
        .view(n_h, n_w, C, p, p)
        .permute(2, 0, 3, 1, 4)
        .reshape(C, n_h * p, n_w * p)
    )
    return img


def repatchify(
        self,
        pixel_value: torch.Tensor,  # [N, C*T*ps*ps]
        grid_thw: torch.Tensor,  # [T, H, W]
        new_patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Re-patchify with zero-padding if needed.

    Args:
        pixel_value: [N, dim], dim = C*T*old_ps*old_ps
        grid_thw:     [3] tensor (T, H, W)
        new_patch_size: desired spatial size

    Returns:
        new_pixel_value: [N', dim'] tensor
        new_grid_thw:     [T, H', W'] tensor
    """
    T, H, W = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    C, t_ps, old_ps = self.patch_embed.in_channels, self.patch_embed.temporal_patch_size, self.patch_size

    # reconstruct volume [C, T, H*ps, W*ps]
    x = pixel_value.view(H, W, C, t_ps, old_ps, old_ps)
    full = x.permute(2, 3, 0, 4, 1, 5).contiguous().view(C, t_ps, H * old_ps, W * old_ps)

    # pad to multiple of new_patch_size
    Hf, Wf = full.shape[2], full.shape[3]
    ph = (new_patch_size - Hf % new_patch_size) % new_patch_size
    pw = (new_patch_size - Wf % new_patch_size) % new_patch_size
    if ph or pw:
        full = F.pad(full, (0, pw, 0, ph))
        Hf += ph;
        Wf += pw

    # compute new grid
    Hn, Wn = Hf // new_patch_size, Wf // new_patch_size

    # extract patches and flatten
    patches = full.unfold(2, new_patch_size, new_patch_size) \
        .unfold(3, new_patch_size, new_patch_size)
    patches = patches.permute(2, 4, 0, 1, 3, 5) \
        .contiguous() \
        .view(-1, C, t_ps, new_patch_size, new_patch_size)
    new_pixel_value = patches.view(patches.size(0), -1)
    new_grid_thw = torch.tensor([T, Hn, Wn], device=grid_thw.device, dtype=torch.long)

    return new_pixel_value, new_grid_thw
