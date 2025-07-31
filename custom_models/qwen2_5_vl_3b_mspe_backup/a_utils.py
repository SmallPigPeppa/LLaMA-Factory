import torch
from typing import Tuple


def flatten_to_merge_idx(grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
    """
    Convert a batch of grid_thw into continuous merged indices.

    Args:
        grid_thw (torch.Tensor): Shape (batch_size, 3), each row is [t, h, w].
        merge_size (int): The size of the merge block.

    Returns:
        torch.Tensor: Flattened indices for the entire batch, continuous.
    """
    batch_size = grid_thw.size(0)
    batch_indices = []
    offset = 0

    for i in range(batch_size):
        t, h, w = grid_thw[i].tolist()
        idx = torch.arange(t * h * w, dtype=torch.long).reshape(t, h, w)
        # Reshape into blocks and rearrange axes for merged blocks
        idx = idx.reshape(t, h // merge_size, merge_size, w // merge_size, merge_size)
        idx = idx.permute(0, 1, 3, 2, 4).contiguous()
        idx = idx.flatten() + offset

        batch_indices.append(idx)
        offset += t * h * w

    return torch.cat(batch_indices, dim=0)


def merge_to_flatten_idx(grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
    """
    Inverse of flatten_to_merge_idx_batch.
    Converts merged indices back into original flattened indices.

    Args:
        grid_thw (torch.Tensor): Shape (batch_size, 3), each row is [t, h, w].
        merge_size (int): The size of the merge block.

    Returns:
        torch.Tensor: Flattened indices in original t*h*w order across the batch.
    """
    batch_size = grid_thw.size(0)
    batch_indices = []
    offset = 0

    for i in range(batch_size):
        t, h, w = grid_thw[i].tolist()
        numel = t * h * w

        # Create the same index grid and perform the same merge permutation
        idx = torch.arange(numel, dtype=torch.long).reshape(t, h, w)
        idx_merged = idx.reshape(t, h // merge_size, merge_size, w // merge_size, merge_size)
        idx_merged = idx_merged.permute(0, 1, 3, 2, 4).contiguous().flatten()

        # Now invert the permutation:
        inverse_idx = torch.empty_like(idx_merged)
        inverse_idx[idx_merged] = torch.arange(numel, dtype=torch.long)

        inverse_idx = inverse_idx + offset
        batch_indices.append(inverse_idx)
        offset += numel

    return torch.cat(batch_indices, dim=0)



def image_to_patches(
    img: torch.Tensor,
    patch_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Split a (C,H,W) torch.Tensor into square patches of size ps x ps.

    Returns:
        patches: (N, C, ps, ps)
        (n_h, n_w): grid dimensions
    """
    C, H, W = img.shape
    p = patch_size
    if H % p or W % p:
        raise ValueError("H and W must be divisible by patch_size")

    n_h, n_w = H // p, W // p
    patches = (
        img
        .view(C, n_h, p, n_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(-1, C, p, p)
    )
    return patches, (n_h, n_w)


def patches_to_image(
    patches: torch.Tensor,
    grid_shape: Tuple[int, int]
) -> torch.Tensor:
    """
    Reconstruct (C,H,W) torch.Tensor from patches (N, C, ps, ps) and grid (n_h, n_w).
    """
    if patches.ndim != 4:
        raise ValueError("patches must have shape (N, C, ps, ps)")
    n_h, n_w = grid_shape
    N, C, p_h, p_w = patches.shape
    if p_h != p_w or N != n_h * n_w:
        raise ValueError("Invalid patches shape or grid_shape")

    img = (
        patches
        .view(n_h, n_w, C, p_h, p_w)
        .permute(2, 0, 3, 1, 4)
        .reshape(C, n_h * p_h, n_w * p_w)
    )
    return img



# Example
if __name__ == "__main__":
    x = torch.randn(3, 256, 256)
    patches, grid = image_to_patches(x, 32)
    print(patches.shape, grid)
    recon = patches_to_image(patches, grid)
    print(recon.shape)
