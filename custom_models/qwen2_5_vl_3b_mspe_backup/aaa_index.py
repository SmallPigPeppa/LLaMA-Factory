import numpy as np


def flatten_to_merge_idx(grid_thw: list[int], merge_size: int) -> np.ndarray:
    t, h, w = grid_thw
    idx = np.arange(t * h * w).reshape(t, h, w)
    # Reshape into blocks and rearrange axes for merged blocks
    idx = idx.reshape(t, h // merge_size, merge_size, w // merge_size, merge_size)
    idx = idx.transpose(0, 1, 3, 2, 4)
    # Flatten to 1D
    return idx.flatten()


import torch


def flatten_to_merge_idx_batch(grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
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


def merge_to_flatten_idx_batch(grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
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


# Example usage:
grid_thw = torch.tensor([[1, 4, 6], [1, 4, 2]])
merge_size = 2
indices = flatten_to_merge_idx_batch(grid_thw, merge_size)
print(indices[:56])
