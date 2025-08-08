import torch
import torch.nn.functional as F
import random
from typing import List, Dict


def random_window_ps(cu_window_seqlens_ps: Dict[int, torch.Tensor], patch_sizes: List[int]) -> List[int]:
    """
    Randomly assign a patch size to each window.

    Args:
        cu_window_seqlens_ps: Maps patch size to [total_windows+1] tensor.
        patch_sizes: List of candidate patch sizes.

    Returns:
        List of patch sizes per window.
    """
    num_windows = cu_window_seqlens_ps[patch_sizes[0]].size(0) - 1
    return [random.choice(patch_sizes) for _ in range(num_windows)]


def random_sample_ps(
    cu_window_seqlens_ps: Dict[int, torch.Tensor],
    patch_sizes: List[int],
    grid_thw_ps: Dict[int, torch.Tensor],
) -> List[int]:
    """
    Assign one random patch size per sample, then apply to its windows.

    Args:
        cu_window_seqlens_ps: Maps patch size to [total_windows+1] tensor.
        patch_sizes: List of candidate patch sizes.
        grid_thw_ps: Maps patch size to [num_samples, 3] tensor.

    Returns:
        List of patch sizes per window.
    """
    base_ps = patch_sizes[0]
    grid_thw = grid_thw_ps[base_ps]
    cu_window_seqlens = cu_window_seqlens_ps[base_ps]

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(0)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    '''
    False: boundaries[i-1] < input[m][n]...[l][x] <= boundaries[i]
    True: boundaries[i-1] <= input[m][n]...[l][x] < boundaries[i]
    compare with window_cu and cu, using False
    e.g. 
    cu_window_seqlens_ps[28]
    tensor([  0,  16,  32,  48,  56,  72,  88, 104, 112, 128, 144, 160, 168, 184,
            200, 216, 224, 240, 256, 272, 280, 288, 296, 304, 308]
    '''
    interval_ids = torch.bucketize(cu_window_seqlens[1:], cu_seqlens[1:], right=False)

    choices = torch.tensor(
        [random.choice(patch_sizes) for _ in range(cu_seqlens.numel() - 1)],
        device=cu_window_seqlens.device
    )

    return choices[interval_ids].tolist()


def random_sample_ps_v2(
    cu_window_seqlens_ps: Dict[int, torch.Tensor],
    patch_sizes: List[int],
    grid_thw_ps: Dict[int, torch.Tensor],
    max_pixels: int,
    min_pixels: int,
    
) -> List[int]:
    """
    Assign one random patch size per sample, then apply to its windows.

    Args:
        cu_window_seqlens_ps: Maps patch size to [total_windows+1] tensor.
        patch_sizes: List of candidate patch sizes.
        grid_thw_ps: Maps patch size to [num_samples, 3] tensor.

    Returns:
        List of patch sizes per window.
    """
    base_ps = patch_sizes[0]
    grid_thw = grid_thw_ps[base_ps]
    cu_window_seqlens = cu_window_seqlens_ps[base_ps]

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(0)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    '''
    False: boundaries[i-1] < input[m][n]...[l][x] <= boundaries[i]
    True: boundaries[i-1] <= input[m][n]...[l][x] < boundaries[i]
    compare with window_cu and cu, using False
    e.g. 
    cu_window_seqlens_ps[28]
    tensor([  0,  16,  32,  48,  56,  72,  88, 104, 112, 128, 144, 160, 168, 184,
            200, 216, 224, 240, 256, 272, 280, 288, 296, 304, 308]
    '''
    interval_ids = torch.bucketize(cu_window_seqlens[1:], cu_seqlens[1:], right=False)

    choices = torch.tensor(
        [random.choice(patch_sizes) for _ in range(cu_seqlens.numel() - 1)],
        device=cu_window_seqlens.device
    )

    return choices[interval_ids].tolist()
