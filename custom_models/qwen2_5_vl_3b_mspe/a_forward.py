import torch
import torch.nn.functional as F


def scale_window_index(ps, min_ps, grid_thw_ps, window_index_ps, start, end, spatial_merge_unit, spatial_merge_size):
    # compute scale factor
    f = ps // min_ps

    # per-sample dims (T, H, W)
    dims = grid_thw_ps[ps]  # [N,3]
    areas = dims[:, 1] * dims[:, 2] // spatial_merge_unit  # windows per frame
    counts = dims[:, 0]  # frames per sample

    # cumulative window counts
    cu = torch.repeat_interleave(areas, counts).cumsum(dim=0, dtype=torch.int32)
    cu = F.pad(cu, (1, 0), value=0)

    # pick indices and sample IDs
    idx = window_index_ps[ps][start:end].type_as(cu)
    # import pdb;pdb.set_trace()
    I = torch.bucketize(idx[0], cu[1:], right=True)
    local = idx - cu[I]

    # unravel to (t,h,w)
    T, H, W = dims[I]
    H, W = H // spatial_merge_size, W // spatial_merge_size
    t, h, w = torch.unravel_index(local, (T.item(), H.item(), W.item()))

    # target grid dims
    H0 = grid_thw_ps[min_ps][I, 1] // spatial_merge_size
    W0 = grid_thw_ps[min_ps][I, 2] // spatial_merge_size

    scale_local = t * (H0 * W0) + (h * f) * W0 + (w * f)
    scale_idx = scale_local + cu[I] * f * f

    # return rescaled indices
    token_itxy = torch.stack([I.expand(t.shape[0]), t, h * f, w * f], dim=1).type_as(cu)
    return scale_idx, token_itxy


def recompose_windows(window_adp_ps, hidden_states_ps, position_embeddings_ps, window_index_ps, cu_window_seqlens_ps,grid_thw_ps,
                      spatial_merge_unit, spatial_merge_size):
    hidden_states_list = []
    position_embeddings_list = []
    window_index_list = []
    token_itxy_list = []
    cu_window_seqlens = [0]

    min_ps = min(window_adp_ps)

    # 遍历每个窗口及其对应的patch size
    for idx, ps in enumerate(window_adp_ps):
        # 获取当前窗口的起止位置
        start = cu_window_seqlens_ps[ps][idx]
        end = cu_window_seqlens_ps[ps][idx + 1]
        win_start = cu_window_seqlens_ps[ps][idx] // spatial_merge_unit
        win_end = cu_window_seqlens_ps[ps][idx + 1] // spatial_merge_unit

        # 提取对应窗口的数据
        hidden_states_list.append(hidden_states_ps[ps][start:end])
        position_embeddings_list.append((
            position_embeddings_ps[ps][0][start:end],
            position_embeddings_ps[ps][1][start:end]
        ))
        # unified to min patchsize
        new_idx, token_itxy = scale_window_index(ps, min_ps, grid_thw_ps, window_index_ps, win_start, win_end,spatial_merge_unit, spatial_merge_size)
        window_index_list.append(new_idx)
        token_itxy_list.append(token_itxy)

        # 更新累计长度
        cu_window_seqlens.append(cu_window_seqlens[-1] + (end - start))

    # 拼接结果
    hidden_states = torch.cat(hidden_states_list, dim=0)
    position_embeddings_cos = torch.cat([emb[0] for emb in position_embeddings_list], dim=0)
    position_embeddings_sin = torch.cat([emb[1] for emb in position_embeddings_list], dim=0)
    # import pdb;pdb.set_trace()
    window_index = torch.cat(window_index_list, dim=0)
    cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=cu_window_seqlens[-1].dtype,device=cu_window_seqlens[-1].device)
    position_embeddings = (position_embeddings_cos, position_embeddings_sin)

    # for token itxy
    token_itxy = torch.stack(token_itxy_list)  # [N, 4]
    I_all = token_itxy[:, 0]
    txy_all = token_itxy[:, 1:]  # [N, 3]
    unique_I = I_all.unique(sorted=True)
    token_itxy = [txy_all[I_all == I] for I in unique_I]

    return hidden_states, position_embeddings, window_index, cu_window_seqlens, token_itxy


def repatchify(
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        old_patch_size: int,
        new_patch_size: int,
        in_channels: int = 3,
        temporal_patch_size: int = 2
) -> [torch.Tensor, torch.Tensor]:
    B = grid_thw.shape[0]
    D = hidden_states.shape[1]

    # 每个 patch 的维度
    C = in_channels
    t = temporal_patch_size
    ps = old_patch_size
    D_per_patch = C * t * ps * ps
    assert D == D_per_patch, f"Input D {D} doesn't match expected {D_per_patch}"

    hidden_states_new = []
    new_grid_thw = []

    offset = 0
    for b in range(B):
        T, H, W = grid_thw[b].tolist()
        num_patches = T * H * W
        hs = hidden_states[offset:offset + num_patches]  # [T*H*W, D]
        offset += num_patches

        # 显式还原为高维结构：[T, H, W, C, t, ps, ps]
        hs_3d = hs.view(T, H, W, C, t, ps, ps)
        hs_3d = hs_3d.permute(0, 3, 4, 1, 5, 2, 6).contiguous()  # [T, C, t, H, ps, W, ps]
        hs_3d = hs_3d.view(T, C * t, H * ps, W * ps)  # [T, C*t, H*ps, W*ps]

        # 计算新 H, W
        H_new = (H * ps) // new_patch_size
        W_new = (W * ps) // new_patch_size

        # 验证可整除
        assert (H * ps) % new_patch_size == 0, f"H={H}, ps={ps}, new_ps={new_patch_size}"
        assert (W * ps) % new_patch_size == 0, f"W={W}, ps={ps}, new_ps={new_patch_size}"

        # 重新划分patch
        hs_repatched = hs_3d.view(
            T,
            C * t,
            H_new,
            new_patch_size,
            W_new,
            new_patch_size
        )  # [T, C*t, H_new, ps_new, W_new, ps_new]

        hs_repatched = hs_repatched.permute(0, 2, 4, 1, 3, 5).contiguous()  # [T, H_new, W_new, C*t, ps_new, ps_new]
        hs_repatched = hs_repatched.view(-1, C * t * new_patch_size * new_patch_size)  # [N_new, D_new]

        hidden_states_new.append(hs_repatched)
        new_grid_thw.append([T, H_new, W_new])

    hidden_states_new = torch.cat(hidden_states_new, dim=0)  # [Nnew, Dnew]
    new_grid_thw = torch.tensor(new_grid_thw, device=grid_thw.device)

    return hidden_states_new, new_grid_thw


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
