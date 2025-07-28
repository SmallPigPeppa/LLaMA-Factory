import torch
import random
from custom_models.qwen2_5_vl_3b_mspe.debug_utilsv2 import repatchify
import torch.nn.functional as F


def recompose_windows(window_adp_ps, hidden_states_ps, position_embeddings_ps, window_index_ps, cu_window_seqlens_ps):
    hidden_states_list = []
    position_embeddings_list = []
    window_index_list = []
    cu_window_seqlens = [0]

    # 每个patch_size的当前窗口位置
    window_cursor_ps = {ps: 0 for ps in hidden_states_ps.keys()}

    # 遍历每个窗口及其对应的patch size
    for ps in window_adp_ps:
        cursor = window_cursor_ps[ps]

        # 获取当前窗口的起止位置
        cu_seqlens = cu_window_seqlens_ps[ps]
        start = cu_seqlens[cursor]
        end = cu_seqlens[cursor + 1]

        # 提取对应窗口的数据
        hidden_states_list.append(hidden_states_ps[ps][start:end])
        position_embeddings_list.append((
            position_embeddings_ps[ps][0][start:end],
            position_embeddings_ps[ps][1][start:end]
        ))
        # window_index_list.append(window_index_ps[ps][start:end])
        # unified to ps=7
        scale = torch.tensor((ps / 7) ** 2, dtype=window_index_ps[ps].dtype, device=window_index_ps[ps].device)
        window_index_list.append(window_index_ps[ps][start:end] * scale)

        # 更新cursor
        window_cursor_ps[ps] += 1

        # 更新累计长度
        cu_window_seqlens.append(cu_window_seqlens[-1] + (end - start))

    # 拼接结果
    hidden_states = torch.cat(hidden_states_list, dim=0)
    position_embeddings_cos = torch.cat([emb[0] for emb in position_embeddings_list], dim=0)
    position_embeddings_sin = torch.cat([emb[1] for emb in position_embeddings_list], dim=0)
    window_index = torch.cat(window_index_list, dim=0)
    cu_window_seqlens= torch.tensor(cu_window_seqlens, dtype=cu_window_seqlens[-1].dtype, device=cu_window_seqlens[-1].device)

    position_embeddings = (position_embeddings_cos, position_embeddings_sin)

    return hidden_states, position_embeddings, window_index, cu_window_seqlens


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


def forward_old(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    return hidden_states


def forward_new(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """

    patch_sizes = [7, 14, 28]
    grid_thw_ps = {}
    hidden_states_ps = {}
    rotary_pos_emb_ps = {}
    window_index_ps = {}
    cu_window_seqlens_ps = {}
    position_embeddings_ps = {}

    # initial
    for ps in patch_sizes:
        m2f = merge_to_flatten_idx(grid_thw=grid_thw, merge_size=self.spatial_merge_size)
        hidden_states_rep = hidden_states[m2f]
        hidden_states_rep, grid_thw_ps[ps] = repatchify(
            hidden_states=hidden_states_rep,
            grid_thw=grid_thw,
            old_patch_size=self.patch_size,
            new_patch_size=ps
        )
        f2m = flatten_to_merge_idx(grid_thw=grid_thw_ps[ps], merge_size=self.spatial_merge_size)
        hidden_states_ps[ps] = hidden_states_rep[f2m]
        hidden_states_ps[ps] = self.patch_embed(hidden_states_ps[ps])
        rotary_pos_emb_ps[ps] = self.rot_pos_emb(grid_thw_ps[ps])
        win, cu = self.get_window_index(grid_thw_ps[ps])
        cu = torch.tensor(
            cu,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu = torch.unique_consecutive(cu)
        window_index_ps[ps] = win
        cu_window_seqlens_ps[ps] = cu

    # reorder for merge
    for ps in patch_sizes:
        seq_len, _ = hidden_states_ps[ps].size()
        hidden_states_ps[ps] = hidden_states_ps[ps].reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states_ps[ps] = hidden_states_ps[ps][window_index_ps[ps], :, :]
        hidden_states_ps[ps] = hidden_states_ps[ps].reshape(seq_len, -1)
        rotary_pos_emb_ps[ps] = rotary_pos_emb_ps[ps].reshape(
            seq_len // self.spatial_merge_unit,
            self.spatial_merge_unit, -1)
        rotary_pos_emb_ps[ps] = rotary_pos_emb_ps[ps][window_index_ps[ps], :, :]
        rotary_pos_emb_ps[ps] = rotary_pos_emb_ps[ps].reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb_ps[ps], rotary_pos_emb_ps[ps]), dim=-1)
        position_embeddings_ps[ps] = (emb.cos(), emb.sin())

    # window patchsize
    window_adp_ps = [random.choice([7, 14, 28]) for _ in range(len(cu_window_seqlens_ps[14]) - 1)]
    hidden_states, position_embeddings, window_index, cu_window_seqlens = recompose_windows(
        window_adp_ps,
        hidden_states_ps,
        position_embeddings_ps,
        window_index_ps,
        cu_window_seqlens_ps
    )

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    updated_cu_seqlens = [cu_window_seqlens[cu_window_seqlens_ps[self.patch_size].index(seq)] for seq in cu_seqlens]

    # window attention
    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            # cu_seqlens_now = cu_seqlens update
            cu_seqlens_now = updated_cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]
    window_index = window_index[reverse_indices]
    token_ithw = None

    return hidden_states, token_ithw
