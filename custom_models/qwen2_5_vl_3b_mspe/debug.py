import torch
import torch.nn.functional as F



def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
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


def forward_mspe(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.
    Returns:
        `torch.Tensor`: hidden_states.
    """

    # 获取窗口划分和patchsize随机
    window_index_list, cu_window_seqlens, patch_sizes, image_index = self.get_window_index_and_patchsizes(grid_thw)
    device = hidden_states.device

    hidden_states_list = []
    rotary_pos_emb_list = []

    for i, (window_indices, patch_size, img_idx) in enumerate(zip(window_index_list, patch_sizes, image_index)):
        # (1) patch_embed
        window_hidden_states = select_from_spatial_merged_windows_index(
            hidden_states, window_indices, self.spatial_merge_unit
        )
        window_patch_embed = self.patch_embed(window_hidden_states, patch_size=patch_size)

        # (2) rotary_pos_emb
        patch_size_ratio = self.patch_size // patch_size
        grid_thw_new = grid_thw[img_idx].clone()
        grid_thw_new[1:3] *= patch_size_ratio
        rotary_pos_emb_new = self.rot_pos_emb(grid_thw_new.unsqueeze(0))
        window_indices_new = update_window_index(window_indices, patch_size_ratio)
        window_rotary_pos_emb = select_from_spatial_merged_windows_index(
            rotary_pos_emb_new, window_indices_new, self.spatial_merge_unit
        )

        hidden_states_list.append(window_patch_embed)
        rotary_pos_emb_list.append(window_rotary_pos_emb)

    # 合并所有 window
    hidden_states = torch.cat(hidden_states_list, dim=0)
    rotary_pos_emb = torch.cat(rotary_pos_emb_list, dim=0)

    cu_window_seqlens = torch.tensor(cu_window_seqlens, device=device, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    # forward blocks
    for layer_num, blk in enumerate(self.blocks):
        cu_seqlens_now = cu_window_seqlens
        position_embeddings = rotary_pos_emb
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    hidden_states = self.merger(hidden_states)

    # 合并所有 window_index，得到在 merge 后的全局 index
    window_index = torch.cat(window_index_list, dim=0)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    return hidden_states


def select_from_spatial_merged_windows_index(hidden_states, window_index, spatial_merge_unit):
    """
    Merge hidden_states along the sequence dimension by spatial_merge_unit,
    select windows by windows_index, then flatten.

    Args:
        hidden_states (Tensor): Shape [seq_len, hidden_dim].
        window_index (Tensor or list): Indices of merged windows to select, shape [num_windows].
        spatial_merge_unit (int): Merge size (number of patches per merged window).

    Returns:
        Tensor: Selected and flattened windows, shape [num_windows * spatial_merge_unit, hidden_dim].
    """
    seq_len, hidden_dim = hidden_states.shape
    assert seq_len % spatial_merge_unit == 0, "seq_len must be divisible by spatial_merge_unit"

    hidden_states = hidden_states.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)

    return hidden_states


def update_window_index(window_index, scale):
    return window_index * scale
