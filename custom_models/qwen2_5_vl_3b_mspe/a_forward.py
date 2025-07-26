import torch

from custom_models.qwen2_5_vl_3b_mspe.debug_utilsv2 import repatchify


def flatten_to_merge_idx(self, grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
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
        grid_thw_ps[ps] = grid_thw[ps]
        hidden_states_ps[ps] = repatchify(hidden_states, grid_thw, patchsize=ps)
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
    window_adp_ps = self.get_window_patchsize(hidden_states_ps, cu_window_seqlens_ps)
    hidden_states = self.recompose(hidden_states_ps, window_adp_ps)
    position_embeddings = self.recompose(position_embeddings_ps, window_adp_ps)
    window_index = self.recompose(window_index_ps, window_adp_ps)
    cu_window_seqlens = self.recompose(cu_window_seqlens_ps, window_adp_ps)
    cu_seqlens = self.recompose(cu_window_seqlens_ps, window_adp_ps)

    # window attention
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
