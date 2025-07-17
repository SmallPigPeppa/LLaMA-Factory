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

    # get window adp patchsize
    window_index_list, cu_window_seqlens, window_patchsize_list, window_grid_thw_list = self.get_window_index_and_patchsizes(
        hidden_states, grid_thw
    )
    grid_txy_coords = self.compute_grid_txy(grid_thw)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    hidden_states_list = []
    pos_emb_list = []

    for i, (window_idx, window_patchsize, window_grid_thw) in enumerate(
            zip(window_index_list, window_patchsize_list, window_grid_thw_list)):

        # window patch_embed
        seq_len, _ = hidden_states.shape
        window_hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        window_hidden_states = window_hidden_states[window_idx, :, :]
        window_hidden_states = window_hidden_states.reshape(len(window_idx) * self.spatial_merge_unit, -1)
        window_hidden_states = self.repatchify(
            pixel_value=window_hidden_states,
            grid_thw=window_grid_thw,
            new_patch_size=window_patchsize
        )
        window_patch_embed = self.patch_embed(window_hidden_states, patch_size=window_patchsize)

        # window pos_embed
        window_grid_txy = grid_txy_coords.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        window_grid_txy = window_grid_txy[window_idx, :, :]
        window_grid_txy = window_grid_txy.reshape(len(window_idx) * self.spatial_merge_unit, -1)
        window_grid_txy = self.recompute_grid_txy(
            grid_txy=window_grid_txy,
            new_patchsize=window_patchsize
        )
        window_pos_emb = self.create_pos_embeds(rotary_pos_emb, window_grid_txy)

        hidden_states_list.append(window_patch_embed)
        pos_emb_list.append(window_pos_emb)

        # update cu_window_seqlens
        cu_window_seqlens[i] = (cu_window_seqlens[i] - cu_window_seqlens[i - 1]) * (
                self.patch_size / window_patchsize_list)

    hidden_states = torch.cat(hidden_states_list, dim=0)
    pos_emb = torch.cat(pos_emb_list, dim=0)

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
        position_embeddings = rotary_pos_emb
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
    # reverse_indices = torch.argsort(window_index)
    reverse_indices = self.get_reverse_indices(patch_xy_list)
    hidden_states = hidden_states[reverse_indices, :]

    return hidden_states

    # patch_xy_coords = self.compute_window_patch_xy_coords(
    #     window_index_list=window_index_list,
    #     window_adp_patchsize=window_adp_patchsize,
    #     base_patchsize=self.patch_size,
    #     min_patchsize=7,
    #     window_size=112,
    #     grid_hw=(grid_thw[:, -2], grid_thw[:, -1]))
