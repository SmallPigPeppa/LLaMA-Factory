import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

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


def forward_mspe(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.
    Returns:
        `torch.Tensor`: hidden_states.
    """

    # 1. get window index and cu_window_seqlens
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    num_windows = cu_window_seqlens.numel() - 1
    window_index_list = [window_index[i * self.spatial_merge_unit: (i + 1) * self.spatial_merge_unit] for i in
                         range(num_windows)]

    # 2. get window patchsize and itxy coords
    window_patchsize_list, window_grid_thw_list = self.get_window_patchsize(grid_thw)
    grid_itxy_coords = self.compute_grid_itxy(grid_thw)

    # 3. get position embed
    tmp_grid_thw = grid_thw.clone()
    tmp_grid_thw[:, 1:] = tmp_grid_thw[:, 1:] * 2
    rotary_pos_emb = self.rot_pos_emb(tmp_grid_thw)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    # 4. compute per window
    hidden_states_list = []
    pos_emb_list = []
    grid_itxy_list = []
    window_seqlens_list = []
    window_imgidx_list = []
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
        window_grid_itxy = grid_itxy_coords.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        window_grid_itxy = window_grid_itxy[window_idx, :, :]
        window_grid_itxy = window_grid_itxy.reshape(len(window_idx) * self.spatial_merge_unit, -1)
        window_grid_itxy = self.recompute_grid_itxy(
            grid_txy=window_grid_itxy,
            new_patchsize=window_patchsize
        )
        flatten_index = self.itxy_to_pos_index(grid_thw=tmp_grid_thw, grid_itxy=window_grid_itxy)
        window_pos_emb = position_embeddings[flatten_index]

        hidden_states_list.append(window_patch_embed)
        pos_emb_list.append(window_pos_emb)
        window_seqlens_list.append(len(window_grid_itxy))
        window_imgidx_list.append(window_grid_itxy[0, 0])
        grid_itxy_list.append(window_grid_itxy)

    # update cu_window_seqlens
    update_cu_window_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(window_seqlens_list), dim=0)))

    # update cu_seqlens
    num_images = max(window_imgidx_list) + 1
    update_cu_seqlens = torch.zeros(num_images + 1, dtype=torch.int32, device=update_cu_window_seqlens.device)
    window_imgidx_tensor = torch.tensor(window_imgidx_list, device=update_cu_window_seqlens.device)
    update_cu_seqlens.scatter_reduce_(
        0,
        window_imgidx_tensor + 1,
        update_cu_window_seqlens[1:] - update_cu_window_seqlens[:-1],
        reduce='sum'
    )
    update_cu_seqlens = update_cu_seqlens.cumsum(dim=0)

    # concate hidden_states pos_emb
    hidden_states = torch.cat(hidden_states_list, dim=0)
    pos_emb = torch.cat(pos_emb_list, dim=0)

    for layer_num, blk in enumerate(self.blocks):
        position_embeddings = rotary_pos_emb
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = update_cu_seqlens
        else:
            cu_seqlens_now = update_cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    hidden_states = self.merger(hidden_states)
    # reverse_indices = torch.argsort(window_index)
    grid_itxy_list = grid_itxy_list.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    grid_itxy_list= grid_itxy_list[:,0,:]
    reverse_indices = self.get_reverse_indices(patch_xy_list)
    hidden_states = hidden_states[reverse_indices, :]

    return hidden_states, grid_itxy_list

    # patch_xy_coords = self.compute_window_patch_xy_coords(
    #     window_index_list=window_index_list,
    #     window_adp_patchsize=window_adp_patchsize,
    #     base_patchsize=self.patch_size,
    #     min_patchsize=7,
    #     window_size=112,
    #     grid_hw=(grid_thw[:, -2], grid_thw[:, -1]))
