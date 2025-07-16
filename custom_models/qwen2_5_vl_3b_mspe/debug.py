import torch
import torch.nn.functional as F
from torch.utils.tensorboard.summary import image_boxes


# def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
#     """
#     Args:
#         hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
#             The final hidden states of the model.
#         grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
#             The temporal, height and width of feature shape of each image in LLM.
#     Returns:
#         `torch.Tensor`: hidden_states.
#     """
#
#     import pdb; pdb.set_trace()
#     # Step 1. 划分每个window的index和每个window的patch size
#     window_index_list, cu_window_seqlens, patch_sizes = self.get_window_index_and_patchsizes(grid_thw)
#     # window_index_list: list[tensor], 每个window的index
#     # cu_window_seqlens: [0, n1, n1+n2, ...] 每个window累加的patch数
#     # patch_sizes: list[int], 每个window的patch size
#
#     device = hidden_states.device
#     dtype = hidden_states.dtype
#     seq_len, _ = hidden_states.size()
#
#     all_window_hidden_states = []
#     all_window_rotary_pos_emb = []
#     position_embeddings_list = []
#
#     cur_start = 0
#
#     for i, (window_indices, patch_size) in enumerate(zip(window_index_list, patch_sizes)):
#         # 当前 window 相关参数
#         window_patch_count = len(window_indices)
#         # 取出属于这个 window 的原始 hidden_states
#         raw_window_hidden = hidden_states[cur_start:cur_start+window_patch_count, :]
#         cur_start += window_patch_count
#
#         # 对每个 window 分别做 patch embedding
#         # 注意如果 patch_size 可变，建议 patch_embed 支持 patch_size 参数
#         window_hidden = self.patch_embed(raw_window_hidden, patch_size=patch_size)
#
#         # 获取 window 内的 grid_thw (t, h, w after split by this patchsize)
#         grid_t, grid_h, grid_w = self.compute_grid_for_window(grid_thw, window_indices, patch_size)  # 你需要实现
#         # 生成 rotary pos emb for this window
#         rotary_pos_emb = self.rot_pos_emb_for_window(grid_t, grid_h, grid_w, patch_size)  # 你需要实现
#
#         # 保存
#         all_window_hidden_states.append(window_hidden)
#         all_window_rotary_pos_emb.append(rotary_pos_emb)
#
#     # 拼接所有 window 处理后的 hidden_states 和 rotary_pos_emb
#     hidden_states = torch.cat(all_window_hidden_states, dim=0)
#     rotary_pos_emb = torch.cat(all_window_rotary_pos_emb, dim=0)
#
#     # 位置编码处理
#     emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
#     position_embeddings = (emb.cos(), emb.sin())
#
#     # cu_window_seqlens 构建（torch.tensor化并去重）
#     cu_window_seqlens = torch.tensor(
#         cu_window_seqlens, device=device, dtype=torch.int32
#     )
#     cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
#
#     # forward blocks
#     for layer_num, blk in enumerate(self.blocks):
#         if layer_num in self.fullatt_block_indexes:
#             cu_seqlens_now = cu_window_seqlens  # or global cu_seqlens if你还有原始全局seqlens
#         else:
#             cu_seqlens_now = cu_window_seqlens
#         if self.gradient_checkpointing and self.training:
#             hidden_states = self._gradient_checkpointing_func(
#                 blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
#             )
#         else:
#             hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)
#
#     hidden_states = self.merger(hidden_states)
#
#     # 由于每个window顺序已经打乱，还原顺序
#     # 构造全局window_index（把window_index_list拼起来）
#     window_index = torch.cat(window_index_list, dim=0)
#     reverse_indices = torch.argsort(window_index)
#     hidden_states = hidden_states[reverse_indices, :]
#
#     return hidden_states

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

    # 1. 获取窗口划分和patchsize随机
    window_index_list, cu_window_seqlens, patch_sizes, image_index = self.get_window_index_and_patchsizes(grid_thw)
    device = hidden_states.device

    all_window_hidden_states = []
    all_window_rotary_pos_emb = []
    all_window_merge_indices = []

    for i, (window_indices, patch_size, img_idx) in enumerate(zip(window_index_list, patch_sizes, image_index)):
        # (1) 先 patch_embed
        window_hidden = self.patch_embed(hidden_states[window_indices, :], patch_size=patch_size)

        # (2) spatial merge
        # 假设 self.spatial_merge_unit > 1 时需要 merge，且 merge 发生在 window 之前
        seq_len = window_hidden.shape[0]
        if self.spatial_merge_unit > 1:
            merge_len = seq_len // self.spatial_merge_unit
            window_hidden = window_hidden.reshape(merge_len, self.spatial_merge_unit, -1)
            # window_indices 已经是 merge 后的 index，直接选取即可
            merged_window_hidden = window_hidden[window_indices, :, :]
            window_hidden = merged_window_hidden.reshape(-1, window_hidden.shape[-1])
        else:
            # 不需要 merge
            window_hidden = window_hidden

        # (3) rotary_pos_emb
        grid_thw_new = grid_thw[img_idx].clone()
        grid_thw_new[1:3] *= self.patch_size // patch_size
        rotary_pos_emb_full = self.rot_pos_emb(grid_thw_new.unsqueeze(0))  # [1, t*h*w, dim]
        rotary_pos_emb_full = rotary_pos_emb_full.view(-1, rotary_pos_emb_full.shape[-1])  # [num_patches, dim]

        if self.spatial_merge_unit > 1:
            # 和上面 hidden 处理一致
            rotary_pos_emb_full = rotary_pos_emb_full.reshape(merge_len, self.spatial_merge_unit, -1)
            rotary_pos_emb = rotary_pos_emb_full[window_indices, :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(-1, rotary_pos_emb_full.shape[-1])
        else:
            rotary_pos_emb = rotary_pos_emb_full[window_indices]

        all_window_hidden_states.append(window_hidden)
        all_window_rotary_pos_emb.append(rotary_pos_emb)
        all_window_merge_indices.append(window_indices)

    # 合并所有 window
    hidden_states = torch.cat(all_window_hidden_states, dim=0)
    rotary_pos_emb = torch.cat(all_window_rotary_pos_emb, dim=0)

    # 位置编码拼接
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_window_seqlens = torch.tensor(cu_window_seqlens, device=device, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    # forward blocks
    for layer_num, blk in enumerate(self.blocks):
        cu_seqlens_now = cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    hidden_states = self.merger(hidden_states)

    # 合并所有 window_index，得到在 merge 后的全局 index
    window_index = torch.cat(all_window_merge_indices, dim=0)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    return hidden_states

