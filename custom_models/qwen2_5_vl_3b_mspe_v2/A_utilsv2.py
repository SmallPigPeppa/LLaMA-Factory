import torch
import torch.nn.functional as F
from typing import Tuple

from s_editckpt_mspe71428 import patch_size


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
    '''
    False: boundaries[i-1] < input[m][n]...[l][x] <= boundaries[i]
    True: boundaries[i-1] <= input[m][n]...[l][x] < boundaries[i]
    compare with idx and cu, using True
    e.g. 
    window_index_ps[28]
    tensor([ 0,  1,  7,  8,  2,  3,  9, 10,  4,  5, 11, 12,  6, 13, 14, 15, 21, 22,
        16, 17, 23, 24, 18, 19, 25, 26, 20, 27, 28, 29, 35, 36, 30, 31, 37, 38,
        32, 33, 39, 40, 34, 41, 42, 43, 49, 50, 44, 45, 51, 52, 46, 47, 53, 54,
        48, 55, 56, 57, 63, 64, 58, 59, 65, 66, 60, 61, 67, 68, 62, 69, 70, 71,
        72, 73, 74, 75, 76])
    cu_window_seqlens_ps[28]
    tensor([  0,  16,  32,  48,  56,  72,  88, 104, 112, 128, 144, 160, 168, 184,
            200, 216, 224, 240, 256, 272, 280, 288, 296, 304, 308]
    '''
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


def split_to_window(
        img_thw: torch.Tensor,  # [B, 3], each row is (gt, gh, gw)
        win_size: int,
        patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reorder patches from grid order (t, h, w) to window-first order.

    Returns:
      x_win:       [N, dim] with the same patches but grouped by windows
      window_hws:  [B_window, 3] giving (t, h, w) size in patches for each window (edges may be smaller)
      coords:      [N, 3] integer tensor, each row is (t, h, w) patch coordinates
    """
    win_token = win_size // patch_size
    win_thw, coord_chunks = [], []

    for img_idx, (gt, gh, gw) in enumerate(img_thw.tolist()):
        gt, gh, gw = int(gt), int(gh), int(gw)

        # 构造三维坐标 grid
        grid = torch.arange(gt * gh * gw, device=img_thw.device).view(gt, gh, gw)
        ts = torch.arange(gt, device=img_thw.device).view(-1, 1, 1).expand(gt, gh, gw)
        hs = torch.arange(gh, device=img_thw.device).view(1, -1, 1).expand(gt, gh, gw)
        ws = torch.arange(gw, device=img_thw.device).view(1, 1, -1).expand(gt, gh, gw)
        imgs = torch.full((gt, gh, gw), img_idx, device=img_thw.device)

        # 按照 h,w 划分 window，t 不切
        for r in range(0, gh, win_token):
            for c in range(0, gw, win_token):
                sub = grid[:, r:r + win_token, c:c + win_token]  # [gt, h_win, w_win]
                sub_ts, sub_hs, sub_ws = ts[:, r:r + win_token, c:c + win_token], \
                    hs[:, r:r + win_token, c:c + win_token], \
                    ws[:, r:r + win_token, c:c + win_token]
                sub_imgs = imgs[:, r:r + win_token, c:c + win_token]

                win_thw.append((sub.size(0), sub.size(1), sub.size(2)))
                win_coords = torch.stack([
                    sub_imgs.reshape(-1),
                    sub_ts.reshape(-1),
                    sub_hs.reshape(-1),
                    sub_ws.reshape(-1)
                ], dim=1)
                # import pdb;pdb.set_trace()
                idx_merge = flatten_to_merge_idx(
                    grid_thw=torch.tensor([sub.size(0), sub.size(1), sub.size(2)]).unsqueeze(0), merge_size=2)
                coord_chunks.append(win_coords[idx_merge])

    coords = torch.cat(coord_chunks, dim=0)  # [N_patch, 4]
    # import pdb;pdb.set_trace()
    # idx_merge = flatten_to_merge_idx(grid_thw=img_thw, merge_size=2)
    # coords = coords[idx_merge]

    win_thw = img_thw.new_tensor(win_thw)  # [B_window, 3]

    return win_thw, coords


def recompose_windows(
        win_adp_ps,
        win_feat_dict,
        pos_emb_dict,
        win_idx_dict,
        win_cu_dict,
        grid_thw_dict,
        spatial_merge_unit,
        spatial_merge_size
):
    win_feat_list = []
    pos_emb_list = []
    win_idx_list = []
    token_itxy_list = []
    win_cu_seq = [0]

    # min_ps = min(window_adp_ps)
    min_ps = min(win_feat_dict.keys())

    # 遍历每个窗口及其对应的patch size
    for idx, ps in enumerate(win_adp_ps):
        # 获取当前窗口的起止位置
        start = win_cu_dict[ps][idx]
        end = win_cu_dict[ps][idx + 1]
        win_start = win_cu_dict[ps][idx] // spatial_merge_unit
        win_end = win_cu_dict[ps][idx + 1] // spatial_merge_unit

        # 提取对应窗口的数据
        # win_feat_list.append(win_feat_dict[ps][start:end])

        win_feat_ps = win_feat_dict[ps][start:end]
        L = win_feat_ps.size(0)
        # 其他特征对齐后加权
        others = []
        for k in win_feat_dict:
            if k == ps: continue
            ss, ee = win_cu_dict[k][idx], win_cu_dict[k][idx + 1]
            feat = win_feat_dict[k][ss:ee]
            feat = feat[:L] if feat.size(0) >= L else F.pad(feat, (0, 0, 0, L - feat.size(0)))
            others.append(feat)
        if others:
            win_feat_ps = win_feat_ps + 0. * torch.mean(torch.stack(others), dim=0)
        win_feat_list.append(win_feat_ps)

        pos_emb_list.append((
            pos_emb_dict[ps][0][start:end],
            pos_emb_dict[ps][1][start:end]
        ))
        # unified to min patchsize
        new_idx, token_itxy = scale_window_index(ps, min_ps, grid_thw_dict, win_idx_dict, win_start, win_end,
                                                 spatial_merge_unit, spatial_merge_size)
        win_idx_list.append(new_idx)
        token_itxy_list.append(token_itxy)

        # 更新累计长度
        win_cu_seq.append(win_cu_seq[-1] + (end - start))

    # 拼接结果
    hidden_states = torch.cat(win_feat_list, dim=0)
    position_embeddings_cos = torch.cat([emb[0] for emb in pos_emb_list], dim=0)
    position_embeddings_sin = torch.cat([emb[1] for emb in pos_emb_list], dim=0)
    # import pdb;pdb.set_trace()
    window_index = torch.cat(win_idx_list, dim=0)
    win_cu_seq = torch.tensor(win_cu_seq, dtype=win_cu_seq[-1].dtype,
                              device=win_cu_seq[-1].device)
    position_embeddings = (position_embeddings_cos, position_embeddings_sin)

    # for token itxy
    token_itxy = torch.cat(token_itxy_list, dim=0)  # [N, 4]

    return hidden_states, position_embeddings, window_index, win_cu_seq, token_itxy


def recompose_windows_v2(
        win_adp_ps,
        win_feat_dict,
        pos_emb_dict,
        win_cu_dict,
        win_coord_dict,
):
    win_feat_list = []
    pos_emb_list = []
    win_idx_list = []
    win_coord_list = []
    win_cu_seq = [0]

    # 遍历每个窗口及其对应的patch size
    for idx, ps in enumerate(win_adp_ps):
        # 获取当前窗口的起止位置
        start = win_cu_dict[ps][idx]
        end = win_cu_dict[ps][idx + 1]

        # 提取对应窗口的数据
        # win_feat_list.append(win_feat_dict[ps][start:end])

        win_feat_ps = win_feat_dict[ps][start:end]
        L = win_feat_ps.size(0)
        # 其他特征对齐后加权
        others = []
        for k in win_feat_dict:
            if k == ps: continue
            ss, ee = win_cu_dict[k][idx], win_cu_dict[k][idx + 1]
            feat = win_feat_dict[k][ss:ee]
            feat = feat[:L] if feat.size(0) >= L else F.pad(feat, (0, 0, 0, L - feat.size(0)))
            others.append(feat)
        if others:
            win_feat_ps = win_feat_ps + 0. * torch.mean(torch.stack(others), dim=0)
        win_feat_list.append(win_feat_ps)

        pos_emb_list.append((
            pos_emb_dict[ps][0][start:end],
            pos_emb_dict[ps][1][start:end]
        ))
        # unified to pixel coord
        coord = win_coord_dict[ps][start:end]
        coord[:, -2:] *= patch_size
        win_idx_list.append(coord)

        # 更新累计长度
        win_cu_seq.append(win_cu_seq[-1] + (end - start))

    # 拼接结果
    hidden_states = torch.cat(win_feat_list, dim=0)
    position_embeddings_cos = torch.cat([emb[0] for emb in pos_emb_list], dim=0)
    position_embeddings_sin = torch.cat([emb[1] for emb in pos_emb_list], dim=0)
    # import pdb;pdb.set_trace()
    win_cu_seq = torch.tensor(win_cu_seq, dtype=win_cu_seq[-1].dtype,device=win_cu_seq[-1].device)
    position_embeddings = (position_embeddings_cos, position_embeddings_sin)

    # for token itxy
    win_coord = torch.cat(win_coord_list, dim=0)  # [N, 4]

    return hidden_states, position_embeddings, win_cu_seq, win_coord


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


def update_ids_masks_labels(
        ids: torch.Tensor,  # (B, N)
        att_masks: torch.Tensor,  # (B, N)
        labels: torch.Tensor,  # (B, N)
        num_image_token: list,  # 只对应“有图像”的样本，顺序与原batch一致
        img_id: int,
):
    """
    有图像: 将 img_id 区块按 num_image_token[ni_idx] 延展/截断
    无图像: 保持整行不变直接加入结果
    最后对齐到 batch 内最大长度（右 pad，补最后一个 token）
    """
    B, N = ids.shape
    out_ids, out_att, out_labels = [], [], []
    new_lengths = []

    ni_idx = 0  # 指向 num_image_token，仅在 has_image=True 的样本上递增

    for i in range(B):
        t_row = ids[i]
        m_row = att_masks[i]
        l_row = labels[i]

        has_image = (t_row == img_id).any()

        if has_image:
            # 找到 img_id 区块
            mask = (t_row == img_id)
            idx = mask.nonzero(as_tuple=True)[0]
            s, e_ix = idx[0].item(), idx[-1].item() + 1
            old_count = e_ix - s
            new_count = int(num_image_token[ni_idx])
            ni_idx += 1

            # 三段拆分
            left_ids, block_ids, right_ids = t_row[:s], t_row[s:e_ix], t_row[e_ix:]
            left_att, block_att, right_att = m_row[:s], m_row[s:e_ix], m_row[e_ix:]
            left_lbl, block_lbl, right_lbl = l_row[:s], l_row[s:e_ix], l_row[e_ix:]

            # 延展/截断
            if new_count > old_count:
                ext = new_count - old_count
                block_ids = torch.cat([block_ids, block_ids[-1:].expand(ext)], dim=0)
                block_att = torch.cat([block_att, block_att[-1:].expand(ext)], dim=0)
                block_lbl = torch.cat([block_lbl, block_lbl[-1:].expand(ext)], dim=0)
            elif new_count < old_count:
                block_ids = block_ids[:new_count]
                block_att = block_att[:new_count]
                block_lbl = block_lbl[:new_count]
            # else: 等长，无需变化

            # 重新拼接
            new_ids_ = torch.cat([left_ids, block_ids, right_ids], dim=0)
            new_att_ = torch.cat([left_att, block_att, right_att], dim=0)
            new_lbl_ = torch.cat([left_lbl, block_lbl, right_lbl], dim=0)

        else:
            # 无图像 → 原样加入结果（不消耗 num_image_token）
            new_ids_ = t_row
            new_att_ = m_row
            new_lbl_ = l_row

        new_lengths.append(new_ids_.shape[0])
        out_ids.append(new_ids_)
        out_att.append(new_att_)
        out_labels.append(new_lbl_)

    # 右侧 pad 到最大长度（用各自最后一个 token 填充）
    max_len = max(new_lengths)
    for i in range(B):
        pad_len = max_len - out_ids[i].shape[0]
        if pad_len > 0:
            out_ids[i] = torch.cat([out_ids[i], out_ids[i][-1:].expand(pad_len)], dim=0)
            out_att[i] = torch.cat([out_att[i], out_att[i][-1:].expand(pad_len)], dim=0)
            out_labels[i] = torch.cat([out_labels[i], out_labels[i][-1:].expand(pad_len)], dim=0)

    # 堆叠
    new_ids = torch.stack(out_ids, dim=0)
    new_att_mask = torch.stack(out_att, dim=0)
    new_labels = torch.stack(out_labels, dim=0)
    return new_ids, new_att_mask, new_labels


def update_position_ids(position_ids, token_itxy, ids, img_id):
    """
    position_ids: torch tensor, shape [3, batch, max_len]
    grid_txy_list: list of torch.Tensor, 每个样本grid_txy: [n_img_token, 3]
    返回新的position_ids
    """

    pos_list = []
    max_len_new = 0
    ti_idx = 0  # token_itxy 的指针
    has_image = (ids == img_id).any(dim=1)

    dim, batch_size, max_len = position_ids.shape

    for b in range(batch_size):
        pos_ids = position_ids[:, b, :]  # [3, max_len]

        if has_image[b]:  # 有图像 → 更新
            grid_txy = token_itxy[ti_idx]  # 对应的图像token
            ti_idx += 1

            n_img_token = grid_txy.shape[0]
            t_row = ids[b]
            mask = (t_row == img_id)
            idx = mask.nonzero(as_tuple=True)[0]
            s, e_ix = idx[0].item(), idx[-1].item() + 1

            Z = max(
                len(torch.unique(pos_ids[0, s:e_ix])),
                len(torch.unique(pos_ids[1, s:e_ix])),
                len(torch.unique(pos_ids[2, s:e_ix]))
            )
            t_max = grid_txy[:, 0].max().item()
            x_max = grid_txy[:, 1].max().item()
            y_max = grid_txy[:, 2].max().item()
            Z_new = int(max(t_max, x_max, y_max)) + 1

            left_pos, img_pos, right_pos = pos_ids[:, :s], pos_ids[:, s:e_ix], pos_ids[:, e_ix:]
            new_img_pos = torch.zeros((3, n_img_token), dtype=pos_ids.dtype, device=pos_ids.device)
            for i in range(n_img_token):
                t, x, y = grid_txy[i].tolist()
                new_img_pos[0, i] = pos_ids[0, s] + t
                new_img_pos[1, i] = pos_ids[1, s] + x
                new_img_pos[2, i] = pos_ids[2, s] + y

            text_mask = (right_pos != 1)  # padding=1
            text_ids = right_pos.clone()
            offset = Z_new - Z
            text_ids[text_mask] += offset

            new_pos = torch.cat([left_pos, new_img_pos, text_ids], dim=1)

        else:  # 无图像 → 原样保留
            new_pos = pos_ids

        pos_list.append(new_pos)
        max_len_new = max(max_len_new, new_pos.shape[1])

    # 最后 padding
    padded_pos = []
    for out in pos_list:
        cur_len = out.shape[1]
        if cur_len < max_len_new:
            pad = max_len_new - cur_len
            last_col = out[:, -1:].expand(3, pad)
            out = torch.cat([out, last_col], dim=1)
        padded_pos.append(out)

    final_out = torch.stack(padded_pos, dim=1)  # [3, batch, max_len_new]
    return final_out
