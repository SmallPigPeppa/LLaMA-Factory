import torch
from typing import List
def update_ids_masks_labels(
        ids: torch.Tensor,         # (B, N)
        att_masks: torch.Tensor,   # (B, N)
        labels: torch.Tensor,      # (B, N)
        num_image_token: list,     # (B)
        img_id: int,
):
    """
    对每个batch，img_id那段token根据num_image_token长度延展/截断，再整体pad到最大长度。
    返回：
        new_ids:      (B, N_new)
        new_att_mask: (B, N_new)
        new_labels:   (B, N_new)
    """
    B, N = ids.shape
    out_ids, out_att, out_labels = [], [], []

    new_lengths = []
    for i in range(B):
        t_row = ids[i]
        m_row = att_masks[i]
        l_row = labels[i]

        # 找到img_id区块
        mask = (t_row == img_id)
        idx = mask.nonzero(as_tuple=True)[0]
        s, e_ix = idx[0].item(), idx[-1].item() + 1
        old_count = e_ix - s
        new_count = num_image_token[i]

        # 分三段：左边、img_id区块、右边
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
        # else new_count == old_count, 无需变化

        # 拼接整行
        new_ids_ = torch.cat([left_ids, block_ids, right_ids], dim=0)
        new_att_ = torch.cat([left_att, block_att, right_att], dim=0)
        new_lbl_ = torch.cat([left_lbl, block_lbl, right_lbl], dim=0)
        new_lengths.append(new_ids_.shape[0])

        out_ids.append(new_ids_)
        out_att.append(new_att_)
        out_labels.append(new_lbl_)

    # 对齐到最大长度（右pad，补最后一个token）
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


def update_input_embeds_ids_masks_labels(
        embeds: torch.Tensor,  # (B, N, C)
        ids: torch.Tensor,  # (B, N)
        att_masks: torch.Tensor,  # (B, N)
        labels: torch.Tensor,  # (B, N)
        num_image_token: List, #(B)
        img_id: int,
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    - For each batch i where flags[i] is True:
        * Replace the slice of image tokens in embeds[i]/ids[i]/att_mask[i]/labels[i]
          with a padded block (last element repeated) of size new_hw[i].
    - For flags[i] False: keep original rows.
    - Finally pad all sequences to the same length (max over B) by repeating each
      sequence's last token/value, and stack into tensors.
    Returns:
      new_embeds:    (B, N_new, C)
      new_ids:       (B, N_new)
      new_att_mask:  (B, N_new)
      new_labels:    (B, N_new)
    """
    B, N, C = embeds.shape
    out_embeds, out_ids, out_att, out_labels = [], [], [], []

    for i in range(B):
        e_row = embeds[i]
        t_row = ids[i]
        m_row = att_masks[i]
        l_row = labels[i]
        mask = (t_row == img_id)
        idx = mask.nonzero(as_tuple=True)[0]
        s, e_ix = idx[0].item(), idx[-1].item() + 1
        old_count = e_ix - s

        new_count = num_image_token[i]

        # slice the original blocks
        block_emb = e_row[s:e_ix]
        block_ids = t_row[s:e_ix]
        block_att = m_row[s:e_ix]
        block_labels = l_row[s:e_ix]

        # extend if needed
        if new_count > old_count:
            ext = new_count - old_count
            block_emb = torch.cat([block_emb, block_emb[-1:].expand(ext, -1)], dim=0)
            block_ids = torch.cat([block_ids, block_ids[-1:].expand(ext)], dim=0)
            block_att = torch.cat([block_att, block_att[-1:].expand(ext)], dim=0)
            block_labels = torch.cat([block_labels, block_labels[-1:].expand(ext)], dim=0)

        # rebuild each sequence
        new_e = torch.cat([e_row[:s], block_emb, e_row[e_ix:]], dim=0)
        new_t = torch.cat([t_row[:s], block_ids, t_row[e_ix:]], dim=0)
        new_m = torch.cat([m_row[:s], block_att, m_row[e_ix:]], dim=0)
        new_l = torch.cat([l_row[:s], block_labels, l_row[e_ix:]], dim=0)

        out_embeds.append(new_e)
        out_ids.append(new_t)
        out_att.append(new_m)
        out_labels.append(new_l)

    # pad to common length
    max_len = max(x.shape[0] for x in out_embeds)
    padded_embeds, padded_ids, padded_att, padded_labels = [], [], [], []

    for e_row, t_row, m_row, l_row in zip(out_embeds, out_ids, out_att, out_labels):
        L = e_row.shape[0]
        if L < max_len:
            pad = max_len - L
            e_row = torch.cat([e_row, e_row[-1:].expand(pad, -1)], dim=0)
            t_row = torch.cat([t_row, t_row[-1:].expand(pad)], dim=0)
            m_row = torch.cat([m_row, m_row[-1:].expand(pad)], dim=0)
            l_row = torch.cat([l_row, l_row[-1:].expand(pad)], dim=0)
        padded_embeds.append(e_row)
        padded_ids.append(t_row)
        padded_att.append(m_row)
        padded_labels.append(l_row)

    new_embeds = torch.stack(padded_embeds, dim=0)  # (B, N_new, C)
    new_ids = torch.stack(padded_ids, dim=0)  # (B, N_new)
    new_att_mask = torch.stack(padded_att, dim=0)  # (B, N_new)
    new_labels = torch.stack(padded_labels, dim=0)  # (B, N_new)

    return new_embeds, new_ids, new_att_mask, new_labels


def update_position_ids(position_ids, token_itxy, ids, img_id):
    """
    position_ids: torch tensor, shape [3, batch, max_len]
    grid_txy_list: list of torch.Tensor, 每个样本grid_txy: [n_img_token, 3]
    返回新的position_ids
    """
    dim, batch_size, max_len = position_ids.shape
    out_list = []
    max_len_new = 0

    for b in range(batch_size):
        pos_ids = position_ids[:, b, :]  # [3, max_len]
        grid_txy = token_itxy[b]  # [n_img_token, 3]
        n_img_token = grid_txy.shape[0]

        # 1. 找到“图像”部分
        t_row = ids[b]
        mask = (t_row == img_id)
        idx = mask.nonzero(as_tuple=True)[0]
        img_start, img_end = idx[0].item(), idx[-1].item() + 1

        # 2. 占位符种类数
        Z = max(
            len(torch.unique(pos_ids[0, img_start:img_end])),
            len(torch.unique(pos_ids[1, img_start:img_end])),
            len(torch.unique(pos_ids[2, img_start:img_end]))
        )
        t_max = grid_txy[:, 0].max().item()
        x_max = grid_txy[:, 1].max().item()
        y_max = grid_txy[:, 2].max().item()
        Z_new = int(max(t_max, x_max, y_max)) + 1

        # 3. 前缀
        pre = pos_ids[:, :img_start]  # [3, img_start]
        # 4. 新的图像部分
        new_img_pos = torch.zeros((3, n_img_token), dtype=pos_ids.dtype, device=pos_ids.device)
        for i in range(n_img_token):
            t, x, y = grid_txy[i].tolist()
            new_img_pos[0, i] = pos_ids[0, img_start] + t
            new_img_pos[1, i] = pos_ids[1, img_start] + x
            new_img_pos[2, i] = pos_ids[2, img_start] + y

        # 5. 后缀文本区（整体偏移Z_new-Z）
        text = pos_ids[:, img_end:]  # [3, 原后缀长]
        text_mask = (text != 1)  # padding=1
        text_ids = text.clone()
        offset = Z_new - Z
        text_ids[text_mask] += offset

        # 6. 拼接
        merged = torch.cat([pre, new_img_pos, text_ids], dim=1)  # [3, 新长度]
        out_list.append(merged)
        max_len_new = max(max_len_new, merged.shape[1])


    # 7. 补pad（统一到 batch 内最大长度），每行repeat last col
    padded_pos = []
    for out in out_list:  # out: [3, cur_len]
        cur_len = out.shape[1]
        if cur_len < max_len_new:
            pad = max_len_new - cur_len
            # repeat last col
            last_col = out[:, -1:].expand(3, pad)  # [3, pad]
            out = torch.cat([out, last_col], dim=1)  # [3, max_len_new]
        padded_pos.append(out)

    final_out = torch.stack(padded_pos, dim=1)  # [3, batch, max_len_new]
    return final_out

    # return torch.cat(out_list, dim=1)  # [3, batch, max_len]