import torch

def update_position_ids(position_ids, grid_txy_list):
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
        grid_txy = grid_txy_list[b]      # [n_img_token, 3]
        n_img_token = grid_txy.shape[0]

        # 1. 找到“图像”部分
        uniq, counts = torch.unique(pos_ids[0][pos_ids[0] != 1], return_counts=True)
        img_id = uniq[counts.argmax()]
        img_mask = (pos_ids[0] == img_id)
        img_start = img_mask.nonzero(as_tuple=True)[0][0].item()
        img_end = img_mask.nonzero(as_tuple=True)[0][-1].item() + 1

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
        text = pos_ids[:, img_end:]     # [3, 原后缀长]
        text_mask = (text != 1)         # padding=1
        text_ids = text.clone()
        offset = Z_new - Z
        text_ids[text_mask] += offset

        # 6. 拼接
        merged = torch.cat([pre, new_img_pos, text_ids], dim=1)  # [3, 新长度]
        out_list.append(merged)
        max_len_new = max(max_len_new, merged.shape[1])

    # 7. 补pad（统一到 batch 内最大长度）
    final_out = torch.ones((3, batch_size, max_len_new), dtype=position_ids.dtype, device=position_ids.device)
    for b, out in enumerate(out_list):
        cur_len = out.shape[1]
        final_out[:, b, :cur_len] = out

    return torch.cat(out_list, dim=1)  # [3, batch, max_len]
