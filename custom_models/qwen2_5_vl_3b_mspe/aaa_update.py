import torch
from typing import List
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