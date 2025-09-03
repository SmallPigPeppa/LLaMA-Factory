import torch
import torch.nn.functional as F

def split_by_cu(feat, cu_seqlens):
    """根据 cu_seqlens 划分 window"""
    return [feat[cu_seqlens[i]:cu_seqlens[i+1]] for i in range(len(cu_seqlens)-1)]

def reshape_to_thw(feat, thw):
    """把 [N,C] reshape 成 [T,H,W,C]"""
    T, H, W = thw
    return feat.view(T, H, W, -1)

def resize_feat_3d(feat, target_thw):
    """把 [T,H,W,C] 通过 avgpool resize 到 target_thw"""
    dtype = feat.dtype
    feat = feat.permute(3, 0, 1, 2).unsqueeze(0)  # [1,C,T,H,W]

    # 转 float32 做 pooling
    feat = feat.to(torch.float32)
    feat = F.adaptive_avg_pool3d(feat, target_thw)

    feat = feat.to(dtype)
    return feat.squeeze(0).permute(1, 2, 3, 0)  # [T,H,W,C]

def window_diff(feat_a, feat_b):
    """计算两个 window feature 的差异"""
    diff = (feat_a.to(torch.float32) - feat_b.to(torch.float32)).abs()
    return diff.mean().item()

def assign_patchsize_from_diff(diffs, patch_sizes=(7, 14)):
    """根据差异值给每个 window 分配 patch size (余数分到第一组)"""
    n = len(diffs)
    base = n // len(patch_sizes)
    remain = n % len(patch_sizes)

    group_sizes = [base + remain] + [base] * (len(patch_sizes) - 1)

    sorted_idx = sorted(range(n), key=lambda i: diffs[i], reverse=True)

    patch_assign = [None] * n
    start = 0
    for group_id, gsize in enumerate(group_sizes):
        for idx in sorted_idx[start:start+gsize]:
            patch_assign[idx] = patch_sizes[group_id]
        start += gsize

    return patch_assign

def get_adp_win_patchsize(win_feat_dict, win_thws_dict, win_cu_dict):
    """
    输入:
        win_feat_dict: {ps: feature tensor}, e.g. {7:..., 14:...}, shape [N,C]
        win_thws_dict: {ps: [(T,H,W), ...]}
        win_cu_dict:   {ps: cu_seqlens}
    输出:
        diffs: 每个 window 的差异值
        patch_assign: 每个 window 的自适应 patch size (7/14)
    """
    # 1. 拆分成 window
    win_feats_7 = split_by_cu(win_feat_dict[7], win_cu_dict[7])
    win_feats_14 = split_by_cu(win_feat_dict[14], win_cu_dict[14])

    diffs = []
    for f7, f14, thw7, thw14 in zip(win_feats_7, win_feats_14, win_thws_dict[7], win_thws_dict[14]):
        # import pdb; pdb.set_trace()
        f7 = reshape_to_thw(f7, thw7)
        f14 = reshape_to_thw(f14, thw14)

        # 对齐到 ps=14 的空间大小
        f7_resized = resize_feat_3d(f7, thw14)
        f14_resized = f14

        diff_val = window_diff(f7_resized, f14_resized)
        diffs.append(diff_val)

    # 2. 根据差异分配 patch size
    patch_assign = assign_patchsize_from_diff(diffs)

    return diffs, patch_assign


def get_infer_adp_win_patchsize(win_feat_dict, win_thws_dict, win_cu_dict):
    """
    输入:
        win_feat_dict: {ps: feature tensor}, e.g. {7:..., 14:...}, shape [N,C]
        win_thws_dict: {ps: [(T,H,W), ...]}
        win_cu_dict:   {ps: cu_seqlens}
    输出:
        diffs: 每个 window 的差异值
        patch_assign: 每个 window 的自适应 patch size (7/14)
    """
    # 1. 拆分成 window
    win_feats_7 = split_by_cu(win_feat_dict[7], win_cu_dict[7])
    win_feats_14 = split_by_cu(win_feat_dict[14], win_cu_dict[14])

    diffs = []
    for f7, f14, thw7, thw14 in zip(win_feats_7, win_feats_14, win_thws_dict[7], win_thws_dict[14]):
        # import pdb; pdb.set_trace()
        f7 = reshape_to_thw(f7, thw7)
        f14 = reshape_to_thw(f14, thw14)

        # 对齐到 ps=14 的空间大小
        f7_resized = resize_feat_3d(f7, thw14)
        f14_resized = f14

        diff_val = window_diff(f7_resized, f14_resized)
        diffs.append(diff_val)

    # 2. 根据差异分配 patch size
    # 2. 按照差异值从大到小排序
    n = len(diffs)
    sorted_idx = sorted(range(n), key=lambda i: diffs[i], reverse=True)
    patch_assign = [14] * n  # 默认都分配 14
    for idx in sorted_idx[:min(12, n)]:
        patch_assign[idx] = 7

    return diffs, patch_assign

