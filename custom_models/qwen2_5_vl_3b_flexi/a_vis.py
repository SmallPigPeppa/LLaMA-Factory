import torch
import matplotlib.pyplot as plt

class DummyRepatchifier:
    def __init__(self, in_channels, temporal_patch_size, patch_size):
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size

    def patchify(self, x):
        """
        将 video 张量 [B, C, T, H, W] 切成 patch：
        输出 hidden_states: [B * T_new * H_new * W_new, C*t*ps*ps]
        和对应的 grid_thw: [B, (T_new, H_new, W_new)]
        """
        B, C, T, H, W = x.shape
        t = self.temporal_patch_size
        ps = self.patch_size
        assert T % t == 0 and H % ps == 0 and W % ps == 0

        Tn = T // t
        Hn = H // ps
        Wn = W // ps

        # 先把 time 维度拉出来
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(B, Tn, t, C, Hn, ps, Wn, ps)
        x = x.permute(0, 1, 4, 6, 3, 2, 5, 7)  # [B, Tn, Hn, Wn, C, t, ps, ps]
        patches = x.reshape(-1, C * t * ps * ps)  # [B*Tn*Hn*Wn, D_per_patch]
        grid_thw = torch.tensor([[Tn, Hn, Wn] for _ in range(B)], device=x.device)
        return patches, grid_thw

    def repatchify(self, hidden_states, grid_thw, new_patch_size):
        # —— 这里直接贴你给的 repatchify 代码 —— #
        old_patch_size = self.patch_size
        B = grid_thw.shape[0]
        C = self.in_channels
        t = self.temporal_patch_size
        ps = old_patch_size
        D_per_patch = C * t * ps * ps
        assert hidden_states.shape[1] == D_per_patch

        hidden_states_new = []
        new_grid_thw = []

        offset = 0
        for b in range(B):
            T, H, W = grid_thw[b].tolist()
            num_patches = T * H * W
            hs = hidden_states[offset:offset + num_patches]
            offset += num_patches

            hs_3d = hs.view(T, H, W, C, t, ps, ps)
            hs_3d = hs_3d.permute(0, 3, 4, 1, 5, 2, 6).contiguous()
            hs_3d = hs_3d.view(T, C * t, H * ps, W * ps)

            H_new = (H * ps) // new_patch_size
            W_new = (W * ps) // new_patch_size
            assert (H * ps) % new_patch_size == 0
            assert (W * ps) % new_patch_size == 0

            hs_repatched = hs_3d.view(
                T,
                C * t,
                H_new,
                new_patch_size,
                W_new,
                new_patch_size
            )
            hs_repatched = hs_repatched.permute(0, 2, 4, 1, 3, 5).contiguous()
            hs_repatched = hs_repatched.view(-1, C * t * new_patch_size * new_patch_size)

            hidden_states_new.append(hs_repatched)
            new_grid_thw.append([T, H_new, W_new])

        hidden_states_new = torch.cat(hidden_states_new, dim=0)
        new_grid_thw = torch.tensor(new_grid_thw, device=grid_thw.device)
        return hidden_states_new, new_grid_thw

def visualize_repatchify_identity():
    ps = 2           # 原始 patch_size
    t  = 1           # temporal_patch_size
    C  = 1           # 单通道，为了简单起见
    B, T, H, W = 1, 4, 8, 8  # 1 段 video，T=4 帧，空间 8×8

    # 随机视频
    x = torch.randn(B, C, T, H, W)

    model = DummyRepatchifier(in_channels=C,
                              temporal_patch_size=t,
                              patch_size=ps)

    # 1) patchify
    hidden, grid = model.patchify(x)
    # 2) repatchify，（这里用 new_patch_size == old_patch_size，等价于“身份映射”）
    hidden2, grid2 = model.repatchify(hidden, grid, new_patch_size=ps)

    # 画一个散点图：横轴原始 hidden，纵轴重划分后 hidden
    h1 = hidden.flatten().cpu().numpy()
    h2 = hidden2.flatten().cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.scatter(h1, h2, s=1)
    mn = min(h1.min(), h2.min())
    mx = max(h1.max(), h2.max())
    plt.plot([mn, mx], [mn, mx], '--')  # 对角线
    plt.xlabel("Original patches")
    plt.ylabel("Repatched patches")
    plt.title("Scatter of hidden vs. repatched hidden")
    plt.show()

    # 你还可以打印最大误差来量化验证
    print("max abs difference:", float(torch.abs(hidden - hidden2).max()))

# 调用
visualize_repatchify_identity()
