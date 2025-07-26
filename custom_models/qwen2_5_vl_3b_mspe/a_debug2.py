import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Dummy patch_embed
class DummyPatchEmbed:
    def __init__(self, in_channels=1, temporal_patch_size=2):
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size

class Visualizer:
    def __init__(self, patch_size=4):
        self.patch_size = patch_size
        self.patch_embed = DummyPatchEmbed()

    def repatchify(self, pixel_value: torch.Tensor, grid_thw: torch.Tensor, new_patch_size: int):
        T, H, W = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
        C, t_ps, old_ps = self.patch_embed.in_channels, self.patch_embed.temporal_patch_size, self.patch_size

        x = pixel_value.view(H, W, C, t_ps, old_ps, old_ps)
        full = x.permute(2, 3, 0, 4, 1, 5).contiguous().view(C, t_ps, H * old_ps, W * old_ps)

        Hf, Wf = full.shape[2], full.shape[3]
        ph = (new_patch_size - Hf % new_patch_size) % new_patch_size
        pw = (new_patch_size - Wf % new_patch_size) % new_patch_size
        if ph or pw:
            full = F.pad(full, (0, pw, 0, ph))
            Hf += ph
            Wf += pw

        Hn, Wn = Hf // new_patch_size, Wf // new_patch_size

        patches = full.unfold(2, new_patch_size, new_patch_size).unfold(3, new_patch_size, new_patch_size)
        patches = patches.permute(2, 4, 0, 1, 3, 5).contiguous().view(-1, C, t_ps, new_patch_size, new_patch_size)
        new_pixel_value = patches.view(patches.size(0), -1)
        new_grid_thw = torch.tensor([T, Hn, Wn], dtype=torch.long)

        return new_pixel_value, new_grid_thw, full

# ========== 小图像：C=1, T=2, H=16, W=16 ==========
volume = torch.arange(1, 1 + 1*2*16*16).float().view(1, 2, 16, 16)

def flatten_patchify(volume, patch_size):
    patches = volume.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return patches.permute(2, 3, 0, 1, 4, 5).contiguous().view(-1, 1*2*patch_size*patch_size)

pixel_value_8 = flatten_patchify(volume, patch_size=8)
grid_thw_8 = torch.tensor([2, 2, 2])

pixel_value_4 = flatten_patchify(volume, patch_size=4)
grid_thw_4 = torch.tensor([2, 4, 4])

vis8 = Visualizer(patch_size=8)
_, _, vol8 = vis8.repatchify(pixel_value_8, grid_thw_8, new_patch_size=8)

vis4 = Visualizer(patch_size=4)
_, _, vol4 = vis4.repatchify(pixel_value_4, grid_thw_4, new_patch_size=4)

# ========= 可视化 =========
fig, axs = plt.subplots(3, 2, figsize=(10, 9))
titles = ["Original", "PatchSize=8", "PatchSize=4"]

for i in range(2):
    axs[0, i].imshow(volume[0, i].numpy(), cmap='gray')
    axs[0, i].set_title(f"{titles[0]} - T{i}")
    axs[0, i].axis('off')

    axs[1, i].imshow(vol8[0, i].numpy(), cmap='gray')
    axs[1, i].set_title(f"{titles[1]} - T{i}")
    axs[1, i].axis('off')

    axs[2, i].imshow(vol4[0, i].numpy(), cmap='gray')
    axs[2, i].set_title(f"{titles[2]} - T{i}")
    axs[2, i].axis('off')

plt.tight_layout()
plt.show()
