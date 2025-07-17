import torch
from typing import Tuple
import torch.nn.functional as F
import numpy as np




def compute_grid_txy(self, grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Create (t, x, y) coords for each patch in a batch.
    Args:
        grid_thw: [B, 3] tensor of [T, H, W].
    Returns:
        [sum(T*H*W), 3] tensor of coords.
    """
    device = grid_thw.device
    coords = []
    for t, h, w in grid_thw.tolist():
        ts = torch.arange(t, device=device)  # 0..T-1
        xs = torch.arange(h, device=device)  # 0..H-1
        ys = torch.arange(w, device=device)  # 0..W-1
        grid = torch.stack(
            torch.meshgrid(ts, xs, ys, indexing='ij'),
            dim=-1
        )  # [T, H, W, 3]
        coords.append(grid.view(-1, 3))  # flatten
    return torch.cat(coords, dim=0)  # [N, 3]


def recompute_grid_xy_new(self, grid_txy: torch.Tensor, new_patchsize: int):
    """
    Recompute grid coordinates to match new patch layout.

    Args:
        grid_txy (torch.Tensor): shape [N, 3], each item is (t, x, y)
        new_patchsize (int): new desired patch size (e.g., 7, 14, 28)

    Returns:
        torch.Tensor: new_grid_txy sorted by x, then y, matching repatchify patch order
    """
    assert new_patchsize in [self.patch_size // 2, self.patch_size, self.patch_size * 2], "Unsupported new_patchsize"

    t, x, y = grid_txy[:, 0], grid_txy[:, 1], grid_txy[:, 2]

    if new_patchsize == self.patch_size:
        new_x = x * 2 + 0.5
        new_y = y * 2 + 0.5
        new_grid_txy = torch.stack([t, new_x, new_y], dim=1)

    elif new_patchsize == self.patch_size // 2:
        coords = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                new_x = x * 2 + dx
                new_y = y * 2 + dy
                coords.append(torch.stack([t, new_x, new_y], dim=1))
        new_grid_txy = torch.cat(coords, dim=0)
    else:  # new_patchsize == self.patch_size * 2
        small_x = x * 2 + 0.5
        small_y = y * 2 + 0.5

        group_coords = {}
        for i in range(len(grid_txy)):
            norm_x = int(x[i] - x.min())
            norm_y = int(y[i] - y.min())
            key = (int(t[i]), norm_x // 2, norm_y // 2)
            if key not in group_coords:
                group_coords[key] = []
            group_coords[key].append(torch.stack([small_x[i], small_y[i]]))

        new_coords = []
        for (t_key, x_key, y_key), points in group_coords.items():
            points_tensor = torch.stack(points)  # shape [n, 2]
            mean_xy = points_tensor.mean(dim=0)
            new_coords.append([t_key, mean_xy[0], mean_xy[1]])

        new_grid_txy = torch.tensor(new_coords, device=grid_txy.device)

    # Sort by t, then x, then y to match repatchify order
    txy = new_grid_txy.cpu().numpy()
    sorted_indices = np.lexsort((txy[:, 2], txy[:, 1], txy[:, 0]))  # y, x, t
    sorted_indices = torch.from_numpy(sorted_indices).to(new_grid_txy.device)
    new_grid_txy = new_grid_txy[sorted_indices]

    return new_grid_txy



def repatchify(
        self,
        pixel_value: torch.Tensor,  # [N, C*T*ps*ps]
        grid_thw: torch.Tensor,  # [T, H, W]
        new_patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Re-patchify with zero-padding if needed.

    Args:
        pixel_value: [N, dim], dim = C*T*old_ps*old_ps
        grid_thw:     [3] tensor (T, H, W)
        new_patch_size: desired spatial size

    Returns:
        new_pixel_value: [N', dim'] tensor
        new_grid_thw:     [T, H', W'] tensor
    """
    T, H, W = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    C, t_ps, old_ps = self.in_channels, self.temporal_patch_size, self.patch_size

    # reconstruct volume [C, T, H*ps, W*ps]
    x = pixel_value.view(H, W, C, t_ps, old_ps, old_ps)
    full = x.permute(2, 3, 0, 4, 1, 5).contiguous().view(C, t_ps, H * old_ps, W * old_ps)

    # pad to multiple of new_patch_size
    Hf, Wf = full.shape[2], full.shape[3]
    ph = (new_patch_size - Hf % new_patch_size) % new_patch_size
    pw = (new_patch_size - Wf % new_patch_size) % new_patch_size
    if ph or pw:
        full = F.pad(full, (0, pw, 0, ph))
        Hf += ph;
        Wf += pw

    # compute new grid
    Hn, Wn = Hf // new_patch_size, Wf // new_patch_size

    # extract patches and flatten
    patches = full.unfold(2, new_patch_size, new_patch_size) \
        .unfold(3, new_patch_size, new_patch_size)
    patches = patches.permute(2, 4, 0, 1, 3, 5) \
        .contiguous() \
        .view(-1, C, t_ps, new_patch_size, new_patch_size)
    new_pixel_value = patches.view(patches.size(0), -1)
    new_grid_thw = torch.Tensor([T, Hn, Wn], device=grid_thw.device)

    return new_pixel_value, new_grid_thw


def recompute_grid_xy(grid_txy: torch.Tensor, old_patchsize: int, new_patchsize: int):
    """
    Recompute grid coordinates to match new patch layout.

    Args:
        grid_txy (torch.Tensor): shape [N, 3], each item is (t, x, y)
        old_patchsize (int): original patch size (e.g., 14)
        new_patchsize (int): new desired patch size (e.g., 7, 14, 28)

    Returns:
        torch.Tensor: new_grid_txy sorted by x, then y, matching repatchify patch order
    """
    assert new_patchsize in [old_patchsize // 2, old_patchsize, old_patchsize * 2], "Unsupported new_patchsize"

    t, x, y = grid_txy[:, 0], grid_txy[:, 1], grid_txy[:, 2]

    if new_patchsize == old_patchsize:
        new_x = x * 2 + 0.5
        new_y = y * 2 + 0.5
        new_grid_txy = torch.stack([t, new_x, new_y], dim=1)

    elif new_patchsize == old_patchsize // 2:
        coords = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                new_x = x * 2 + dx
                new_y = y * 2 + dy
                coords.append(torch.stack([t, new_x, new_y], dim=1))
        new_grid_txy = torch.cat(coords, dim=0)
    else:  # new_patchsize == old_patchsize * 2
        small_x = x * 2 + 0.5
        small_y = y * 2 + 0.5

        group_coords = {}
        for i in range(len(grid_txy)):
            norm_x = int(x[i] - x.min())
            norm_y = int(y[i] - y.min())
            key = (int(t[i]), norm_x // 2, norm_y // 2)
            if key not in group_coords:
                group_coords[key] = []
            group_coords[key].append(torch.stack([small_x[i], small_y[i]]))

        new_coords = []
        for (t_key, x_key, y_key), points in group_coords.items():
            points_tensor = torch.stack(points)  # shape [n, 2]
            mean_xy = points_tensor.mean(dim=0)
            new_coords.append([t_key, mean_xy[0], mean_xy[1]])

        new_grid_txy = torch.tensor(new_coords, device=grid_txy.device)

    # Sort by t, then x, then y to match repatchify order
    txy = new_grid_txy.cpu().numpy()
    sorted_indices = np.lexsort((txy[:, 2], txy[:, 1], txy[:, 0]))  # y, x, t
    sorted_indices = torch.from_numpy(sorted_indices).to(new_grid_txy.device)
    new_grid_txy = new_grid_txy[sorted_indices]

    return new_grid_txy


# Example usage:
if __name__ == '__main__':
    grid_txy = torch.tensor([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1]
    ])
    grid_txy = torch.tensor([
        [0, 5, 7],
        [0, 5, 8],
        [0, 6, 7],
        [0, 6, 8],
        [1, 5, 7],
        [1, 5, 8],
        [1, 6, 7],
        [1, 6, 8],
    ])

    print("Original grid_txy:")
    print(grid_txy)

    print("\nPatchsize from 14 to 7:")
    print(recompute_grid_xy(grid_txy, 14, 7))

    print("\nPatchsize unchanged (14):")
    print(recompute_grid_xy(grid_txy, 14, 14))

    print("\nPatchsize from 14 to 28:")
    print(recompute_grid_xy(grid_txy, 14, 28))


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")
#
#     batch = torch.tensor([[1, 24, 28],
#                           [1, 20, 28]], device=device)
#     coords = grid_coords(batch)
#
#     print(coords.shape)  # torch.Size([1232, 3])
#     print(coords[29])  # tensor([0, 0, 0], device=device)
