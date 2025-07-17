import torch
from typing import Tuple
import torch.nn.functional as F
import numpy as np

def itxy_to_flatten_index(grid_thw: torch.Tensor, grid_itxy: torch.Tensor) -> torch.Tensor:
    image_idx, t, x, y = grid_itxy.unbind(dim=1)
    H, W = grid_thw[image_idx, 1], grid_thw[image_idx, 2]
    patch_per_image = grid_thw.prod(dim=1)
    offset = torch.cumsum(torch.cat([patch_per_image.new_zeros(1), patch_per_image[:-1]]), dim=0)
    return offset[image_idx] + t * H * W + x * W + y




def compute_grid_itxy(self, grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Create (image_idx, t, x, y) coords for each patch in a batch.
    Args:
        grid_thw: [B, 3] tensor of [T, H, W] per image.
    Returns:
        [sum(T*H*W), 4] tensor of coords: [image_idx, t, x, y].
    """
    device = grid_thw.device
    coords = []
    for idx, (t, h, w) in enumerate(grid_thw.tolist()):
        ts = torch.arange(t, device=device)
        xs = torch.arange(h, device=device)
        ys = torch.arange(w, device=device)
        grid = torch.stack(
            torch.meshgrid(ts, xs, ys, indexing='ij'),
            dim=-1
        )  # [T, H, W, 3]
        grid = grid.view(-1, 3)  # [T*H*W, 3]
        image_idx = torch.full((grid.shape[0], 1), idx, device=device, dtype=torch.long)
        grid_with_idx = torch.cat([image_idx, grid], dim=1)  # [T*H*W, 4]
        coords.append(grid_with_idx)
    return torch.cat(coords, dim=0)  # [N, 4]


def recompute_grid_ityx_self(self, grid_ityx: torch.Tensor, new_patchsize: int) -> torch.Tensor:
    """
    Recompute grid coordinates to match new patch layout, keeping image_idx.

    Args:
        grid_ityx (torch.Tensor): shape [N, 4], each item is (image_idx, t, x, y)
        new_patchsize (int): new desired patch size (e.g., 7, 14, 28)

    Returns:
        torch.Tensor: new_grid_ityx sorted by image_idx, t, x, y
    """
    old_patchsize = self.patch_size
    assert new_patchsize in [old_patchsize // 2, old_patchsize, old_patchsize * 2], "Unsupported new_patchsize"

    image_idx = grid_ityx[:, 0]
    t = grid_ityx[:, 1]
    x = grid_ityx[:, 2]
    y = grid_ityx[:, 3]

    if new_patchsize == old_patchsize:
        new_x = x * 2
        new_y = y * 2
        new_grid = torch.stack([image_idx, t, new_x, new_y], dim=1)

    elif new_patchsize == old_patchsize // 2:
        coords = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                new_x = x * 2 + dx
                new_y = y * 2 + dy
                coords.append(torch.stack([image_idx, t, new_x, new_y], dim=1))
        new_grid = torch.cat(coords, dim=0)

    else:  # new_patchsize == old_patchsize * 2
        new_x = x * 2
        new_y = y * 2

        new_coords = []
        for img in image_idx.unique():
            img_mask = image_idx == img
            img_t = t[img_mask]
            img_x = new_x[img_mask]
            img_y = new_y[img_mask]

            group_coords = {}
            for i in range(len(img_t)):
                norm_x = int(x[img_mask][i] - x[img_mask].min())
                norm_y = int(y[img_mask][i] - y[img_mask].min())
                key = (int(img_t[i]), norm_x // 2, norm_y // 2)
                if key not in group_coords:
                    group_coords[key] = []
                # ensure float during mean calculation
                group_coords[key].append(torch.stack([img_x[i].float(), img_y[i].float()]))

            for (t_key, x_key, y_key), points in group_coords.items():
                points_tensor = torch.stack(points)  # shape [n, 2]
                mean_xy = points_tensor.mean(dim=0).round().to(grid_ityx.dtype)
                new_coords.append([img.item(), t_key, mean_xy[0], mean_xy[1]])

        new_grid = torch.tensor(new_coords, device=grid_ityx.device)

    # Sort by image_idx, then t, then x, then y
    itxy = new_grid.cpu().numpy()
    sorted_indices = np.lexsort((itxy[:, 3], itxy[:, 2], itxy[:, 1], itxy[:, 0]))
    sorted_indices = torch.from_numpy(sorted_indices).to(new_grid.device)
    new_grid = new_grid[sorted_indices]

    return new_grid


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


def recompute_grid_ityx(grid_ityx: torch.Tensor, old_patchsize: int, new_patchsize: int) -> torch.Tensor:
    """
    Recompute grid coordinates to match new patch layout, keeping image_idx.

    Args:
        grid_ityx (torch.Tensor): shape [N, 4], each item is (image_idx, t, x, y)
        old_patchsize (int): original patch size (e.g., 14)
        new_patchsize (int): new desired patch size (e.g., 7, 14, 28)

    Returns:
        torch.Tensor: new_grid_ityx sorted by image_idx, t, x, y
    """
    assert new_patchsize in [old_patchsize // 2, old_patchsize, old_patchsize * 2], "Unsupported new_patchsize"

    image_idx = grid_ityx[:, 0]
    t = grid_ityx[:, 1]
    x = grid_ityx[:, 2]
    y = grid_ityx[:, 3]

    if new_patchsize == old_patchsize:
        new_x = x * 2
        new_y = y * 2
        new_grid = torch.stack([image_idx, t, new_x, new_y], dim=1)

    elif new_patchsize == old_patchsize // 2:
        coords = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                new_x = x * 2 + dx
                new_y = y * 2 + dy
                coords.append(torch.stack([image_idx, t, new_x, new_y], dim=1))
        new_grid = torch.cat(coords, dim=0)

    else:  # new_patchsize == old_patchsize * 2
        new_x = x * 2
        new_y = y * 2

        new_coords = []
        for img in image_idx.unique():
            img_mask = image_idx == img
            img_t = t[img_mask]
            img_x = new_x[img_mask]
            img_y = new_y[img_mask]

            group_coords = {}
            for i in range(len(img_t)):
                norm_x = int(x[img_mask][i] - x[img_mask].min())
                norm_y = int(y[img_mask][i] - y[img_mask].min())
                key = (int(img_t[i]), norm_x // 2, norm_y // 2)
                if key not in group_coords:
                    group_coords[key] = []
                # ensure float during mean calculation
                group_coords[key].append(torch.stack([img_x[i].float(), img_y[i].float()]))

            for (t_key, x_key, y_key), points in group_coords.items():
                points_tensor = torch.stack(points)  # shape [n, 2]
                mean_xy = points_tensor.mean(dim=0).round().to(grid_ityx.dtype)
                new_coords.append([img.item(), t_key, mean_xy[0], mean_xy[1]])

        new_grid = torch.tensor(new_coords, device=grid_ityx.device)

    # Sort by image_idx, then t, then x, then y
    itxy = new_grid.cpu().numpy()
    sorted_indices = np.lexsort((itxy[:, 3], itxy[:, 2], itxy[:, 1], itxy[:, 0]))
    sorted_indices = torch.from_numpy(sorted_indices).to(new_grid.device)
    new_grid = new_grid[sorted_indices]

    return new_grid


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_thw = torch.tensor([
        [1, 24, 28],  # → 672 patches
        [1, 20, 28],  # → 560 patches
    ], device=device)

    window_grid_itxy = torch.tensor([
        [0, 0, 0, 0],  # → index 0
        [0, 0, 0, 1],  # → index 1
        [1, 0, 0, 0],  # → index 672
    ], device=device)

    indices = itxy_to_flatten_index(grid_thw, window_grid_itxy)
    print(indices)  # tensor([  0,   1, 672], device='cuda:0')

if __name__ == '__main__':
    # Simulate 2 images with identical txy layout
    grid_ityx = torch.tensor([
        [0, 0, 5, 7],
        [0, 0, 5, 8],
        [0, 0, 6, 7],
        [0, 0, 6, 8],
        [1, 0, 5, 7],
        [1, 0, 5, 8],
        [1, 0, 6, 7],
        [1, 0, 6, 8],
    ])

    print("Original grid_ityx:")
    print(grid_ityx)

    print("\nPatchsize from 14 to 7:")
    print(recompute_grid_ityx(grid_ityx, 14, 7))

    print("\nPatchsize unchanged (14):")
    print(recompute_grid_ityx(grid_ityx, 14, 14))

    print("\nPatchsize from 14 to 28:")
    print(recompute_grid_ityx(grid_ityx, 14, 28))
