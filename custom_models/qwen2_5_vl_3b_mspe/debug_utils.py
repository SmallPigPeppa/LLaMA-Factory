from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def compute_window_patch_xy_coords(
        window_indices: List[List[int]],
        patch_sizes: List[int],
        base_patchsize: int,
        min_patchsize: int,
        window_size: int,
        img_shape: Tuple[int, int]
) -> List[List[Tuple[float, float]]]:
    """
    For each window, compute center (x,y) of its sub-patches,
    normalized to the finest grid (min_patchsize).
    Supports windows truncated at image edges.
    """
    rows, cols = img_shape
    centers_all: List[List[Tuple[float, float]]] = []

    for idxs, p in zip(window_indices, patch_sizes):
        # determine top-left of this window in grid coords
        flat0 = idxs[0]
        r0, c0 = divmod(flat0, cols)

        # actual physical size of this window (px)
        win_w = min(window_size, (cols - c0) * base_patchsize)
        win_h = min(window_size, (rows - r0) * base_patchsize)

        # how many sub-patches fit in x/y
        nx = int(win_w // p)
        ny = int(win_h // p)

        centers: List[Tuple[float, float]] = []
        for iy in range(ny):
            for ix in range(nx):
                x_phys = c0 * base_patchsize + (ix + 0.5) * p
                y_phys = r0 * base_patchsize + (iy + 0.5) * p
                x = x_phys / min_patchsize - 0.5
                y = y_phys / min_patchsize - 0.5
                centers.append((x, y))

        centers_all.append(centers)

    return centers_all


def generate_window_indices(
        img_shape: Tuple[int, int],
        window_size: int,
        base_patchsize: int
) -> List[List[int]]:
    """
    Partition the img patch grid into square windows of size `window_size`,
    returning for each window the list of valid flat indices (row-major).
    Supports edge windows that may be truncated.
    """
    rows, cols = img_shape
    step = window_size // base_patchsize
    windows = []
    for r0 in range(0, rows, step):
        for c0 in range(0, cols, step):
            idxs: List[int] = []
            for dr in range(step):
                for dc in range(step):
                    rr, cc = r0 + dr, c0 + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        idxs.append(rr * cols + cc)
            if idxs:
                windows.append(idxs)
    return windows





def visualize_windows(
        img_shape: Tuple[int, int],
        window_size: int,
        patch_sizes: List[int],
        base_patchsize: int,
        min_patchsize: int
) -> None:
    """
    Draw the patch grid, each window's subdivisions and sub-patch centers.
    """
    windows = generate_window_indices(img_shape, window_size, base_patchsize)
    centers = compute_window_patch_xy_coords(
        windows, patch_sizes,
        base_patchsize, min_patchsize,
        window_size, img_shape
    )

    rows, cols = img_shape
    scale = base_patchsize / min_patchsize

    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_aspect('equal')

    # draw base grid
    for x in range(cols + 1):
        ax.plot([x * scale - 0.5] * 2, [-0.5, rows * scale - 0.5], 'k-')
    for y in range(rows + 1):
        ax.plot([-0.5, cols * scale - 0.5], [y * scale - 0.5] * 2, 'k-')

    cmap = plt.cm.get_cmap('tab10', len(windows))
    for i, (idxs, p, ctrs) in enumerate(zip(windows, patch_sizes, centers)):
        # top-left
        r0, c0 = divmod(idxs[0], cols)
        # sub-patch grid size
        g = p / min_patchsize

        # draw sub-patch boxes
        win_w = min(window_size, (cols - c0) * base_patchsize)
        win_h = min(window_size, (rows - r0) * base_patchsize)
        nx = int(win_w // p)
        ny = int(win_h // p)
        for iy in range(ny):
            for ix in range(nx):
                x0 = c0 * scale - 0.5 + ix * g
                y0 = r0 * scale - 0.5 + iy * g
                ax.add_patch(Rectangle(
                    (x0, y0), g, g,
                    fill=False, edgecolor=cmap(i), lw=2
                ))

        # draw centers
        for x, y in ctrs:
            ax.plot(x, y, 'o', color=cmap(i))
            ax.text(x, y, f"({x:.1f},{y:.1f})",
                    fontsize=8, ha='left', va='bottom',
                    color=cmap(i))

    ax.invert_yaxis()
    ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    img_shape = (5, 18)
    window_size = 56
    base_patchsize = 14
    min_patchsize = 7
    # step = 56/14 = 4 → columns 0,4,8,12,16 → 5 windows
    patch_sizes = [56, 14, 7, 14, 14, 14, 14, 7, 14, 14]

    visualize_windows(
        img_shape, window_size,
        patch_sizes,
        base_patchsize, min_patchsize
    )
