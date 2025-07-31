from typing import List, Tuple, Union
from itertools import accumulate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def generate_window_indices(
        img_shape: Tuple[int, int],
        window_size: int,
        base_patchsize: int
) -> List[List[int]]:
    """
    单张图：将 patch grid 划分成 window，并返回每个 window 中所有 patch 的扁平索引（row-major）。
    支持边界截断。
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


def generate_batch_window_indices(
        grid_thw: List[Union[Tuple[int, int, int], Tuple[int, int]]],
        window_size: int,
        base_patchsize: int
) -> List[List[int]]:
    """
    Batch 版本：输入每张图的 (t,h,w) 或 (h,w)，
    按顺序将所有图的 patchlist 拼成一个大列表，并为每个 window
    生成“全局”扁平索引。返回值长度 = 所有图的 window 数之和。
    """
    # 先计算每张图的 patch 数，构造 offset
    patch_counts = []
    for dims in grid_thw:
        if len(dims) == 3:
            _, h, w = dims
        else:
            h, w = dims
        patch_counts.append(h * w)
    offsets = [0] + list(accumulate(patch_counts))

    all_windows: List[List[int]] = []
    for img_idx, dims in enumerate(grid_thw):
        # 取出单张图的 (h,w)
        if len(dims) == 3:
            _, rows, cols = dims
        else:
            rows, cols = dims

        # 生成这张图的 window（局部索引）
        local_windows = generate_window_indices((rows, cols), window_size, base_patchsize)
        base_off = offsets[img_idx]

        # 转成全局索引并累加
        for win in local_windows:
            all_windows.append([idx + base_off for idx in win])

    return all_windows


def compute_window_patch_xy_coords(
        window_index_list: List[List[int]],
        window_adp_patchsize: List[int],
        base_patchsize: int,
        min_patchsize: int,
        window_size: int,
        grid: Union[
            Tuple[int, int],
            List[Union[Tuple[int, int, int], Tuple[int, int]]]
        ]
) -> List[List[Tuple[float, float]]]:
    """
    支持单图或 Batch：
    - 单图：grid = (rows,cols)
    - Batch：grid = [(t1,h1,w1), (t2,h2,w2), ...] 或 [(h1,w1),(h2,w2),...]

    输出：每个 window 下所有 sub-patch 的中心 (x,y)，归一化到最细粒度 min_patchsize。
    """
    patch_xy_coords: List[List[Tuple[float, float]]] = []

    # 准备 batch 模式下的 offset、rows_list, cols_list
    if isinstance(grid, list):
        # 统一抽取 (rows,cols) 并计算 offsets
        rows_list, cols_list = [], []
        for dims in grid:
            if len(dims) == 3:
                _, r, c = dims
            else:
                r, c = dims
            rows_list.append(r)
            cols_list.append(c)
        patch_counts = [r * c for r, c in zip(rows_list, cols_list)]
        offsets = [0] + list(accumulate(patch_counts))
    else:
        # 单图模式
        rows_list, cols_list, offsets = [grid[0]], [grid[1]], [0, grid[0] * grid[1]]

    # 对每个 window 逐个处理
    for idxs, p in zip(window_index_list, window_adp_patchsize):
        flat0 = idxs[0]
        # 找到所属图像的序号 img_i，使 offsets[img_i] <= flat0 < offsets[img_i+1]
        # 单图模式下 img_i 恒为 0
        img_i = 0
        if len(offsets) > 2:
            # batch
            # 找到最大的 k 使 offsets[k] <= flat0
            # 最简单：线性查找
            for k in range(len(offsets) - 1):
                if offsets[k] <= flat0 < offsets[k + 1]:
                    img_i = k
                    break
        row_cnt = rows_list[img_i]
        col_cnt = cols_list[img_i]
        rel0 = flat0 - offsets[img_i]
        r0, c0 = divmod(rel0, col_cnt)

        # 物理尺寸
        win_w = min(window_size, (col_cnt - c0) * base_patchsize)
        win_h = min(window_size, (row_cnt - r0) * base_patchsize)
        nx = int(win_w // p)
        ny = int(win_h // p)

        coords: List[Tuple[float, float]] = []
        for iy in range(ny):
            for ix in range(nx):
                x_phys = c0 * base_patchsize + (ix + 0.5) * p
                y_phys = r0 * base_patchsize + (iy + 0.5) * p
                x = x_phys / min_patchsize - 0.5
                y = y_phys / min_patchsize - 0.5
                coords.append((x, y))
        patch_xy_coords.append(coords)

    return patch_xy_coords


def visualize_batch_windows(
        grid_thw: List[Union[Tuple[int, int, int], Tuple[int, int]]],
        window_size: int,
        patch_sizes: List[int],
        base_patchsize: int,
        min_patchsize: int
) -> None:
    """
    对一个 batch（多图）可视化：
    - 将所有 window 叠加到一个大扁平 patchlist 上计算中心点
    - 然后分图绘制，保留和单图时一样的风格
    """
    # 1) 生成全局 window 索引
    windows = generate_batch_window_indices(grid_thw, window_size, base_patchsize)
    # 2) 计算所有 window 的中心点
    centers = compute_window_patch_xy_coords(
        windows, patch_sizes,
        base_patchsize, min_patchsize,
        window_size, grid_thw
    )

    # 分离出每张图的窗口数量
    if isinstance(grid_thw, list):
        # 每张图的窗口数量 = len(generate_window_indices)
        per_img_counts = [
            len(generate_window_indices(
                (dims[1], dims[2]) if len(dims) == 3 else dims,
                window_size, base_patchsize
            )) for dims in grid_thw
        ]
    else:
        per_img_counts = [len(windows)]

    # 绘制
    n = len(per_img_counts)
    fig, axs = plt.subplots(1, n, figsize=(18 * n, 5))
    if n == 1:
        axs = [axs]

    start = 0
    scale = base_patchsize / min_patchsize
    for i, count in enumerate(per_img_counts):
        ax = axs[i]
        # 当前图的 dims
        dims = grid_thw[i]
        rows, cols = (dims[1], dims[2]) if len(dims) == 3 else dims

        # 基础网格
        for x in range(cols + 1):
            ax.plot([x * scale - 0.5] * 2, [-0.5, rows * scale - 0.5], 'k-')
        for y in range(rows + 1):
            ax.plot([-0.5, cols * scale - 0.5], [y * scale - 0.5] * 2, 'k-')

        # 只画这一张图的 window
        end = start + count
        cmap = plt.cm.get_cmap('tab10', count)
        for j, ((idxs, p, ctrs), color) in enumerate(
                zip(
                    zip(windows[start:end], patch_sizes[start:end], centers[start:end]),
                    [cmap(k) for k in range(count)]
                )):
            flat0 = idxs[0] - (offsets := [0] + list(accumulate([r * c for r, c in [
                ((d[1], d[2]) if len(d) == 3 else d) for d in grid_thw
            ]])))[i]
            # 恢复相对坐标
            rel0 = flat0
            r0, c0 = divmod(rel0, cols)

            win_w = min(window_size, (cols - c0) * base_patchsize)
            win_h = min(window_size, (rows - r0) * base_patchsize)
            nx = int(win_w // p)
            ny = int(win_h // p)
            g = p / min_patchsize

            # 子 patch 方框
            for iy in range(ny):
                for ix in range(nx):
                    x0 = c0 * scale - 0.5 + ix * g
                    y0 = r0 * scale - 0.5 + iy * g
                    ax.add_patch(Rectangle(
                        (x0, y0), g, g,
                        fill=False, edgecolor=color, lw=1.5
                    ))

            # 标记中心
            for x, y in ctrs:
                ax.plot(x, y, 'o', color=color, markersize=4)

        ax.set_title(f"Image {i}")
        ax.invert_yaxis()
        ax.axis('off')
        start = end

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # ==== 示例：batch size = 2 ====
    # 假设两张图的 patch grid 分别是 5×18 和 5×18 (t 忽略)
    batch_grid = [(1, 5, 18), (1, 5, 18)]
    window_size = 56
    base_patchsize = 14
    min_patchsize = 7

    # 每张图会有 2×5 = 10 个 window，总共 20
    # 这里演示：两张图沿用同样的 patch_sizes 模式
    single_img_patch_sizes = [56, 14, 7, 14, 14, 14, 14, 7, 14, 14]
    patch_sizes = single_img_patch_sizes * 2

    visualize_batch_windows(
        batch_grid,
        window_size,
        patch_sizes,
        base_patchsize,
        min_patchsize
    )
