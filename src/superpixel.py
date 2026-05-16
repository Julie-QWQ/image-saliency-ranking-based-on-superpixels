import numpy as np
from skimage.segmentation import slic


def compute_slic(image, num_segments, compactness, sigma, max_num_iter, start_label):
    labels = slic(
        image,
        n_segments=num_segments,
        compactness=compactness,
        sigma=sigma,
        max_num_iter=max_num_iter,
        start_label=start_label,
    )
    return labels


def build_adjacency(label_map, connectivity=8):
    """优化版的邻接图构建，减少循环次数"""
    h, w = label_map.shape
    adjacency = {}

    # 使用偏移量数组来检查邻居
    if connectivity == 8:
        offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    else:
        offsets = [(-1,0), (1,0), (0,-1), (0,1)]

    # 只遍历超像素边界，而不是所有像素
    unique_labels = np.unique(label_map)

    for label in unique_labels:
        label = int(label)
        neighbors = set()

        # 找到当前超像素的所有像素位置
        rows, cols = np.where(label_map == label)

        # 只检查边界像素（有不同邻居的像素）
        for y, x in zip(rows, cols):
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    neighbor = int(label_map[ny, nx])
                    if neighbor != label:
                        neighbors.add(neighbor)

        adjacency[label] = neighbors

    return adjacency


def n_ring_neighbors(adjacency, node, n_ring):
    visited = {node}
    frontier = {node}
    for _ in range(n_ring):
        next_frontier = set()
        for u in frontier:
            for v in adjacency.get(u, []):
                if v not in visited:
                    visited.add(v)
                    next_frontier.add(v)
        frontier = next_frontier
    visited.remove(node)
    return visited


def _neighbors(connectivity):
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
