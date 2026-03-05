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
    h, w = label_map.shape
    adjacency = {}
    for y in range(h):
        for x in range(w):
            current = int(label_map[y, x])
            if current not in adjacency:
                adjacency[current] = set()
            for dy, dx in _neighbors(connectivity):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    neighbor = int(label_map[ny, nx])
                    if neighbor != current:
                        adjacency[current].add(neighbor)
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


def n_ring_neighbors_weighted(adjacency, node, n_ring):
    """
    带距离权重的 N-ring 邻域

    返回邻居节点和对应的权重（距离越近权重越高）

    Args:
        adjacency: 邻接表
        node: 目标节点
        n_ring: 邻域半径

    Returns:
        List[(neighbor_id, weight)]: 邻居节点和权重列表
    """
    visited = {node}
    frontier = {node}
    distances = {node: 0}

    for ring in range(n_ring):
        next_frontier = set()
        for u in frontier:
            for v in adjacency.get(u, []):
                if v not in visited:
                    visited.add(v)
                    next_frontier.add(v)
                    distances[v] = ring + 1
        frontier = next_frontier

    # 计算权重（距离越近权重越高）
    weighted_neighbors = []
    for neighbor in visited - {node}:
        distance = distances[neighbor]
        weight = 1.0 / distance  # 距离为1权重为1，距离为2权重为0.5
        weighted_neighbors.append((neighbor, weight))

    return weighted_neighbors


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
