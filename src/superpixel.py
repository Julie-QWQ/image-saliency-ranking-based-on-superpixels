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
