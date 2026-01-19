import os

import cv2
import numpy as np
import torch

from .superpixel import build_adjacency, compute_slic, n_ring_neighbors
from .utils import ensure_dir, resize_image, to_tensor


def predict_image(model, image, slic_cfg, mask_cfg, input_size, device):
    label_map = compute_slic(image, **slic_cfg)
    adjacency = build_adjacency(label_map, connectivity=8)
    heatmap = np.zeros(label_map.shape, dtype=np.float32)

    with torch.no_grad():
        for sp_id in np.unique(label_map):
            target_mask = (label_map == sp_id).astype(np.uint8)
            context_mask = _context_mask(label_map, adjacency, sp_id, mask_cfg["n_ring"])

            xa = _apply_mask(image, target_mask)
            xb = _apply_mask(image, context_mask)
            xc = image

            xa = resize_image(xa, input_size)
            xb = resize_image(xb, input_size)
            xc = resize_image(xc, input_size)

            xa_t = to_tensor(xa).unsqueeze(0).to(device)
            xb_t = to_tensor(xb).unsqueeze(0).to(device)
            xc_t = to_tensor(xc).unsqueeze(0).to(device)

            logit = model(xa_t, xb_t, xc_t)
            prob = torch.sigmoid(logit).cpu().numpy().item()
            heatmap[label_map == sp_id] = prob

    return heatmap


def predict_multiscale(model, image, slic_cfg, mask_cfg, input_size, device, k_list):
    heatmaps = []
    for k in k_list:
        cfg = dict(slic_cfg)
        cfg["num_segments"] = int(k)
        heatmaps.append(predict_image(model, image, cfg, mask_cfg, input_size, device))
    return np.mean(heatmaps, axis=0)


def save_visuals(image, heatmap, out_dir, name):
    ensure_dir(out_dir)
    heat = (heatmap * 255).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * image + 0.4 * heat_color).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, f"{name}_heat.png"), cv2.cvtColor(heat_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f"{name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def _apply_mask(image, mask):
    mask_3 = np.repeat(mask[:, :, None], 3, axis=2)
    return image * mask_3


def _context_mask(label_map, adjacency, sp_id, n_ring):
    neighbors = n_ring_neighbors(adjacency, sp_id, n_ring)
    context = (label_map == sp_id).astype(np.uint8)
    for n in neighbors:
        context[label_map == n] = 1
    return context
