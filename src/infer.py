import os

import cv2
import numpy as np
import torch

from .superpixel import build_adjacency, compute_slic, n_ring_neighbors
from .utils import ensure_dir, resize_image, to_tensor


def predict_image(model, image, slic_cfg, mask_cfg, input_size, device, batch_size=64):
    label_map = compute_slic(image, **slic_cfg)
    adjacency = build_adjacency(label_map, connectivity=8)
    heatmap = np.zeros(label_map.shape, dtype=np.float32)

    with torch.no_grad():
        sp_ids = np.unique(label_map).tolist()
        xc = resize_image(image, input_size)
        xc_t = to_tensor(xc).unsqueeze(0).to(device)
        fc = model.branch_c(xc_t)

        for start in range(0, len(sp_ids), batch_size):
            batch_ids = sp_ids[start : start + batch_size]
            xa_list = []
            xb_list = []
            for sp_id in batch_ids:
                target_mask = (label_map == sp_id).astype(np.uint8)
                context_mask = _context_mask(label_map, adjacency, sp_id, mask_cfg["n_ring"])
                xa = _apply_mask(image, target_mask)
                xb = _apply_mask(image, context_mask)
                xa_list.append(resize_image(xa, input_size))
                xb_list.append(resize_image(xb, input_size))

            xa_t = torch.stack([to_tensor(xa) for xa in xa_list]).to(device)
            xb_t = torch.stack([to_tensor(xb) for xb in xb_list]).to(device)
            fa = model.branch_a(xa_t)
            fb = model.branch_b(xb_t)
            fc_rep = fc.repeat(len(batch_ids), 1)
            z = torch.relu(model.fc7(torch.cat([fa, fb, fc_rep], dim=1)))
            logits = model.head(z).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            for sp_id, prob in zip(batch_ids, probs):
                heatmap[label_map == sp_id] = float(prob)

    return heatmap


def predict_multiscale(model, image, slic_cfg, mask_cfg, input_size, device, k_list, batch_size=64):
    heatmaps = []
    for k in k_list:
        cfg = dict(slic_cfg)
        cfg["num_segments"] = int(k)
        heatmaps.append(predict_image(model, image, cfg, mask_cfg, input_size, device, batch_size=batch_size))
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
