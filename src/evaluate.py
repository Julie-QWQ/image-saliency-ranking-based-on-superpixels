import glob
import os

import numpy as np
import torch

from .infer import predict_multiscale, predict_image
from .utils import compute_metrics, read_image_rgb, read_mask_binary, save_json


def evaluate(model, images_dir, masks_dir, cfg, device):
    images = _list_images(images_dir)
    masks = _match_masks(images, masks_dir)
    metrics = []

    slic_cfg = cfg["slic"]
    mask_cfg = cfg["masking"]
    input_size = cfg["model"]["input_size"]
    k_list = cfg["inference"]["multi_scale"]

    for img_path, mask_path in zip(images, masks):
        image = read_image_rgb(img_path)
        gt = read_mask_binary(mask_path)
        if k_list:
            heatmap = predict_multiscale(model, image, slic_cfg, mask_cfg, input_size, device, k_list)
        else:
            heatmap = predict_image(model, image, slic_cfg, mask_cfg, input_size, device)
        metrics.append(compute_metrics(heatmap, gt))

    return _aggregate(metrics)


def _aggregate(metrics_list):
    if not metrics_list:
        return {"mae": 0.0, "iou": 0.0, "f1": 0.0}
    mae = np.mean([m["mae"] for m in metrics_list])
    iou = np.mean([m["iou"] for m in metrics_list])
    f1 = np.mean([m["f1"] for m in metrics_list])
    return {"mae": float(mae), "iou": float(iou), "f1": float(f1)}


def _list_images(images_dir):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    images = []
    for pattern in patterns:
        images.extend(glob.glob(os.path.join(images_dir, pattern)))
    images.sort()
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return images


def _match_masks(images, masks_dir):
    masks = []
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        candidates = [
            os.path.join(masks_dir, f"{name}.png"),
            os.path.join(masks_dir, f"{name}.jpg"),
            os.path.join(masks_dir, f"{name}.bmp"),
        ]
        match = None
        for c in candidates:
            if os.path.exists(c):
                match = c
                break
        if match is None:
            raise FileNotFoundError(f"Mask not found for {img_path}")
        masks.append(match)
    return masks
