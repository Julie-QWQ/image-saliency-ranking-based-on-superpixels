import glob
import os

import numpy as np
import torch
from tqdm import tqdm

from .infer import predict_image
from .utils import compute_metrics, read_image_rgb, read_mask_binary, save_json


def _prepare_image_data(args):
    """并行准备图像数据（CPU密集型）- 顶层函数以便序列化"""
    img_path, mask_path = args
    return {
        'img_path': img_path,
        'mask_path': mask_path,
        'image': read_image_rgb(img_path),
        'gt': read_mask_binary(mask_path)
    }


def evaluate(model, images_dir, masks_dir, cfg, device, max_images=None):
    images = _list_images(images_dir)
    masks = _match_masks(images, masks_dir)
    if max_images is not None:
        images = images[:max_images]
        masks = masks[:max_images]

    slic_cfg = cfg["slic"]
    mask_cfg = cfg["masking"]
    input_size = cfg["model"]["input_size"]
    k_list = cfg["inference"]["multi_scale"]
    batch_size = cfg["inference"].get("batch_size", 64)
    num_workers = cfg["train"].get("val_workers", 8)  # 数据准备并行数
    cache_dir = cfg["paths"].get("cache_dir")

    metrics = []
    batch_size_preprocess = num_workers  # 每次并行准备的图像数

    with tqdm(total=len(images), desc="validate", leave=False) as pbar:
        for start_idx in range(0, len(images), batch_size_preprocess):
            end_idx = min(start_idx + batch_size_preprocess, len(images))
            batch_images = images[start_idx:end_idx]
            batch_masks = masks[start_idx:end_idx]

            # 并行准备数据（读取图像、超像素计算等）
            from concurrent.futures import ProcessPoolExecutor
            args_list = list(zip(batch_images, batch_masks))

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                prepared_data = list(executor.map(_prepare_image_data, args_list))

            # GPU推理部分（串行，但数据已准备好）
            for data in prepared_data:
                # 验证时使用单尺度推理（更快），避免多尺度并行问题
                heatmap = predict_image(
                    model,
                    data['image'],
                    slic_cfg,
                    mask_cfg,
                    input_size,
                    device,
                    batch_size=batch_size,
                    cache_dir=cache_dir,
                    image_path=data['img_path'],
                )

                metrics.append(compute_metrics(heatmap, data['gt']))
                pbar.update(1)
                pbar.set_postfix(count=len(metrics))

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
