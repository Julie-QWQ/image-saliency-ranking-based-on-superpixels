import json
import logging
import os
import random
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_run_dir(base_dir, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{name}_{timestamp}")
    ensure_dir(run_dir)
    return run_dir


def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def read_image_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_mask_binary(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = (mask > 127).astype(np.uint8)
    return mask


def resize_image(image, size):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def to_tensor(image):
    return torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def compute_metrics(pred, gt, threshold=0.5):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    mae = np.mean(np.abs(pred - gt))
    pred_bin = (pred >= threshold).astype(np.uint8)
    gt_bin = (gt >= 0.5).astype(np.uint8)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = intersection / (union + 1e-8)
    precision = intersection / (pred_bin.sum() + 1e-8)
    recall = intersection / (gt_bin.sum() + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"mae": float(mae), "iou": float(iou), "f1": float(f1)}
