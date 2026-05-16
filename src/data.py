import glob
import os

import hashlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .superpixel import build_adjacency, compute_slic, n_ring_neighbors
from .utils import read_image_rgb, read_mask_binary, resize_image, to_tensor


class SuperpixelSaliencyDataset(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        slic_cfg,
        label_cfg,
        mask_cfg,
        input_size,
        cache_dir=None,
        num_workers=4,
    ):
        self.images = _list_images(images_dir)
        self.masks = _match_masks(self.images, masks_dir)
        self.slic_cfg = slic_cfg
        self.label_cfg = label_cfg
        self.mask_cfg = mask_cfg
        self.input_size = input_size
        self.cache_dir = cache_dir
        self._num_workers = num_workers

        self._label_maps = []
        self._adjacency = []
        self.samples = []
        self._logger = logging.getLogger()

        self._build_index()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, sp_id, label = self.samples[idx]
        # 动态加载图像，而不是缓存（节省内存）
        image = read_image_rgb(self.images[img_idx])
        label_map = self._label_maps[img_idx]
        adjacency = self._adjacency[img_idx]

        target_mask = (label_map == sp_id).astype(np.uint8)
        context_mask = self._context_mask(target_mask, label_map, adjacency, sp_id)

        branch_a = _apply_mask(image, target_mask)
        branch_b = _apply_mask(image, context_mask)
        branch_c = image

        branch_a = resize_image(branch_a, self.input_size)
        branch_b = resize_image(branch_b, self.input_size)
        branch_c = resize_image(branch_c, self.input_size)

        return (
            to_tensor(branch_a),
            to_tensor(branch_b),
            to_tensor(branch_c),
            torch.tensor(label, dtype=torch.float32),
        )

    def _build_index(self):
        total_images = len(self.images)
        num_workers = getattr(self, '_num_workers', 4)  # 默认4个进程

        if num_workers > 0 and total_images > 10:
            self._logger.info("building superpixels with %d workers...", num_workers)
            # 准备参数
            args_list = [
                (img_path, mask_path, self.slic_cfg, self.label_cfg, self.cache_dir)
                for img_path, mask_path in zip(self.images, self.masks)
            ]

            # 多进程处理
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(
                    executor.map(_process_single_image, args_list),
                    total=total_images,
                    desc="build dataset",
                    leave=False
                ))

            # 创建路径到索引的映射
            path_to_idx = {img_path: idx for idx, img_path in enumerate(self.images)}

            # 处理结果并按原始顺序存储
            temp_results = {img_path: (label_map, sp_ids, sp_labels)
                           for img_path, (label_map, sp_ids, sp_labels) in results}

            for img_path in self.images:
                if img_path not in temp_results:
                    self._logger.error("Missing result for %s", img_path)
                    continue
                label_map, sp_ids, sp_labels = temp_results[img_path]
                adjacency = build_adjacency(label_map, connectivity=8)

                self._label_maps.append(label_map)
                self._adjacency.append(adjacency)

                valid_before = len(self.samples)
                for sp_id, label in zip(sp_ids, sp_labels):
                    self.samples.append((len(self._label_maps) - 1, int(sp_id), float(label)))
        else:
            # 单进程处理（原有逻辑）
            self._logger.info("building superpixels (single process)...")
            progress = tqdm(zip(self.images, self.masks), total=total_images, desc="build dataset", leave=False)
            for img_path, mask_path in progress:
                cache_payload = self._load_cache(img_path, mask_path)
                if cache_payload is None:
                    image = read_image_rgb(img_path)
                    mask = read_mask_binary(mask_path)
                    label_map = compute_slic(image, **self.slic_cfg)
                    sp_ids, sp_labels = _label_superpixels(label_map, mask, self.label_cfg)
                    self._save_cache(img_path, mask_path, label_map, sp_ids, sp_labels)
                else:
                    label_map, sp_ids, sp_labels = cache_payload
                adjacency = build_adjacency(label_map, connectivity=8)

                self._label_maps.append(label_map)
                self._adjacency.append(adjacency)

                valid_before = len(self.samples)
                for sp_id, label in zip(sp_ids, sp_labels):
                    self.samples.append((len(self._label_maps) - 1, int(sp_id), float(label)))
                added = len(self.samples) - valid_before
                progress.set_postfix(added=added)

        self._logger.info("dataset build done, images=%d samples=%d", total_images, len(self.samples))

    def _cache_key(self, img_path, mask_path):
        parts = [
            os.path.abspath(img_path),
            str(os.path.getmtime(img_path)),
            os.path.abspath(mask_path),
            str(os.path.getmtime(mask_path)),
            str(self.slic_cfg),
            str(self.label_cfg),
        ]
        digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
        return digest

    def _cache_path(self, img_path, mask_path):
        if not self.cache_dir:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{self._cache_key(img_path, mask_path)}.npz")

    def _load_cache(self, img_path, mask_path):
        cache_path = self._cache_path(img_path, mask_path)
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            data = np.load(cache_path)
            label_map = data["label_map"]
            sp_ids = data["sp_ids"]
            sp_labels = data["sp_labels"]
            return label_map, sp_ids, sp_labels
        except Exception as exc:
            self._logger.info("cache load failed: %s (%s)", cache_path, exc)
            return None

    def _save_cache(self, img_path, mask_path, label_map, sp_ids, sp_labels):
        cache_path = self._cache_path(img_path, mask_path)
        if not cache_path:
            return
        np.savez_compressed(cache_path, label_map=label_map, sp_ids=sp_ids, sp_labels=sp_labels)

    def _context_mask(self, target_mask, label_map, adjacency, sp_id):
        context = target_mask.copy()
        n_ring = self.mask_cfg["n_ring"]
        neighbors = n_ring_neighbors(adjacency, sp_id, n_ring)
        return _expand_mask(context, label_map, neighbors)


def _apply_mask(image, mask):
    mask_3 = np.repeat(mask[:, :, None], 3, axis=2)
    return image * mask_3


def _expand_mask(base_mask, label_map, neighbors):
    expanded = base_mask.copy()
    for n in neighbors:
        expanded[label_map == n] = 1
    return expanded


def _process_single_image(args):
    """处理单张图像的超像素计算（用于多进程）"""
    img_path, mask_path, slic_cfg, label_cfg, cache_dir = args

    def _cache_key(img_path, mask_path, slic_cfg, label_cfg):
        parts = [
            os.path.abspath(img_path),
            str(os.path.getmtime(img_path)),
            os.path.abspath(mask_path),
            str(os.path.getmtime(mask_path)),
            str(slic_cfg),
            str(label_cfg),
        ]
        digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
        return digest

    def _cache_path(img_path, mask_path, slic_cfg, label_cfg, cache_dir):
        if not cache_dir or not img_path:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        key = _cache_key(img_path, mask_path, slic_cfg, label_cfg)
        return os.path.join(cache_dir, f"{key}.npz")

    def _load_cache(img_path, mask_path, slic_cfg, label_cfg, cache_dir):
        cache_path = _cache_path(img_path, mask_path, slic_cfg, label_cfg, cache_dir)
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            data = np.load(cache_path)
            return data["label_map"], data["sp_ids"], data["sp_labels"]
        except Exception:
            return None

    def _save_cache(img_path, mask_path, slic_cfg, label_cfg, cache_dir, label_map, sp_ids, sp_labels):
        cache_path = _cache_path(img_path, mask_path, slic_cfg, label_cfg, cache_dir)
        if not cache_path:
            return
        np.savez_compressed(cache_path, label_map=label_map, sp_ids=sp_ids, sp_labels=sp_labels)

    # 检查缓存
    cache_payload = _load_cache(img_path, mask_path, slic_cfg, label_cfg, cache_dir)
    if cache_payload is not None:
        return img_path, cache_payload

    # 计算超像素
    image = read_image_rgb(img_path)
    mask = read_mask_binary(mask_path)
    label_map = compute_slic(image, **slic_cfg)

    # 计算标签
    sp_ids = []
    sp_labels = []
    for sp_id in np.unique(label_map):
        sp_mask = label_map == sp_id
        ratio = mask[sp_mask].mean() if sp_mask.any() else 0.0
        if ratio >= label_cfg["tau_pos"]:
            label = 1.0
        elif ratio <= label_cfg["tau_neg"]:
            label = 0.0
        else:
            continue
        sp_ids.append(int(sp_id))
        sp_labels.append(float(label))

    _save_cache(img_path, mask_path, slic_cfg, label_cfg, cache_dir, label_map, sp_ids, sp_labels)
    return img_path, (label_map, np.array(sp_ids), np.array(sp_labels))


def _label_from_ratio(ratio, label_cfg):
    if ratio >= label_cfg["tau_pos"]:
        return 1.0
    if ratio <= label_cfg["tau_neg"]:
        return 0.0
    return None


def _label_superpixels(label_map, mask, label_cfg):
    sp_ids = []
    sp_labels = []
    for sp_id in np.unique(label_map):
        sp_mask = label_map == sp_id
        ratio = mask[sp_mask].mean() if sp_mask.any() else 0.0
        label = _label_from_ratio(ratio, label_cfg)
        if label is None:
            continue
        sp_ids.append(int(sp_id))
        sp_labels.append(float(label))
    return np.array(sp_ids, dtype=np.int32), np.array(sp_labels, dtype=np.float32)


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
