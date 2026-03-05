import glob
import os

import hashlib
import logging
import os

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
    ):
        self.images = _list_images(images_dir)
        self.masks = _match_masks(self.images, masks_dir)
        self.slic_cfg = slic_cfg
        self.label_cfg = label_cfg
        self.mask_cfg = mask_cfg
        self.input_size = input_size
        self.cache_dir = cache_dir

        self._image_cache = []
        self._mask_cache = []
        self._label_maps = []
        self._adjacency = []
        self.samples = []
        self._logger = logging.getLogger()

        self._build_index()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, sp_id, label = self.samples[idx]
        image = self._image_cache[img_idx]
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
        self._logger.info("building superpixels and samples...")
        progress = tqdm(zip(self.images, self.masks), total=total_images, desc="build dataset", leave=False)
        for img_path, mask_path in progress:
            image = read_image_rgb(img_path)
            cache_payload = self._load_cache(img_path, mask_path)
            if cache_payload is None:
                mask = read_mask_binary(mask_path)
                label_map = compute_slic(image, **self.slic_cfg)
                sp_ids, sp_labels = _label_superpixels(label_map, mask, self.label_cfg)
                self._save_cache(img_path, mask_path, label_map, sp_ids, sp_labels)
            else:
                label_map, sp_ids, sp_labels = cache_payload
                mask = read_mask_binary(mask_path)
            adjacency = build_adjacency(label_map, connectivity=8)
            self._image_cache.append(image)
            self._mask_cache.append(mask)
            self._label_maps.append(label_map)
            self._adjacency.append(adjacency)

            valid_before = len(self.samples)
            for sp_id, label in zip(sp_ids, sp_labels):
                self.samples.append((len(self._image_cache) - 1, int(sp_id), float(label)))
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
