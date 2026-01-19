import os
import random
import shutil


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_msra_pairs(root):
    pairs = []
    for name in os.listdir(root):
        if not name.lower().endswith(".jpg"):
            continue
        base = os.path.splitext(name)[0]
        img_path = os.path.join(root, name)
        mask_path = os.path.join(root, f"{base}.png")
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    return pairs


def copy_pairs(pairs, images_dir, masks_dir):
    ensure_dir(images_dir)
    ensure_dir(masks_dir)
    for img_path, mask_path in pairs:
        shutil.copy2(img_path, os.path.join(images_dir, os.path.basename(img_path)))
        shutil.copy2(mask_path, os.path.join(masks_dir, os.path.basename(mask_path)))


def copy_duts(images_src, masks_src, images_dir, masks_dir):
    ensure_dir(images_dir)
    ensure_dir(masks_dir)
    for name in os.listdir(images_src):
        if not name.lower().endswith(".jpg"):
            continue
        base = os.path.splitext(name)[0]
        img_path = os.path.join(images_src, name)
        mask_path = os.path.join(masks_src, f"{base}.png")
        if os.path.exists(mask_path):
            shutil.copy2(img_path, os.path.join(images_dir, name))
            shutil.copy2(mask_path, os.path.join(masks_dir, f"{base}.png"))


def main():
    random.seed(42)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    msra_root = os.path.join(project_root, "data", "raw", "MSRA-B", "MSRA-B")
    duts_root = os.path.join(project_root, "data", "raw", "DUTS-TE", "DUTS-TE")
    duts_images = os.path.join(duts_root, "DUTS-TE-Image")
    duts_masks = os.path.join(duts_root, "DUTS-TE-Mask")

    pairs = list_msra_pairs(msra_root)
    if not pairs:
        raise RuntimeError("MSRA-B not found or empty")

    random.shuffle(pairs)
    small_total = 500
    train_count = 400
    val_count = 100
    small_pairs = pairs[:small_total]
    train_pairs = small_pairs[:train_count]
    val_pairs = small_pairs[train_count:train_count + val_count]

    train_images = os.path.join(project_root, "data", "train", "images")
    train_masks = os.path.join(project_root, "data", "train", "masks")
    val_images = os.path.join(project_root, "data", "val", "images")
    val_masks = os.path.join(project_root, "data", "val", "masks")
    test_images = os.path.join(project_root, "data", "test", "images")
    test_masks = os.path.join(project_root, "data", "test", "masks")

    copy_pairs(train_pairs, train_images, train_masks)
    copy_pairs(val_pairs, val_images, val_masks)

    if not (os.path.isdir(duts_images) and os.path.isdir(duts_masks)):
        raise RuntimeError("DUTS-TE not found or invalid")
    copy_duts(duts_images, duts_masks, test_images, test_masks)

    print("Prepared: train/val from MSRA-B (small subset), test from DUTS-TE (medium)")


if __name__ == "__main__":
    main()
