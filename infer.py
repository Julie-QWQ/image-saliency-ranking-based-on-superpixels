import argparse
import glob
import os

import torch
from tqdm import tqdm

from src.infer import predict_multiscale, predict_image, save_visuals
from src.model import MultiBranchNet
from src.utils import ensure_dir, get_device, load_config, read_image_rgb


def list_images(images_dir):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    images = []
    for pattern in patterns:
        images.extend(glob.glob(os.path.join(images_dir, pattern)))
    images.sort()
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["runtime"]["device"])

    model = MultiBranchNet(cfg["model"]["feature_dim"], cfg["model"]["mlp_hidden"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    images_dir = cfg["paths"][f"{args.split}_images"]
    images = list_images(images_dir)

    out_dir = args.output or os.path.join(cfg["paths"]["output_dir"], f"infer_{args.split}")
    ensure_dir(out_dir)

    slic_cfg = cfg["slic"]
    mask_cfg = cfg["masking"]
    input_size = cfg["model"]["input_size"]
    k_list = cfg["inference"]["multi_scale"]

    progress = tqdm(images, desc=f"infer {args.split}", leave=False)
    for img_path in progress:
        name = os.path.splitext(os.path.basename(img_path))[0]
        image = read_image_rgb(img_path)
        if k_list:
            heatmap = predict_multiscale(model, image, slic_cfg, mask_cfg, input_size, device, k_list)
        else:
            heatmap = predict_image(model, image, slic_cfg, mask_cfg, input_size, device)
        if cfg["inference"]["save_visuals"]:
            save_visuals(image, heatmap, out_dir, name)


if __name__ == "__main__":
    main()
