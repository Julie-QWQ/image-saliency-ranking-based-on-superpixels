import argparse
import os

import torch

from src.evaluate import evaluate
from src.model import MultiBranchNet
from src.utils import get_device, load_config, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["runtime"]["device"])

    model = MultiBranchNet(cfg["model"]["feature_dim"], cfg["model"]["mlp_hidden"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    images_dir = cfg["paths"][f"{args.split}_images"]
    masks_dir = cfg["paths"][f"{args.split}_masks"]

    metrics = evaluate(model, images_dir, masks_dir, cfg, device, max_images=args.max_images)

    output_path = args.output or os.path.join(cfg["paths"]["output_dir"], f"metrics_{args.split}.json")
    save_json(metrics, output_path)
    print(metrics)


if __name__ == "__main__":
    main()
