import os

import torch
from torch.utils.data import DataLoader

from .data import SuperpixelSaliencyDataset
from .evaluate import evaluate
from .model import MultiBranchNet
from .utils import (
    ensure_dir,
    get_device,
    load_config,
    save_json,
    set_seed,
    setup_logger,
    setup_run_dir,
)


def train(config_path):
    cfg = load_config(config_path)
    set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    device = get_device(cfg["runtime"]["device"])

    run_dir = setup_run_dir(cfg["paths"]["output_dir"], "train")
    logger = setup_logger(os.path.join(run_dir, "train.log"))
    save_json(cfg, os.path.join(run_dir, "config.json"))

    train_dataset = SuperpixelSaliencyDataset(
        cfg["paths"]["train_images"],
        cfg["paths"]["train_masks"],
        cfg["slic"],
        cfg["labels"],
        cfg["masking"],
        cfg["model"]["input_size"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )

    model = MultiBranchNet(cfg["model"]["feature_dim"], cfg["model"]["mlp_hidden"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for step, (xa, xb, xc, y) in enumerate(train_loader, start=1):
            xa = xa.to(device)
            xb = xb.to(device)
            xc = xc.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(xa, xb, xc)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % cfg["train"]["log_interval"] == 0:
                avg_loss = running_loss / cfg["train"]["log_interval"]
                logger.info("epoch %d step %d loss %.4f", epoch, step, avg_loss)
                running_loss = 0.0

        if epoch % cfg["train"]["save_interval"] == 0:
            ckpt_path = os.path.join(run_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

        val_metrics = _maybe_eval(model, cfg, device)
        if val_metrics:
            logger.info("epoch %d val_mae %.4f val_iou %.4f val_f1 %.4f", epoch, val_metrics["mae"], val_metrics["iou"], val_metrics["f1"])
            save_json(val_metrics, os.path.join(run_dir, f"val_metrics_epoch_{epoch}.json"))

    final_path = os.path.join(run_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info("training done, final checkpoint %s", final_path)


def _maybe_eval(model, cfg, device):
    val_images = cfg["paths"]["val_images"]
    val_masks = cfg["paths"]["val_masks"]
    if not (os.path.exists(val_images) and os.path.exists(val_masks)):
        return None
    model.eval()
    return evaluate(model, val_images, val_masks, cfg, device)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
