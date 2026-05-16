import os

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def train(config_path, no_val=False, resume_path=None):
    cfg = load_config(config_path)
    set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    device = get_device(cfg["runtime"]["device"])

    run_dir = setup_run_dir(cfg["paths"]["output_dir"], "train")
    logger = setup_logger(os.path.join(run_dir, "train.log"))
    save_json(cfg, os.path.join(run_dir, "config.json"))
    logger.info("config loaded, device=%s, run_dir=%s", device, run_dir)

    # Initialize wandb
    wandb_enabled = cfg.get("wandb", {}).get("enable", True)
    if wandb_enabled:
        wandb.init(
            project=cfg["wandb"]["project"],
            name=os.path.basename(run_dir),
            config=cfg,
            dir=run_dir,
            mode=cfg["wandb"].get("mode", "online"),
        )
        logger.info("wandb initialized: project=%s, mode=%s", cfg["wandb"]["project"], cfg["wandb"].get("mode", "online"))
    else:
        logger.info("wandb disabled")

    logger.info("building train dataset...")
    train_dataset = SuperpixelSaliencyDataset(
        cfg["paths"]["train_images"],
        cfg["paths"]["train_masks"],
        cfg["slic"],
        cfg["labels"],
        cfg["masking"],
        cfg["model"]["input_size"],
        cache_dir=cfg["paths"].get("cache_dir"),
        num_workers=cfg["train"].get("dataloader_num_workers", 4),
    )
    logger.info("train dataset ready, samples=%d", len(train_dataset))
    logger.info("building dataloader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    logger.info("dataloader ready, batch_size=%d", cfg["train"]["batch_size"])

    logger.info("building model...")
    model = MultiBranchNet(cfg["model"]["feature_dim"], cfg["model"]["mlp_hidden"]).to(device)
    logger.info("model ready")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    logger.info("optimizer and loss ready")

    start_epoch = 1
    loss_history = []
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(resume_path)
        logger.info("resuming from checkpoint: %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        loss_history = ckpt.get("loss_history", [])
    val_history = {"mae": [], "iou": [], "f1": []}
    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        step_count = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False, dynamic_ncols=True)
        for step, (xa, xb, xc, y) in enumerate(progress, start=1):
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
            epoch_loss += loss.item()
            step_count += 1
            if step % cfg["train"]["log_interval"] == 0:
                avg_loss = running_loss / cfg["train"]["log_interval"]
                progress.set_postfix(loss=f"{avg_loss:.4f}")
                running_loss = 0.0

        if epoch % cfg["train"]["save_interval"] == 0:
            ckpt_path = os.path.join(run_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            resume_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss_history": loss_history,
                },
                resume_path,
            )
            logger.info("checkpoint saved: %s", resume_path)

        val_metrics = None
        if epoch % cfg["train"].get("val_interval", 1) == 0:
            logger.info("running validation...")
            val_metrics = _maybe_eval(model, cfg, device, no_val=no_val)
        if val_metrics:
            logger.info("epoch %d val - mae: %.4f, iou: %.4f, f1: %.4f",
                       epoch, val_metrics["mae"], val_metrics["iou"], val_metrics["f1"])
            save_json(val_metrics, os.path.join(run_dir, f"val_metrics_epoch_{epoch}.json"))
            val_history["mae"].append(val_metrics["mae"])
            val_history["iou"].append(val_metrics["iou"])
            val_history["f1"].append(val_metrics["f1"])
        if step_count > 0:
            avg_epoch_loss = epoch_loss / step_count
            loss_history.append(avg_epoch_loss)
            if val_metrics:
                logger.info("epoch %d - train: %.4f, val_mae: %.4f, val_iou: %.4f",
                           epoch, avg_epoch_loss, val_metrics["mae"], val_metrics["iou"])
            else:
                logger.info("epoch %d - train: %.4f", epoch, avg_epoch_loss)

        # Log to wandb
        log_dict = {"epoch": epoch, "train/loss": avg_epoch_loss}
        if val_metrics:
            log_dict.update({
                "val/mae": val_metrics["mae"],
                "val/iou": val_metrics["iou"],
                "val/f1": val_metrics["f1"],
            })
        if wandb_enabled:
            wandb.log(log_dict)

    final_path = os.path.join(run_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info("training done, final checkpoint: %s", final_path)
    if wandb_enabled:
        wandb.finish()


def _maybe_eval(model, cfg, device, no_val=False):
    if no_val:
        return None
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
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument("--resume", default=None, help="path to checkpoint_epoch_*.pt")
    args = parser.parse_args()
    train(args.config, no_val=args.no_val, resume_path=args.resume)


if __name__ == "__main__":
    main()
