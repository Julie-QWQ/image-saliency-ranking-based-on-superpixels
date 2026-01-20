import os

import torch
from matplotlib import pyplot as plt
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


def train(config_path):
    cfg = load_config(config_path)
    set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    device = get_device(cfg["runtime"]["device"])

    run_dir = setup_run_dir(cfg["paths"]["output_dir"], "train")
    logger = setup_logger(os.path.join(run_dir, "train.log"))
    save_json(cfg, os.path.join(run_dir, "config.json"))
    logger.info("config loaded, device=%s, run_dir=%s", device, run_dir)

    logger.info("building train dataset...")
    train_dataset = SuperpixelSaliencyDataset(
        cfg["paths"]["train_images"],
        cfg["paths"]["train_masks"],
        cfg["slic"],
        cfg["labels"],
        cfg["masking"],
        cfg["model"]["input_size"],
    )
    logger.info("train dataset ready, samples=%d", len(train_dataset))
    logger.info("building dataloader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
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

    loss_history = []
    val_history = {"mae": [], "iou": [], "f1": []}
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        step_count = 0
        logger.info("epoch %d start", epoch)
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
                logger.debug("epoch %d step %d loss %.4f", epoch, step, avg_loss)
                progress.set_postfix(loss=f"{avg_loss:.4f}")
                running_loss = 0.0

        if epoch % cfg["train"]["save_interval"] == 0:
            ckpt_path = os.path.join(run_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info("checkpoint saved: %s", ckpt_path)

        logger.info("running validation (if available)...")
        val_metrics = _maybe_eval(model, cfg, device)
        if val_metrics:
            logger.info("epoch %d val_mae %.4f val_iou %.4f val_f1 %.4f", epoch, val_metrics["mae"], val_metrics["iou"], val_metrics["f1"])
            save_json(val_metrics, os.path.join(run_dir, f"val_metrics_epoch_{epoch}.json"))
            logger.info("validation metrics saved")
            val_history["mae"].append(val_metrics["mae"])
            val_history["iou"].append(val_metrics["iou"])
            val_history["f1"].append(val_metrics["f1"])
        if step_count > 0:
            avg_epoch_loss = epoch_loss / step_count
            loss_history.append(avg_epoch_loss)
            logger.info("epoch %d avg_loss %.4f end", epoch, avg_epoch_loss)

    final_path = os.path.join(run_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info("training done, final checkpoint %s", final_path)
    _save_loss_plot(loss_history, run_dir)
    _save_val_plot(val_history, run_dir)


def _save_loss_plot(loss_history, run_dir):
    if not loss_history:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    out_path = os.path.join(run_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _save_val_plot(val_history, run_dir):
    if not val_history["mae"]:
        return
    epochs = range(1, len(val_history["mae"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_history["mae"], marker="o", label="MAE")
    plt.plot(epochs, val_history["iou"], marker="o", label="IoU")
    plt.plot(epochs, val_history["f1"], marker="o", label="F1")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.title("validation metrics")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    out_path = os.path.join(run_dir, "val_metrics_curve.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


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
