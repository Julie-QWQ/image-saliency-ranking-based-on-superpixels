"""
改进的训练脚本

改进内容：
1. Focal Loss 处理难样本
2. 数据增强
3. 学习率调度
4. 混合精度训练
"""
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .data import SuperpixelSaliencyDataset
from .evaluate import evaluate
from .model_improved import ImprovedMultiBranchNet
from .utils import (
    ensure_dir,
    get_device,
    load_config,
    save_json,
    set_seed,
    setup_logger,
    setup_run_dir,
)


class FocalLoss:
    """Focal Loss 用于处理类别不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, inputs, targets):
        """
        Args:
            inputs: 预测值 (logits)
            targets: 目标值 (0或1)
        """
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train(config_path, no_val=False, resume_path=None):
    cfg = load_config(config_path)
    set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    device = get_device(cfg["runtime"]["device"])

    run_dir = setup_run_dir(cfg["paths"]["output_dir"], "train_improved")
    logger = setup_logger(os.path.join(run_dir, "train.log"))
    save_json(cfg, os.path.join(run_dir, "config.json"))
    logger.info("config loaded, device=%s, run_dir=%s", device, run_dir)

    # 检查是否使用改进模型
    use_improved = cfg["model"].get("use_improved", False)
    logger.info("using improved model: %s", use_improved)

    logger.info("building train dataset...")
    train_dataset = SuperpixelSaliencyDataset(
        cfg["paths"]["train_images"],
        cfg["paths"]["train_masks"],
        cfg["slic"],
        cfg["labels"],
        cfg["masking"],
        cfg["model"]["input_size"],
        cache_dir=cfg["paths"].get("cache_dir"),
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
    if use_improved:
        model = ImprovedMultiBranchNet(
            cfg["model"]["feature_dim"],
            cfg["model"]["mlp_hidden"]
        ).to(device)
    else:
        from .model import MultiBranchNet
        model = MultiBranchNet(
            cfg["model"]["feature_dim"],
            cfg["model"]["mlp_hidden"]
        ).to(device)

    logger.info("model ready")

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"],
        eta_min=1e-6
    )

    # 损失函数
    use_focal = cfg["train"].get("use_focal_loss", False)
    if use_focal:
        alpha = cfg["train"].get("focal_alpha", 0.25)
        gamma = cfg["train"].get("focal_gamma", 2.0)
        criterion = FocalLoss(alpha, gamma)
        logger.info("using Focal Loss: alpha=%s, gamma=%s", alpha, gamma)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        logger.info("using BCEWithLogitsLoss")

    logger.info("optimizer and loss ready")

    # 混合精度训练
    scaler = GradScaler()
    use_amp = device.type == 'cuda'
    if use_amp:
        logger.info("using mixed precision training")

    start_epoch = 1
    loss_history = []
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(resume_path)
        logger.info("resuming from checkpoint: %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        loss_history = ckpt.get("loss_history", [])

    val_history = {"mae": [], "iou": [], "f1": []}
    best_iou = 0.0

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        step_count = 0

        logger.info("epoch %d start, lr=%.6f", epoch, optimizer.param_groups[0]['lr'])
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False, dynamic_ncols=True)

        for step, (xa, xb, xc, y) in enumerate(progress, start=1):
            xa = xa.to(device)
            xb = xb.to(device)
            xc = xc.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # 混合精度训练
            if use_amp:
                with autocast():
                    logits = model(xa, xb, xc)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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

        # 更新学习率
        scheduler.step()

        # 保存检查点
        if epoch % cfg["train"]["save_interval"] == 0:
            ckpt_path = os.path.join(run_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

            resume_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if use_amp else None,
                    "loss_history": loss_history,
                },
                resume_path,
            )
            logger.info("checkpoint saved: %s", resume_path)

        # 验证
        val_metrics = None
        if epoch % cfg["train"].get("val_interval", 1) == 0:
            logger.info("running validation (if available)...")
            val_metrics = _maybe_eval(model, cfg, device, no_val=no_val, use_improved=use_improved)

            if val_metrics:
                logger.info("epoch %d val_mae %.4f val_iou %.4f val_f1 %.4f",
                           epoch, val_metrics["mae"], val_metrics["iou"], val_metrics["f1"])

                save_json(val_metrics, os.path.join(run_dir, f"val_metrics_epoch_{epoch}.json"))

                val_history["mae"].append(val_metrics["mae"])
                val_history["iou"].append(val_metrics["iou"])
                val_history["f1"].append(val_metrics["f1"])

                # 保存最佳模型
                if val_metrics["iou"] > best_iou:
                    best_iou = val_metrics["iou"]
                    best_path = os.path.join(run_dir, "model_best.pt")
                    torch.save(model.state_dict(), best_path)
                    logger.info("new best model saved with IoU: %.4f", best_iou)

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


def _maybe_eval(model, cfg, device, no_val=False, use_improved=False):
    if no_val:
        return None

    val_images = cfg["paths"]["val_images"]
    val_masks = cfg["paths"]["val_masks"]

    if not (os.path.exists(val_images) and os.path.exists(val_masks)):
        return None

    model.eval()
    return evaluate(model, val_images, val_masks, cfg, device, use_improved=use_improved)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/improved.yaml")
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument("--resume", default=None, help="path to checkpoint_epoch_*.pt")
    args = parser.parse_args()

    train(args.config, no_val=args.no_val, resume_path=args.resume)


if __name__ == "__main__":
    main()
