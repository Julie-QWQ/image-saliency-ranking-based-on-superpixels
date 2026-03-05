#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
from pathlib import Path
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.data import SuperpixelSaliencyDataset
from src.model import MultiBranchNet as OriginalModel
from src.model_ultra_efficient import UltraEfficientNet as ImprovedModel
from src.dss_baseline import DSSBaseline
from src.utils import load_config, set_seed, get_device, read_image_rgb, read_mask_binary, compute_metrics
from src.infer import predict_image
from src.experiment_utils import ExperimentLogger, compute_average_metrics, generate_experiment_report, save_experiment_results
from tqdm import tqdm

class FullExperimentRunner:
    def __init__(self, base_output_dir="outputs/full_experiment"):
        self.base_dir = Path(base_output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.run_dir / "experiment.log"
        self.logger = ExperimentLogger(str(log_file))
        self.results = {'timestamp': self.timestamp, 'models': {}}

    def log(self, message):
        self.logger.log(message)

    def close(self):
        self.logger.close()

    def train_model(self, model_type='original', epochs=3, resume_checkpoint=None):
        self.log("=" * 70)
        self.log(f"Training {model_type} model")
        self.log("=" * 70)

        cfg = load_config("configs/improved.yaml" if model_type == 'improved' else "configs/default.yaml")
        device = get_device(cfg['runtime']['device'])
        set_seed(cfg['seed'], cfg['runtime']['deterministic'])
        self.log(f"Device: {device}")

        Model = ImprovedModel if model_type == 'improved' else OriginalModel
        model = Model(cfg['model']['feature_dim'], cfg['model']['mlp_hidden']).to(device)
        params = sum(p.numel() for p in model.parameters())
        self.log(f"Model parameters: {params:,} ({params/1e6:.2f}M)")

        # 断点续训
        start_epoch = 1
        if resume_checkpoint:
            checkpoint_path = resume_checkpoint
            if not os.path.exists(checkpoint_path):
                # 尝试在当前运行目录查找
                checkpoint_path = self.run_dir / resume_checkpoint
                if not os.path.exists(checkpoint_path):
                    self.log(f"Warning: Checkpoint not found: {resume_checkpoint}")
                    resume_checkpoint = None

            if resume_checkpoint and os.path.exists(checkpoint_path):
                self.log(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)

                # 检查是否是完整 checkpoint（包含 optimizer 等）
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    start_epoch = checkpoint.get('epoch', 1) + 1
                    best_loss = checkpoint.get('best_loss', float('inf'))
                    self.log(f"Resumed from epoch {start_epoch-1}, best_loss={best_loss:.4f}")
                else:
                    # 只是模型权重
                    model.load_state_dict(checkpoint)
                    best_loss = float('inf')
                    self.log(f"Loaded model weights from checkpoint")
            else:
                best_loss = float('inf')
        else:
            best_loss = float('inf')

        try:
            train_dataset = SuperpixelSaliencyDataset(
                cfg['paths']['train_images'], cfg['paths']['train_masks'],
                cfg['slic'], cfg['labels'], cfg['masking'], cfg['model']['input_size'],
                cache_dir=cfg['paths'].get('cache_dir')
            )
            self.log(f"Training samples: {len(train_dataset)}")
            train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True)
        except Exception as e:
            self.log(f"Warning: Cannot create dataset: {e}")
            train_loader = None

        if train_loader:
            self.log(f"\nStarting training ({epochs} epochs, from epoch {start_epoch})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])

            # 如果是完整 checkpoint，恢复 optimizer
            if resume_checkpoint and os.path.exists(resume_checkpoint):
                checkpoint = torch.load(resume_checkpoint, map_location=device)
                if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if cfg['train'].get('use_focal_loss', False):
                from src.train_improved import FocalLoss
                criterion = FocalLoss(cfg['train'].get('focal_alpha', 0.25), cfg['train'].get('focal_gamma', 2.0))
            else:
                criterion = torch.nn.BCEWithLogitsLoss()

            model.train()

            for epoch in range(start_epoch, epochs + 1):
                total_loss, count = 0.0, 0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, ncols=80)
                for xa, xb, xc, y in pbar:
                    xa, xb, xc, y = xa.to(device), xb.to(device), xc.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xa, xb, xc), y)
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    total_loss += loss.item()
                    count += 1
                avg_loss = total_loss/count
                self.log(f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f}")

                # 每个 epoch 保存模型权重
                epoch_path = self.run_dir / f"model_{model_type}_epoch{epoch}.pt"
                torch.save(model.state_dict(), epoch_path)
                self.log(f"Checkpoint saved: {epoch_path}")

                # 保存完整 checkpoint（包含 optimizer 等状态，用于断点续训）
                full_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'cfg': cfg
                }
                full_path = self.run_dir / f"model_{model_type}_epoch{epoch}_full.pt"
                torch.save(full_checkpoint, full_path)

                # 保存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = self.run_dir / f"model_{model_type}_best.pt"
                    torch.save(model.state_dict(), best_path)
                    self.log(f"Best model saved (loss={best_loss:.4f}): {best_path}")

        # 最终模型
        model_path = self.run_dir / f"model_{model_type}_final.pt"
        torch.save(model.state_dict(), model_path)
        self.log(f"Final model saved: {model_path}")
        return model, model_path

    def evaluate_model(self, model, model_name, max_images=50, save_vis=10):
        self.log("\n" + "=" * 70)
        self.log(f"Evaluating {model_name}")
        self.log("=" * 70)

        cfg = load_config("configs/default.yaml")
        device = get_device(cfg['runtime']['device'])
        model.eval()

        images_dir = cfg['paths']['val_images']
        masks_dir = cfg['paths']['val_masks']

        if not os.path.exists(images_dir):
            self.log(f"Warning: {images_dir} does not exist")
            return None

        import glob
        images = []
        for pattern in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            images.extend(glob.glob(os.path.join(images_dir, pattern)))
        images = sorted(images)[:max_images]

        if not images:
            self.log("Warning: No images found")
            return None

        self.log(f"Test images: {len(images)}")

        vis_path = None
        if save_vis > 0:
            vis_path = self.run_dir / "visualizations" / model_name
            vis_path.mkdir(parents=True, exist_ok=True)
            self.log(f"Visualizations save to: {vis_path}")

        metrics_list = []
        slic_cfg = cfg['slic']
        mask_cfg = cfg['masking']
        input_size = cfg['model']['input_size']

        for i, img_path in enumerate(images):
            try:
                name = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(masks_dir, f"{name}.png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(masks_dir, f"{name}.jpg")
                if not os.path.exists(mask_path):
                    continue

                image = read_image_rgb(img_path)
                gt_mask = read_mask_binary(mask_path)

                if 'dss' in model_name.lower():
                    saliency_map = model.predict(image, input_size, device)
                else:
                    saliency_map = predict_image(model, image, slic_cfg, mask_cfg, input_size, device)

                metrics = compute_metrics(saliency_map, gt_mask)
                metrics_list.append(metrics)

                if vis_path and i < save_vis:
                    try:
                        from src.experiment_utils import save_experiment_visuals
                        save_experiment_visuals(image, saliency_map, vis_path, name)
                    except Exception as e:
                        self.log(f"Warning: Failed to save visualization ({name}): {e}")

                if (i + 1) % 10 == 0:
                    self.log(f"Processed: {i + 1}/{len(images)}")
            except Exception as e:
                self.log(f"Error: {e}")

        if not metrics_list:
            return None

        avg_metrics = compute_average_metrics(metrics_list)
        self.log(f"\nEvaluation results: MAE={avg_metrics['mae']:.4f}, IoU={avg_metrics['iou']:.4f}, F1={avg_metrics['f1']:.4f}")
        return avg_metrics

    def run_full_experiment(self, train_epochs=3, resume_checkpoint=None):
        start_time = time.time()
        self.log("\n" + "=" * 70)
        self.log("Starting Full Comparison Experiment")
        self.log("=" * 70)
        self.log(f"\nTimestamp: {self.timestamp}")
        self.log(f"Training epochs: {train_epochs}")
        if resume_checkpoint:
            self.log(f"Resume from: {resume_checkpoint}")
        self.log("")

        try:
            original_model, _ = self.train_model('original', epochs=train_epochs, resume_checkpoint=resume_checkpoint)
            improved_model, _ = self.train_model('improved', epochs=train_epochs, resume_checkpoint=resume_checkpoint)

            device = get_device('auto')
            dss_model = DSSBaseline().to(device)
            dss_params = sum(p.numel() for p in dss_model.parameters())
            self.log(f"\nDSS model parameters: {dss_params:,}")
            self.results['models']['dss'] = {'params': dss_params}

            self.log("\n[Evaluating all models]")
            original_metrics = self.evaluate_model(original_model, "original")
            if original_metrics:
                self.results['models']['original'] = {'metrics': original_metrics}

            improved_metrics = self.evaluate_model(improved_model, "improved")
            if improved_metrics:
                self.results['models']['improved'] = {'metrics': improved_metrics}

            dss_metrics = self.evaluate_model(dss_model, "dss")
            if dss_metrics:
                self.results['models']['dss']['metrics'] = dss_metrics

            save_experiment_results(self.results, str(self.run_dir))
            generate_experiment_report(self.results, str(self.run_dir / "experiment_report.md"), self.timestamp)
            self.log("\nResults and report saved")
        except Exception as e:
            self.log(f"\nError: {e}")
            import traceback
            self.log(traceback.format_exc())

        elapsed_time = time.time() - start_time
        self.log("\n" + "=" * 70)
        self.log(f"Experiment completed! Total time: {elapsed_time/60:.1f} minutes")
        self.log("=" * 70)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run full comparison experiment')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--save_vis', type=int, default=10, help='Number of prediction images to save')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (path or filename in run dir)')
    args = parser.parse_args()

    runner = FullExperimentRunner()
    try:
        runner.run_full_experiment(train_epochs=args.epochs, resume_checkpoint=args.resume)
    finally:
        runner.close()

if __name__ == '__main__':
    main()
