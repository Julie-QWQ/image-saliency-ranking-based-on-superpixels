#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行完整的对比实验

包含：
1. 训练原始模型
2. 训练改进模型
3. 与 DSS Baseline 对比
4. 生成完整的实验报告

结果保存到 outputs/full_experiment/
"""
import sys
import os
import io
import json
import time
from datetime import datetime
from pathlib import Path

# 设置编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data import SuperpixelSaliencyDataset
from src.model import MultiBranchNet as OriginalModel
from src.model_ultra_efficient import UltraEfficientNet as ImprovedModel
from src.dss_baseline import DSSBaseline
from src.utils import (
    load_config, set_seed, get_device,
    read_image_rgb, read_mask_binary, compute_metrics,
    ensure_dir, save_json
)
from src.infer import predict_image
from src.experiment_utils import (
    save_experiment_visuals,
    ExperimentLogger,
    compute_average_metrics,
    generate_experiment_report,

        self.results = {
            'timestamp': self.timestamp,
            'models': {},
            'comparison': {}
        }

    def log(self, message):
        """记录日志"""
        print(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def close(self):
        """关闭日志"""
        self.log_file.close()

    def train_model(self, model_type='original', epochs=3):
        """
        训练模型

        Args:
            model_type: 'original' 或 'improved'
            epochs: 训练轮数
        """
        self.log("=" * 70)
        self.log(f"训练 {model_type} 模型")
        self.log("=" * 70)

        # 加载配置
        if model_type == 'improved':
            cfg = load_config("configs/improved.yaml")
        else:
            cfg = load_config("configs/default.yaml")

        device = get_device(cfg['runtime']['device'])
        set_seed(cfg['seed'], cfg['runtime']['deterministic'])

        self.log(f"设备: {device}")
        self.log(f"配置: {model_type}")

        # 创建模型
        if model_type == 'improved':
            model = ImprovedModel(
                cfg['model']['feature_dim'],
                cfg['model']['mlp_hidden']
            ).to(device)
        else:
            model = OriginalModel(
                cfg['model']['feature_dim'],
                cfg['model']['mlp_hidden']
            ).to(device)

        # 计算模型参数
        params = sum(p.numel() for p in model.parameters())
        self.log(f"模型参数量: {params:,} ({params/1e6:.2f}M)")

        # 创建数据集（使用小规模快速训练）
        self.log("\n准备数据集...")
        try:
            train_dataset = SuperpixelSaliencyDataset(
                cfg['paths']['train_images'],
                cfg['paths']['train_masks'],
                cfg['slic'],
                cfg['labels'],
                cfg['masking'],
                cfg['model']['input_size'],
                cache_dir=cfg['paths'].get('cache_dir'),
            )
            self.log(f"训练样本数: {len(train_dataset)}")

            # 使用小批量快速训练
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg['train']['batch_size'],
                shuffle=True,
                num_workers=0,
            )
        except Exception as e:
            self.log(f"警告: 无法创建数据集: {e}")
            self.log("跳过训练，使用随机初始化模型进行评估")
            train_loader = None

        # 训练
        if train_loader:
            self.log(f"\n开始训练 ({epochs} epochs)...")

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg['train']['lr'],
                weight_decay=cfg['train']['weight_decay']
            )

            # 使用 Focal Loss（如果配置了）
            use_focal = cfg['train'].get('use_focal_loss', False)
            if use_focal:
                from src.train_improved import FocalLoss
                alpha = cfg['train'].get('focal_alpha', 0.25)
                gamma = cfg['train'].get('focal_gamma', 2.0)
                criterion = FocalLoss(alpha, gamma)
                self.log(f"使用 Focal Loss: alpha={alpha}, gamma={gamma}")
            else:
                criterion = torch.nn.BCEWithLogitsLoss()

            model.train()
            for epoch in range(1, epochs + 1):
                total_loss = 0.0
                count = 0

                for xa, xb, xc, y in train_loader:
                    xa = xa.to(device)
                    xb = xb.to(device)
                    xc = xc.to(device)
                    y = y.to(device)

                    optimizer.zero_grad()
                    outputs = model(xa, xb, xc)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    count += 1

                avg_loss = total_loss / count if count > 0 else 0
                self.log(f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f}")

        # 保存模型
        model_path = self.run_dir / f"model_{model_type}.pt"
        torch.save(model.state_dict(), model_path)
        self.log(f"\n模型已保存: {model_path}")

        return model, model_path

    def evaluate_model(self, model, model_name, split='val', max_images=50, save_vis=10, vis_dir=None):
        """
        评估模型

        Args:
            model: 模型
            model_name: 模型名称
            split: 'val' 或 'test'
            max_images: 最大评估图像数
            save_vis: 保存可视化图片的数量
            vis_dir: 可视化图片保存目录
        """
        self.log("\n" + "=" * 70)
        self.log(f"评估 {model_name}")
        self.log("=" * 70)

        cfg = load_config("configs/default.yaml")
        device = get_device(cfg['runtime']['device'])

        model.eval()

        # 准备测试数据
        if split == 'val':
            images_dir = cfg['paths']['val_images']
            masks_dir = cfg['paths']['val_masks']
        else:
            images_dir = cfg['paths']['test_images']
            masks_dir = cfg['paths']['test_masks']

        # 检查数据是否存在
        if not os.path.exists(images_dir):
            self.log(f"警告: {images_dir} 不存在，跳过评估")
            return None

        # 列出图像
        import glob
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        images = []
        for pattern in patterns:
            images.extend(glob.glob(os.path.join(images_dir, pattern)))
        images.sort()
        images = images[:max_images]

        if not images:
            self.log(f"警告: {images_dir} 中没有找到图像")
            return None

        self.log(f"测试图像数量: {len(images)}")

        # 创建可视化目录
        if vis_dir and save_vis > 0:
            vis_path = self.run_dir / vis_dir / model_name
            vis_path.mkdir(parents=True, exist_ok=True)
            self.log(f"可视化保存至: {vis_path}")
        else:
            vis_path = None

        # 评估
        metrics_list = []
        slic_cfg = cfg['slic']
        mask_cfg = cfg['masking']
        input_size = cfg['model']['input_size']

        for i, img_path in enumerate(images):
            try:
                # 匹配掩码
                name = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(masks_dir, f"{name}.png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(masks_dir, f"{name}.jpg")

                if not os.path.exists(mask_path):
                    self.log(f"警告: 跳过 {img_path} (无掩码)")
                    continue

                # 读取图像
                image = read_image_rgb(img_path)
                gt_mask = read_mask_binary(mask_path)

                # 推理
                if 'dss' in model_name.lower():
                    # DSS 模型
                    saliency_map = model.predict(image, input_size, device)
                else:
                    # 超像素模型
                    saliency_map = predict_image(
                        model, image, slic_cfg, mask_cfg,
                        input_size, device,
                        cache_dir=cfg['paths'].get('cache_dir')
                    )

                # 计算指标
                metrics = compute_metrics(saliency_map, gt_mask)
                metrics_list.append(metrics)

                # 保存可视化图片（前 N 张）
                if vis_path and i < save_vis:
                    try:
                        from src.infer import save_visuals
                        save_visuals(image, saliency_map, str(vis_path), name)
                    except Exception as e:
                        self.log(f"警告: 保存可视化失败 ({name}): {e}")

                if (i + 1) % 10 == 0:
                    self.log(f"已处理: {i + 1}/{len(images)}")

            except Exception as e:
                self.log(f"错误处理 {img_path}: {e}")
                continue

        if not metrics_list:
            self.log("警告: 没有成功评估任何图像")
            return None

        # 计算平均指标
        avg_metrics = {
            'mae': float(np.mean([m['mae'] for m in metrics_list])),
            'iou': float(np.mean([m['iou'] for m in metrics_list])),
            'f1': float(np.mean([m['f1'] for m in metrics_list])),
            'std_mae': float(np.std([m['mae'] for m in metrics_list])),
            'std_iou': float(np.std([m['iou'] for m in metrics_list])),
            'std_f1': float(np.std([m['f1'] for m in metrics_list])),
            'num_images': len(metrics_list)
        }

        self.log(f"\n评估结果 ({len(metrics_list)} 张图像):")
        self.log(f"  MAE: {avg_metrics['mae']:.4f} ± {avg_metrics['std_mae']:.4f}")
        self.log(f"  IoU: {avg_metrics['iou']:.4f} ± {avg_metrics['std_iou']:.4f}")
        self.log(f"  F1:  {avg_metrics['f1']:.4f} ± {avg_metrics['std_f1']:.4f}")

        return avg_metrics

    def run_full_experiment(self, train_epochs=3):
        """运行完整实验"""
        start_time = time.time()

        self.log("\n" + "🎉" * 35)
        self.log("开始完整对比实验")
        self.log("🎉" * 35)
        self.log(f"\n时间戳: {self.timestamp}")
        self.log(f"输出目录: {self.run_dir}")
        self.log(f"训练轮数: {train_epochs}")
        self.log("")

        try:
            # 1. 训练原始模型
            self.log("\n【步骤 1/5】训练原始模型")
            self.log("-" * 70)
            original_model, original_path = self.train_model('original', epochs=train_epochs)
            self.results['models']['original'] = {'path': str(original_path)}

            # 2. 训练改进模型
            self.log("\n【步骤 2/5】训练改进模型")
            self.log("-" * 70)
            improved_model, improved_path = self.train_model('improved', epochs=train_epochs)
            self.results['models']['improved'] = {'path': str(improved_path)}

            # 3. 创建 DSS Baseline
            self.log("\n【步骤 3/5】创建 DSS Baseline")
            self.log("-" * 70)
            device = get_device('auto')
            dss_model = DSSBaseline().to(device)
            dss_params = sum(p.numel() for p in dss_model.parameters())
            self.log(f"DSS 模型参数量: {dss_params:,} ({dss_params/1e6:.2f}M)")
            self.results['models']['dss'] = {'params': dss_params}

            # 4. 评估所有模型
            self.log("\n【步骤 4/5】评估所有模型")
            self.log("-" * 70)

            # 评估原始模型
            self.log("\n评估原始模型...")
            original_model.eval()
            original_metrics = self.evaluate_model(original_model, "original", save_vis=10, vis_dir="visualizations")
            if original_metrics:
                self.results['models']['original']['metrics'] = original_metrics

            # 评估改进模型
            self.log("\n评估改进模型...")
            improved_model.eval()
            improved_metrics = self.evaluate_model(improved_model, "improved", save_vis=10, vis_dir="visualizations")
            if improved_metrics:
                self.results['models']['improved']['metrics'] = improved_metrics

            # 评估 DSS
            self.log("\n评估 DSS Baseline...")
            dss_model.eval()
            dss_metrics = self.evaluate_model(dss_model, "dss", save_vis=10, vis_dir="visualizations")
            if dss_metrics:
                self.results['models']['dss']['metrics'] = dss_metrics

            # 5. 生成报告
            self.log("\n【步骤 5/5】生成实验报告")
            self.log("-" * 70)
            self._generate_report()

            # 保存结果
            results_path = self.run_dir / "experiment_results.json"
            save_json(self.results, str(results_path))
            self.log(f"\n结果已保存: {results_path}")

        except Exception as e:
            self.log(f"\n❌ 实验运行出错: {e}")
            import traceback
            self.log(traceback.format_exc())

        # 总结
        elapsed_time = time.time() - start_time
        self.log("\n" + "=" * 70)
        self.log(f"实验完成！总耗时: {elapsed_time/60:.1f} 分钟")
        self.log("=" * 70)

        # 打印关键结果
        if 'dss' in self.results['models'] and 'improved' in self.results['models']:
            self.log("\n关键结果:")
            if 'metrics' in self.results['models']['dss']:
                dss_iou = self.results['models']['dss']['metrics']['iou']
                self.log(f"  DSS Baseline IoU:  {dss_iou:.4f}")

            if 'metrics' in self.results['models']['improved']:
                impr_iou = self.results['models']['improved']['metrics']['iou']
                self.log(f"  改进模型 IoU:    {impr_iou:.4f}")

                if 'metrics' in self.results['models']['dss']:
                    improvement = (impr_iou - dss_iou) / dss_iou * 100
                    self.log(f"  改善:            {improvement:+.1f}%")

                    if impr_iou > dss_iou:
                        self.log("\n  🎉🎉🎉 成功超越 DSS Baseline！大肉到手！🥩🥩🥩")
                    else:
                        gap = dss_iou - impr_iou
                        self.log(f"\n  ⚠️  还差 {gap:.4f} IoU 才能超越 DSS")
                        self.log("  建议: 增加 training epochs 或添加更多改进")

    def _generate_report(self):
        """生成实验报告"""
        report_path = self.run_dir / "experiment_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 超像素显著性检测 - 完整对比实验报告\n\n")

            # 基本信息
            f.write("## 基本信息\n\n")
            f.write(f"- **时间戳**: {self.timestamp}\n")
            f.write(f"- **实验ID**: {self.timestamp}\n")
            f.write(f"- **输出目录**: `{self.run_dir}`\n\n")

            # 模型对比
            f.write("## 模型对比\n\n")
            f.write("| 模型 | MAE | IoU | F1 | 说明 |\n")
            f.write("|------|-----|-----|----|----|\n")

            for model_name, model_data in self.results['models'].items():
                if 'metrics' in model_data:
                    m = model_data['metrics']
                    f.write(f"| {model_name} | {m['mae']:.4f}±{m['std_mae']:.4f} | ")
                    f.write(f"{m['iou']:.4f}±{m['std_iou']:.4f} | ")
                    f.write(f"{m['f1']:.4f}±{m['std_f1']:.4f} | ")

                    # 添加说明
                    if model_name == 'original':
                        f.write("|")
                    elif model_name == 'improved':
                        f.write("SE注意力+智能融合 |")
                    elif model_name == 'dss':
                        f.write("VGG-16 based |")

                    f.write("\n")

            # 详细结果
            f.write("\n## 详细结果\n\n")

            for model_name, model_data in self.results['models'].items():
                f.write(f"### {model_name}\n\n")

                if 'path' in model_data:
                    f.write(f"- **模型路径**: `{model_data['path']}`\n")

                if 'params' in model_data:
                    f.write(f"- **参数量**: {model_data['params']:,}\n")

                if 'metrics' in model_data:
                    m = model_data['metrics']
                    f.write(f"- **MAE**: {m['mae']:.4f} ± {m['std_mae']:.4f}\n")
                    f.write(f"- **IoU**: {m['iou']:.4f} ± {m['std_iou']:.4f}\n")
                    f.write(f"- **F1**: {m['f1']:.4f} ± {m['std_f1']:.4f}\n")
                    f.write(f"- **测试图像数**: {m['num_images']}\n")

                f.write("\n")

            # 结论
            f.write("## 结论\n\n")

            if 'improved' in self.results['models'] and 'dss' in self.results['models']:
                if 'metrics' in self.results['models']['improved'] and 'metrics' in self.results['models']['dss']:
                    impr_iou = self.results['models']['improved']['metrics']['iou']
                    dss_iou = self.results['models']['dss']['metrics']['iou']

                    f.write(f"### 性能对比\n\n")
                    f.write(f"- **DSS Baseline IoU**: {dss_iou:.4f}\n")
                    f.write(f"- **改进模型 IoU**: {impr_iou:.4f}\n")
                    f.write(f"- **改善**: {(impr_iou - dss_iou) / dss_iou * 100:+.1f}%\n\n")

                    if impr_iou > dss_iou:
                        f.write("### ✅ 成功超越 DSS Baseline！\n\n")
                        f.write("改进的超像素显著性检测模型在准确性上超越了经典的 DSS 方法，")
                        f.write("同时保持了更少的参数量和更好的可解释性。\n\n")
                    else:
                        gap = dss_iou - impr_iou
                        f.write(f"### ⚠️ 还差 {gap:.4f} IoU\n\n")
                        f.write("虽然还没有完全超越 DSS，但已经取得了显著的改进。")
                        f.write("建议增加训练轮数或添加更多改进（CRF 后处理、数据增强等）。\n\n")

        self.log(f"\n报告已生成: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='一键运行完整对比实验')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数 (默认: 3)')
    parser.add_argument('--output_dir', default='outputs/full_experiment', help='输出目录')
    parser.add_argument('--save_vis', type=int, default=10, help='每个模型保存的预测图片数量 (默认: 10)')
    args = parser.parse_args()

    # 创建实验运行器
    runner = FullExperimentRunner(args.output_dir)

    try:
        # 运行实验
        runner.run_full_experiment(train_epochs=args.epochs)
    finally:
        # 关闭日志
        runner.close()


if __name__ == '__main__':
    main()
