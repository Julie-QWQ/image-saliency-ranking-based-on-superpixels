#!/usr/bin/env python3
"""
DSS Baseline vs 超像素方法 - 计算量与准确性对比实验

使用方法:
    python compare.py \
        --config configs/comparison.yaml \
        --checkpoint outputs/train_*/model_final.pt \
        --num_images 10 \
        --output_dir outputs/comparison
"""
import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.comparison import ComparisonEvaluator, _list_images, _match_masks
from src.model import MultiBranchNet
from src.dss_baseline import DSSBaseline
from src.utils import get_device, load_config


def main():
    parser = argparse.ArgumentParser(description='计算量对比实验')
    parser.add_argument('--config', default='configs/comparison.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', required=True,
                       help='超像素模型检查点路径')
    parser.add_argument('--dss_weights', default=None,
                       help='DSS 预训练权重路径（可选，默认使用 VGG-16 预训练）')
    parser.add_argument('--num_images', type=int, default=10,
                       help='测试图像数量（默认：10）')
    parser.add_argument('--output_dir', default='outputs/comparison',
                       help='输出目录（默认：outputs/comparison）')
    args = parser.parse_args()

    print("=" * 60)
    print("计算量与准确性对比实验")
    print("=" * 60)

    # 加载配置
    print(f"\n[1/5] 加载配置: {args.config}")
    cfg = load_config(args.config)
    device = get_device(cfg['runtime']['device'])
    print(f"  设备: {device}")

    # 加载超像素模型
    print(f"\n[2/5] 加载超像素模型: {args.checkpoint}")
    superpixel_model = MultiBranchNet(
        cfg['model']['feature_dim'],
        cfg['model']['mlp_hidden']
    ).to(device)

    if not os.path.exists(args.checkpoint):
        print(f"  错误: 检查点文件不存在: {args.checkpoint}")
        sys.exit(1)

    try:
        state_dict = torch.load(args.checkpoint, map_location=device)
        superpixel_model.load_state_dict(state_dict)
        print(f"  ✓ 超像素模型加载成功")
    except Exception as e:
        print(f"  错误: 无法加载超像素模型: {e}")
        sys.exit(1)

    superpixel_model.eval()

    # 加载 DSS Baseline
    print(f"\n[3/5] 加载 DSS Baseline")
    dss_model = DSSBaseline(pretrained_path=args.dss_weights).to(device)
    print(f"  ✓ DSS Baseline 初始化成功")

    # 准备测试数据
    print(f"\n[4/5] 准备测试数据")
    test_images = _list_images(cfg['paths']['test_images'])
    test_masks = _match_masks(test_images, cfg['paths']['test_masks'])

    num_images = min(args.num_images, len(test_images))
    test_images = test_images[:num_images]
    test_masks = test_masks[:num_images]

    print(f"  测试图像数量: {len(test_images)}")

    # 运行对比实验
    print(f"\n[5/5] 运行对比实验")
    evaluator = ComparisonEvaluator(superpixel_model, dss_model, cfg, device)

    try:
        results, summary = evaluator.run_comparison(test_images, test_masks)
    except Exception as e:
        print(f"\n错误: 实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 保存结果
    print(f"\n保存结果到: {args.output_dir}")
    evaluator.save_results(results, summary, args.output_dir)

    # 打印摘要
    print("\n" + "=" * 60)
    print("实验完成！结果摘要:")
    print("=" * 60)
    print(f"\n计算量对比:")
    print(f"  DSS Baseline:   {summary['dss']['flops_formatted']}")
    print(f"  超像素方法:     {summary['superpixel']['flops_formatted']}")
    print(f"  比值:           {summary['comparison']['flops_ratio']:.2f}x")

    print(f"\n准确性对比:")
    print(f"  指标    DSS Baseline    超像素方法    改善")
    print(f"  MAE     {summary['dss']['mae']:.4f}         {summary['superpixel']['mae']:.4f}         {summary['comparison']['mae_improvement']*100:+.2f}%")
    print(f"  IoU     {summary['dss']['iou']:.4f}         {summary['superpixel']['iou']:.4f}         {summary['comparison']['iou_improvement']*100:+.2f}%")
    print(f"  F1      {summary['dss']['f1']:.4f}         {summary['superpixel']['f1']:.4f}         {summary['comparison']['f1_improvement']*100:+.2f}%")

    print(f"\n输出文件:")
    print(f"  - {args.output_dir}/comparison_results.json  (详细数据)")
    print(f"  - {args.output_dir}/comparison_report.md    (可视化报告)")
    print("=" * 60)


if __name__ == '__main__':
    main()
