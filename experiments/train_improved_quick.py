#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练改进模型的快速启动脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train_improved import main
import argparse

if __name__ == '__main__':
    # 使用改进配置
    config_path = "configs/improved.yaml"

    print("=" * 70)
    print("启动改进模型训练")
    print("=" * 70)
    print(f"\n配置文件: {config_path}")
    print("\n改进内容:")
    print("  ✓ SE 通道注意力")
    print("  ✓ 智能融合机制")
    print("  ✓ 2-ring 邻域")
    print("  ✓ Focal Loss")
    print("  ✓ 混合精度训练")
    print("\n预期目标: IoU > 0.80 (超越 DSS Baseline)")
    print("=" * 70)
    print()

    # 模拟命令行参数
    sys.argv = ['train_improved.py', '--config', config_path]

    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
