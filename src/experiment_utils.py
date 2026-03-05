"""
实验工具模块 - 用于完整实验的辅助函数

包含：
- 可视化保存辅助函数
- 实验运行辅助类
"""
import os
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch


def save_experiment_visuals(image, saliency_map, output_dir: Path, name: str):
    """
    保存实验预测图片（热力图和叠加图）
    
    Args:
        image: 原始图像 (H, W, 3) numpy array
        saliency_map: 显著性图 (H, W) numpy array
        output_dir: 输出目录
        name: 图片名称（不含扩展名）
    """
    from .infer import save_visuals
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_visuals(image, saliency_map, str(output_dir), name)


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, log_file_path: str):
        """
        初始化日志记录器
        
        Args:
            log_file_path: 日志文件路径
        """
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def log(self, message: str):
        """记录日志到文件和控制台"""
        print(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()
    
    def close(self):
        """关闭日志文件"""
        self.log_file.close()


def compute_average_metrics(metrics_list: list) -> Dict:
    """
    计算平均指标和标准差
    
    Args:
        metrics_list: 指标字典列表
        
    Returns:
        包含均值和标准差的字典
    """
    if not metrics_list:
        return None
    
    return {
        'mae': float(np.mean([m['mae'] for m in metrics_list])),
        'iou': float(np.mean([m['iou'] for m in metrics_list])),
        'f1': float(np.mean([m['f1'] for m in metrics_list])),
        'std_mae': float(np.std([m['mae'] for m in metrics_list])),
        'std_iou': float(np.std([m['iou'] for m in metrics_list])),
        'std_f1': float(np.std([m['f1'] for m in metrics_list])),
        'num_images': len(metrics_list)
    }


def format_metrics_table(model_results: Dict, model_names: list = None) -> str:
    """
    格式化指标为 Markdown 表格
    
    Args:
        model_results: 模型结果字典 {model_name: {metrics: {...}}}
        model_names: 模型名称列表（指定顺序）
        
    Returns:
        Markdown 表格字符串
    """
    if model_names is None:
        model_names = list(model_results.keys())
    
    lines = []
    lines.append("| 模型 | MAE | IoU | F1 | 说明 |")
    lines.append("|------|-----|----|----|----|")
    
    for model_name in model_names:
        if model_name not in model_results:
            continue
        
        data = model_results[model_name]
        if 'metrics' not in data:
            continue
        
        m = data['metrics']
        
        # 模型名称
        name_display = {
            'original': '原始超像素',
            'improved': '改进超像素',
            'dss': 'DSS Baseline'
        }.get(model_name, model_name)
        
        # 说明
        description = {
            'original': '-',
            'improved': 'SE注意力+智能融合',
            'dss': 'VGG-16 based'
        }.get(model_name, '')
        
        line = f"| {name_display} | {m['mae']:.4f}±{m['std_mae']:.4f} | "
        line += f"{m['iou']:.4f}±{m['std_iou']:.4f} | "
        line += f"{m['f1']:.4f}±{m['std_f1']:.4f} | {description} |"
        
        lines.append(line)
    
    return '\n'.join(lines)


def generate_comparison_summary(results: Dict) -> str:
    """
    生成对比总结文本
    
    Args:
        results: 实验结果字典
        
    Returns:
        总结文本字符串
    """
    lines = []
    
    if 'improved' in results['models'] and 'dss' in results['models']:
        if 'metrics' in results['models']['improved'] and 'metrics' in results['models']['dss']:
            impr_iou = results['models']['improved']['metrics']['iou']
            dss_iou = results['models']['dss']['metrics']['iou']
            
            lines.append("### 性能对比\n\n")
            lines.append(f"- **DSS Baseline IoU**: {dss_iou:.4f}\n")
            lines.append(f"- **改进模型 IoU**: {impr_iou:.4f}\n")
            lines.append(f"- **改善**: {(impr_iou - dss_iou) / dss_iou * 100:+.1f}%\n\n")
            
            if impr_iou > dss_iou:
                lines.append("### ✅ 成功超越 DSS Baseline！\n\n")
                lines.append("改进的超像素显著性检测模型在准确性上超越了经典的 DSS 方法，")
                lines.append("同时保持了更少的参数量和更好的可解释性。\n\n")
            else:
                gap = dss_iou - impr_iou
                lines.append(f"### ⚠️ 还差 {gap:.4f} IoU\n\n")
                lines.append("虽然还没有完全超越 DSS，但已经取得了显著的改进。")
                lines.append("建议增加训练轮数或添加更多改进（CRF 后处理、数据增强等）。\n\n")
    
    return ''.join(lines)


def save_experiment_results(results: Dict, output_dir: str, filename: str = "experiment_results.json"):
    """
    保存实验结果到 JSON 文件
    
    Args:
        results: 结果字典
        output_dir: 输出目录
        filename: 文件名
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return output_path


def generate_experiment_report(results: Dict, output_path: str, timestamp: str):
    """
    生成完整的实验报告 Markdown 文件
    
    Args:
        results: 实验结果字典
        output_path: 输出文件路径
        timestamp: 时间戳
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 超像素显著性检测 - 完整对比实验报告\n\n")
        
        # 基本信息
        f.write("## 基本信息\n\n")
        f.write(f"- **时间戳**: {timestamp}\n")
        f.write(f"- **实验ID**: {timestamp}\n\n")
        
        # 模型对比
        f.write("## 模型对比\n\n")
        f.write(format_metrics_table(results['models']))
        f.write("\n\n")
        
        # 详细结果
        f.write("## 详细结果\n\n")
        
        for model_name, model_data in results['models'].items():
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
        f.write(generate_comparison_summary(results))
    
    return output_path
