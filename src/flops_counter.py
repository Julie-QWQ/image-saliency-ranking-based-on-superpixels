"""通用 FLOPs 计数器"""
import torch
from thop import profile

def count_superpixel_inference_flops(model, num_superpixels=200):
    """计算超像素推理 FLOPs"""
    # 单分支 FLOPs
    branch_flops, _ = profile(model.branch_a, (torch.randn(1, 3, 224, 224),), verbose=False)
    
    # 融合层 FLOPs (估算)
    if hasattr(model, 'smart_fusion'):
        fusion_flops = 384 * 64 + 64 * 1  # 智能融合
    elif hasattr(model, 'attention_fusion'):
        fusion_flops = 384 * 64 + 64 * 1  # 注意力融合
    else:
        fusion_flops = 384 * 64 + 64 * 1  # 原始融合
    
    total_flops = (num_superpixels * 2 + 1) * branch_flops + num_superpixels * fusion_flops
    
    # 格式化
    if total_flops >= 1e12:
        fmt = f"{total_flops / 1e12:.3f}T"
    elif total_flops >= 1e9:
        fmt = f"{total_flops / 1e9:.3f}G"
    elif total_flops >= 1e6:
        fmt = f"{total_flops / 1e6:.3f}M"
    else:
        fmt = str(total_flops)
    
    return {'total': total_flops, 'total_formatted': fmt}
