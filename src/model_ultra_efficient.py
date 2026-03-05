"""
超高效改进模型 - 目标是接近 DSS 的计算量同时超越其准确性

核心策略：
1. 保持轻量级分支（单分支 ~100M FLOPs）
2. 使用高效的注意力机制
3. 通过巧妙的架构设计而非暴力堆叠层数来提升性能
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightBranch(nn.Module):
    """
    轻量级分支 - 目标单分支 ~100M FLOPs

    与原始 BranchNet 类似的计算量，但添加关键改进
    """
    def __init__(self, feature_dim=128):
        super().__init__()

        # 使用深度可分离卷积减少计算量
        self.dw_conv1 = nn.Conv2d(3, 32, 3, padding=1, groups=1)  # 深度卷积
        self.pw_conv1 = nn.Conv2d(32, 32, 1)  # 逐点卷积
        self.bn1 = nn.BatchNorm2d(32)

        self.dw_conv2 = nn.Conv2d(32, 64, 3, padding=1, groups=1)
        self.pw_conv2 = nn.Conv2d(64, 64, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.dw_conv3 = nn.Conv2d(64, 128, 3, padding=1, groups=1)
        self.pw_conv3 = nn.Conv2d(128, 128, 1)
        self.bn3 = nn.BatchNorm2d(128)

        # SE 注意力（计算量很小）
        self.se = SEBlock(128, reduction=16)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        # 深度可分离卷积 + 池化
        x = F.relu(self.bn1(self.pw_conv1(self.dw_conv1(x))))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.pw_conv2(self.dw_conv2(x))))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.pw_conv3(self.dw_conv3(x))))
        x = self.se(x)
        x = F.max_pool2d(x, 2)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class SEBlock(nn.Module):
    """SE 注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()[:2]
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class SmartFusion(nn.Module):
    """
    智能融合机制 - 低计算量高效果

    使用通道注意力而非复杂的注意力机制
    """
    def __init__(self, feature_dim):
        super().__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim * 3, feature_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim // 4, feature_dim * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, fa, fb, fc):
        # 拼接
        fused = torch.cat([fa, fb, fc], dim=1).unsqueeze(2)  # (B, C*3, 1)

        # 计算注意力权重
        weights = self.channel_attention(fused)  # (B, C*3, 1)
        weights = weights.squeeze(2)  # (B, C*3)

        # 应用注意力
        fused = torch.cat([fa, fb, fc], dim=1)
        fused = fused * weights

        return fused


class UltraEfficientNet(nn.Module):
    """
    超高效多分支网络

    设计目标：
    - FLOPs: ~20G (200 超像素)
    - 参数量: ~0.5M
    - IoU: >0.82 (超越 DSS 的 0.80)

    策略：
    1. 深度可分离卷积大幅减少计算量
    2. SE 注意力提升特征表达能力
    3. 智能融合提升多分支协作
    """
    def __init__(self, feature_dim=128, mlp_hidden=64):
        super().__init__()

        # 三个轻量级分支
        self.branch_a = LightweightBranch(feature_dim)
        self.branch_b = LightweightBranch(feature_dim)
        self.branch_c = LightweightBranch(feature_dim)

        # 智能融合
        self.smart_fusion = SmartFusion(feature_dim)

        # 融合层
        self.fc7 = nn.Sequential(
            nn.Linear(feature_dim * 3, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # 输出头
        self.head = nn.Linear(mlp_hidden, 1)

    def forward(self, xa, xb, xc):
        fa = self.branch_a(xa)
        fb = self.branch_b(xb)
        fc = self.branch_c(xc)

        z = self.smart_fusion(fa, fb, fc)
        z = self.fc7(z)
        out = self.head(z).squeeze(1)

        return out


# 兼容性别名
MultiBranchNet = UltraEfficientNet
BranchNet = LightweightBranch
