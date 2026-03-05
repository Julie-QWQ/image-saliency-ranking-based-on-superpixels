"""
DSS (Deeply Supervised Salient Object Detection) Baseline - 修复版
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class DSSBaseline(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        
        # 骨干网络 - 简化的 VGG-16，只用到 pool5
        self.backbone = self._make_backbone()
        
        # 短连接模块
        self.short_conn3 = nn.Conv2d(256, 64, kernel_size=1)   # pool3 -> 64
        self.short_conn4 = nn.Conv2d(512, 128, kernel_size=1)  # pool4 -> 128
        self.short_conn5 = nn.Conv2d(512, 256, kernel_size=1)  # pool5 -> 256
        
        # 批归一化
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        
        # 侧输出层
        self.side_output3 = nn.Conv2d(64, 1, kernel_size=1)
        self.side_output4 = nn.Conv2d(128, 1, kernel_size=1)
        self.side_output5 = nn.Conv2d(256, 1, kernel_size=1)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        if pretrained_path and Path(pretrained_path).exists():
            self.load_dss_weights(pretrained_path)
        else:
            self._load_vgg_weights()
    
    def _make_backbone(self):
        """创建 VGG-16 骨干网络"""
        layers = []
        
        # Block 1: conv1_2 + pool (64 channels)
        layers.extend([
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        ])
        
        # Block 2: conv2_2 + pool (128 channels)
        layers.extend([
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        ])
        
        # Block 3: conv3_3 + pool (256 channels)
        layers.extend([
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        ])
        
        # Block 4: conv4_3 + pool (512 channels)
        layers.extend([
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        ])
        
        # Block 5: conv5_3 + pool (512 channels)
        layers.extend([
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        ])
        
        return nn.Sequential(*layers)
    
    def _load_vgg_weights(self):
        try:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)
            vgg_dict = vgg.state_dict()
            model_dict = self.state_dict()
            
            pretrained_dict = {}
            for k, v in vgg_dict.items():
                if 'features' in k:
                    new_key = k.replace('features.', 'backbone.')
                    if new_key in model_dict and model_dict[new_key].shape == v.shape:
                        pretrained_dict[new_key] = v
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print("DSS: VGG-16 预训练权重加载成功")
        except Exception as e:
            print(f"DSS: 无法加载预训练权重: {e}，使用随机初始化")
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # 手动提取特征，确保正确的通道数
        pool_feats = []
        pool_count = 0
        
        for module in self.backbone:
            x = module(x)
            if isinstance(module, nn.MaxPool2d):
                pool_count += 1
                # 只保存 pool3, pool4, pool5
                if pool_count >= 3:
                    pool_feats.append(x)
        
        # pool_feats 现在是 [pool3(256), pool4(512), pool5(512)]
        
        # 短连接 + 侧输出
        sides = []
        
        # Pool3 分支
        feat3 = self.bn3(self.short_conn3(pool_feats[0]))
        sides.append(self.side_output3(feat3))
        
        # Pool4 分支
        feat4 = self.bn4(self.short_conn4(pool_feats[1]))
        sides.append(self.side_output4(feat4))
        
        # Pool5 分支
        feat5 = self.bn5(self.short_conn5(pool_feats[2]))
        sides.append(self.side_output5(feat5))
        
        # 上采样到输入尺寸
        up_sides = []
        for side in sides:
            up_sides.append(F.interpolate(side, size=input_size, mode='bilinear', align_corners=True))
        
        # 融合
        fused = torch.cat(up_sides, dim=1)
        saliency_map = self.fusion(fused)
        
        return saliency_map

    def load_dss_weights(self, path):
        """加载 DSS 预训练权重"""
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            print(f"DSS: 权重加载成功: {path}")
        except Exception as e:
            print(f"DSS: 警告 - 无法加载权重 {path}: {e}")

    def predict(self, image, input_size=224, device='cuda'):
        """
        推理接口，与项目风格一致
        
        Args:
            image: RGB 图像 (H, W, 3), numpy array
            input_size: 网络输入尺寸
            device: 计算设备
        
        Returns:
            heatmap: 显著性热力图 (H, W), numpy array
        """
        from .utils import resize_image, to_tensor
        
        self.eval()
        with torch.no_grad():
            # 预处理
            x = resize_image(image, input_size)
            x = to_tensor(x).unsqueeze(0).to(device)
            
            # 推理
            saliency_map = self.forward(x)
            
            # 后处理
            saliency_map = torch.sigmoid(saliency_map)
            saliency_map = saliency_map.squeeze().cpu().numpy()
            
            # resize 回原始尺寸
            import cv2
            h, w = image.shape[:2]
            saliency_map = cv2.resize(saliency_map, (w, h))
        
        return saliency_map
