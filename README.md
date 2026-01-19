# 基于超像素的显著性检测

本仓库使用 PyTorch 实现了多分支网络，用于超像素级显著性检测。

## 安装
安装依赖：
```
pip install -r requirements.txt
```

## 快速开始（虚拟数据）
```
python scripts/make_dummy_data.py
python train.py --config configs/default.yaml
```

## 准备数据集（小规模 + 中规模）
```
# 自动下载（可选加 --extract 解压）：
python scripts/download_datasets.py

# 将 MSRA-B 和 DUTS-TE 的压缩包下载到 data/raw 后，运行：
python scripts/prepare_datasets.py
```

## 推理
```
python infer.py --config configs/default.yaml --checkpoint outputs/train_*/model_final.pt --split test
```

## 评估
```
python evaluate.py --config configs/default.yaml --checkpoint outputs/train_*/model_final.pt --split test
```
