# 基于超像素的显著性检测

本仓库使用 PyTorch 实现了多分支网络，用于超像素级显著性检测。

## 安装

安装依赖：

```bash
pip install -r requirements.txt
```

## 准备数据集（小规模 + 中规模）

```python
# 自动下载（可选加 --extract 解压）：
python scripts/download_datasets.py

# 将 MSRA-B 和 DUTS-TE 的压缩包下载到 data/raw 后，运行：
python scripts/prepare_datasets.py
```

## 训练

```python
python train.py --config configs/default.yaml
```

## 推理

```python
python infer.py --config configs/default.yaml --checkpoint outputs/train_*/model_final.pt --split test
```

## 评估

```python
python evaluate.py --config configs/default.yaml --checkpoint outputs/train_*/model_final.pt --split test
```

## 配置说明（configs/default.yaml）

- `seed`：随机种子。
- `paths`：数据与输出路径（train/val/test 的 images 与 masks，以及 `output_dir`）。
- `slic`：SLIC 超像素参数（数量、紧致度、平滑、迭代次数、起始标签）。
- `labels`：超像素标签阈值（`tau_pos`/`tau_neg`，其余为忽略）。
- `masking`：N-ring 邻域范围。
- `model`：网络结构参数（输入尺寸、特征维度、融合层隐藏维度）。
- `train`：训练超参数（batch size、学习率、权重衰减、训练轮数、日志与保存间隔）。
- `inference`：推理设置（多尺度超像素列表、是否保存可视化、同图内 batch 大小）。
- `runtime`：运行时设置（设备、确定性开关）。
