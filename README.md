# 基于超像素的显著性检测

本仓库使用 PyTorch 实现了多分支网络，用于超像素级显著性检测。

## 安装

安装依赖：

```bash
pip install -r requirements.txt
```

## 准备数据集（小规模 + 中规模）

```bash
# 自动下载（可选加 --extract 解压）：
python scripts/download_datasets.py

# 将 MSRA-B 和 DUTS-TE 的压缩包下载到 data/raw 后，运行：
python scripts/prepare_datasets.py
```

## 训练

```bash
python train.py --config configs/default.yaml
```

## 断点续训

```bash
python train.py --config configs/default.yaml --resume outputs/train_*/checkpoint_epoch_*.pt
```

说明：`--resume` 会加载 checkpoint 中的模型与优化器状态，从上次 epoch 继续训练。

## 推理

```bash
python infer.py --config configs/default.yaml --checkpoint outputs/train_*/model_final.pt --split test
```

## 评估

```bash
python evaluate.py --config configs/default.yaml --checkpoint outputs/train_*/model_final.pt --split test
```

## 可选参数示例

```bash
# 只训练不验证
python train.py --config configs/default.yaml --no_val

# 每 N 个 epoch 验证一次（在配置中设置）
# train:
#   val_interval: 20

# 子集评估（加速）
python evaluate.py --config configs/default.yaml --checkpoint outputs/train_*/model_final.pt --split test --max_images 200
```

## 配置说明（configs/default.yaml）

- `seed`：随机种子。
- `paths`：数据与输出路径（train/val/test 的 images 与 masks，以及 `output_dir`）。
- `paths.cache_dir`：超像素/样本索引缓存目录（训练/评估/推理都会复用）。
- `slic`：SLIC 超像素参数（数量、紧致度、平滑、迭代次数、起始标签）。
- `labels`：超像素标签阈值（`tau_pos`/`tau_neg`，其余为忽略）。
- `masking`：N-ring 邻域范围。
- `model`：网络结构参数（输入尺寸、特征维度、融合层隐藏维度）。
- `train`：训练超参数（batch size、学习率、权重衰减、训练轮数、日志/保存/验证间隔）。
- `train.val_interval`：每隔 N 个 epoch 才进行一次验证。
- `inference`：推理设置（多尺度超像素列表、是否保存可视化、同图内 batch 大小）。
- `runtime`：运行时设置（设备、确定性开关）。
