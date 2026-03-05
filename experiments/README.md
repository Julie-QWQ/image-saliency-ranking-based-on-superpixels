# 对比实验脚本

本文件夹包含所有用于对比实验的脚本。

## 📁 脚本列表

### 1. `run_full_experiment.py` - 完整实验（推荐）

**功能**：训练 + 评估 + 可视化的完整流程

**输出**：
- `outputs/full_experiment_<timestamp>/`
  - `experiment_results.json` - 详细指标数据
  - `experiment_report.md` - 可读报告
  - `experiment.log` - 运行日志
  - `model_original.pt` - 原始模型
  - `model_improved.pt` - 改进模型
  - `visualizations/` - 预测图片
    - `original/` - 原始模型的预测
    - `improved/` - 改进模型的预测
    - `dss/` - DSS Baseline 的预测

**运行**：
```bash
# 默认：3 epochs，保存前 10 张预测图
python experiments/run_full_experiment.py --epochs 3 --save_vis 10

# 自定义：更多轮数，更多图片
python experiments/run_full_experiment.py --epochs 10 --save_vis 20

# 只保存指标，不保存图片
python experiments/run_full_experiment.py --epochs 3 --save_vis 0
```

**参数**：
- `--epochs N`: 训练轮数（默认: 3）
- `--output_dir PATH`: 输出目录（默认: `outputs/full_experiment`）
- `--save_vis N`: 每个模型保存的预测图片数量（默认: 10）

**预测图片格式**：
- `{image_name}_heat.png` - 热力图（JET 彩色映射）
- `{image_name}_overlay.png` - 叠加图（原始图 + 热力图）

---

### 2. `quick_eval.py` - 快速评估

**功能**：无需训练，快速验证架构改进

**输出**：
- `outputs/quick_eval_<timestamp>/`
  - `quick_eval_results.json` - 架构对比
  - `quick_eval_report.md` - 预期性能分析

**运行**：
```bash
python experiments/quick_eval.py
```

**特点**：
- ✅ 30 秒内完成
- ✅ 无需训练数据
- ✅ 预测性能提升

---

### 3. `train_improved_quick.py` - 单独训练改进模型

**功能**：只训练改进的超像素模型

**输出**：
- `outputs/train_improved_<timestamp>/`
  - `model_improved.pt` - 训练好的模型
  - `model_best.pt` - 最佳模型
  - 训练日志和曲线

**运行**：
```bash
python experiments/train_improved_quick.py
```

---

## 📊 输出说明

### 指标（Metrics）
所有实验都会计算以下指标：
- **MAE** (Mean Absolute Error) - 平均绝对误差，越低越好
- **IoU** (Intersection over Union) - 交并比，越高越好
- **F1 Score** - F1 分数，越高越好

### 可视化图片（Visualizations）
当 `--save_vis > 0` 时，会保存预测图片：
- **热力图**：显著性概率的彩色映射（0=蓝色，1=红色）
- **叠加图**：原始图像和热力图的融合（60% 原图 + 40% 热力图）

---

## 🎯 使用流程

### 完整验证流程

1. **快速验证**（30 秒）
   ```bash
   python experiments/quick_eval.py
   ```

2. **完整实验**（10-30 分钟）
   ```bash
   python experiments/run_full_experiment.py --epochs 5 --save_vis 10
   ```

3. **查看结果**
   ```bash
   # 查看报告
   cat outputs/full_experiment_<timestamp>/experiment_report.md

   # 查看预测图片
   ls outputs/full_experiment_<timestamp>/visualizations/
   ```

---

## 📝 对比的其他脚本

根目录下的核心脚本：
- `train.py` - 训练原始模型
- `evaluate.py` - 评估单个模型
- `infer.py` - 单张图像推理

本目录下的脚本：
- `compare.py` - DSS vs Superpixel 对比工具
- `run_full_experiment.py` - 完整实验流程
- `quick_eval.py` - 快速架构验证
- `train_improved_quick.py` - 快速训练改进模型
