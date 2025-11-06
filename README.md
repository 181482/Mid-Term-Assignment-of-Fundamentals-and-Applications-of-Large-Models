# Mid-Term-Assignment-of-Fundamentals-and-Applications-of-Large-Models

本项目实现了一个完整的 Transformer 模型用于英中机器翻译，包含完整的训练、评估和可视化流程。

## 项目概述

- **任务**: 英语→中文机器翻译
- **数据集**: IWSLT17 (zh-en)
- **模型**: 标准 Transformer 架构 (Encoder-Decoder)
- **特点**: 
  - 使用 jieba 中文分词
  - 支持混合精度训练 (AMP)
  - 完整的评估指标 (Loss, PPL, BLEU)
  - 丰富的可视化功能
  - 支持从checkpoint继续训练
  - 消融实验支持

## 硬件要求

### 推荐配置
- **GPU**: NVIDIA GPU with 8GB+ VRAM (推荐 RTX 3060 或更好)
- **RAM**: 16GB+
- **存储**: 5GB+ 可用空间
- **CUDA**: 11.0+

### 最低配置
- **GPU**: NVIDIA GPU with 6GB VRAM (可能需要减小batch size)
- **RAM**: 8GB
- **CPU训练**: 可行但极慢(不推荐)

## 快速开始

### 1. 环境安装

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

数据集应放在 `datasets/zh-en/` 目录下:

### 一键运行完整流程

```bash
# 运行完整的训练+评估+可视化流程
python src/run_full_pipeline.py --device cuda

# 或使用脚本
bash scripts/run_all.sh --device cuda
```

### 单独运行各模块

```bash
# 1. 数据质量检查
python src/inspect_data.py

# 2. 查看数据样例
python src/show_samples.py

# 3. 训练模型
bash scripts/run.sh --mode train --device cuda

# 4. 消融实验
bash scripts/run.sh --mode ablation --device cuda

# 5. 生成翻译样例
python src/generate_samples.py

# 6. 生成可视化
python src/visualize_results.py
```

## 数据集信息

**IWSLT17 中英翻译数据集**

| 分割 | 句对数 | 描述 |
|------|--------|------|
| 训练集 | 231,240 | train.tags.zh-en.* |
| 验证集 | 877 | dev2010 |
| 测试集 | 4,674 | tst2010-2012 |
https://huggingface.co/datasets/IWSLT/iwslt2017

## 项目结构

```
.
├── src/
│   ├── models/
│   │   ├── attention.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── positional_encoding.py
│   │   └── transformer.py
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── trainer.py
│   │   └── visualization.py
│   ├── run_full_pipeline.py
│   ├── inspect_data.py
│   ├── show_samples.py
│   ├── visualize_results.py
│   └── config.py
├── scripts/
│   ├── run.sh
│   ├── run_all.sh
├── results/
├── requirements.txt
└── README.md
```
