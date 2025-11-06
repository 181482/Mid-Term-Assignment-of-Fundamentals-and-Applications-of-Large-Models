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

## 数据集信息

**IWSLT17 中英翻译数据集**

https://huggingface.co/datasets/IWSLT/iwslt2017

| 分割 | 句对数 | 描述 |
|------|--------|------|
| 训练集 | 231,240 | train.tags.zh-en.* |
| 验证集 | 877 | dev2010 |
| 测试集 | 4,674 | tst2010-2012 |

## 快速开始

### 1. 环境安装

# 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 数据准备

数据集应放在 `datasets/zh-en/` 目录下:
```
datasets/zh-en/
├── train.tags.zh-en.en
├── train.tags.zh-en.zh
├── IWSLT17.TED.dev2010.zh-en.en.xml
├── IWSLT17.TED.dev2010.zh-en.zh.xml
├── IWSLT17.TED.tst2010.zh-en.en.xml
├── IWSLT17.TED.tst2010.zh-en.zh.xml
├── IWSLT17.TED.tst2011.zh-en.en.xml
├── IWSLT17.TED.tst2011.zh-en.zh.xml
├── IWSLT17.TED.tst2012.zh-en.en.xml
└── IWSLT17.TED.tst2012.zh-en.zh.xml
```

### 一键运行完整流程

```bash
# 运行完整的训练+评估+可视化流程
# CUDA (推荐)
python src/run_full_pipeline.py --device cuda

# CPU (不推荐,极慢)
python src/run_full_pipeline.py --device cpu

# 或使用脚本
bash scripts/run_all.sh --device cuda
```

### 单独运行各模块

```bash
# 1. 数据质量检查
python src/inspect_data.py

# 2. 查看数据样例
python src/show_samples.py

# 3. 训练模型（固定随机种子42）
bash scripts/run.sh --mode train --device cuda

# 4. 消融实验
bash scripts/run.sh --mode ablation --device cuda

# 5. 生成翻译样例
python src/generate_samples.py

# 6. 生成可视化
python src/visualize_results.py

# 7. 自动从最新checkpoint继续
python src/continue_training.py --epochs 50 --device cuda

# 8. 清理旧数据
rm -rf checkpoints/*
rm -rf results/*
rm -rf wandb/*
rm -rf datasets/zh-en/cache/*
```

## 项目结构

```
transformer/
├── src/
│   ├── models/
│   │   ├── attention.py          # Multi-Head Attention
│   │   ├── encoder.py             # Transformer Encoder
│   │   ├── decoder.py             # Transformer Decoder
│   │   ├── positional_encoding.py # 位置编码
│   │   └── transformer.py         # 完整模型
│   ├── utils/
│   │   ├── data_loader.py         # 数据加载器(支持缓存)
│   │   ├── trainer.py             # 训练器
│   │   └── visualization.py       # 可视化工具
│   ├── config.py                  # 配置文件
│   ├── train.py                   # 训练脚本
│   ├── continue_training.py       # 继续训练脚本
│   ├── ablation.py                # 消融实验
│   ├── generate_samples.py        # 生成翻译样例
│   ├── inspect_data.py            # 数据检查
│   ├── show_samples.py            # 显示数据样例
│   ├── visualize_results.py       # 综合可视化
│   ├── run_full_pipeline.py       # 完整流程脚本
│   └── analyze_ablation.py        # 消融实验分析
├── scripts/
│   ├── run.sh                     # 主运行脚本
│   └── run_all.sh                 # 完整流程脚本
├── datasets/zh-en/                # 数据集目录
│   └── cache/                     # 数据缓存
├── checkpoints/                   # 模型检查点
├── results/                       # 结果输出
│   ├── visualizations/            # 可视化图表
│   ├── ablation/                  # 消融实验结果
│   ├── translation_samples.md     # 翻译样例
│   └── metrics.csv                # 训练指标
├── requirements.txt               # 依赖列表
└── README.md                      # 本文件
```
