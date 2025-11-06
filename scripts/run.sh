#!/bin/bash

# 创建数据目录
mkdir -p .data

# 设置默认参数
MODE="train"
CONFIG="config/default.yaml"
DEVICE="cuda"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --mode)
        MODE="$2"
        shift
        shift
        ;;
        --config)
        CONFIG="$2"
        shift
        shift
        ;;
        --device)
        DEVICE="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# 设置 Python 路径
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# 运行相应的模式
case $MODE in
    "train")
        python src/train.py --config $CONFIG --device $DEVICE
        ;;
    "ablation")
        python src/ablation.py --config $CONFIG --device $DEVICE
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac
