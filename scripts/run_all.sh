#!/bin/bash

# 设置默认设备
DEVICE="cuda"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --device)
        DEVICE="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        echo "Usage: bash scripts/run_all.sh --device [cuda|cpu]"
        exit 1
        ;;
    esac
done

# 设置Python路径
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# 运行完整流程
python src/run_full_pipeline.py --device $DEVICE
