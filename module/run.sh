#!/bin/bash

# ===========================
# JAX/PYSCFAD 分子训练启动脚本
# ===========================

# 选择 GPU（如需多卡可调整CUDA_VISIBLE_DEVICES，也可不用export直接在命令前加 CUDA_VISIBLE_DEVICES=3）
export CUDA_VISIBLE_DEVICES=3

# 线程等资源环境变量（按硬件实际情况调整）
export OMP_NUM_THREADS=96   
export MKL_NUM_THREADS=96
export OPENBLAS_NUM_THREADS=96
export NUMEXPR_NUM_THREADS=96

# JAX 平台优先用GPU
export JAX_PLATFORM_NAME=gpu

# PySCF 静音
export PYSCF_VERBOSE=0
export PYSCF_PRINT=0

# 日志目录准备
mkdir -p logs

# 当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_deepsrsh_${TIMESTAMP}.log"

echo "启动 DeepRSH 水分子训练..."
echo "训练日志文件: $LOG_FILE"

nohup python train.py > $LOG_FILE 2>&1 &

# 获取训练进程PID
TRAINING_PID=$!

echo "训练PID: $TRAINING_PID"
echo "查看训练日志: tail -f $LOG_FILE"
echo "停止训练: kill $TRAINING_PID"

# 保存PID到文件
echo $TRAINING_PID > training.pid