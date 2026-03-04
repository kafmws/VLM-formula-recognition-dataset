#!/bin/bash

# 创建日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/[SFT]internvl3_1b_${TIMESTAMP}.log"

# 设置环境变量
# export ENABLE_AUDIO_OUTPUT=False
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export SWANLAB_API_KEY=obxN0xQlG4pz1QfQjM8AB


# 设置随机端口号，避免端口冲突
export MASTER_PORT=$((10000 + RANDOM % 50000))


# 先打印启动信息
echo "Starting training..."
echo "Log file: $LOG_FILE"
echo "Using port: $MASTER_PORT"

# 启动训练并获取PID
DEFAULT_PYTHON_FILE="/root/dev/baseline/ft.py"

if [ $# -eq 0 ]; then
    echo "执行默认文件: $DEFAULT_PYTHON_FILE"
    nohup python "$DEFAULT_PYTHON_FILE" "$@" > "$LOG_FILE" 2>&1 &
else
    PYTHON_FILE="$1"
    shift  # 移除第一个参数（文件名），剩下的作为 Python 脚本的参数
    echo "执行文件: $PYTHON_FILE，参数: $@"
    nohup python "$PYTHON_FILE" "$@" > "$LOG_FILE" 2>&1 &
fi

# 获取PID并等待一下确保进程启动
TRAIN_PID=$!
sleep 2

# 检查进程是否还在运行
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "Training started successfully with PID $TRAIN_PID"
    echo "To view logs in real-time, use:"
    echo "tail -f $LOG_FILE"
    echo ""
    echo "To stop training, use:"
    echo "kill $TRAIN_PID"
else
    echo "Failed to start training process"
    echo "Check log file for errors: $LOG_FILE"
fi

swift export \
    --adapters /root/dev/swift_output/all_fc_lora_ft5/v2-20260227-221139/checkpoint-4500 \
    --merge_lora true && python /root/dev/baseline/upload_model.py