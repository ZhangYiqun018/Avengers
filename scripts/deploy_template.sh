#!/bin/bash
# 环境变量设置
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_ATTENTION_BACKEND=XFORMERS
export TORCH_USE_CUDA_DSA=1
# export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# 模型路径
MODEL=/path/to/your/model

# 日志目录
ROOT="logs"
mkdir -p vllm_logs/$ROOT

# 端口设置
PORT=8000

# GPU设置
GPU_COUNTS=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)

echo "Starting API server on port $PORT..."

# 通用参数
COMMON_ARGS="--model $MODEL \
    --trust-remote-code \
    --seed 42 \
    --enforce-eager \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.93 \
    --tensor-parallel-size $GPU_COUNTS \
    --served-model-name your-model-name"

# 可选：添加特定功能
# --enable-prefix-caching \
# --enable-chunked-prefill \
# --enable-auto-tool-choice \
# --tool-call-parser hermes \

# 启动服务器
python -m vllm.entrypoints.openai.api_server \
    $COMMON_ARGS \
    --port $PORT

echo "API server started on port $PORT" 