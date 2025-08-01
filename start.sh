#!/bin/bash

# NVIDIA 4090 优化启动脚本
# 为 Qwen3-8B QLoRA 微调进行完整优化

echo "=== NVIDIA 4090 优化启动脚本 ==="
echo "平台: Linux"
echo "优化: 启用所有可用优化选项"
echo

# 检查CUDA是否可用
echo "检查CUDA可用性..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未检测到NVIDIA GPU或CUDA驱动"
    exit 1
fi

# 显示GPU信息
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n 1
echo

# 设置CUDA相关环境变量以优化4090性能
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# 设置PyTorch相关优化
export TORCH_LOGS=""
export TORCHDYNAMO_DISABLE=0
export TORCH_COMPILE_DISABLE=0
export TRITON_DISABLE=0
export TORCH_USE_TRITON=1
export PYTORCH_DISABLE_TRITON=0

# 设置其他环境变量
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2

# 检查Unsloth是否已安装
echo "检查Unsloth安装状态..."
if python -c "import unsloth" &> /dev/null; then
    echo "Unsloth: 已安装"
    USE_UNSLOTH="true"
else
    echo "Unsloth: 未安装，将使用普通4bit量化"
    USE_UNSLOTH="false"
fi
echo

# 构建训练命令
CMD=(
    python finetune/qlora_qwen3.py
    --use_unsloth "$USE_UNSLOTH"  # 启用 Unsloth 加速
    --use_qlora true
    --data_path training_data.jsonl
    --output_dir finetune/models/qwen3-8b-qlora
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 16
    --learning_rate 2e-4
    --num_train_epochs 3
    --lora_r 16
    --lora_alpha 32
    --lora_dropout 0.05
    --logging_steps 20
    --save_steps 200
    --warmup_ratio 0.05
    --lr_scheduler_type cosine
    --gradient_checkpointing true
    --tf32 true
    --merge_and_save true
    --fp16 true  # 启用自动混合精度 (AMP)
    --optim adamw_torch_fused  # 使用优化的优化器
    --dataloader_pin_memory false  # 禁用pin_memory以获得更好的性能
    --dataloader_num_workers 0  # 设置为0以避免多进程问题
)

# 显示将要执行的命令
echo "将要执行的命令:"
echo "${CMD[@]}"
echo

echo "环境变量设置:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "  TORCHDYNAMO_DISABLE=$TORCHDYNAMO_DISABLE"
echo "  TRITON_DISABLE=$TRITON_DISABLE"
echo "  USE_UNSLOTH=$USE_UNSLOTH"
echo

echo "开始训练..."
echo

# 执行训练
"${CMD[@]}"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo
    echo "=== 训练成功完成 ==="
    echo "模型已保存到: finetune/models/qwen3-8b-qlora"
    if [ "$USE_UNSLOTH" = "true" ]; then
        echo "提示: 您已使用 Unsloth 加速训练，推理时也建议使用 Unsloth 加载模型以获得最佳性能"
    fi
else
    echo
    echo "=== 训练失败 ==="
    exit 1
fi