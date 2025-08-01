#!/usr/bin/env python3
"""
Windows 兼容性启动脚本
专门为 Windows 平台优化，避免 Triton/torch.compile 相关问题
"""

import subprocess
import sys
import os

def main():
    # 设置环境变量以避免 Windows 兼容性问题
    env = os.environ.copy()
    env.update({
        "TORCH_LOGS": "",
        "TORCHDYNAMO_DISABLE": "1", 
        "TORCH_COMPILE_DISABLE": "1",
        "CUDA_LAUNCH_BLOCKING": "0",  # 关闭调试模式以提高性能
        "TRITON_DISABLE": "1",  # 禁用 Triton
        "TORCH_USE_TRITON": "0",  # 不使用 Triton
        "PYTORCH_DISABLE_TRITON": "1",  # 禁用 PyTorch 中的 Triton
        "TF_ENABLE_ONEDNN_OPTS": "0",  # 禁用TensorFlow oneDNN自定义操作以减少警告
        "TF_CPP_MIN_LOG_LEVEL": "2",  # 减少TensorFlow警告信息
    })
    
    # 构建命令行参数
    cmd = [
        sys.executable, 
        "finetune/qlora_qwen3.py",
        "--use_unsloth", "false",  # 默认禁用 Unsloth 避免 Triton 问题
        "--use_qlora", "true",
        "--data_path", "training_data.jsonl",
        "--output_dir", "finetune/models/qwen3-8b-qlora",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16", 
        "--learning_rate", "2e-4",
        "--num_train_epochs", "3",
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--lora_dropout", "0.05",
        "--logging_steps", "20",
        "--local_dir", "qwen3-8b-base",
        "--save_steps", "200",
        "--warmup_ratio", "0.05",
        "--lr_scheduler_type", "cosine",
        "--gradient_checkpointing",
        "--merge_and_save",
        "--fp16", "true",  # 启用自动混合精度 (AMP)
        "--optim", "adamw_torch_fused",  # 使用优化的优化器
        "--dataloader_pin_memory", "false",  # 禁用pin_memory以避免Windows兼容性问题
        "--dataloader_num_workers", "0"  # 设置为0以避免Windows兼容性问题
    ]
    
    # 检查是否尝试启用 unsloth
    use_unsloth = False
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--use_unsloth" and i < len(sys.argv) - 1:
            use_unsloth = sys.argv[i + 1].lower() == "true"
            break
    
    # 如果尝试启用 unsloth，在 Windows 上给出警告
    if use_unsloth:
        print("警告: 在 Windows 上使用 Unsloth 可能会导致兼容性问题，将自动回退到普通 4bit 量化")
        # 在命令中确保禁用 unsloth
        for i, arg in enumerate(cmd):
            if arg == "--use_unsloth":
                cmd[i + 1] = "false"
    
    print("启动 QLoRA 微调 (Windows 兼任模式)...")
    print("命令:", " ".join(cmd))
    print("环境变量:")
    for k, v in env.items():
        if k.startswith(("TORCH", "CUDA", "TORCHDYNAMO")):
            print(f"  {k}={v}")
    print()
    
    # 执行训练
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print("训练完成!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"训练失败，退出码: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("训练被用户中断")
        return 1

if __name__ == "__main__":
    sys.exit(main())