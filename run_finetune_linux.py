#!/usr/bin/env python3
"""
Linux 兼容性启动脚本
专门为 Linux 平台优化，支持 Unsloth 加速
"""

import subprocess
import sys
import os

def main():
    # 设置环境变量以优化 Linux 性能
    env = os.environ.copy()
    env.update({
        "TORCH_LOGS": "",
        "CUDA_LAUNCH_BLOCKING": "0",  # 不阻塞 CUDA
        "TRITON_DISABLE": "0",  # 启用 Triton
        "TORCH_USE_TRITON": "1",  # 使用 Triton
        "PYTORCH_DISABLE_TRITON": "0",  # 不禁用 PyTorch 中的 Triton
    })
    
    # 构建命令行参数
    cmd = [
        sys.executable, 
        "finetune/qlora_qwen3.py",
        "--use_unsloth", "true",  # 启用 Unsloth 加速
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
        "--save_steps", "200",
        "--warmup_ratio", "0.05",
        "--lr_scheduler_type", "cosine",
        "--gradient_checkpointing",  # 不需要额外的true值
        "--merge_and_save",  # 不需要额外的true值
        "--fp16", "true",  # 启用自动混合精度 (AMP)
        "--optim", "adamw_torch_fused",  # 使用优化的优化器
        "--dataloader_pin_memory", "false",  # 默认禁用pin_memory
        "--dataloader_num_workers", "0"  # 默认设置为0
    ]
    
    print("启动 QLoRA 微调 (Linux 兼容模式)..."),
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