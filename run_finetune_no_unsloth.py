

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' no unsloth量化版本的一键QLora微调'''

import subprocess
import sys
import os
import argparse


def main():
    """运行QLoRA微调的主函数

    功能：
        - 解析命令行参数
        - 构建命令行参数
        - 执行训练命令并处理结果
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='QLoRA微调脚本')
    
    # 添加命令行参数
    parser.add_argument('--use_unsloth', type=str, default='false', choices=['true', 'false'],
                        help='是否使用unsloth (default: false)')
    parser.add_argument('--use_qlora', type=str, default='true', choices=['true', 'false'],
                        help='是否使用qlora (default: true)')
    parser.add_argument('--data_path', type=str, default='training_data.jsonl',
                        help='训练数据路径 (default: training_data.jsonl)')
    parser.add_argument('--output_dir', type=str, default='finetune/models/qwen3-8b-qlora',
                        help='输出目录 (default: finetune/models/qwen3-8b-qlora)')
    parser.add_argument('--per_device_train_batch_size', type=str, default='1',
                        help='每个设备的训练批次大小 (default: 1)')
    parser.add_argument('--gradient_accumulation_steps', type=str, default='16',
                        help='梯度累积步数 (default: 16)')
    parser.add_argument('--learning_rate', type=str, default='2e-4',
                        help='学习率 (default: 2e-4)')
    parser.add_argument('--num_train_epochs', type=str, default='3',
                        help='训练轮数 (default: 3)')
    parser.add_argument('--lora_r', type=str, default='16',
                        help='LoRA的秩 (default: 16)')
    parser.add_argument('--lora_alpha', type=str, default='32',
                        help='LoRA的alpha值 (default: 32)')
    parser.add_argument('--lora_dropout', type=str, default='0.05',
                        help='LoRA的dropout率 (default: 0.05)')
    parser.add_argument('--logging_steps', type=str, default='20',
                        help='日志记录步数 (default: 20)')
    parser.add_argument('--local_dir', type=str, default='qwen3-8b-base',
                        help='本地模型目录 (default: qwen3-8b-base)')
    parser.add_argument('--save_steps', type=str, default='200',
                        help='保存模型步数 (default: 200)')
    parser.add_argument('--warmup_ratio', type=str, default='0.05',
                        help='预热比例 (default: 0.05)')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                        help='学习率调度器类型 (default: cosine)')
    parser.add_argument('--no-gradient_checkpointing', action='store_true',
                        help='不使用梯度检查点 (default: 使用)')
    parser.add_argument('--no-merge_and_save', action='store_true',
                        help='不合并并保存模型 (default: 合并并保存)')
    parser.add_argument('--fp16', type=str, default='true', choices=['true', 'false'],
                        help='是否使用fp16 (default: true)')
    parser.add_argument('--optim', type=str, default='adamw_torch_fused',
                        help='优化器 (default: adamw_torch_fused)')
    parser.add_argument('--dataloader_pin_memory', type=str, default='false', choices=['true', 'false'],
                        help='是否固定数据加载器内存 (default: false)')
    parser.add_argument('--dataloader_num_workers', type=str, default='0',
                        help='数据加载器工作线程数 (default: 0)')
    
    # 解析命令行参数
    args = parser.parse_args()

    env = os.environ.copy()
    env.update({
        "TORCH_LOGS": "",
        "TORCHDYNAMO_DISABLE": "1",
        "TORCH_COMPILE_DISABLE": "1",
        "CUDA_LAUNCH_BLOCKING": "0",  # 关闭调试模式以提高性能
        "TRITON_DISABLE": "1",  # 禁用Triton
        "TORCH_USE_TRITON": "0",  # 不使用Triton
        "PYTORCH_DISABLE_TRITON": "1",  # 禁用PyTorch中的Triton
        "TF_ENABLE_ONEDNN_OPTS": "0",  # 禁用TensorFlow oneDNN自定义操作以减少警告
        "TF_CPP_MIN_LOG_LEVEL": "2",  # 减少TensorFlow警告信息
    })

    # 构建命令行参数
    cmd = [
        sys.executable,
        "finetune/qlora_qwen3.py",
        "--use_unsloth", args.use_unsloth,
        "--use_qlora", args.use_qlora,
        "--data_path", args.data_path,
        "--output_dir", args.output_dir,
        "--per_device_train_batch_size", args.per_device_train_batch_size,
        "--gradient_accumulation_steps", args.gradient_accumulation_steps,
        "--learning_rate", args.learning_rate,
        "--num_train_epochs", args.num_train_epochs,
        "--lora_r", args.lora_r,
        "--lora_alpha", args.lora_alpha,
        "--lora_dropout", args.lora_dropout,
        "--logging_steps", args.logging_steps,
        "--local_dir", args.local_dir,
        "--save_steps", args.save_steps,
        "--warmup_ratio", args.warmup_ratio,
        "--lr_scheduler_type", args.lr_scheduler_type,
    ]
    
    # 添加标志参数
    if not args.no_gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if not args.no_merge_and_save:
        cmd.append("--merge_and_save")
    
    # 添加剩余参数
    cmd.extend([
        "--fp16", args.fp16,
        "--optim", args.optim,
        "--dataloader_pin_memory", args.dataloader_pin_memory,
        "--dataloader_num_workers", args.dataloader_num_workers
    ])


    print("启动QLoRA微调(Windows兼容模式)...")
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