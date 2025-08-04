#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""no unsloth量化版本的一键QLora微调（已支持 MoE 参数透传）"""

import subprocess
import sys
import os
import argparse


def main():
    """运行QLoRA微调的主函数（不使用 Unsloth）

    功能：
        - 解析命令行参数
        - 构建命令行参数
        - 执行训练命令并处理结果
    """
    parser = argparse.ArgumentParser(description="QLoRA微调脚本（no-unsloth）")

    # 基础与资源相关
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="HF 仓库ID，默认 Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="qwen3-30b-a3b-instruct",
        help="本地模型目录（默认：qwen3-30b-a3b-instruct）",
    )

    # 训练与QLoRA开关
    parser.add_argument(
        "--use_unsloth",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否使用unsloth (default: false)",
    )
    parser.add_argument(
        "--use_qlora",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否使用qlora (default: true)",
    )

    # 数据与输出
    parser.add_argument(
        "--data_path",
        type=str,
        default="training_data.jsonl",
        help="训练数据路径 (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help="验证数据文件路径，None表示不使用验证集",
    )
    parser.add_argument(
        "--max_samples", type=str, default=None, help="最大训练样本数，None表示用全部"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=str,
        default=None,
        help="最大验证样本数，None表示用全部",
    )
    parser.add_argument(
        "--model_max_length",
        type=str,
        default="2048",
        help="最大序列长度 (default: 2048)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetune/models/qwen3-30b-a3b-qlora",
        help="输出目录 (default: finetune/models/qwen3-30b-a3b-qlora)",
    )
    parser.add_argument("--seed", type=str, default="42", help="随机种子 (default: 42)")

    # 训练超参
    parser.add_argument(
        "--per_device_train_batch_size",
        type=str,
        default="1",
        help="每个设备的训练批次大小 (default: 1)",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=str,
        default="1",
        help="每个设备的验证批次大小 (default: 1)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=str,
        default="16",
        help="梯度累积步数 (default: 16)",
    )
    parser.add_argument(
        "--learning_rate", type=str, default="2e-4", help="学习率 (default: 2e-4)"
    )
    parser.add_argument(
        "--num_train_epochs", type=str, default="3", help="训练轮数 (default: 3)"
    )
    parser.add_argument(
        "--max_steps", type=str, default="-1", help="最大步数，-1不限制 (default: -1)"
    )

    # LoRA 参数
    parser.add_argument(
        "--lora_r", type=str, default="16", help="LoRA的秩 (default: 16)"
    )
    parser.add_argument(
        "--lora_alpha", type=str, default="32", help="LoRA的alpha值 (default: 32)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=str,
        default="0.05",
        help="LoRA的dropout率 (default: 0.05)",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="稠密模型 LoRA 目标模块（逗号分隔）",
    )
    parser.add_argument(
        "--weight_decay", type=str, default="0.0", help="权重衰减 (default: 0.0)"
    )

    # MoE 相关参数（与 finetune/qlora_qwen3.py 对齐）
    parser.add_argument(
        "--moe_enable",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否启用 MoE 注入逻辑",
    )
    parser.add_argument(
        "--moe_lora_scope",
        type=str,
        default="expert_only",
        choices=["expert_only", "router_only", "all"],
        help="LoRA 注入范围（默认 expert_only）",
    )
    parser.add_argument(
        "--moe_expert_patterns",
        type=str,
        default="experts.ffn.(gate_proj|up_proj|down_proj),layers.[0-9]+.mlp.experts.[0-9]+.(w1|w2|w3)",
        help="专家线性层模式，多个正则用逗号分隔（兼容 Qwen-MoE / Mixtral）",
    )
    parser.add_argument(
        "--moe_router_patterns",
        type=str,
        default="router.(gate|dense)",
        help="路由/门控线性层模式，多个正则逗号分隔",
    )
    parser.add_argument(
        "--moe_max_experts_lora",
        type=str,
        default="-1",
        help="每层最多注入 LoRA 的专家个数，-1 表示全部",
    )
    parser.add_argument(
        "--moe_dry_run",
        type=str,
        default="false",
        choices=["true", "false"],
        help="仅打印匹配到的模块并退出（Dry-Run）",
    )

    # 模型加载精度
    parser.add_argument(
        "--load_precision",
        type=str,
        default="fp16",
        choices=["int8", "int4", "fp16"],
        help="模型加载精度：int8、int4 或 fp16 (default: fp16)",
    )

    # 其余训练设置
    parser.add_argument(
        "--logging_steps",
        type=str,
        default="1",
        help="日志记录步数，设置为1以每step输出loss",
    )
    parser.add_argument("--eval_steps", type=str, default="50", help="验证间隔步数")
    parser.add_argument("--save_steps", type=str, default="200", help="保存模型步数")
    parser.add_argument("--save_total_limit", type=str, default="2", help="最多保存数")
    parser.add_argument("--warmup_ratio", type=str, default="0.05", help="预热比例")
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='从指定的检查点恢复训练，可以是本地路径或"latest"',
    )
    parser.add_argument(
        "--no-gradient_checkpointing",
        action="store_true",
        help="不使用梯度检查点 (default: 使用)",
    )
    parser.add_argument(
        "--no-merge_and_save",
        action="store_true",
        help="不合并并保存模型 (default: 合并并保存)",
    )
    parser.add_argument(
        "--fp16",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否使用fp16 (default: true)",
    )
    parser.add_argument("--optim", type=str, default="adamw_torch_fused", help="优化器")
    parser.add_argument(
        "--dataloader_pin_memory",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否固定数据加载器内存 (default: false)",
    )
    parser.add_argument(
        "--dataloader_num_workers", type=str, default="0", help="DataLoader 工作线程数"
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=str,
        default="2",
        help="DataLoader 预取因子 (default: 2)",
    )

    args = parser.parse_args()

    env = os.environ.copy()
    env.update({})

    # 清理环境变量中可能存在的triton相关设置
    for key in list(env.keys()):
        if "triton" in key.lower():
            env.pop(key, None)

    # 构建命令行参数（将新增参数透传给 finetune/qlora_qwen3.py）
    cmd = [
        sys.executable,
        "finetune/qlora_qwen3.py",
        "--repo_id",
        args.repo_id,
        "--local_dir",
        args.local_dir,
        "--use_unsloth",
        args.use_unsloth,
        "--use_qlora",
        args.use_qlora,
        "--data_path",
        args.data_path,
        "--output_dir",
        args.output_dir,
        "--per_device_train_batch_size",
        args.per_device_train_batch_size,
        "--per_device_eval_batch_size",
        args.per_device_eval_batch_size,
        "--gradient_accumulation_steps",
        args.gradient_accumulation_steps,
        "--learning_rate",
        args.learning_rate,
        "--num_train_epochs",
        args.num_train_epochs,
        "--max_steps",
        args.max_steps,
        "--lora_r",
        args.lora_r,
        "--lora_alpha",
        args.lora_alpha,
        "--lora_dropout",
        args.lora_dropout,
        "--target_modules",
        args.target_modules,
        "--weight_decay",
        args.weight_decay,
        "--load_precision",
        args.load_precision,
        "--logging_steps",
        args.logging_steps,
        "--eval_steps",
        args.eval_steps,
        "--save_steps",
        args.save_steps,
        "--save_total_limit",
        args.save_total_limit,
        "--warmup_ratio",
        args.warmup_ratio,
        "--lr_scheduler_type",
        args.lr_scheduler_type,
        "--seed",
        args.seed,
        "--model_max_length",
        args.model_max_length,
        # MoE 相关
        "--moe_enable",
        args.moe_enable,
        "--moe_lora_scope",
        args.moe_lora_scope,
        "--moe_expert_patterns",
        args.moe_expert_patterns,
        "--moe_router_patterns",
        args.moe_router_patterns,
        "--moe_max_experts_lora",
        args.moe_max_experts_lora,
        "--moe_dry_run",
        args.moe_dry_run,
    ]

    # 添加可选参数（仅当值不为None时）
    if args.eval_data_path is not None:
        cmd.extend(["--eval_data_path", args.eval_data_path])
    if args.max_samples is not None:
        cmd.extend(["--max_samples", args.max_samples])
    if args.max_eval_samples is not None:
        cmd.extend(["--max_eval_samples", args.max_eval_samples])
    if args.resume_from_checkpoint is not None:
        cmd.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])

    if not args.no_gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if not args.no_merge_and_save:
        cmd.append("--merge_and_save")

    cmd.extend(
        [
            "--fp16",
            args.fp16,
            "--optim",
            args.optim,
            "--dataloader_pin_memory",
            args.dataloader_pin_memory,
            "--dataloader_num_workers",
            args.dataloader_num_workers,
            "--dataloader_prefetch_factor",
            args.dataloader_prefetch_factor,
        ]
    )

    print("启动QLoRA微调(no-unsloth)...")
    print("命令:", " ".join(cmd))
    print("环境变量(关键项):")
    for k, v in env.items():
        if k.startswith(("TORCH", "CUDA", "TORCHDYNAMO")):
            print(f"  {k}={v}")
    print()

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
