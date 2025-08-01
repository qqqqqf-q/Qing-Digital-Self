#!/usr/bin/env python3
"""
 QLoRA 微调脚本
"""

import argparse
import json
import math
import os
import platform
import sys
from modelspace import snapshot_download
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# 移除可能存在的 TORCH_LOGS 环境变量
os.environ.pop("TORCH_LOGS", None)

# 设置默认环境变量 - 可能在导入 torch 之前被覆盖
# 禁用各种可能导致兼容性问题的优化
os.environ.update({
    "TORCHDYNAMO_DISABLE": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "TRITON_DISABLE": "1",
    "TORCH_USE_TRITON": "0",
    "PYTORCH_DISABLE_TRITON": "1",
    "TF_ENABLE_ONEDNN_OPTS": "0",  # 禁用TensorFlow oneDNN自定义操作以减少警告
    "TF_CPP_MIN_LOG_LEVEL": "2",   # 减少TensorFlow警告信息
})

import torch
from torch.utils.data import Dataset


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

def print_rank0(*args, **kwargs):
    """仅在主进程打印信息"""
    print(*args, **kwargs)

def log_gpu_memory_usage(step_name: str) -> None:
    """记录GPU显存使用情况

    Args:
        step_name: 当前步骤名称，用于日志标识
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print_rank0(
            f"{step_name} - GPU显存使用: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB"
        )
    else:
        print_rank0(f"{step_name} - CUDA不可用")

def safe_dtype_preference() -> Dict[str, Union[torch.dtype, bool]]:
    """选择合适的数据类型

    根据GPU是否支持bfloat16来决定使用哪种数据类型

    Returns:
        包含数据类型配置的字典
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"torch_dtype": torch.bfloat16, "bf16": True, "fp16": False}
    else:
        return {"torch_dtype": torch.float16, "bf16": False, "fp16": True}

class JsonlSFTDataset(Dataset):
    """JSONL 格式的监督微调数据集

    支持多种数据格式：
    1. messages 格式: 包含角色和内容的对话列表
    2. 指令格式: 包含 instruction、input 和 output
    3. 简单输入输出格式: 包含 input 和 output
    4. 纯文本格式: 包含 text 字段
    """
    def __init__(self, path: str, eos_token: str, max_samples: Optional[int] = None):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = self._convert_example(obj, eos_token)
                if text:
                    self.samples.append(text)
                if max_samples and len(self.samples) >= max_samples:
                    break

    def _convert_example(self, obj: Dict[str, Any], eos_token: str) -> Optional[str]:
        # 处理 messages 格式
        if "messages" in obj and isinstance(obj["messages"], list):
            text = self._join_messages(obj["messages"])
            return text + eos_token if eos_token else text

        # 处理指令格式
        if "instruction" in obj and "output" in obj:
            ins = obj["instruction"]
            inp = obj.get("input", "")
            out = obj["output"]
            if inp:
                text = f"Instruction: {ins}\nInput: {inp}\nAnswer: {out}"
            else:
                text = f"Instruction: {ins}\nAnswer: {out}"
            return text + eos_token if eos_token else text

        # 处理简单的 input/output 格式
        if "input" in obj and "output" in obj:
            text = f"Input: {obj['input']}\nAnswer: {obj['output']}"
            return text + eos_token if eos_token else text

        # 处理纯文本格式
        if "text" in obj:
            text = str(obj["text"])
            return text + eos_token if eos_token else text

        return None

    @staticmethod
    def _join_messages(msgs: List[Dict[str, str]]) -> str:
        """将消息列表转换为字符串

        Args:
            msgs: 包含消息的字典列表，每个字典应包含role和content

        Returns:
            格式化后的消息字符串
        """
        buf = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            buf.append(f"{role}: {content}")
        return "\n".join(buf)

    def __len__(self) -> int:
        """返回数据集样本数量"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """获取指定索引的样本

        Args:
            idx: 样本索引

        Returns:
            包含text字段的字典
        """
        return {"text": self.samples[idx]}


@dataclass
class CollatorForCausalLM:
    """因果语言模型的数据整理器

    用于将文本数据转换为模型训练所需的张量批次
    """
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """处理输入特征并返回模型训练所需的批次数据

        Args:
            features: 包含文本数据的特征列表

        Returns:
            包含input_ids、attention_mask和labels的张量字典
        """
        texts = [f["text"] for f in features]
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

def load_model_and_tokenizer(
    base_dir: str,
    use_qlora: bool = True,
    use_unsloth: bool = False,
    use_gradient_checkpointing: bool = True
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """加载模型和分词器，支持 Unsloth 和普通 8bit 量化

    Args:
        base_dir: 模型基础目录
        use_qlora: 是否使用QLoRA
        use_unsloth: 是否尝试使用Unsloth加速
        use_gradient_checkpointing: 是否使用梯度检查点

    Returns:
        模型和分词器的元组
    """
    print_rank0("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_dir,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 尝试使用 Unsloth 加载模型
    if use_unsloth:
        try:
            print_rank0("尝试使用 Unsloth 加载模型...")
            from unsloth import FastLanguageModel

            # 启用 Triton 和相关优化
            os.environ.update({
                "TORCHDYNAMO_DISABLE": "0",
                "TORCH_COMPILE_DISABLE": "0",
                "TRITON_DISABLE": "0",
                "TORCH_USE_TRITON": "1",
                "PYTORCH_DISABLE_TRITON": "0",
            })

            model, tokenizer = FastLanguageModel.from_pretrained(
                base_dir,
                dtype=None,
                load_in_8bit=True,
                trust_remote_code=True,
            )

            print_rank0("成功使用 Unsloth 加载模型")
            return model, tokenizer
        except Exception as e:
            print_rank0(f"Unsloth 加载失败: {e}")
            print_rank0("回退到普通 8bit 量化加载...")

    # 普通 8bit 量化加载
    print_rank0("加载模型 (8bit量化)...")
    dtype_kwargs = safe_dtype_preference()

    # 加载模型配置并设置return_dict=True
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        base_dir,
        trust_remote_code=True
    )
    config.return_dict = True

    # 强制使用 8-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    print_rank0(f"使用量化配置: 8bit量化")

    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=dtype_kwargs["torch_dtype"],
        config=config,  # 添加配置
    )

    # 准备模型进行训练
    if use_qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

        # 配置 LoRA
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "lm_head"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        print_rank0(f"QLoRA 配置: {lora_config}")
    else:
        # 非量化训练准备
        if use_gradient_checkpointing:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()

    print_rank0("模型已准备好进行训练")

    return model, tokenizer

def apply_lora(
    model: AutoModelForCausalLM,
    target_modules: List[str],
    r: int,
    alpha: int,
    dropout: float
) -> PeftModel:
    """应用 LoRA 适配器

    Args:
        model: 基础模型
        target_modules: 要应用LoRA的模块列表
        r: LoRA秩
        alpha: LoRA alpha参数
        dropout: LoRA dropout率

    Returns:
        应用了LoRA的模型
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return model

def merge_and_save(base_dir: str, output_dir: str) -> str:
    """合并 LoRA 权重并保存完整模型

    Args:
        base_dir: 基础模型目录
        output_dir: LoRA模型输出目录

    Returns:
        合并后模型的保存路径
    """
    print_rank0("合并 LoRA 权重...")

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载 PEFT 模型
    peft_model = PeftModel.from_pretrained(
        model,
        output_dir,
        device_map="auto"
    )

    # 合并权重
    merged_model = peft_model.merge_and_unload()

    # 保存合并后的模型
    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)

    # 保存分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_dir,
        use_fast=True,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(merged_dir)

    print_rank0(f"合并后的模型已保存到: {merged_dir}")
    return merged_dir

def parse_args() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        包含命令行参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description="简化版 QLoRA 微调 Qwen3-8B")

    # 模型相关
    parser.add_argument(
        "--repo_id", 
        type=str, 
        default="Qwen/Qwen3-8B-Base",
        help="基础模型的HuggingFace仓库ID (默认: Qwen/Qwen3-8B-Base)"
    )
    parser.add_argument(
        "--local_dir", 
        type=str, 
        default="qwen3-8b-base",
        help="本地模型存储目录 (默认: qwen3-8b-base)"
    )
    parser.add_argument(
        "--trust_remote_code", 
        action="store_true", 
        default=True,
        help="是否信任远程代码 (默认: True)"
    )
    parser.add_argument(
        "--use_unsloth", 
        type=lambda x: str(x).lower() == "true", 
        default=False, 
        help="是否使用 Unsloth 加速 (默认: False)"
    )
    parser.add_argument(
        "--use_qlora", 
        type=lambda x: str(x).lower() == "true", 
        default=True, 
        help="是否使用 8bit 量化 (默认: True)"
    )

    # 数据相关
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="training_data.jsonl",
        help="训练数据文件路径 (默认: training_data.jsonl)"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="最大训练样本数，None表示使用全部数据 (默认: None)"
    )
    parser.add_argument(
        "--model_max_length", 
        type=int, 
        default=2048,
        help="模型最大序列长度 (默认: 2048)"
    )

    # 训练相关
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="finetune/models/qwen3-8b-qlora",
        help="模型输出目录 (默认: finetune/models/qwen3-8b-qlora)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子 (默认: 42)"
    )

    # LoRA 参数
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16,
        help="LoRA秩 (默认: 16)"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32,
        help="LoRA alpha参数 (默认: 32)"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.05,
        help="LoRA dropout率 (默认: 0.05)"
    )
    parser.add_argument(
        "--target_modules", 
        type=str, 
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="应用LoRA的目标模块 (默认: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj)"
    )

    # 训练超参数
    parser.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=1,
        help="每个设备的训练批次大小 (默认: 1)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=16,
        help="梯度累积步数 (默认: 16)"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-4,
        help="学习率 (默认: 2e-4)"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0,
        help="权重衰减 (默认: 0.0)"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=float, 
        default=3.0,
        help="训练轮数 (默认: 3.0)"
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=-1,
        help="最大训练步数，-1表示不限制 (默认: -1)"
    )
    parser.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=0.05,
        help="学习率预热比例 (默认: 0.05)"
    )
    parser.add_argument(
        "--lr_scheduler_type", 
        type=str, 
        default="cosine",
        help="学习率调度器类型 (默认: cosine)"
    )

    # 其他设置
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=20,
        help="日志记录步数间隔 (默认: 20)"
    )
    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=200,
        help="模型保存步数间隔 (默认: 200)"
    )
    parser.add_argument(
        "--save_total_limit", 
        type=int, 
        default=2,
        help="最多保存模型数量 (默认: 2)"
    )
    parser.add_argument(
        "--gradient_checkpointing", 
        action="store_true", 
        default=True,
        help="是否使用梯度检查点 (默认: True)"
    )
    parser.add_argument(
        "--merge_and_save", 
        action="store_true", 
        default=True,
        help="训练完成后是否合并LoRA权重并保存完整模型 (默认: True)"
    )
    parser.add_argument(
        "--fp16", 
        type=lambda x: str(x).lower() == "true", 
        default=True, 
        help="是否使用FP16混合精度训练 (默认: True)"
    )
    parser.add_argument(
        "--optim", 
        type=str, 
        default="adamw_torch_fused", 
        help="优化器类型 (默认: adamw_torch_fused)"
    )
    parser.add_argument(
        "--dataloader_pin_memory", 
        type=lambda x: str(x).lower() == "true", 
        default=False, 
        help="是否在数据加载器中使用pin_memory (默认: False)"
    )
    parser.add_argument(
        "--dataloader_num_workers", 
        type=int, 
        default=0, 
        help="数据加载器的工作线程数 (默认: 0)"
    )

    return parser.parse_args()

def main() -> None:
    """主函数，执行QLoRA微调流程"""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print_rank0("=== QLoRA 微调开始 ===")
    print_rank0(f"平台: {platform.system()}")
    print_rank0(f"配置: 使用Unsloth={args.use_unsloth}, 使用8bit量化={args.use_qlora}")

    # 禁用动态编译和相关优化
    try:
        torch._dynamo.config.disable = True
        torch._dynamo.config.suppress_errors = True
    except Exception as e:
        print_rank0(f"禁用动态编译失败: {e}")

    # 设置精度 - 使用新的TF32控制API
    try:
        # 使用新的API设置TF32行为
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception as e:
        print_rank0(f"设置TF32行为失败: {e}")


    # 步骤 1: 下载模型
    print_rank0("步骤 1/5: 下载或复用基础模型...")
    log_gpu_memory_usage("开始下载模型")

    # 检查本地是否已存在模型
    if os.path.exists(args.local_dir) and os.listdir(args.local_dir):
        print_rank0(f"复用本地模型: {args.local_dir}")
        base_dir = args.local_dir
    else:
        print_rank0(f"从远程下载模型到: {args.local_dir}")
        base_dir = snapshot_download(
            model_id=args.repo_id,
            local_dir=args.local_dir,
        )
    print_rank0(f"基础模型目录: {base_dir}")
    log_gpu_memory_usage("模型下载完成")


    # 步骤 2: 加载模型和分词器
    print_rank0("步骤 2/5: 加载模型和分词器...")
    log_gpu_memory_usage("开始加载模型")
    model, tokenizer = load_model_and_tokenizer(
        base_dir,
        use_qlora=args.use_qlora,
        use_unsloth=args.use_unsloth,
        use_gradient_checkpointing=args.gradient_checkpointing
    )
    tokenizer.model_max_length = args.model_max_length
    log_gpu_memory_usage("模型加载完成")


    # 步骤 3: 准备数据
    print_rank0("步骤 3/5: 准备训练数据...")
    log_gpu_memory_usage("开始准备数据")
    train_dataset = JsonlSFTDataset(
        args.data_path,
        eos_token=tokenizer.eos_token or "",
        max_samples=args.max_samples
    )
    data_collator = CollatorForCausalLM(tokenizer=tokenizer)
    print_rank0(f"训练样本数量: {len(train_dataset)}")
    log_gpu_memory_usage("数据准备完成")


    # 步骤 4: 应用 LoRA
    print_rank0("步骤 4/5: 应用 LoRA...")
    log_gpu_memory_usage("开始应用LoRA")
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    model = apply_lora(model, target_modules, args.lora_r, args.lora_alpha, args.lora_dropout)
    model.train()  # 确保模型处于训练模式
    log_gpu_memory_usage("LoRA应用完成")


    # 步骤 5: 训练
    print_rank0("步骤 5/5: 开始训练...")
    log_gpu_memory_usage("开始训练前")

    dtype_kwargs = safe_dtype_preference()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=dtype_kwargs["bf16"],
        fp16=dtype_kwargs["fp16"],
        optim=args.optim,  # 使用命令行参数指定的优化器
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # 开始训练
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log_gpu_memory_usage("训练完成")

    print_rank0("训练完成!")

    # 合并并保存完整模型
    if args.merge_and_save:
        merged_dir = merge_and_save(base_dir, args.output_dir)
        print_rank0(f"完整模型已保存到: {merged_dir}")

    print_rank0("所有步骤完成!")

    # 显式清理资源
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 如果使用了 Unsloth，显示提示信息
    if args.use_unsloth:
        print_rank0("提示: 您已使用 Unsloth 加速训练，推理时也建议使用 Unsloth 加载模型以获得最佳性能")

if __name__ == "__main__":
    main()