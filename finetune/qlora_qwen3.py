#!/usr/bin/env python3
"""
QLoRA 微调脚本
"""
import os
import sys

# os.environ["TORCH_LOGS"] = ""
# os.environ["PYTORCH_DISABLE_LOGGING"] = "1"
# 尝试导入 unsloth，必须在所有其他库之前导入
try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError as e:
    UNSLOTH_AVAILABLE = False
    FastLanguageModel = None
    print(f"警告: 无法导入 unsloth ({e})，将使用标准 PEFT 训练")

import argparse
import json
import math
import platform


# 尝试导入 modelscope.snapshot_download，失败则禁用自动下载但不影响后续流程
try:
    from modelscope import snapshot_download as _ms_snapshot_download

    def snapshot_download(model_id: str, local_dir: str) -> str:
        return _ms_snapshot_download(model_id=model_id, local_dir=local_dir)

    MODELSPACE_AVAILABLE = True
except Exception as exc:
    MODELSPACE_AVAILABLE = False

    def snapshot_download(model_id: str, local_dir: str) -> str:
        # 安全回退：不执行任何下载，直接返回本地目录；调用方需确保目录已准备好
        logger.warning(
            zhcn=f"未能导入 modelscope（{repr(exc)}），跳过自动下载，依赖本地目录: {local_dir}",
            en=f"Failed to import modelscope ({repr(exc)}), skipping auto-download, relying on local directory: {local_dir}",
        )
        return local_dir


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# 移除可能存在的 TORCH_LOGS 环境变量
os.environ.pop("TORCH_LOGS", None)

# 设置默认环境变量 - 必须在导入 torch 之前设置
# 禁用各种可能导致兼容性问题的优化
# 设置CPU并行化
cpu_count = os.cpu_count()
os.environ.update(
    {
        "PYTORCH_ENABLE_BACKWARD_COMPATIBILITY": "0",
        "CUDA_LAUNCH_BLOCKING": "0",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "OMP_NUM_THREADS": str(cpu_count),
        "MKL_NUM_THREADS": str(cpu_count),
        "OPENBLAS_NUM_THREADS": str(cpu_count),
        "TOKENIZERS_PARALLELISM": "false",
    }
)

# 清理可能存在的triton相关环境变量
for key in list(os.environ.keys()):
    if "triton" in key.lower():
        os.environ.pop(key, None)

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


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger.logger import logger


def log_gpu_memory_usage(step_name: str) -> None:
    """记录GPU显存使用情况

    Args:
        step_name: 当前步骤名称，用于日志标识
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            zhcn=f"{step_name} - GPU显存使用: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB",
            en=f"{step_name} - GPU memory usage: allocated {allocated:.2f} GB, reserved {reserved:.2f} GB",
        )
    else:
        logger.warning(
            zhcn=f"{step_name} - CUDA不可用", en=f"{step_name} - CUDA not available"
        )


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
    use_gradient_checkpointing: bool = True,
    load_precision: str = "fp16",
    use_flash_attention_2: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """加载模型和分词器，支持 Unsloth 和普通量化，支持多种加载精度。
    注意：不在此函数内注入 LoRA，统一在主流程步骤 4 进行一次注入，避免重复。

    Args:
        base_dir: 模型目录路径
        use_qlora: 是否使用QLoRA量化
        use_unsloth: 是否使用Unsloth加速
        use_gradient_checkpointing: 是否使用梯度检查点
        load_precision: 加载精度 ("fp16", "int8", "int4")
        use_flash_attention_2: 是否使用FlashAttention2（仅支持fp16/bf16，Unsloth内置优化）
    """
    logger.info(zhcn="加载分词器...", en="Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_dir, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # FlashAttention2 配置检查
    attn_implementation = None
    if use_flash_attention_2:
        if load_precision == "fp32":
            logger.warning(
                zhcn="FlashAttention2 不支持 fp32 精度，将禁用 FlashAttention2",
                en="FlashAttention2 does not support fp32 precision, disabling FlashAttention2",
            )
            use_flash_attention_2 = False
        elif use_unsloth:
            logger.info(
                zhcn="Unsloth 内置 Flash Attention 优化，无需额外配置",
                en="Unsloth has built-in Flash Attention optimization, no additional configuration needed",
            )
        else:
            attn_implementation = "flash_attention_2"
            logger.info(
                zhcn="启用 FlashAttention2 加速注意力计算",
                en="Enabling FlashAttention2 to accelerate attention computation",
            )

    # 尝试使用 Unsloth 加载模型
    if use_unsloth and UNSLOTH_AVAILABLE:
        try:
            logger.info(
                zhcn="尝试使用 Unsloth 加载模型...",
                en="Attempting to load model with Unsloth...",
            )

            # 启用 Triton 和相关优化（仅在使用unsloth时）
            os.environ.update(
                {
                    "TORCHDYNAMO_DISABLE": "0",
                    "TORCH_COMPILE_DISABLE": "0",
                    "TRITON_DISABLE": "0",
                    "TORCH_USE_TRITON": "1",
                    "PYTORCH_DISABLE_TRITON": "0",
                }
            )

            # Unsloth 根据load_precision选择量化模式
            if load_precision == "int4":
                model, tokenizer = FastLanguageModel.from_pretrained(
                    base_dir,
                    dtype=None,
                    load_in_4bit=True,  # 启用4bit
                    load_in_8bit=False,  # 禁用8bit
                    trust_remote_code=True,
                )
                logger.info(
                    zhcn="成功使用 Unsloth 加载模型 (4bit)",
                    en="Successfully loaded model with Unsloth (4bit)",
                )
            elif load_precision == "int8":
                model, tokenizer = FastLanguageModel.from_pretrained(
                    base_dir,
                    dtype=None,
                    load_in_4bit=False,  # 禁用4bit
                    load_in_8bit=True,  # 启用8bit
                    trust_remote_code=True,
                )
                logger.info(
                    zhcn="成功使用 Unsloth 加载模型 (8bit)",
                    en="Successfully loaded model with Unsloth (8bit)",
                )
            else:  # fp16
                model, tokenizer = FastLanguageModel.from_pretrained(
                    base_dir,
                    dtype=None,
                    load_in_4bit=False,  # 禁用4bit
                    load_in_8bit=False,  # 禁用8bit
                    trust_remote_code=True,
                )
                logger.info(
                    zhcn="成功使用 Unsloth 加载模型 (fp16)",
                    en="Successfully loaded model with Unsloth (fp16)",
                )
            return model, tokenizer
        except Exception as e:
            logger.error(
                zhcn=f"Unsloth 加载失败: {e}", en=f"Unsloth loading failed: {e}"
            )
            logger.info(
                zhcn="回退到普通量化加载...",
                en="Falling back to standard quantization loading...",
            )
            # 恢复原始环境变量设置
            os.environ.update(
                {
                    "TORCHDYNAMO_DISABLE": "1",
                    "TORCH_COMPILE_DISABLE": "1",
                    "TRITON_DISABLE": "1",
                    "TORCH_USE_TRITON": "0",
                    "PYTORCH_DISABLE_TRITON": "1",
                }
            )
    elif use_unsloth and not UNSLOTH_AVAILABLE:
        logger.warning(
            zhcn="请求使用 Unsloth 但未安装，回退到普通量化加载...",
            en="Unsloth requested but not installed, falling back to standard quantization loading...",
        )

    # 普通量化加载
    dtype_kwargs = safe_dtype_preference()

    # 加载模型配置并设置return_dict=True
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base_dir, trust_remote_code=True)
    config.return_dict = True

    # 根据load_precision选择量化配置
    if load_precision == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_kwargs["torch_dtype"],
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info(
            zhcn="使用量化配置: 4bit量化",
            en="Using quantization config: 4bit quantization",
        )
    elif load_precision == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        logger.info(
            zhcn="使用量化配置: 8bit量化",
            en="Using quantization config: 8bit quantization",
        )
    else:  # fp16
        bnb_config = None
        logger.info(
            zhcn="使用量化配置: fp16无量化",
            en="Using quantization config: fp16 without quantization",
        )

    # 构建模型加载参数
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype_kwargs["torch_dtype"],
        "config": config,
    }

    # 添加FlashAttention2支持
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model = AutoModelForCausalLM.from_pretrained(base_dir, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_dir, **model_kwargs)

    # 训练前准备（不注入 LoRA）
    if use_qlora or load_precision in ["int4", "int8"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gradient_checkpointing
        )
    else:
        if use_gradient_checkpointing:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()

    # 确保模型处于训练模式
    model.train()

    logger.info(
        zhcn="模型已准备好进行训练（未注入LoRA，稍后统一注入）",
        en="Model is ready for training (LoRA not injected yet, will be injected uniformly later)",
    )

    # 调试：检查加载后的模型参数状态
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        zhcn=f"加载后模型状态 - 总参数: {total_params}, 可训练参数: {trainable_params}",
        en=f"Model state after loading - Total params: {total_params}, Trainable params: {trainable_params}",
    )

    return model, tokenizer


def apply_lora(
    model: AutoModelForCausalLM,
    target_modules: List[str],
    r: int,
    alpha: int,
    dropout: float,
    use_unsloth: bool = False,
    use_gradient_checkpointing: bool = True,
) -> PeftModel:
    """应用 LoRA 适配器（统一入口）

    注意：target_modules 可以来自稠密模型的固定列表，或基于 MoE 模式构建的模块路径。
    """
    if not target_modules:
        raise ValueError(
            "LoRA target_modules 为空，无法注入。请检查参数或 MoE 模式匹配。"
        )

    logger.info(
        zhcn=f"LoRA 目标模块数量: {len(target_modules)}",
        en=f"Number of LoRA target modules: {len(target_modules)}",
    )
    if len(target_modules) <= 50:
        logger.info(
            zhcn=f"LoRA 目标模块示例: {target_modules[:50]}",
            en=f"LoRA target modules examples: {target_modules[:50]}",
        )

    if use_unsloth and UNSLOTH_AVAILABLE:
        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=target_modules,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=3407,
        )
    else:
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

    # 确保LoRA参数可训练（解决梯度错误问题）
    for name, param in model.named_parameters():
        if "lora_" in name.lower():
            param.requires_grad = True

    return model


def merge_and_save(base_dir: str, output_dir: str) -> str:
    """合并 LoRA 权重并保存完整模型

    Args:
        base_dir: 基础模型目录
        output_dir: LoRA模型输出目录

    Returns:
        合并后模型的保存路径
    """
    logger.info(zhcn="合并 LoRA 权重...", en="Merging LoRA weights...")

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, device_map="auto", trust_remote_code=True
    )

    # 创建 offload 目录
    offload_dir = os.path.join(output_dir, "offload")
    os.makedirs(offload_dir, exist_ok=True)

    # 加载 PEFT 模型
    peft_model = PeftModel.from_pretrained(
        model, output_dir, device_map="auto", offload_dir=offload_dir
    )

    # 合并权重
    merged_model = peft_model.merge_and_unload()

    # 保存合并后的模型
    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)

    # 保存分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_dir, use_fast=True, trust_remote_code=True
    )
    tokenizer.save_pretrained(merged_dir)

    logger.info(
        zhcn=f"合并后的模型已保存到: {merged_dir}",
        en=f"Merged model saved to: {merged_dir}",
    )
    return merged_dir


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="QLoRA 微调（支持稠密与 MoE）")

    # 模型相关
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Qwen/Qwen3-8B-Base",
        help="基础模型或 MoE 模型的仓库ID",
    )
    parser.add_argument(
        "--local_dir", type=str, default="qwen3-8b-base", help="本地模型存储目录"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="是否信任远程代码",
    )
    parser.add_argument(
        "--use_unsloth",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="是否使用 Unsloth 加速",
    )
    parser.add_argument(
        "--use_qlora",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="是否使用 8bit 量化（QLoRA）",
    )

    # 数据相关
    parser.add_argument(
        "--data_path", type=str, default="training_data.jsonl", help="训练数据文件路径"
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help="验证数据文件路径，None表示不使用验证集",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="最大训练样本数，None 表示用全部"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="最大验证样本数，None 表示用全部",
    )
    parser.add_argument(
        "--model_max_length", type=int, default=2048, help="最大序列长度"
    )

    # 训练相关
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetune/models/qwen3-8b-qlora",
        help="输出目录",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="稠密模型的 LoRA 目标模块列表（逗号分隔）",
    )

    # MoE 支持参数
    parser.add_argument(
        "--moe_enable",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="是否启用 MoE 注入逻辑",
    )
    parser.add_argument(
        "--moe_lora_scope",
        type=str,
        default="expert_only",
        choices=["expert_only", "router_only", "all"],
        help="LoRA 注入范围：仅专家、仅路由或全部",
    )
    parser.add_argument(
        "--moe_expert_patterns",
        type=str,
        default="experts.ffn.(gate_proj|up_proj|down_proj),layers.[0-9]+.mlp.experts.[0-9]+.(w1|w2|w3)",
        help="专家线性层匹配模式，支持多个正则，用逗号分隔；默认兼容 Qwen-MoE 与 Mixtral",
    )
    parser.add_argument(
        "--moe_router_patterns",
        type=str,
        default="router.(gate|dense)",
        help="路由/门控线性层匹配模式，多个正则逗号分隔",
    )
    parser.add_argument(
        "--moe_max_experts_lora",
        type=int,
        default=-1,
        help="每层最多注入 LoRA 的专家数，-1 表示全部",
    )
    parser.add_argument(
        "--moe_dry_run",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="仅打印匹配到的模块，不执行训练",
    )

    # 训练超参数
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=1, help="每卡 batch size"
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=1, help="每卡验证 batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数"
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大步数，-1 不限制")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="预热比例")
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型"
    )

    # 模型加载精度
    parser.add_argument(
        "--load_precision",
        type=str,
        default="fp16",
        choices=["int8", "int4", "fp16"],
        help="模型加载精度：int8、int4 或 fp16 (default: fp16)",
    )

    # 其他设置
    parser.add_argument(
        "--logging_steps", type=int, default=1, help="日志间隔，设置为1以每step输出loss"
    )
    parser.add_argument("--eval_steps", type=int, default=50, help="验证间隔步数")
    parser.add_argument("--save_steps", type=int, default=200, help="保存间隔")
    parser.add_argument("--save_total_limit", type=int, default=2, help="最多保存数")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="是否使用梯度检查点",
    )
    parser.add_argument(
        "--merge_and_save",
        action="store_true",
        default=True,
        help="训练后是否合并 LoRA 并保存完整模型",
    )
    parser.add_argument(
        "--fp16",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="是否使用 FP16",
    )
    parser.add_argument("--optim", type=str, default="adamw_torch_fused", help="优化器")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从指定的检查点恢复训练，可以是本地路径或'latest'",
    )
    parser.add_argument(
        "--use_flash_attention_2",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="是否使用 FlashAttention2 加速注意力计算（需要 fp16 或 bf16）",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="DataLoader pin_memory",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="DataLoader 工作线程数 (0表示使用主进程，推荐设置为CPU核心数-2)",
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=2,
        help="DataLoader 预取因子 (default: 2)",
    )

    return parser.parse_args()


def _split_patterns(p: str) -> List[str]:
    return [s.strip() for s in p.split(",") if s.strip()]


def _is_linear_like(m: torch.nn.Module) -> bool:
    import torch.nn as nn

    linear_like = isinstance(m, nn.Linear) or m.__class__.__name__.lower() in [
        "qlinear",
        "linear8bitlt",
        "bnblinear",
        "lora_linear",
    ]
    # 一些 MoE 实现自定义 Linear 名称，这里采用类名包含 "linear" 的宽松判断
    return linear_like or ("linear" in m.__class__.__name__.lower())


def detect_moe(model: torch.nn.Module) -> Dict[str, Any]:
    hints = []
    for name, module in model.named_modules():
        low = name.lower()
        if any(k in low for k in ["experts", "moe", "router"]):
            hints.append(name)
    return {"is_moe": len(hints) > 0, "hints": hints[:50]}


def build_moe_target_modules(
    model: torch.nn.Module,
    scope: str,
    expert_patterns: List[str],
    router_patterns: List[str],
    max_experts_per_layer: int = -1,
) -> List[str]:
    import re

    expert_res = [re.compile(p) for p in expert_patterns] if expert_patterns else []
    router_res = [re.compile(p) for p in router_patterns] if router_patterns else []
    targets: List[str] = []

    def match_any(res_list, text: str) -> bool:
        return any(r.search(text) for r in res_list)

    # 粗略按层聚合的键：提取 "layers.N" 或 "h.N" 段
    def layer_key(name: str) -> str:
        import re as _re

        m = _re.search(r"(layers\.\d+|h\.\d+)", name)
        return m.group(1) if m else "global"

    per_layer_expert_count: Dict[str, int] = {}

    for n, m in model.named_modules():
        if not _is_linear_like(m):
            continue
        chosen = False
        if scope in ("expert_only", "all") and expert_res and match_any(expert_res, n):
            if max_experts_per_layer >= 0:
                lk = layer_key(n)
                c = per_layer_expert_count.get(lk, 0)
                if c >= max_experts_per_layer:
                    chosen = False
                else:
                    per_layer_expert_count[lk] = c + 1
                    chosen = True
            else:
                chosen = True
        if scope in ("router_only", "all") and router_res and match_any(router_res, n):
            chosen = True
        if chosen:
            targets.append(n)

    # 去重
    seen = set()
    uniq = []
    for t in targets:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def pretty_print_targets(title: str, targets: List[str], max_show: int = 80) -> None:
    logger.info(
        zhcn=f"{title}: 共 {len(targets)} 个", en=f"{title}: {len(targets)} total"
    )
    if len(targets) <= max_show:
        for t in targets:
            logger.info(zhcn=f"  - {t}", en=f"  - {t}")
    else:
        for t in targets[:max_show]:
            logger.info(zhcn=f"  - {t}", en=f"  - {t}")
        logger.info(
            zhcn=f"  ... 其余 {len(targets) - max_show} 项已省略",
            en=f"  ... {len(targets) - max_show} more items omitted",
        )


def main() -> None:
    """主函数，执行QLoRA微调流程"""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    logger.info(zhcn="=== QLoRA 微调开始 ===", en="=== QLoRA Fine-tuning Started ===")
    logger.info(zhcn=f"平台: {platform.system()}", en=f"Platform: {platform.system()}")
    logger.info(
        zhcn=f"配置: 使用Unsloth={args.use_unsloth}, 使用qlora={args.use_qlora}",
        en=f"Config: Use Unsloth={args.use_unsloth}, Use 8bit quantization={args.use_qlora}",
    )

    # 禁用动态编译和相关优化
    try:
        torch._dynamo.config.disable = True
        torch._dynamo.config.suppress_errors = True
    except Exception as e:
        logger.error(
            zhcn=f"禁用动态编译失败: {e}",
            en=f"Failed to disable dynamic compilation: {e}",
        )

    # 设置精度 - 使用新的TF32控制API
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception as e:
        logger.error(
            zhcn=f"设置TF32行为失败: {e}", en=f"Failed to set TF32 behavior: {e}"
        )

    # 强制PyTorch使用所有CPU核心
    try:
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        torch.set_num_threads(cpu_count)
        torch.set_num_interop_threads(min(cpu_count, 8))
        logger.info(
            zhcn=f"已设置PyTorch使用{cpu_count}个CPU核心",
            en=f"Set PyTorch to use {cpu_count} CPU cores",
        )
    except Exception as e:
        logger.error(
            zhcn=f"设置PyTorch线程数失败: {e}",
            en=f"Failed to set PyTorch thread count: {e}",
        )

    # 步骤 1: 下载模型
    logger.info(
        zhcn="步骤 1/5: 下载或复用基础模型...",
        en="Step 1/5: Download or reuse base model...",
    )
    log_gpu_memory_usage("开始下载模型")

    if os.path.exists(args.local_dir) and os.listdir(args.local_dir):
        logger.info(
            zhcn=f"复用本地模型: {args.local_dir}",
            en=f"Reusing local model: {args.local_dir}",
        )
        base_dir = args.local_dir
    else:
        logger.info(
            zhcn=f"从远程下载模型到: {args.local_dir}",
            en=f"Downloading model from remote to: {args.local_dir}",
        )
        base_dir = snapshot_download(
            model_id=args.repo_id,
            local_dir=args.local_dir,
        )
    logger.info(
        zhcn=f"基础模型目录: {base_dir}", en=f"Base model directory: {base_dir}"
    )
    log_gpu_memory_usage("模型下载完成")

    # 步骤 2: 加载模型和分词器
    logger.info(
        zhcn="步骤 2/5: 加载模型和分词器...", en="Step 2/5: Load model and tokenizer..."
    )
    log_gpu_memory_usage("开始加载模型")
    model, tokenizer = load_model_and_tokenizer(
        base_dir,
        use_qlora=args.use_qlora,
        use_unsloth=args.use_unsloth,
        use_gradient_checkpointing=args.gradient_checkpointing,
        load_precision=args.load_precision,
        use_flash_attention_2=args.use_flash_attention_2,
    )
    tokenizer.model_max_length = args.model_max_length
    log_gpu_memory_usage("模型加载完成")

    # 步骤 3: 准备数据
    logger.info(
        zhcn="步骤 3/5: 准备训练数据...", en="Step 3/5: Prepare training data..."
    )
    log_gpu_memory_usage("开始准备数据")

    # 优化DataLoader设置以减少显存占用
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    # 减少DataLoader工作线程数以降低显存占用
    if (
        not hasattr(args, "dataloader_num_workers")
        or args.dataloader_num_workers is None
    ):
        args.dataloader_num_workers = min(4, max(1, cpu_count // 2))  # 减少工作线程
    args.dataloader_pin_memory = True
    logger.info(
        zhcn=f"设置DataLoader工作线程数为: {args.dataloader_num_workers} (CPU核心数: {cpu_count}), pin_memory: {args.dataloader_pin_memory}",
        en=f"Setting DataLoader worker threads to: {args.dataloader_num_workers} (CPU cores: {cpu_count}), pin_memory: {args.dataloader_pin_memory}",
    )

    train_dataset = JsonlSFTDataset(
        args.data_path,
        eos_token=tokenizer.eos_token or "",
        max_samples=args.max_samples,
    )

    eval_dataset = None
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        logger.info(zhcn="准备验证数据...", en="Preparing validation data...")
        eval_dataset = JsonlSFTDataset(
            args.eval_data_path,
            eos_token=tokenizer.eos_token or "",
            max_samples=args.max_eval_samples,
        )
        logger.info(
            zhcn=f"验证样本数量: {len(eval_dataset)}",
            en=f"Validation samples: {len(eval_dataset)}",
        )
    else:
        logger.warning(
            zhcn="未提供验证数据路径，跳过验证集",
            en="No validation data path provided, skipping validation set",
        )

    data_collator = CollatorForCausalLM(tokenizer=tokenizer)
    logger.info(
        zhcn=f"训练样本数量: {len(train_dataset)}",
        en=f"Training samples: {len(train_dataset)}",
    )
    log_gpu_memory_usage("数据准备完成")

    # 步骤 4: 应用 LoRA（统一入口，支持 MoE）
    logger.info(zhcn="步骤 4/5: 应用 LoRA...", en="Step 4/5: Apply LoRA...")
    log_gpu_memory_usage("开始应用LoRA")

    if args.moe_enable:
        info = detect_moe(model)
        logger.info(
            zhcn=f"MoE 检测结果: is_moe={info['is_moe']}",
            en=f"MoE detection result: is_moe={info['is_moe']}",
        )
        if info["hints"]:
            pretty_print_targets("MoE 模块路径提示", info["hints"])

        expert_patterns = _split_patterns(args.moe_expert_patterns)
        router_patterns = _split_patterns(args.moe_router_patterns)
        moe_targets = build_moe_target_modules(
            model=model,
            scope=args.moe_lora_scope,
            expert_patterns=expert_patterns,
            router_patterns=router_patterns,
            max_experts_per_layer=args.moe_max_experts_lora,
        )
        pretty_print_targets("匹配到的 LoRA 注入目标（MoE）", moe_targets, max_show=120)

        if args.moe_dry_run:
            logger.warning(
                zhcn="MoE dry-run 模式启用，仅打印匹配模块并退出。",
                en="MoE dry-run mode enabled, only printing matched modules and exiting.",
            )
            return

        model = apply_lora(
            model,
            moe_targets,
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
            use_unsloth=args.use_unsloth,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        target_modules = [
            m.strip() for m in args.target_modules.split(",") if m.strip()
        ]
        pretty_print_targets(
            "匹配到的 LoRA 注入目标（稠密）", target_modules, max_show=120
        )
        model = apply_lora(
            model,
            target_modules,
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
            use_unsloth=args.use_unsloth,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    model.train()

    # 调试信息：检查可训练参数
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(
        zhcn=f"可训练参数: {trainable_params} || 全部参数: {all_param} || 可训练比例: {100 * trainable_params / all_param:.2f}%",
        en=f"Trainable params: {trainable_params} || All params: {all_param} || Trainable ratio: {100 * trainable_params / all_param:.2f}%",
    )

    # 验证是否有可训练参数
    if trainable_params == 0:
        raise ValueError(
            "没有检测到可训练参数！请检查LoRA配置是否正确。 / No trainable parameters detected! Please check if LoRA configuration is correct."
        )

    log_gpu_memory_usage("LoRA应用完成")

    # 步骤 5: 训练
    logger.info(zhcn="步骤 5/5: 开始训练...", en="Step 5/5: Start training...")

    if args.resume_from_checkpoint:
        logger.info(
            zhcn=f"将从检查点恢复训练: {args.resume_from_checkpoint}",
            en=f"Will resume training from checkpoint: {args.resume_from_checkpoint}",
        )
    if eval_dataset is not None:
        logger.info(
            zhcn=f"将在每 {args.eval_steps} 步后进行验证",
            en=f"Will perform validation every {args.eval_steps} steps",
        )
    logger.info(
        zhcn=f"将每 {args.logging_steps} 步输出训练loss",
        en=f"Will output training loss every {args.logging_steps} steps",
    )
    log_gpu_memory_usage("开始训练前")

    dtype_kwargs = safe_dtype_preference()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=dtype_kwargs["bf16"],
        fp16=dtype_kwargs["fp16"],
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=(
            min(2, args.dataloader_prefetch_factor)
            if args.dataloader_prefetch_factor
            else 2
        ),  # 限制预取因子以减少显存占用
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log_gpu_memory_usage("训练完成")

    logger.info(zhcn="训练完成!", en="Training completed!")

    if args.merge_and_save:
        merged_dir = merge_and_save(base_dir, args.output_dir)
        logger.info(
            zhcn=f"完整模型已保存到: {merged_dir}",
            en=f"Complete model saved to: {merged_dir}",
        )

    logger.info(zhcn="所有步骤完成!", en="All steps completed!")

    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.use_unsloth:
        logger.info(
            zhcn="提示: 您已使用 Unsloth 加速训练，推理时也建议使用 Unsloth 加载模型以获得最佳性能",
            en="Tip: You have used Unsloth for accelerated training, it is also recommended to use Unsloth to load the model during inference for optimal performance",
        )


if __name__ == "__main__":
    main()
