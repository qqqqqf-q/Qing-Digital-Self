#!/usr/bin/env python3
"""
从checkpoint合并LoRA权重的脚本
支持从训练检查点恢复并合并完整模型
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

# 导入自定义Logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger.logger import Logger
from utils.config.config import Config

# 初始化配置和日志器
config = Config()
logger = Logger('MergeScript')





def merge_and_save_from_checkpoint(
    base_model_path: str,
    checkpoint_path: str,
    output_path: str,
    trust_remote_code: bool = True,
    load_precision: str = "fp16"
) -> str:
    """
    从checkpoint合并LoRA权重并保存完整模型

    Args:
        base_model_path: 基础模型路径
        checkpoint_path: 检查点路径（包含LoRA权重）
        output_path: 输出目录
        trust_remote_code: 是否信任远程代码

    Returns:
        合并后模型的保存路径
    """
    logger.info(
        zhcn="=== 开始从checkpoint合并LoRA权重 ===",
        en="=== Starting LoRA weight merging from checkpoint ==="
    )
    
    # 检查路径是否存在
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"基础模型路径不存在: {base_model_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点路径不存在: {checkpoint_path}")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(
        zhcn=f"基础模型: {base_model_path}",
        en=f"Base model: {base_model_path}"
    )
    logger.info(
        zhcn=f"检查点路径: {checkpoint_path}",
        en=f"Checkpoint path: {checkpoint_path}"
    )
    logger.info(
        zhcn=f"输出目录: {output_path}",
        en=f"Output directory: {output_path}"
    )
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        logger.info(
            zhcn=f"使用GPU: {torch.cuda.get_device_name()}",
            en=f"Using GPU: {torch.cuda.get_device_name()}"
        )
    else:
        logger.info(
            zhcn="使用CPU进行合并",
            en="Using CPU for merging"
        )
    
    # 加载基础模型
    logger.info(
        zhcn="加载基础模型...",
        en="Loading base model..."
    )
    if load_precision == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=trust_remote_code
        )
    elif load_precision == "int4":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=trust_remote_code
        )
    else:  # fp16
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    
    # 加载分词器
    logger.info(
        zhcn="加载分词器...",
        en="Loading tokenizer..."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code
    )
    
    # 检查checkpoint中是否有LoRA配置
    lora_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(lora_config_path):
        # 尝试在checkpoint目录的子目录中查找
        possible_lora_dirs = [d for d in os.listdir(checkpoint_path) 
                            if os.path.isdir(os.path.join(checkpoint_path, d)) 
                            and os.path.exists(os.path.join(checkpoint_path, d, "adapter_config.json"))]
        
        if possible_lora_dirs:
            checkpoint_path = os.path.join(checkpoint_path, possible_lora_dirs[0])
            logger.info(
                zhcn=f"找到LoRA配置目录: {checkpoint_path}",
                en=f"Found LoRA config directory: {checkpoint_path}"
            )
        else:
            raise FileNotFoundError(f"在{checkpoint_path}中未找到LoRA配置(adapter_config.json)")
    
    # 加载PEFT模型
    logger.info(
        zhcn="加载LoRA权重...",
        en="Loading LoRA weights..."
    )
    peft_model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
        device_map="auto"
    )
    
    # 合并权重
    logger.info(
        zhcn="合并LoRA权重...",
        en="Merging LoRA weights..."
    )
    merged_model = peft_model.merge_and_unload()
    
    # 保存合并后的模型
    logger.info(
        zhcn="保存合并后的模型...",
        en="Saving merged model..."
    )
    merged_model.save_pretrained(output_path)
    
    # 保存分词器
    logger.info(
        zhcn="保存分词器...",
        en="Saving tokenizer..."
    )
    tokenizer.save_pretrained(output_path)
    
    # 清理内存
    del model
    del peft_model
    del merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(
        zhcn=f"合并完成！完整模型已保存到: {output_path}",
        en=f"Merging completed! Complete model saved to: {output_path}"
    )
    return output_path


def auto_detect_checkpoint(base_output_dir: str) -> str:
    """
    自动检测最新的checkpoint目录

    Args:
        base_output_dir: 基础输出目录

    Returns:
        最新的checkpoint路径
    """
    if not os.path.exists(base_output_dir):
        raise FileNotFoundError(f"输出目录不存在: {base_output_dir}")
    
    # 查找checkpoint目录
    checkpoints = []
    for item in os.listdir(base_output_dir):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(base_output_dir, item)):
            checkpoints.append(item)
    
    if not checkpoints:
        raise FileNotFoundError(f"在{base_output_dir}中未找到checkpoint目录")
    
    # 按checkpoint编号排序
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(base_output_dir, latest_checkpoint)
    
    logger.info(
        zhcn=f"自动检测到最新checkpoint: {latest_checkpoint}",
        en=f"Auto-detected latest checkpoint: {latest_checkpoint}"
    )
    return checkpoint_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从checkpoint合并LoRA权重")
    
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        required=True,
        help="基础模型路径"
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str,
        help="检查点路径（包含LoRA权重）。如果未指定，将尝试从--base_output_dir自动检测"
    )
    parser.add_argument(
        "--base_output_dir", 
        type=str,
        help="基础输出目录（用于自动检测最新checkpoint）"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="合并后模型的输出目录"
    )
    parser.add_argument(
        "--trust_remote_code", 
        action="store_true", 
        default=True,
        help="是否信任远程代码"
    )
    parser.add_argument(
        "--load_precision", 
        type=str, 
        default="fp16", 
        choices=["int8", "int4", "fp16"],
        help="模型加载精度：int8、int4 或 fp16 (default: fp16)"
    )
    
    args = parser.parse_args()
    
    try:
        # 确定checkpoint路径
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
        elif args.base_output_dir:
            checkpoint_path = auto_detect_checkpoint(args.base_output_dir)
        else:
            raise ValueError("必须指定--checkpoint_path或--base_output_dir之一")
        
        # 执行合并
        merge_and_save_from_checkpoint(
            base_model_path=args.base_model_path,
            checkpoint_path=checkpoint_path,
            output_path=args.output_path,
            trust_remote_code=args.trust_remote_code,
            load_precision=args.load_precision
        )
        
        logger.info(
            zhcn="=== 合并完成 ===",
            en="=== Merging Completed ==="
        )
        
    except Exception as e:
        logger.error(
            zhcn=f"合并过程中出现错误: {str(e)}",
            en=f"Error occurred during merging: {str(e)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()