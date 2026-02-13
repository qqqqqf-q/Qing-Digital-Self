#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 setting.jsonc 生成 LLaMA Factory 的训练配置（llamaboard 风格）

输出示例参考: saves/*/llora/train_2025-09-06-*/llamaboard_config.yaml
生成路径: config/finetune-config.yaml

仅做字段映射与格式化，不改动原始数据与 CLI。
"""
from __future__ import annotations

import os
from typing import Dict, Any, List

from .config import get_config
from .path_resolver import resolve_train_data_path


def _derive_model_name(cfg: Dict[str, Any]) -> str:
    repo = cfg.get("model_repo") or ""
    path = cfg.get("model_path") or ""
    name = ""
    if repo:
        name = repo.strip("/").split("/")[-1]
    if not name and path:
        name = os.path.basename(os.path.normpath(path))
    return name or "model"


def _as_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val]
    return [str(val)]


def _join_modules(val: Any) -> str:
    if isinstance(val, list):
        return ",".join(str(x) for x in val if x)
    if isinstance(val, str):
        return val
    return ""


def _guess_quant_bit(cfg: Dict[str, Any]) -> str:
    t = str(cfg.get("finetuning_type", "")).lower()
    if t == "qlora":
        return "4"
    load_precision = str(cfg.get("load_precision", "")).lower()
    if load_precision in ("int4", "nf4", "fp4"):
        return "4"
    if load_precision in ("int8", "fp8"):
        return "8"
    return "none"


def _compute_type(cfg: Dict[str, Any]) -> str:
    # 优先按 fp16 标志；否则退回 auto
    fp16 = bool(cfg.get("fp16", False))
    return "fp16" if fp16 else "auto"


def _fmt_scalar(v: Any) -> str:
    # 简单的 YAML 标量格式化：数字/布尔保持原样，其他统一单引号
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    # 单引号内将单引号转义为 ''
    s = s.replace("'", "''")
    return f"'{s}'"


def _emit_lines(flat: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    # 确保有序：top.* 在前，train.* 在后
    for section in ("top.", "train."):
        for k in sorted([kk for kk in flat if kk.startswith(section)]):
            v = flat[k]
            if isinstance(v, list):
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"- {_fmt_scalar(item)}")
            else:
                lines.append(f"{k}: {_fmt_scalar(v)}")
    return lines


def build_llamafactory_config() -> Dict[str, Any]:
    """根据 setting.jsonc 构建扁平 key 的训练配置字典。"""
    c = get_config()
    cfg = c.all()

    # 顶层
    top: Dict[str, Any] = {}
    top["top.booster"] = "auto"
    top["top.model_name"] = _derive_model_name(cfg)
    top["top.template"] = cfg.get("template", "qwen")
    top["top.finetuning_type"] = cfg.get("finetuning_type", "lora")
    top["top.quantization_bit"] = _guess_quant_bit(cfg)
    top["top.quantization_method"] = "bitsandbytes"
    top["top.rope_scaling"] = "none"
    # checkpoint path 仅作为占位，可由 WebUI 选择具体路径
    top["top.checkpoint_path"] = []

    # 训练
    train: Dict[str, Any] = {}
    train["train.compute_type"] = _compute_type(cfg)
    train["train.batch_size"] = int(cfg.get("per_device_train_batch_size", 1))
    train["train.gradient_accumulation_steps"] = int(cfg.get("gradient_accumulation_steps", 1))
    train["train.learning_rate"] = float(cfg.get("learning_rate", 2e-4))
    train["train.logging_steps"] = int(cfg.get("logging_steps", 10))
    train["train.save_steps"] = int(cfg.get("save_steps", 100))
    train["train.cutoff_len"] = int(cfg.get("messages_max_length", 2048))
    # 数据集在 Board 中也是列表。支持占位符 LATEST：自动选择 runs/chat 下最新的 sft/train.jsonl
    resolved = resolve_train_data_path(cfg.get("data_path", "dataset/sft.jsonl"), runs_root=cfg.get("runs_root", "./runs"))
    train["train.dataset"] = _as_list(resolved.value)
    train["train.dataset_dir"] = "data"

    train["train.lora_rank"] = int(cfg.get("lora_r", 16))
    train["train.lora_alpha"] = int(cfg.get("lora_alpha", 32))
    train["train.lora_dropout"] = float(cfg.get("lora_dropout", 0.05))
    train["train.lora_target"] = _join_modules(cfg.get("lora_target_modules", [])) or "all"

    # 调度/其他
    train["train.lr_scheduler_type"] = cfg.get("lr_scheduler", "cosine")
    train["train.max_grad_norm"] = "1.0"
    train["train.packing"] = False
    train["train.report_to"] = False
    train["train.warmup_steps"] = int(cfg.get("warmup_steps", 0))

    flat: Dict[str, Any] = {}
    flat.update(top)
    flat.update(train)
    return flat


def write_finetune_yaml(output_path: str = os.path.join("config", "finetune-config.yaml")) -> str:
    """生成并写入 YAML 样式文本，返回写入路径。"""
    flat = build_llamafactory_config()
    lines = _emit_lines(flat)
    outdir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(outdir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return output_path
