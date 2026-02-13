from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ResolvedPath:
    """解析结果（用于生成配置与训练脚本共用）。"""

    value: str
    reason: str


def resolve_train_data_path(data_path: Optional[str], runs_root: str = "./runs") -> ResolvedPath:
    """解析训练数据路径。

    支持:
    - 显式路径: 文件存在则直接返回
    - 占位符 LATEST: 解析到 runs/chat 下最新的 sft/train.jsonl
    - legacy 回退: dataset/sft.jsonl（仅当存在且 LATEST 未命中）
    """
    raw = (data_path or "").strip()
    if not raw:
        return ResolvedPath(value="", reason="empty")

    if raw.lower() != "latest":
        return ResolvedPath(value=raw, reason="explicit")

    latest = _find_latest_chat_sft(Path(runs_root))
    if latest:
        return ResolvedPath(value=latest, reason="latest_runs_chat")

    legacy = Path("dataset") / "sft.jsonl"
    if legacy.exists() and legacy.is_file():
        return ResolvedPath(value=str(legacy.as_posix()), reason="legacy_dataset_sft")

    raise FileNotFoundError(
        "data_path=LATEST 但未找到可用训练集。\n"
        f"- 期望: {Path(runs_root) / 'chat' / '<run_id>' / 'sft' / 'train.jsonl'}\n"
        "- 你可以先运行: python cli.py data clean raw\n"
        "- 或者在 setting.jsonc 里把 data_path 改为具体 jsonl 路径"
    )


def _find_latest_chat_sft(runs_root: Path) -> Optional[str]:
    chat_root = runs_root / "chat"
    if not chat_root.exists() or not chat_root.is_dir():
        return None

    candidates: list[Path] = []
    for run_dir in chat_root.iterdir():
        if not run_dir.is_dir():
            continue
        p = run_dir / "sft" / "train.jsonl"
        if p.exists() and p.is_file():
            candidates.append(p)

    if not candidates:
        return None

    latest = max(candidates, key=lambda x: x.stat().st_mtime)
    # 尽量输出相对路径，避免工作目录不同导致混乱
    try:
        rel = os.path.relpath(latest, Path.cwd())
        return Path(rel).as_posix()
    except Exception:
        return latest.as_posix()

