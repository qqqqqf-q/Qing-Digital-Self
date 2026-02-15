"""
数据处理命令

提供QQ数据提取、清洗、格式转换、合并等数据处理功能。
支持批量处理和进度监控。
"""

import os
import json
import sys
import csv
import re
import shutil
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ..core.base import BaseCommand
from ..core.exceptions import DataProcessingError, ValidationError, FileOperationError
from ..core.helpers import (
    format_time_duration, 
    format_file_size, 
    get_file_stats, 
    ensure_directory,
    format_progress_bar
)
from ..interface.validators import validate_path, validate_positive_int
from utils.config.config import get_config


@dataclass
class _EstimateMessage:
    """估算阶段使用的轻量消息结构"""
    role: str
    content: str


@dataclass
class _EstimateQaPair:
    """估算阶段使用的轻量问答结构"""
    id: str
    messages: List[_EstimateMessage]
    images: Optional[List[str]] = None


class DataCommand(BaseCommand):
    """数据处理命令"""
    
    def __init__(self):
        super().__init__("data", "数据处理")

    def _print_migration_tip(self, message: str) -> None:
        """输出迁移提示（既打日志也打印到stdout，避免被quiet吞掉）"""
        self.logger.warning(message)
        print(f"迁移提示: {message}")

    def _canonical_source_type(self, source_type: Optional[str]) -> Optional[str]:
        """规范化数据源类型"""
        if not source_type:
            return None

        normalized = str(source_type).strip().lower()
        if normalized in {"tg", "telegram"}:
            return "telegram"
        if normalized in {"wx", "wechat"}:
            return "wechat"
        if normalized == "qq":
            return "qq"
        return None

    def _sanitize_run_tag(self, run_tag: Optional[str]) -> Optional[str]:
        if not run_tag:
            return None
        cleaned = re.sub(r"[^0-9a-zA-Z_-]+", "_", str(run_tag).strip())
        cleaned = cleaned.strip("_")
        return cleaned or None

    def _generate_run_id(self, run_id: Optional[str], run_tag: Optional[str] = None) -> str:
        """生成 run_id：YYYYMMDD_HHMMSS[_tag]"""
        if run_id and str(run_id).strip():
            return str(run_id).strip()

        base = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = self._sanitize_run_tag(run_tag)
        return f"{base}_{tag}" if tag else base

    def _runs_root(self) -> Path:
        return Path(self.config.get("runs_root", "./runs"))

    def _data_root(self) -> Path:
        return Path(self.config.get("data_root", "./data"))

    def _default_chat_original_dir(self, source_type: Optional[str]) -> Path:
        source = source_type or "qq"
        return self._data_root() / "chat" / source / "original"

    def _legacy_chat_original_dir(self) -> Path:
        return Path("./dataset/original")

    def _legacy_chat_csv_dir(self) -> Path:
        return Path("dataset/csv")

    def _is_legacy_path(self, path: str) -> bool:
        normalized = str(path).replace("\\", "/").lower()
        return "dataset/original" in normalized or "dataset/csv" in normalized or "dataset/sft.jsonl" in normalized

    def _resolve_extract_data_dir(self, args: argparse.Namespace) -> Tuple[str, bool]:
        """解析 extract 的 data_dir，并在必要时回退 legacy 目录"""
        explicit_data_dir = getattr(args, "data_dir", None)
        if explicit_data_dir:
            return explicit_data_dir, self._is_legacy_path(explicit_data_dir)

        configured = self.config.get("data_dir")
        if configured:
            configured_path = Path(str(configured))
            if configured_path.exists():
                return str(configured_path), self._is_legacy_path(str(configured_path))

            legacy = self._legacy_chat_original_dir()
            if legacy.exists():
                self._print_migration_tip(
                    f"配置 data_dir 指向 {configured_path} 但目录不存在，当前回退读取旧目录 {legacy}（后续建议迁移）"
                )
                return str(legacy), True

            return str(configured_path), self._is_legacy_path(str(configured_path))

        source_type = self._canonical_source_type(getattr(args, "source_type", None))
        candidate = self._default_chat_original_dir(source_type)
        if candidate.exists():
            return str(candidate), False

        legacy = self._legacy_chat_original_dir()
        if legacy.exists():
            self._print_migration_tip(
                f"默认数据目录已迁移到 {self._default_chat_original_dir(source_type)}，"
                f"当前回退读取旧目录 {legacy}（后续建议迁移）"
            )
            return str(legacy), True

        return str(candidate), False

    def _resolve_extract_output_dir(self, args: argparse.Namespace, source_type: Optional[str]) -> Tuple[str, Optional[str], bool]:
        """解析 extract 的输出目录。默认输出到 runs/chat/<run_id>/csv"""
        explicit_output = getattr(args, "output", None)
        if explicit_output:
            if self._is_legacy_path(explicit_output):
                self._print_migration_tip(
                    f"检测到输出仍指向旧目录 {explicit_output}，建议改为 {self._runs_root() / 'chat' / '<run_id>' / 'csv'}"
                )
            return explicit_output, None, self._is_legacy_path(explicit_output)

        run_id = self._generate_run_id(
            getattr(args, "run_id", None),
            getattr(args, "run_tag", None) or (f"chat_{source_type}" if source_type else None),
        )
        output_dir = self._runs_root() / "chat" / run_id / "csv"
        return str(output_dir), run_id, False

    def _find_latest_chat_run_id(self) -> Optional[str]:
        chat_root = self._runs_root() / "chat"
        if not chat_root.exists() or not chat_root.is_dir():
            return None

        candidates = [p for p in chat_root.iterdir() if p.is_dir()]
        if not candidates:
            return None

        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return latest.name

    def _find_latest_openai_distill_run_id(self) -> Optional[str]:
        distill_root = self._runs_root() / "openai-distill"
        if not distill_root.exists() or not distill_root.is_dir():
            return None

        candidates = [p for p in distill_root.iterdir() if p.is_dir()]
        if not candidates:
            return None

        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return latest.name

    def _try_parse_run_id_from_path(self, path: str) -> Optional[str]:
        normalized = Path(path).as_posix().replace("\\", "/")
        parts = [p for p in normalized.split("/") if p]
        # .../runs/chat/<run_id>/csv 或 .../runs/chat/<run_id>/sft/...
        runs_root_name = self._runs_root().name or "runs"
        for chat_index in range(len(parts) - 2, 0, -1):
            if parts[chat_index] != "chat":
                continue
            if parts[chat_index - 1] != runs_root_name:
                continue
            run_id = parts[chat_index + 1]
            if run_id in {"csv", "sft", "stats"}:
                continue
            return run_id
        return None

    def _resolve_clean_input_path(self, args: argparse.Namespace) -> Tuple[str, Optional[str], bool]:
        """解析 clean 的输入CSV目录。优先级: --input > --run-id > latest run > legacy dataset/csv"""
        explicit_input = getattr(args, "input", None)
        if explicit_input:
            return explicit_input, self._try_parse_run_id_from_path(explicit_input), self._is_legacy_path(explicit_input)

        run_id = getattr(args, "run_id", None)
        if run_id:
            input_dir = self._runs_root() / "chat" / str(run_id) / "csv"
            if not input_dir.exists():
                raise ValidationError(f"指定的 run_id '{run_id}' 对应的CSV目录不存在: {input_dir}")
            return str(input_dir), str(run_id), False

        latest_run = self._find_latest_chat_run_id()
        if latest_run:
            input_dir = self._runs_root() / "chat" / latest_run / "csv"
            return str(input_dir), latest_run, False

        legacy = self._legacy_chat_csv_dir()
        if legacy.exists():
            prog = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "cli.py"
            self._print_migration_tip(
                f"未找到 runs/chat 下的运行产物，当前回退读取旧目录 {legacy}；建议先运行 `{prog} data extract` 生成 runs/ 结构"
            )
            return str(legacy), None, True

        return str(legacy), None, True

    def _resolve_clean_output_path(self, args: argparse.Namespace, run_id_hint: Optional[str]) -> Tuple[str, str, bool]:
        """解析 clean 的输出SFT文件。默认输出到 runs/chat/<run_id>/sft/train.jsonl"""
        explicit_output = getattr(args, "output", None)
        if explicit_output:
            if self._is_legacy_path(explicit_output):
                self._print_migration_tip(
                    f"检测到输出仍指向旧目录 {explicit_output}，建议改为 {self._runs_root() / 'chat' / '<run_id>' / 'sft' / 'train.jsonl'}"
                )
            run_id = getattr(args, "run_id", None) or run_id_hint or self._generate_run_id(None, "chat")
            return explicit_output, str(run_id), self._is_legacy_path(explicit_output)

        run_id = getattr(args, "run_id", None) or run_id_hint or self._generate_run_id(None, "chat")
        output_path = self._runs_root() / "chat" / str(run_id) / "sft" / "train.jsonl"
        return str(output_path), str(run_id), False
    
    def execute(self, args: argparse.Namespace) -> int:
        """执行数据处理命令"""
        action = getattr(args, 'data_action', None)
        
        if action == 'extract':
            return self._extract_data(args)
        elif action == 'clean':
            return self._clean_data(args)
        elif action == 'convert':
            return self._convert_data(args)
        elif action == 'merge':
            return self._merge_data(args)
        elif action == 'preview':
            return self._preview_data(args)
        elif action == 'stats':
            return self._show_stats(args)
        elif action == 'migrate-layout':
            return self._migrate_layout_v2(args)
        elif action == 'openai-distill':
            return self._openai_distill(args)
        elif action == 'openai-clean':
            return self._openai_clean(args)
        else:
            self.logger.error("未指定数据操作")
            return 1
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """验证命令参数"""
        action = getattr(args, 'data_action', None)
        
        if action == 'extract':
            self._validate_extract_args(args)
        elif action == 'clean':
            self._validate_clean_args(args)
        elif action == 'convert':
            self._validate_convert_args(args)
        elif action == 'merge':
            self._validate_merge_args(args)
        elif action in ['preview', 'stats']:
            self._validate_input_file_args(args)
        elif action == 'migrate-layout':
            self._validate_migrate_layout_v2_args(args)
        elif action == 'openai-distill':
            self._validate_openai_distill_args(args)
        elif action == 'openai-clean':
            self._validate_openai_clean_args(args)

    def _validate_migrate_layout_v2_args(self, args: argparse.Namespace) -> None:
        """验证目录迁移参数（Data Layout v2）"""
        mode = str(getattr(args, "mode", "move") or "move").lower()
        if mode not in {"move", "copy"}:
            raise ValidationError("mode 仅支持: move / copy")

    def _migrate_layout_v2(self, args: argparse.Namespace) -> int:
        """将 legacy 的 dataset/openai_data 迁移到 data/runs（默认仅输出计划）"""
        apply_changes = bool(getattr(args, "apply", False))
        mode = str(getattr(args, "mode", "move") or "move").lower()
        force = bool(getattr(args, "force", False))
        skip_openai = bool(getattr(args, "skip_openai", False))

        data_root = Path(getattr(args, "data_root", None) or self._data_root())
        runs_root = Path(getattr(args, "runs_root", None) or self._runs_root())

        run_id = self._generate_run_id(
            getattr(args, "run_id", None),
            getattr(args, "run_tag", None) or "legacy",
        )

        plan = self._build_layout_v2_migration_plan(
            data_root=data_root,
            runs_root=runs_root,
            run_id=run_id,
            skip_openai=skip_openai,
        )

        self._print_layout_v2_migration_plan(plan, data_root=data_root, runs_root=runs_root, run_id=run_id, mode=mode, apply_changes=apply_changes)
        if not plan["ops"]:
            self.logger.warning("未发现可迁移的 legacy 数据（dataset/openai_data 都不存在或为空）")
            return 0

        if not apply_changes:
            return 0

        self._apply_layout_v2_migration_plan(plan, mode=mode, force=force)
        self._write_layout_v2_migration_manifest(
            runs_root=runs_root,
            run_id=run_id,
            mode=mode,
            skip_openai=skip_openai,
            moved_items=plan["moved_items"],
        )
        return 0

    def _build_layout_v2_migration_plan(
        self,
        data_root: Path,
        runs_root: Path,
        run_id: str,
        skip_openai: bool,
    ) -> Dict[str, Any]:
        ops: List[Dict[str, Any]] = []
        moved_items: List[Dict[str, str]] = []

        def add_op(src: Path, dst: Path, kind: str, description: str) -> None:
            ops.append(
                {
                    "src": src,
                    "dst": dst,
                    "kind": kind,
                    "description": description,
                }
            )
            moved_items.append({"src": str(src).replace("\\", "/"), "dst": str(dst).replace("\\", "/"), "kind": kind})

        # 1) dataset/original -> data/chat/{qq,telegram,wechat}/original
        legacy_original = Path("dataset") / "original"
        if legacy_original.exists() and legacy_original.is_dir():
            qq_original = data_root / "chat" / "qq" / "original"
            tg_original = data_root / "chat" / "telegram" / "original"
            wx_original = data_root / "chat" / "wechat" / "original"

            for item in sorted(legacy_original.iterdir(), key=lambda p: p.name.lower()):
                if item.is_dir() and item.name.lower() == "wechat":
                    add_op(item, wx_original, "dir", "迁移 WeChat 原始目录到 data/chat/wechat/original")
                    continue

                if item.is_dir() and (item.name.startswith("ChatExport_") or item.name.startswith("TG_ChatExport_")):
                    add_op(item, tg_original / item.name, "dir", "迁移 Telegram ChatExport 目录到 data/chat/telegram/original")
                    continue

                if item.is_file() and item.suffix.lower() in {".db", ".sql", ".sqlite", ".sqlite3"}:
                    add_op(item, qq_original / item.name, "file", "迁移 QQ 数据库/SQL 到 data/chat/qq/original")
                    continue

                # 兜底：未知文件/目录也归到 qq/original，避免丢数据
                add_op(item, qq_original / item.name, "dir" if item.is_dir() else "file", "迁移 legacy original 的其它内容到 data/chat/qq/original")

        # 2) dataset/media -> data/chat/media（优先使用配置里的 media_dir）
        legacy_media = Path("dataset") / "media"
        if legacy_media.exists() and legacy_media.is_dir():
            configured_media_dir = Path(self.config.get("media_dir", str(data_root / "chat" / "media")))
            # 迁移场景下，配置可能仍指向 legacy 目录；避免 src==dst 导致递归合并
            try:
                media_dir = data_root / "chat" / "media" if configured_media_dir.resolve() == legacy_media.resolve() else configured_media_dir
            except Exception:
                media_dir = data_root / "chat" / "media" if str(configured_media_dir).replace("\\", "/").strip("./") == "dataset/media" else configured_media_dir
            add_op(legacy_media, media_dir, "dir", "迁移媒体目录到 data/chat/media（或配置 media_dir）")

        # 3) openai_data -> data/openai-export
        legacy_openai = Path("openai_data")
        if not skip_openai and legacy_openai.exists() and legacy_openai.is_dir():
            add_op(legacy_openai, data_root / "openai-export", "dir", "迁移 ChatGPT 导出到 data/openai-export")

        # 4) dataset/csv -> runs/chat/<run_id>/csv
        legacy_csv = Path("dataset") / "csv"
        if legacy_csv.exists() and legacy_csv.is_dir():
            add_op(legacy_csv, runs_root / "chat" / run_id / "csv", "dir", "归档 legacy CSV 到 runs/chat/<run_id>/csv")

        # 5) dataset/sft.jsonl -> runs/chat/<run_id>/sft/train.jsonl
        legacy_sft = Path("dataset") / "sft.jsonl"
        if legacy_sft.exists() and legacy_sft.is_file():
            add_op(legacy_sft, runs_root / "chat" / run_id / "sft" / "train.jsonl", "file", "归档 legacy SFT 到 runs/chat/<run_id>/sft/train.jsonl")

        # 6) dataset/sft_scored.csv -> runs/chat/<run_id>/stats/sft_scored.csv
        legacy_scored = Path("dataset") / "sft_scored.csv"
        if legacy_scored.exists() and legacy_scored.is_file():
            add_op(legacy_scored, runs_root / "chat" / run_id / "stats" / "sft_scored.csv", "file", "归档 legacy scored CSV 到 runs/chat/<run_id>/stats/")

        # 7) dataset/backup + dataset/*.bak -> runs/_archive/dataset_backup_<run_id>/
        legacy_backup_dir = Path("dataset") / "backup"
        archive_root = runs_root / "_archive" / f"dataset_backup_{run_id}"
        if legacy_backup_dir.exists() and legacy_backup_dir.is_dir():
            add_op(legacy_backup_dir, archive_root / "backup", "dir", "迁移 legacy backup 到 runs/_archive/")

        legacy_root_baks = sorted((Path("dataset")).glob("*.bak"), key=lambda p: p.name.lower())
        for bak_file in legacy_root_baks:
            if bak_file.is_file():
                add_op(bak_file, archive_root / "bak" / bak_file.name, "file", "迁移 legacy 根目录 .bak 到 runs/_archive/")

        return {"ops": ops, "moved_items": moved_items}

    def _print_layout_v2_migration_plan(
        self,
        plan: Dict[str, Any],
        data_root: Path,
        runs_root: Path,
        run_id: str,
        mode: str,
        apply_changes: bool,
    ) -> None:
        ops: List[Dict[str, Any]] = plan.get("ops", [])
        print("\n=== Data Layout v2 迁移计划 ===")
        print(f"模式: {mode} | 执行: {'apply' if apply_changes else 'dry-run'} | run_id: {run_id}")
        print(f"data_root: {data_root}")
        print(f"runs_root: {runs_root}")
        print("迁移映射摘要:")
        print("- dataset/original -> data/chat/{qq,telegram,wechat}/original")
        print("- dataset/media -> data/chat/media（或配置 media_dir）")
        print("- dataset/csv -> runs/chat/<run_id>/csv")
        print("- dataset/sft.jsonl -> runs/chat/<run_id>/sft/train.jsonl")
        print("- dataset/sft_scored.csv -> runs/chat/<run_id>/stats/sft_scored.csv")
        print("- dataset/backup + dataset/*.bak -> runs/_archive/dataset_backup_<run_id>/")
        print("- openai_data -> data/openai-export（可选）")
        print(f"待处理项: {len(ops)}")
        if not apply_changes:
            print("提示: 这是 dry-run；如确认无误，追加 --apply 执行实际迁移。")
        print("=============================\n")

    def _apply_layout_v2_migration_plan(self, plan: Dict[str, Any], mode: str, force: bool) -> None:
        ops: List[Dict[str, Any]] = plan.get("ops", [])
        for op in ops:
            src = Path(op["src"])
            dst = Path(op["dst"])
            kind = op.get("kind", "file")
            desc = op.get("description", "")
            if not src.exists():
                self.logger.warning(f"跳过（不存在）: {src}")
                continue

            self.logger.info(f"{desc}: {src} -> {dst}")
            if kind == "dir":
                self._transfer_dir(src, dst, mode=mode, force=force)
            else:
                self._transfer_file(src, dst, mode=mode, force=force)

        # 清理空的 legacy 目录（仅 move 才做）
        if mode == "move":
            self._remove_empty_dirs([Path("dataset") / "original", Path("dataset") / "media"])

    def _transfer_dir(self, src: Path, dst: Path, mode: str, force: bool) -> None:
        if not src.is_dir():
            raise FileOperationError("源路径不是目录", str(src))

        ensure_directory(dst.parent)

        if dst.exists() and not dst.is_dir():
            raise FileOperationError("目标路径不是目录", str(dst))

        if not dst.exists():
            if mode == "move":
                shutil.move(str(src), str(dst))
            else:
                shutil.copytree(str(src), str(dst))
            return

        # 目标已存在：合并迁移（默认跳过同名冲突，--force 才覆盖）
        for child in sorted(src.iterdir(), key=lambda p: p.name.lower()):
            child_dst = dst / child.name
            if child.is_dir():
                self._transfer_dir(child, child_dst, mode=mode, force=force)
            else:
                self._transfer_file(child, child_dst, mode=mode, force=force)

        if mode == "move":
            self._try_remove_dir(src)

    def _transfer_file(self, src: Path, dst: Path, mode: str, force: bool) -> None:
        if not src.is_file():
            raise FileOperationError("源路径不是文件", str(src))

        ensure_directory(dst.parent)

        if dst.exists():
            if not force:
                self.logger.warning(f"跳过（目标已存在，未指定 --force）: {dst}")
                if mode == "move":
                    return
                return
            try:
                if dst.is_file():
                    dst.unlink()
                else:
                    shutil.rmtree(str(dst))
            except Exception as exc:
                raise FileOperationError(f"无法覆盖目标: {exc}", str(dst)) from exc

        if mode == "move":
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

    def _try_remove_dir(self, path: Path) -> None:
        try:
            if path.exists() and path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except Exception:
            return

    def _remove_empty_dirs(self, dirs: List[Path]) -> None:
        for d in dirs:
            self._try_remove_dir(d)

    def _write_layout_v2_migration_manifest(
        self,
        runs_root: Path,
        run_id: str,
        mode: str,
        skip_openai: bool,
        moved_items: List[Dict[str, str]],
    ) -> None:
        manifest_path = runs_root / "chat" / run_id / "manifest.json"
        ensure_directory(manifest_path.parent)
        payload = {
            "pipeline": "chat",
            "run_id": run_id,
            "kind": "layout_migration_v2",
            "mode": mode,
            "skip_openai": skip_openai,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "items": moved_items,
        }
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已写入 manifest: {manifest_path}")
        except Exception as exc:
            self.logger.warning(f"写入 manifest 失败: {exc}")

    def _resolve_openai_export_input(self, args: argparse.Namespace, data_root: Path) -> Path:
        explicit = getattr(args, "input", None)
        if explicit:
            return Path(str(explicit))

        candidate = data_root / "openai-export" / "conversations.json"
        if candidate.exists():
            return candidate

        candidate_dir = data_root / "openai-export"
        if candidate_dir.exists() and candidate_dir.is_dir():
            for p in candidate_dir.rglob("conversations.json"):
                if p.is_file():
                    return candidate_dir

        legacy = Path("openai_data") / "conversations.json"
        if legacy.exists():
            self._print_migration_tip(
                f"检测到仍在使用 legacy 导出 {legacy}，建议迁移到 {candidate}"
            )
            return legacy

        return candidate_dir if candidate_dir.exists() else candidate

    def _validate_openai_distill_args(self, args: argparse.Namespace) -> None:
        data_root = Path(getattr(args, "data_root", None) or self._data_root())
        input_path = self._resolve_openai_export_input(args, data_root=data_root)
        validate_path(str(input_path), must_exist=True)

        try:
            from process_data.openai_export_distill import discover_openai_export_sources
        except Exception as e:
            raise ValidationError(f"无法加载 OpenAI 导出解析模块: {e}")

        sources = discover_openai_export_sources(input_path)
        if not sources:
            raise ValidationError(f"未在目录中找到 conversations.json: {input_path}")

        max_chars = int(getattr(args, "max_chars", 20000) or 20000)
        max_messages = int(getattr(args, "max_messages", 80) or 80)
        if max_chars <= 0:
            raise ValidationError("max_chars 必须为正整数")
        if max_messages <= 0:
            raise ValidationError("max_messages 必须为正整数")

        pii_policy = str(getattr(args, "pii_policy", "mask") or "mask").lower()
        if pii_policy not in {"mask", "drop", "keep"}:
            raise ValidationError("pii_policy 仅支持: mask / drop / keep")

    def _openai_distill(self, args: argparse.Namespace) -> int:
        """从 OpenAI-Export(conversations.json) 生成 SFT 训练集（文本版）。"""
        try:
            from process_data.openai_export_distill import DistillOptions, distill_openai_export
        except Exception as e:
            raise DataProcessingError(f"无法加载 OpenAI 导出解析模块: {e}")

        data_root = Path(getattr(args, "data_root", None) or self._data_root())
        runs_root = Path(getattr(args, "runs_root", None) or self._runs_root())
        input_path = self._resolve_openai_export_input(args, data_root=data_root)

        run_id = self._generate_run_id(
            getattr(args, "run_id", None),
            getattr(args, "run_tag", None) or "openai4o",
        )
        output_root = runs_root / "openai-distill" / run_id

        allow_models_raw = str(getattr(args, "allow_models", "") or "")
        allow_models = {m.strip() for m in allow_models_raw.split(",") if m.strip()}
        options = DistillOptions(
            allow_models=allow_models,
            cutoff_ts=getattr(args, "cutoff_ts", None),
            pii_policy=str(getattr(args, "pii_policy", "mask") or "mask").lower(),
            keep_system=bool(getattr(args, "keep_system", False)),
            keep_code=bool(getattr(args, "keep_code", False)),
            keep_tool=bool(getattr(args, "keep_tool", False)),
            max_chars=int(getattr(args, "max_chars", 20000) or 20000),
            max_messages=int(getattr(args, "max_messages", 80) or 80),
        )

        distill_openai_export(
            input_path=input_path,
            output_root=output_root,
            run_id=run_id,
            options=options,
        )

        self.logger.info(f"run_id: {run_id}")
        print(f"run_id: {run_id}")
        print(f"SFT: {(output_root / 'sft' / 'text.jsonl').as_posix()}")
        return 0

    def _resolve_openai_clean_input(self, args: argparse.Namespace, runs_root: Path) -> Path:
        explicit = getattr(args, "input", None)
        if explicit:
            return Path(str(explicit))

        distill_run_id = getattr(args, "distill_run_id", None)
        if distill_run_id:
            candidate = runs_root / "openai-distill" / str(distill_run_id) / "sft" / "text.jsonl"
            if not candidate.exists():
                raise ValidationError(f"指定的 distill_run_id '{distill_run_id}' 不存在: {candidate}")
            return candidate

        latest = self._find_latest_openai_distill_run_id()
        if latest:
            return runs_root / "openai-distill" / latest / "sft" / "text.jsonl"

        return runs_root / "openai-distill" / "LATEST" / "sft" / "text.jsonl"

    def _validate_openai_clean_args(self, args: argparse.Namespace) -> None:
        runs_root = Path(getattr(args, "runs_root", None) or self._runs_root())
        input_path = self._resolve_openai_clean_input(args, runs_root=runs_root)
        validate_path(str(input_path), must_exist=True)

        workers = getattr(args, "workers", None)
        if workers is not None:
            validate_positive_int(workers, "workers")

        max_chars = getattr(args, "max_chars", None)
        if max_chars is not None:
            validate_positive_int(max_chars, "max_chars")

        max_messages = getattr(args, "max_messages", None)
        if max_messages is not None:
            validate_positive_int(max_messages, "max_messages")

        max_tokens = getattr(args, "max_tokens", None)
        if max_tokens is not None:
            validate_positive_int(max_tokens, "max_tokens")

        max_samples = getattr(args, "max_samples", None)
        if max_samples is not None:
            validate_positive_int(max_samples, "max_samples")

        temperature = getattr(args, "temperature", None)
        if temperature is not None:
            try:
                float(temperature)
            except (TypeError, ValueError):
                raise ValidationError("temperature 必须为数字")

        base_prompt_file = getattr(args, "base_prompt_file", None)
        if base_prompt_file:
            validate_path(str(base_prompt_file), must_exist=True)

    def _openai_clean(self, args: argparse.Namespace) -> int:
        """对 openai-distill 产物做 LLM 清洗，去除技术/工具/搜索痕迹。"""
        try:
            from process_data.openai_export_llm_clean import OpenAICleanOptions, clean_openai_sft_jsonl_with_llm
        except Exception as e:
            raise DataProcessingError(f"无法加载 OpenAI LLM 清洗模块: {e}")

        runs_root = Path(getattr(args, "runs_root", None) or self._runs_root())
        input_path = self._resolve_openai_clean_input(args, runs_root=runs_root)

        run_id = self._generate_run_id(
            getattr(args, "run_id", None),
            getattr(args, "run_tag", None) or "openai-clean",
        )
        output_root = runs_root / "openai-clean" / run_id

        sft_system_prompt: Optional[str] = None
        if bool(getattr(args, "no_base_prompt", False)):
            sft_system_prompt = "*"
        else:
            prompt_file = getattr(args, "base_prompt_file", None)
            if prompt_file:
                with open(str(prompt_file), "r", encoding="utf-8") as f:
                    sft_system_prompt = f.read()
            else:
                sft_system_prompt = getattr(args, "base_prompt", None)
                if sft_system_prompt is None:
                    sft_system_prompt = self.config.get("openai_sft_system_prompt", "*")

        model = getattr(args, "model", None) or self.config.get("OpenAI_Model")
        temperature = getattr(args, "temperature", None)
        if temperature is None:
            temperature = 0.2
        max_tokens = getattr(args, "max_tokens", None)
        if max_tokens is None:
            max_tokens = 4096
        workers = getattr(args, "workers", None)
        if workers is None:
            workers = self.config.get("clean_workers", 4)
        max_chars = getattr(args, "max_chars", None)
        if max_chars is None:
            max_chars = 20000
        max_messages = getattr(args, "max_messages", None)
        if max_messages is None:
            max_messages = 80

        options_kwargs: Dict[str, Any] = {
            "model": model,
            "temperature": float(temperature or 0.2),
            "max_tokens": int(max_tokens or 4096),
            "workers": int(workers or 4),
            "max_chars": int(max_chars or 20000),
            "max_messages": int(max_messages or 80),
            "max_samples": getattr(args, "max_samples", None),
        }
        if sft_system_prompt is not None:
            options_kwargs["base_prompt"] = sft_system_prompt

        options = OpenAICleanOptions(**options_kwargs)

        clean_openai_sft_jsonl_with_llm(
            input_path=Path(input_path),
            output_root=output_root,
            run_id=run_id,
            options=options,
        )

        self.logger.info(f"run_id: {run_id}")
        print(f"run_id: {run_id}")
        print(f"SFT: {(output_root / 'sft' / 'train.jsonl').as_posix()}")
        return 0
	    
    def _validate_extract_args(self, args: argparse.Namespace) -> None:
        """验证数据提取参数"""
        # 获取数据源类型
        source_type = self._canonical_source_type(getattr(args, 'source_type', None))
        data_dir, _ = self._resolve_extract_data_dir(args)
        
        # 验证数据目录存在
        if not os.path.exists(data_dir):
            raise ValidationError(f"数据目录不存在: {data_dir}")
        
        # 根据数据源类型进行特定验证
        if source_type == 'qq':
            qq_c2c_db_path = getattr(args, 'qq_c2c_db_path', None) or self.config.get('qq_c2c_db_path') or self.config.get('qq_db_path')
            qq_group_db_path = getattr(args, 'qq_group_db_path', None) or self.config.get('qq_group_db_path')

            if qq_c2c_db_path and not os.path.exists(qq_c2c_db_path):
                raise ValidationError(f"QQ私聊数据库/SQL文件不存在: {qq_c2c_db_path}")
            if qq_group_db_path and not os.path.exists(qq_group_db_path):
                raise ValidationError(f"QQ群聊数据库/SQL文件不存在: {qq_group_db_path}")
        
        # 输出路径验证
        output_path = getattr(args, 'output')
        if output_path:
            validate_path(output_path, must_exist=False, check_parent=True)
    
    def _validate_clean_args(self, args: argparse.Namespace) -> None:
        """验证数据清洗参数"""
        method = getattr(args, 'clean_method', None)
        
        if method == 'estimate':
            self._validate_clean_estimate_args(args)
            return
        
        input_path, run_id_hint, _ = self._resolve_clean_input_path(args)
        output_path, _, _ = self._resolve_clean_output_path(args, run_id_hint)
        
        if method == 'rellm':
            validate_path(output_path, must_exist=False)
            validate_path(input_path, must_exist=True)
            scored_path = getattr(args, 'scored', None) or os.path.splitext(output_path)[0] + "_scored.csv"
            validate_path(scored_path, must_exist=True)
            accept_score = getattr(args, 'accept_score', None)
            if accept_score is not None:
                validate_positive_int(accept_score, "accept_score")
            return
        
        validate_path(input_path, must_exist=True)
        validate_path(output_path, must_exist=False)
        
        if hasattr(args, 'batch_size') and getattr(args, 'batch_size', None) is not None:
            validate_positive_int(args.batch_size, "batch_size")
        
        if hasattr(args, 'workers') and getattr(args, 'workers', None) is not None:
            validate_positive_int(args.workers, "workers")
    
    def _validate_clean_estimate_args(self, args: argparse.Namespace) -> None:
        """验证清洗估算参数"""
        estimate_method = getattr(args, 'estimate_method', None)
        if estimate_method != 'llm':
            raise ValidationError("估算目前仅支持 llm 策略")
        
        input_path, _, _ = self._resolve_clean_input_path(args)
        validate_path(input_path, must_exist=True)
        
        if hasattr(args, 'batch_size') and getattr(args, 'batch_size', None) is not None:
            validate_positive_int(args.batch_size, "batch_size")
        
        if hasattr(args, 'workers') and getattr(args, 'workers', None) is not None:
            validate_positive_int(args.workers, "workers")
    
    def _validate_convert_args(self, args: argparse.Namespace) -> None:
        """验证数据转换参数"""
        validate_path(args.input, must_exist=True)
        validate_path(args.output, must_exist=False, check_parent=True)
        
        valid_formats = ['chatml', 'alpaca', 'sharegpt']
        if hasattr(args, 'format') and args.format not in valid_formats:
            raise ValidationError(f"无效的数据格式: {args.format}, 支持的格式: {valid_formats}")
    
    def _validate_merge_args(self, args: argparse.Namespace) -> None:
        """验证数据合并参数"""
        for input_file in args.inputs:
            validate_path(input_file, must_exist=True)
        
        validate_path(args.output, must_exist=False, check_parent=True)
    
    def _validate_input_file_args(self, args: argparse.Namespace) -> None:
        """验证输入文件参数"""
        validate_path(args.input, must_exist=True)
    
    def _extract_data(self, args: argparse.Namespace) -> int:
        """从聊天数据中提取数据（支持QQ和Telegram）"""
        try:
            self.logger.info("开始从聊天数据中提取数据...")
            
            # 准备参数
            source_type = self._canonical_source_type(getattr(args, 'source_type', None))
            data_dir, used_legacy = self._resolve_extract_data_dir(args)
            output_path, run_id, _ = self._resolve_extract_output_dir(args, source_type)

            if used_legacy and source_type:
                self._print_migration_tip(f"建议将 {source_type} 原始数据放到 {self._default_chat_original_dir(source_type)}")
            
            # 确保输出目录存在
            ensure_directory(output_path)
            
            # 构建提取命令参数
            extract_args = {
                'data_dir': data_dir,
                'output_dir': output_path,
                'source_type': source_type,
                # QQ相关参数
                'qq_c2c_db_path': getattr(args, 'qq_c2c_db_path', None) or self.config.get('qq_c2c_db_path') or self.config.get('qq_db_path'),
                'qq_group_db_path': getattr(args, 'qq_group_db_path', None) or self.config.get('qq_group_db_path'),
                'qq_number_ai': getattr(args, 'qq_number_ai', None) or self.config.get('qq_number_ai'),
                # Telegram相关参数
                'telegram_chat_id': getattr(args, 'telegram_chat_id', None) or self.config.get('telegram_chat_id'),
                'tg_data_dir': getattr(args, 'tg_data_dir', None) or self.config.get('tg_data_dir')
            }
            
            # 执行数据提取
            result = self._execute_unified_data_extraction(extract_args)
            
            if result == 0:
                # 显示提取结果统计
                self._show_extraction_stats(output_path)
                if run_id:
                    self.logger.info(f"run_id: {run_id}")
                    print(f"run_id: {run_id}")
                self.logger.info(f"数据提取完成: {output_path}")
            
            return result
            
        except Exception as e:
            raise DataProcessingError(f"数据提取失败: {e}")
    
    def _execute_unified_data_extraction(self, extract_args: Dict[str, Any]) -> int:
        """执行统一数据提取过程"""
        try:
            # 导入统一解析器
            from process_data.chat_parser.generate_parser import UnifiedParser
            
            # 创建统一解析器
            unified_parser = UnifiedParser(
                data_dir=extract_args['data_dir'],
                output_dir=extract_args['output_dir']
            )
            
            # 构建解析参数
            parse_kwargs = {}
            
            # 添加QQ相关参数
            if extract_args.get('qq_c2c_db_path'):
                parse_kwargs['qq_c2c_db_path'] = extract_args['qq_c2c_db_path']
            if extract_args.get('qq_group_db_path'):
                parse_kwargs['qq_group_db_path'] = extract_args['qq_group_db_path']
            if extract_args.get('qq_number_ai'):
                parse_kwargs['qq_number_ai'] = extract_args['qq_number_ai']
            
            # 添加Telegram相关参数
            if extract_args.get('telegram_chat_id'):
                parse_kwargs['telegram_chat_id'] = extract_args['telegram_chat_id']
            if extract_args.get('tg_data_dir'):
                parse_kwargs['tg_data_dir'] = extract_args['tg_data_dir']
            
            # 执行解析
            if extract_args.get('source_type'):
                # 指定数据源类型
                from process_data.chat_parser.generate_parser import DataSourceType
                
                if extract_args['source_type'] == 'qq':
                    source_type = DataSourceType.QQ
                elif extract_args['source_type'] in ['tg', 'telegram']:
                    source_type = DataSourceType.TELEGRAM
                elif extract_args['source_type'] in ['wx', 'wechat']:
                    source_type = DataSourceType.WECHAT
                else:
                    raise DataProcessingError(f"不支持的数据源类型: {extract_args['source_type']}")
                
                return unified_parser.parse_with_source_type(source_type, **parse_kwargs)
            else:
                # 自动检测数据源
                return unified_parser.parse_auto(**parse_kwargs)
            
        except ImportError as e:
            self.logger.error(f"无法导入统一解析器: {e}")
            # 降级到原有的QQ解析器
            return self._fallback_to_qq_parser(extract_args)
        except Exception as e:
            raise DataProcessingError(f"统一数据提取执行失败: {e}")
    
    def _fallback_to_qq_parser(self, extract_args: Dict[str, Any]) -> int:
        """降级到原有的QQ解析器方法"""
        try:
            # 导入QQ解析器
            from process_data.chat_parser.qq_parser import QQParser
            from process_data.chat_parser.qq_group_parser import QQGroupParser
            
            qq_c2c_db_path = extract_args.get('qq_c2c_db_path')
            qq_group_db_path = extract_args.get('qq_group_db_path')
            if not qq_c2c_db_path and not qq_group_db_path:
                self.logger.error("降级到QQ解析器时未指定QQ私聊/群聊数据库路径")
                return 1

            if qq_c2c_db_path:
                QQParser(
                    db_path=qq_c2c_db_path,
                    output_dir=extract_args['output_dir'],
                    qq_number_ai=extract_args.get('qq_number_ai')
                ).parse_all()

            if qq_group_db_path:
                QQGroupParser(
                    db_path=qq_group_db_path,
                    output_dir=extract_args['output_dir'],
                    qq_number_ai=extract_args.get('qq_number_ai')
                ).parse_all()
            return 0
            
        except ImportError as e:
            self.logger.error(f"无法导入QQ解析模块: {e}")
            # 继续降级到直接调用脚本
            return self._fallback_extract_data(extract_args)
        except Exception as e:
            raise DataProcessingError(f"QQ解析器降级失败: {e}")
    def _fallback_extract_data(self, extract_args: Dict[str, Any]) -> int:
        """降级的数据提取方法，直接调用脚本"""
        try:
            # 构建命令行参数
            cmd = [
                sys.executable,
                'generate_training_data.py',
                '--extract-only'
            ]
            
            if extract_args.get('qq_c2c_db_path'):
                cmd.extend(['--qq-db-path', extract_args['qq_c2c_db_path']])
            
            if extract_args.get('qq_number_ai'):
                cmd.extend(['--qq-number-ai', extract_args['qq_number_ai']])
            
            if extract_args.get('output_dir'):
                cmd.extend(['--output', extract_args['output_dir']])
            
            # 执行命令
            self.logger.info(f"执行命令: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            # 实时显示输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.logger.info(output.strip())
                    print(output.strip())
            
            return_code = process.poll()
            
            if return_code != 0:
                stderr = process.stderr.read()
                self.logger.error(f"数据提取失败: {stderr}")
            
            return return_code
            
        except Exception as e:
            raise DataProcessingError(f"降级数据提取失败: {e}")
    
    def _show_extraction_stats(self, output_path: str) -> None:
        """显示提取结果统计"""
        try:
            if not os.path.exists(output_path):
                return
            
            file_stats = get_file_stats(output_path)
            self.logger.info(f"提取结果统计: {output_path}")
            print(f"\n提取结果统计:")
            print(f"输出文件: {output_path}")
            print(f"文件大小: {file_stats['size']}")
            print(f"修改时间: {file_stats['modified']}")
            
            # 尝试统计记录数
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    if output_path.endswith('.json'):
                        data = json.load(f)
                        if isinstance(data, list):
                            self.logger.info(f"提取记录数量: {len(data)}")
                            print(f"记录数量: {len(data)}")
                        elif isinstance(data, dict) and 'messages' in data:
                            self.logger.info(f"提取记录数量: {len(data['messages'])}")
                            print(f"记录数量: {len(data['messages'])}")
                    elif output_path.endswith('.jsonl'):
                        count = sum(1 for _ in f)
                        self.logger.info(f"提取记录数量: {count}")
                        print(f"记录数量: {count}")
            except:
                pass
                
        except Exception as e:
            self.logger.warning(f"显示统计信息失败: {e}")
    
    def _clean_data(self, args: argparse.Namespace) -> int:
        """清洗训练数据"""
        try:
            method = getattr(args, 'clean_method', None)
            if not method:
                self.logger.error("未指定清洗方法，请使用 'raw' 或 'llm'")
                return 1
                
            self.logger.info(f"开始清洗训练数据，使用方法: {method}")
            
            if method == 'estimate':
                return self._clean_data_estimate(args)
            
            input_path, run_id_hint, used_legacy_input = self._resolve_clean_input_path(args)
            output_path, run_id, used_legacy_output = self._resolve_clean_output_path(args, run_id_hint)
            
            if used_legacy_input:
                self._print_migration_tip(f"clean 的默认输入已迁移到 {self._runs_root() / 'chat' / '<run_id>' / 'csv'}，当前仍在读取旧目录")
            if used_legacy_output:
                self._print_migration_tip(f"clean 的默认输出已迁移到 {self._runs_root() / 'chat' / '<run_id>' / 'sft' / 'train.jsonl'}，当前仍在写旧目录")

            self.logger.info(f"run_id: {run_id}")
            print(f"run_id: {run_id}")
            self.logger.info(f"输出路径: {output_path}")
            
            # 确保输出目录存在
            ensure_directory(os.path.dirname(output_path))
            
            # 根据清洗方法执行
            if method == 'rellm':
                accept_score = getattr(args, 'accept_score', None) or self.config.get('accept_score', 2)
                scored_path = getattr(args, 'scored', None) or os.path.splitext(output_path)[0] + "_scored.csv"
                self.logger.info(f"输入路径: {input_path}")
                self.logger.info(f"打分结果路径: {scored_path}")
                self.logger.info(f"目标分数阈值: {accept_score}")
                if not os.path.exists(scored_path):
                    raise FileOperationError("打分结果文件不存在", scored_path)
                result = self._clean_data_rellm(scored_path, input_path, output_path, accept_score)
            else:
                self.logger.info(f"输入路径: {input_path}")
                
                # 检查输入路径
                if not os.path.exists(input_path):
                    raise FileOperationError("输入路径不存在", input_path)
                
                if method == 'llm':
                    # 获取LLM清洗策略参数
                    batch_size = getattr(args, 'batch_size', None) or self.config.get('clean_batch_size', 10)
                    workers = getattr(args, 'workers', None) or self.config.get('clean_workers', 4)
                    self.logger.info(f"批处理大小: {batch_size}")
                    self.logger.info(f"工作线程数: {workers}")
                    parser = getattr(args, 'parser', None) or self.config.get('llm_parser', 'scoring')
                    accept_score = getattr(args, 'accept_score', None) or self.config.get('accept_score', 2)
                    result = self._clean_data_llm(input_path, output_path, batch_size, workers, parser, accept_score)
                else:  # raw
                    result = self._clean_data_raw(input_path, output_path)
            
            if result == 0:
                if os.path.exists(output_path):
                    output_stats = get_file_stats(output_path)
                    self.logger.info(f"数据清洗完成: {output_path} ({output_stats['size']})")
            
            return result
            
        except Exception as e:
            raise DataProcessingError(f"数据清洗失败: {e}")

    def _clean_data_estimate(self, args: argparse.Namespace) -> int:
        """估算LLM清洗资源开销"""
        try:
            estimate_method = getattr(args, 'estimate_method', None)
            if estimate_method != 'llm':
                self.logger.error("当前仅支持 llm 估算策略")
                return 1
            
            input_path, _, _ = self._resolve_clean_input_path(args)
            parser = getattr(args, 'parser', 'scoring')
            batch_size = getattr(args, 'batch_size', None) or self.config.get('clean_batch_size', 10)
            workers = getattr(args, 'workers', None) or self.config.get('clean_workers', 4)
            accept_score = getattr(args, 'accept_score', None) or self.config.get('accept_score', 2)
            
            self.logger.info(f"估算输入路径: {input_path}")
            self.logger.info(f"估算处理策略: {parser}")
            self.logger.info(f"估算批处理大小: {batch_size}")
            self.logger.info(f"估算工作线程数(对齐配置使用): {workers}")
            self.logger.info(f"估算分数阈值: {accept_score}")
            
            return self._estimate_llm_clean(
                input_path=input_path,
                batch_size=batch_size,
                parser=parser,
                accept_score=accept_score
            )
        except Exception as e:
            raise DataProcessingError(f"清洗资源估算失败: {e}")

    def _clean_data_llm(self, input_path: str, output_path: str, batch_size: int, workers: int, parser: str = 'scoring', accept_score: int = 2) -> int:
        """使用LLM清洗数据"""
        try:
            from process_data.generate_chatml_llm import LLMDataProcessor
            
            self.logger.info(f"开始LLM清洗，策略: {parser}")
            if parser in ('scoring', 'default'):
                self.logger.info(f"分数阈值: {accept_score}")
                self.logger.info(f"批处理大小: {batch_size}")
                self.logger.info(f"工作线程数: {workers}")
            
            # 创建LLM数据处理器
            processor = LLMDataProcessor(
                parser=parser,
                accept_score=accept_score,
                batch_size=batch_size,
                workers=workers
            )

            # 处理文件
            scored_csv = os.path.splitext(output_path)[0] + "_scored.csv"
            result = processor.process_file(input_path, output_path, scored_csv=scored_csv)

            if result == 0:
                self.logger.info(f"LLM清洗完成: {output_path}")
                self.logger.info(f"完整打分结果: {scored_csv}")
            else:
                self.logger.error("LLM清洗失败")
                # 失败时回退到raw方法
                self.logger.warning("回退到raw清洗方法")
                return self._clean_data_raw(input_path, output_path)
            
            return result
            
        except ImportError as e:
            self.logger.error(f"无法导入LLM清洗模块: {e}")
            self.logger.warning("回退到raw清洗方法")
            return self._clean_data_raw(input_path, output_path)
        except Exception as e:
            self.logger.error(f"LLM清洗失败: {e}")
            self.logger.warning("回退到raw清洗方法")
            return self._clean_data_raw(input_path, output_path)
    
    def _estimate_llm_clean(self, input_path: str, batch_size: int, parser: str, accept_score: int) -> int:
        """估算LLM清洗时的字符传输量"""
        try:
            if parser != 'scoring':
                raise ValidationError("估算暂时仅支持 scoring 策略")
            
            clean_set_args = self.config.get('clean_set_args', {})
            if not isinstance(clean_set_args, dict):
                clean_set_args = {}
            openai_api = clean_set_args.get('openai_api', {})
            if not isinstance(openai_api, dict):
                openai_api = {}
            configured_batch = openai_api.get('clean_batch_size')
            fallback_batch = self.config.get('clean_batch_size', 10)
            candidate_batch = batch_size or configured_batch or fallback_batch
            try:
                actual_batch_size = int(candidate_batch)
            except (TypeError, ValueError):
                actual_batch_size = fallback_batch
            actual_batch_size = max(1, actual_batch_size)
            
            scoring_prompt = None
            qa_pairs: List[Any] = []
            
            try:
                from process_data.generate_chatml_llm import LLMDataProcessor  # type: ignore
                processor = LLMDataProcessor(
                    parser=parser,
                    accept_score=accept_score,
                    batch_size=actual_batch_size
                )
                qa_pairs = processor._load_qa_pairs(input_path)
                scoring_prompt = processor.strategy._build_scoring_prompt()
            except ImportError as import_error:
                if 'pandas' not in str(import_error):
                    raise DataProcessingError(f"无法加载LLM估算模块: {import_error}") from import_error
                self.logger.warning("检测到缺少pandas，使用轻量估算逻辑")
                scoring_prompt = self._fallback_scoring_prompt()
                qa_pairs = self._load_qa_pairs_for_estimate(input_path)
            except Exception as module_error:
                raise DataProcessingError(f"初始化LLM估算模块失败: {module_error}") from module_error
            
            if not qa_pairs:
                self.logger.warning("没有找到可估算的问答对")
                print("未找到可估算的问答对")
                return 0
            
            prompt_prefix = "请评估以下问答对：\n"
            
            total_input_english = 0
            total_input_chinese = 0
            total_output_english = 0
            total_output_chinese = 0
            skipped_with_images = 0
            effective_pairs = 0
            request_batches = 0
            
            for batch in self._iter_batches(qa_pairs, actual_batch_size):
                qa_list = []
                for qa in batch:
                    if getattr(qa, 'images', None):
                        skipped_with_images += 1
                        continue
                    
                    user_msg = next((msg.content for msg in qa.messages if msg.role == 'user'), '')
                    assistant_msg = next((msg.content for msg in qa.messages if msg.role == 'assistant'), '')
                    qa_list.append({
                        "id": qa.id,
                        "Q": user_msg,
                        "A": assistant_msg
                    })
                
                if not qa_list:
                    continue
                
                qa_list_json = json.dumps(qa_list, ensure_ascii=False)
                eng_in, chi_in = self._count_char_types(scoring_prompt + prompt_prefix + qa_list_json)
                total_input_english += eng_in
                total_input_chinese += chi_in
                
                estimated_output = json.dumps(
                    [{"id": item["id"], "score": 0} for item in qa_list],
                    ensure_ascii=False
                )
                eng_out, chi_out = self._count_char_types(estimated_output)
                total_output_english += eng_out
                total_output_chinese += chi_out
                
                effective_pairs += len(qa_list)
                request_batches += 1
            
            if request_batches == 0:
                self.logger.warning("所有问答均包含图片或无有效内容，未产生估算请求")
                print("所有问答均被跳过，未产生估算请求")
                print("模型输入字符总数: 0")
                print("模型输出字符总数: 0")
                print("模型输入字符预估Token数(英0.3/中0.6): 0.00")
                print("模型输出字符预估Token数(英0.3/中0.6): 0.00")
                return 0
            
            total_input_chars = total_input_english + total_input_chinese
            total_output_chars = total_output_english + total_output_chinese
            input_tokens = self._estimate_tokens(total_input_english, total_input_chinese)
            output_tokens = self._estimate_tokens(total_output_english, total_output_chinese)
            
            print(f"估算批次数: {request_batches}")
            print(f"参与估算的问答数量: {effective_pairs}")
            if skipped_with_images:
                print(f"包含图片而跳过的问答数量: {skipped_with_images}")
            print(f"模型输入字符总数: {total_input_chars}")
            print(f"模型输出字符总数: {total_output_chars}")
            print(f"模型输入字符预估Token数(英0.3/中0.6): {input_tokens:.2f}")
            print(f"模型输出字符预估Token数(英0.3/中0.6): {output_tokens:.2f}")
            
            self.logger.info(
                f"估算完成，批次数: {request_batches}, 输入字符: {total_input_chars}, 输出字符: {total_output_chars}\n"
                f"输入token: {input_tokens:.2f}\n输出token: {output_tokens:.2f}"
            )
            return 0
        
        except ValidationError:
            raise
        except Exception as e:
            raise DataProcessingError(f"LLM清洗字符估算失败: {e}")
    
    def _fallback_scoring_prompt(self) -> str:
        """当无法导入LLM模块时使用的默认打分提示词"""
        return """# 角色
你是一个数据质量评估员。

# 任务
你的任务是评估下面提供的聊天记录的**逻辑性**、**相关性**以及**风格代表性**。目标是识别并过滤掉那些回答与问题**明显不匹配**、**逻辑严重混乱**的样本，筛选出具有人类聊天风格独特性与辨识度的样本。请根据以下核心评估点给出一个1到5的整数分数，并将该分数与原始 `id` 一起输出。

**重要考量:**
1.  **简短回答的有效性:** 请注意，诸如“好的”、“是的”、“收到”、“嗯”、“知道了”等简短的肯定、确认或应答，在合适的语境下是完全**有逻辑且相关的**。**不要仅仅因为回答简短就将其评为低分。** 只有当这类简短回答与【问题/上下文 Q】**明显不符**时，才应考虑低分。
2.  **处理错别字和自我纠正:** 聊天记录中可能包含常见的打字错误（错别字）或用户先打错字随后又自行纠正的情况（例如，发送“我想去1楼”紧接着又发送“*2楼”进行更正）。在评估时，请**聚焦于用户想要表达的最终意图和信息的核心内容**，而**不应仅仅因为存在错别字或纠正过程就判定为低质量**。

# 核心评估点 (请在心中衡量)
1.  **相关性 (Relevance):** 【回答 A】是否直接回应或恰当地衔接了【问题/上下文 Q】？只有当两者**明显矛盾**或**完全不相关**时，才应给出低分。
2.  **逻辑性 (Coherence):** 【回答 A】是否在语义与结构上自洽？当回答**逻辑混乱**或**与上下文冲突**时，才应给出低分。
3. **风格代表性 (Style Representativeness):** 关注回答是否展现出自然的人类聊天风格，例如特定语气、情绪表达、俚语或口头禅等。风格代表性是获得5分的必要条件，但不是低分的主要依据。

# 评分标准 (1-5分)
*   **1分 (极差):** 聊天记录完全不相关，或逻辑严重混乱。
*   **2分 (差):** 大部分问答相关性低，存在明显逻辑问题。
*   **3分 (中等):** 相关性尚可但不突出，逻辑基本成立。
*   **4分 (良好):** 问答高度相关，逻辑清晰。
*   **5分 (优秀):** 满足4分标准，并展现明显的人类聊天风格特征。

# 输出要求
请严格按照以下 JSON 格式输出，每条记录仅包含原始 `id` 与评估的整数分数 `score`，不要包含任何额外说明：
[
  {
    "id": "<这里填入第1条输入数据的id值>",
    "score": <1-5的整数评分>
  },
  {
    "id": "<这里填入第2条输入数据的id值>",
    "score": <1-5的整数评分>
  }
  …
]"""
    
    def _load_qa_pairs_for_estimate(self, input_path: str) -> List[_EstimateQaPair]:
        """在缺少依赖时加载问答数据用于估算"""
        qa_pairs: List[_EstimateQaPair] = []
        
        if os.path.isfile(input_path):
            lower_name = input_path.lower()
            if lower_name.endswith('.jsonl'):
                qa_pairs.extend(self._load_qa_pairs_from_jsonl_for_estimate(input_path))
            elif lower_name.endswith('.csv'):
                qa_pairs.extend(self._load_qa_pairs_from_csv_for_estimate(input_path))
            else:
                raise DataProcessingError(f"不支持的估算输入格式: {input_path}")
        elif os.path.isdir(input_path):
            csv_files = self._collect_csv_files_for_estimate(input_path)
            for csv_file in csv_files:
                qa_pairs.extend(self._load_qa_pairs_from_csv_for_estimate(csv_file))
            if not csv_files:
                raise DataProcessingError("估算目录中未找到CSV文件")
        else:
            raise DataProcessingError(f"估算输入路径无效: {input_path}")
        
        return qa_pairs
    
    def _load_qa_pairs_from_jsonl_for_estimate(self, file_path: str) -> List[_EstimateQaPair]:
        """备用JSONL加载逻辑"""
        qa_pairs: List[_EstimateQaPair] = []
        try:
            with open(file_path, 'r', encoding='utf-8') as fp:
                for idx, line in enumerate(fp, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    raw_messages = []
                    if isinstance(data, dict):
                        if 'messages' in data and isinstance(data['messages'], list):
                            raw_messages = data['messages']
                        elif 'conversations' in data and isinstance(data['conversations'], list):
                            raw_messages = data['conversations']
                    
                    messages: List[_EstimateMessage] = []
                    for msg in raw_messages:
                        role = msg.get('role') or msg.get('from') or 'user'
                        content = msg.get('content') or msg.get('value') or ''
                        content = str(content).strip()
                        if not content:
                            continue
                        messages.append(_EstimateMessage(role=role, content=content))
                    
                    if messages:
                        qa_pairs.append(_EstimateQaPair(
                            id=f"{os.path.basename(file_path)}_{idx}",
                            messages=messages,
                            images=data.get('images')
                        ))
        except OSError as exc:
            raise DataProcessingError(f"读取JSONL失败: {exc}") from exc
        
        return qa_pairs
    
    def _load_qa_pairs_from_csv_for_estimate(self, csv_path: str) -> List[_EstimateQaPair]:
        """备用CSV加载逻辑"""
        qa_pairs: List[_EstimateQaPair] = []
        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as fp:
                reader = csv.DictReader(fp)
                if not reader.fieldnames:
                    return qa_pairs
                
                conversations: List[List[_EstimateMessage]] = []
                current: List[_EstimateMessage] = []
                
                for row in reader:
                    if row is None:
                        continue
                    message_raw = row.get('msg') or row.get('message') or ''
                    message = str(message_raw).strip()
                    if not message:
                        continue
                    
                    sender_flag = row.get('is_sender', 0)
                    try:
                        is_sender = int(float(sender_flag))
                    except (TypeError, ValueError):
                        is_sender = 0
                    
                    role = 'assistant' if is_sender == 1 else 'user'
                    current.append(_EstimateMessage(role=role, content=message))
                    
                    if role == 'user' and len(current) > 1 and current[-2].role == 'assistant':
                        if len(current) > 1:
                            conversations.append(current[:-1])
                        current = [current[-1]]
                
                if current:
                    conversations.append(current)
                
                for index, conv in enumerate(conversations):
                    if len(conv) < 2:
                        continue
                    roles = {item.role for item in conv}
                    if not {'user', 'assistant'}.issubset(roles):
                        continue
                    qa_pairs.append(_EstimateQaPair(
                        id=f"{os.path.basename(csv_path)}_{index}",
                        messages=conv
                    ))
        
        except OSError as exc:
            raise DataProcessingError(f"读取CSV失败: {exc}") from exc
        
        return qa_pairs
    
    @staticmethod
    def _collect_csv_files_for_estimate(root_path: str) -> List[str]:
        """收集目录下所有CSV文件"""
        csv_files: List[str] = []
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.lower().endswith('.csv'):
                    csv_files.append(os.path.join(dirpath, filename))
        return csv_files
    
    @staticmethod
    def _count_char_types(text: str) -> Tuple[int, int]:
        """统计文本中英文字符与中文字符数量"""
        english = 0
        chinese = 0
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                chinese += 1
            else:
                english += 1
        return english, chinese
    
    @staticmethod
    def _iter_batches(items: List[Any], batch_size: int):
        """生成固定大小的批次"""
        size = max(1, batch_size or 1)
        for start in range(0, len(items), size):
            yield items[start:start + size]
    
    @staticmethod
    def _estimate_tokens(english_chars: int, chinese_chars: int) -> float:
        """根据经验系数估算token数量"""
        english = max(0, english_chars or 0)
        chinese = max(0, chinese_chars or 0)
        return english * 0.3 + chinese * 0.6

    def _clean_data_rellm(self, scored_path: str, input_path: str, output_path: str, accept_score: int) -> int:
        """基于已有打分结果重新筛选数据"""
        try:
            if os.path.exists(output_path):
                backup_path = f"{output_path}.{datetime.now().strftime('%Y%m%d%H%M%S')}.bak"
                shutil.copy(output_path, backup_path)
                self.logger.info(f"已备份现有输出文件: {backup_path}")
            
            accepted_scores: Dict[str, float] = {}
            total_records = 0
            
            with open(scored_path, 'r', encoding='utf-8', newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                fieldnames = {name.strip() for name in (reader.fieldnames or []) if name}
                required_fields = {'id', 'score'}
                missing_fields = required_fields - fieldnames
                if missing_fields:
                    raise DataProcessingError(f"打分结果缺少必要字段: {', '.join(sorted(missing_fields))}")
                
                for row in reader:
                    total_records += 1
                    score_raw = row.get('score')
                    uid = str(row.get('id') or '').strip()
                    if not uid:
                        continue
                    try:
                        score_value = float(score_raw)
                    except (TypeError, ValueError):
                        self.logger.debug(f"跳过无法解析分数的记录: {row}")
                        continue
                    if score_value >= accept_score:
                        accepted_scores[uid] = score_value
            
            if not accepted_scores:
                self.logger.warning("没有记录满足当前分数阈值，未生成新数据")
                return 0
            
            self.logger.info(f"打分文件共 {total_records} 条，满足阈值的记录 {len(accepted_scores)} 条")
            
            target_indices: Dict[str, Set[int]] = defaultdict(set)
            for uid in accepted_scores.keys():
                if '_' not in uid:
                    continue
                file_name, idx_str = uid.rsplit('_', 1)
                try:
                    target_indices[file_name].add(int(idx_str))
                except ValueError:
                    self.logger.debug(f"跳过无法解析编号的记录ID: {uid}")
                    continue
            
            system_prompt = (self.config.get('system_prompt', '') or '').strip()
            include_system = system_prompt and system_prompt != "*"
            
            def replace_spaces_with_newlines(content: str) -> str:
                if not content:
                    return ''
                content = re.sub(r'[。！？；，、：] +', lambda m: m.group(0)[0] + '\n', content)
                content = re.sub(r' +', '\n', content)
                return content.strip()
            
            def process_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
                processed: List[Dict[str, str]] = []
                current_role = None
                current_chunks: List[str] = []
                
                for msg in messages:
                    content = replace_spaces_with_newlines(msg['content'])
                    role = msg['role']
                    if not content:
                        continue
                    if role == current_role:
                        current_chunks.append(content)
                    else:
                        if current_chunks and current_role:
                            processed.append({"role": current_role, "content": "\n".join(current_chunks)})
                        current_role = role
                        current_chunks = [content]
                
                if current_chunks and current_role:
                    processed.append({"role": current_role, "content": "\n".join(current_chunks)})
                
                return processed
            
            selected = 0
            found_ids: Set[str] = set()
            
            def handle_conversation(file_name: str, conv_index: int, messages: List[Dict[str, str]], writer) -> None:
                nonlocal selected
                if conv_index not in target_indices.get(file_name, set()):
                    return
                conv_id = f"{file_name}_{conv_index}"
                score = accepted_scores.get(conv_id)
                if score is None:
                    return
                found_ids.add(conv_id)
                
                processed_msgs = process_messages(messages)
                if len(processed_msgs) < 2:
                    return
                
                output_messages: List[Dict[str, str]] = []
                if include_system:
                    output_messages.append({"role": "system", "content": system_prompt})
                output_messages.extend(processed_msgs)
                
                writer.write(json.dumps({"messages": output_messages}, ensure_ascii=False) + '\n')
                selected += 1
            
            def iter_csv_files(path: str) -> List[str]:
                if os.path.isfile(path) and path.lower().endswith('.csv'):
                    return [path]
                csv_files: List[str] = []
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith('.csv'):
                            csv_files.append(os.path.join(root, file))
                return csv_files
            
            def load_conversations_from_csv(csv_path: str, writer) -> None:
                file_name = os.path.basename(csv_path)
                indices_for_file = target_indices.get(file_name)
                if not indices_for_file:
                    return
                conv_index = 0
                conversation: List[Dict[str, str]] = []
                
                with open(csv_path, 'r', encoding='utf-8', newline='') as fp:
                    reader = csv.DictReader(fp)
                    if not reader.fieldnames or 'msg' not in reader.fieldnames:
                        return
                    
                    for row in reader:
                        msg = str(row.get('msg') or '').strip()
                        if not msg:
                            continue
                        is_sender_raw = row.get('is_sender', 0)
                        try:
                            is_sender = int(float(is_sender_raw))
                        except (TypeError, ValueError):
                            is_sender = 0
                        role = 'assistant' if is_sender == 1 else 'user'
                        
                        conversation.append({"role": role, "content": msg})
                        
                        if role == 'user' and len(conversation) > 1 and conversation[-2]['role'] == 'assistant':
                            finished = conversation[:-1]
                            if any(m['role'] == 'user' for m in finished) and any(m['role'] == 'assistant' for m in finished):
                                handle_conversation(file_name, conv_index, finished, writer)
                            conv_index += 1
                            conversation = [conversation[-1]]
                    
                    if conversation and any(m['role'] == 'user' for m in conversation) and any(m['role'] == 'assistant' for m in conversation):
                        handle_conversation(file_name, conv_index, conversation, writer)
            
            with open(output_path, 'w', encoding='utf-8') as out_fp:
                csv_files = iter_csv_files(input_path)
                if not csv_files:
                    raise DataProcessingError("未在输入路径中找到任何CSV文件")
                
                for csv_file in csv_files:
                    load_conversations_from_csv(csv_file, out_fp)
            
            if selected == 0:
                self.logger.warning("没有问答对满足筛选条件或未找到匹配的原始记录")
            
            missing = sorted(set(accepted_scores.keys()) - found_ids)
            if missing:
                self.logger.warning(f"有 {len(missing)} 条满足分数阈值的记录未在原始数据中匹配到，示例: {missing[:5]}")
            
            self.logger.info(f"重新筛选完成: {selected}/{len(accepted_scores)} 条记录写入 (阈值 {accept_score})")
            return 0
        
        except DataProcessingError:
            raise
        except Exception as e:
            raise DataProcessingError(f"重新筛选打分数据失败: {e}") from e
    
    def _clean_data_raw(self, input_path: str, output_path: str) -> int:
        """使用原始算法清洗数据"""
        try:
            # 直接导入并调用ChatMLGenerator，避免子进程编码问题
            from process_data.generate_chatml_raw import ChatMLGenerator
            
            self.logger.info(f"开始原始算法清洗")
            self.logger.info(f"输入路径: {input_path}")
            self.logger.info(f"输出路径: {output_path}")
            
            generator = ChatMLGenerator(input_path=input_path, output_path=output_path)
            generator.run()
            
            return 0
            
        except Exception as e:
            raise DataProcessingError(f"原始算法清洗失败: {e}")
    
    def _convert_data(self, args: argparse.Namespace) -> int:
        """转换数据格式"""
        try:
            self.logger.info("开始转换数据格式...")
            
            input_path = args.input
            output_path = args.output
            target_format = getattr(args, 'format', 'chatml')
            
            # 确保输出目录存在
            ensure_directory(os.path.dirname(output_path))
            
            # 检查输入文件
            input_stats = get_file_stats(input_path)
            self.logger.info(f"输入文件: {input_path} ({input_stats['size']})")
            
            # 执行格式转换
            result = self._execute_format_conversion(input_path, output_path, target_format)
            
            if result == 0:
                output_stats = get_file_stats(output_path)
                self.logger.info(f"格式转换完成: {output_path} ({output_stats['size']})")
            
            return result
            
        except Exception as e:
            raise DataProcessingError(f"数据转换失败: {e}")
    
    def _execute_format_conversion(self, input_path: str, output_path: str, target_format: str) -> int:
        """执行格式转换"""
        try:
            # 目前只支持chatml格式转换，其他格式暂不支持
            if target_format == 'chatml':
                # 使用ChatMLGenerator进行chatml格式转换
                from process_data.generate_chatml_raw import ChatMLGenerator
                
                self.logger.info(f"使用ChatMLGenerator进行{target_format}格式转换")
                generator = ChatMLGenerator(input_path=input_path, output_path=output_path)
                generator.run()
                return 0
            else:
                # 其他格式暂不支持，返回错误
                self.logger.error(f"暂不支持 {target_format} 格式转换")
                self.logger.info("目前只支持 chatml 格式转换")
                raise ValidationError(f"暂不支持的格式: {target_format}，目前只支持 chatml 格式")
            
        except Exception as e:
            raise DataProcessingError(f"格式转换执行失败: {e}")
    
    def _merge_data(self, args: argparse.Namespace) -> int:
        """合并多源数据"""
        try:
            self.logger.info("开始合并多源数据...")
            
            input_files = args.inputs
            output_path = args.output
            deduplicate = getattr(args, 'deduplicate', False)
            
            # 确保输出目录存在
            ensure_directory(os.path.dirname(output_path))
            
            # 检查所有输入文件
            for input_file in input_files:
                if not os.path.exists(input_file):
                    raise FileOperationError(f"输入文件不存在", input_file)
                
                file_stats = get_file_stats(input_file)
                self.logger.info(f"输入文件: {input_file} ({file_stats['size']})")
            
            # 执行合并
            result = self._execute_data_merge(input_files, output_path, deduplicate)
            
            if result == 0:
                output_stats = get_file_stats(output_path)
                self.logger.info(f"数据合并完成: {output_path} ({output_stats['size']})")
            
            return result
            
        except Exception as e:
            raise DataProcessingError(f"数据合并失败: {e}")
    
    def _execute_data_merge(self, input_files: List[str], output_path: str, deduplicate: bool) -> int:
        """执行数据合并"""
        try:
            merged_data = []
            seen_items = set() if deduplicate else None
            
            for input_file in input_files:
                self.logger.info(f"处理文件: {input_file}")
                
                # 读取文件数据
                with open(input_file, 'r', encoding='utf-8') as f:
                    if input_file.endswith('.jsonl'):
                        data = [json.loads(line) for line in f]
                    else:
                        data = json.load(f)
                
                if not isinstance(data, list):
                    self.logger.warning(f"跳过非数组格式文件: {input_file}")
                    continue
                
                # 添加数据
                for item in data:
                    if deduplicate:
                        # 简单的去重逻辑
                        item_hash = hash(json.dumps(item, sort_keys=True))
                        if item_hash not in seen_items:
                            merged_data.append(item)
                            seen_items.add(item_hash)
                    else:
                        merged_data.append(item)
                
                self.logger.info(f"已处理 {len(data)} 条记录")
            
            # 保存合并结果
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.endswith('.jsonl'):
                    for item in merged_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                else:
                    json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
            removed_count = sum(len(data) for data in []) - len(merged_data) if deduplicate else 0
            self.logger.info(f"合并完成，总记录数: {len(merged_data)}")
            if deduplicate and removed_count > 0:
                self.logger.info(f"去重移除记录数: {removed_count}")
            
            return 0
            
        except Exception as e:
            raise DataProcessingError(f"数据合并执行失败: {e}")
    
    def _preview_data(self, args: argparse.Namespace) -> int:
        """预览数据样本"""
        try:
            input_path = args.input
            count = getattr(args, 'count', 5)
            
            if not os.path.exists(input_path):
                raise FileOperationError("输入文件不存在", input_path)
            
            self.logger.info(f"预览数据文件: {input_path}")
            
            # 读取数据
            with open(input_path, 'r', encoding='utf-8') as f:
                if input_path.endswith('.jsonl'):
                    data = []
                    for i, line in enumerate(f):
                        if i >= count:
                            break
                        data.append(json.loads(line))
                else:
                    all_data = json.load(f)
                    if isinstance(all_data, list):
                        data = all_data[:count]
                    else:
                        data = [all_data]
            
            # 显示预览
            self.logger.info(f"数据预览: 显示前{len(data)}条记录")
            print(f"\n数据预览 (前 {len(data)} 条记录):")
            print("=" * 80)
            
            for i, item in enumerate(data, 1):
                print(f"\n记录 {i}:")
                print("-" * 40)
                if isinstance(item, dict):
                    for key, value in item.items():
                        # 限制显示长度
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        print(f"  {key}: {value}")
                else:
                    print(f"  {item}")
            
            print("=" * 80)
            return 0
            
        except Exception as e:
            raise DataProcessingError(f"数据预览失败: {e}")
    
    def _show_stats(self, args: argparse.Namespace) -> int:
        """显示数据统计"""
        try:
            input_path = args.input
            
            if not os.path.exists(input_path):
                raise FileOperationError("输入文件不存在", input_path)
            
            file_stats = get_file_stats(input_path)
            
            self.logger.info(f"数据统计: {input_path}")
            print(f"\n数据统计: {input_path}")
            print("=" * 80)
            print(f"文件大小: {file_stats['size']}")
            print(f"修改时间: {file_stats['modified']}")
            
            # 读取并分析数据
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    if input_path.endswith('.jsonl'):
                        record_count = 0
                        total_length = 0
                        
                        for line in f:
                            record_count += 1
                            data = json.loads(line)
                            if isinstance(data, dict) and 'content' in data:
                                total_length += len(str(data['content']))
                            elif isinstance(data, str):
                                total_length += len(data)
                        
                        self.logger.info(f"数据记录数量: {record_count}")
                        print(f"记录数量: {record_count}")
                        if record_count > 0:
                            self.logger.info(f"数据平均长度: {total_length // record_count} 字符")
                            print(f"平均长度: {total_length // record_count} 字符")
                        
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.logger.info(f"数据记录数量: {len(data)}")
                            print(f"记录数量: {len(data)}")
                            if data:
                                # 分析数据结构
                                sample = data[0]
                                if isinstance(sample, dict):
                                    self.logger.info(f"数据字段: {list(sample.keys())}")
                                    print("字段统计:")
                                    for key in sample.keys():
                                        print(f"  - {key}")
                        elif isinstance(data, dict):
                            self.logger.info(f"数据类型: 单个对象, 字段: {list(data.keys())}")
                            print("数据类型: 单个对象")
                            print("字段统计:")
                            for key in data.keys():
                                print(f"  - {key}")
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON解析错误: {e}")
                print(f"JSON解析错误: {e}")
            except Exception as e:
                self.logger.error(f"统计分析错误: {e}")
                print(f"统计分析错误: {e}")
            
            print("=" * 80)
            return 0
            
        except Exception as e:
            raise DataProcessingError(f"数据统计失败: {e}")
    
    def _execute_subprocess(self, cmd: List[str], operation: str) -> int:
        """执行子进程命令"""
        try:
            self.logger.info(f"执行{operation}: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # 实时显示输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.logger.info(output.strip())
                    print(output.strip())
            
            return_code = process.poll()
            
            if return_code != 0:
                self.logger.error(f"{operation}失败，退出码: {return_code}")
            
            return return_code
            
        except Exception as e:
            raise DataProcessingError(f"{operation}子进程执行失败: {e}")
