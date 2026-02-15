from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from utils.logger.logger import get_logger


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff]")
_SEDIMENT_RE = re.compile(r"sediment://\S+")
_MULTI_BLANK_LINES_RE = re.compile(r"\n{4,}")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CN_MOBILE_RE = re.compile(r"\b1[3-9]\d{9}\b")
_ID_CARD_RE = re.compile(r"\b\d{17}[\dXx]\b")
_CREDIT_CARD_RE = re.compile(r"\b\d{16,19}\b")
_API_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9]{10,})\b")


@dataclass(frozen=True)
class NormalizedMessage:
    role: str
    content: str
    content_type: str
    create_time: Optional[float] = None
    source_id: Optional[str] = None


@dataclass(frozen=True)
class NormalizedConversation:
    conversation_id: str
    model: str
    create_time: Optional[float]
    update_time: Optional[float]
    messages: List[NormalizedMessage]
    user_context: Optional[str] = None
    images: Optional[List[str]] = None


@dataclass(frozen=True)
class DistillOptions:
    allow_models: Set[str]
    cutoff_ts: Optional[float] = None
    keep_system: bool = False
    keep_code: bool = False
    keep_tool: bool = False
    pii_policy: str = "mask"  # mask / drop / keep
    max_chars: int = 20000
    max_messages: int = 80


@dataclass(frozen=True)
class OpenAIExportSource:
    """一次 ChatGPT 导出源（以 conversations.json 为入口）。"""

    conversations_path: Path
    export_root: Path
    chat_html_path: Optional[Path] = None


@dataclass
class DistillStats:
    total_conversations: int = 0
    kept_conversations: int = 0
    dropped_conversations: int = 0
    dropped_by_reason: Dict[str, int] = field(default_factory=dict)
    normalized_messages: int = 0
    kept_messages: int = 0
    masked_pii_conversations: int = 0
    masked_pii_hits: int = 0

    def drop(self, reason: str) -> None:
        self.dropped_conversations += 1
        self.dropped_by_reason[reason] = self.dropped_by_reason.get(reason, 0) + 1


def discover_openai_export_sources(input_path: Path) -> List[OpenAIExportSource]:
    """发现 OpenAI 导出源。

    - input_path 为文件：按单文件处理
    - input_path 为目录：递归查找所有 conversations.json
    """

    def to_source(path: Path) -> OpenAIExportSource:
        export_root = path.parent
        chat_html = export_root / "chat.html"
        return OpenAIExportSource(
            conversations_path=path,
            export_root=export_root,
            chat_html_path=chat_html if chat_html.exists() else None,
        )

    if input_path.is_file():
        return [to_source(input_path)]

    if not input_path.is_dir():
        return []

    candidates = [p for p in input_path.rglob("conversations.json") if p.is_file()]
    candidates.sort(key=lambda p: p.as_posix().lower())

    sources: List[OpenAIExportSource] = []
    seen: Set[str] = set()
    for p in candidates:
        try:
            key = p.resolve().as_posix()
        except Exception:
            key = p.absolute().as_posix()
        if key in seen:
            continue
        seen.add(key)
        sources.append(to_source(p))
    return sources


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_json_array_items(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """流式读取 JSON 数组文件，逐个 yield 元素。"""
    decoder = json.JSONDecoder()
    buffer = ""

    with open(path, "r", encoding="utf-8") as f:
        # 找到数组起始 '['
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"文件为空或不是 JSON 数组: {path}")
            buffer += chunk
            if buffer.startswith("\ufeff"):
                buffer = buffer.lstrip("\ufeff")
            start = buffer.find("[")
            if start >= 0:
                buffer = buffer[start + 1 :]
                break
            if len(buffer) > chunk_size * 2:
                buffer = buffer[-chunk_size:]

        # 逐个解析元素
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk
                continue

            if buffer[0] == "]":
                break
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue

            try:
                obj, idx = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = f.read(chunk_size)
                if not chunk:
                    snippet = buffer[:200].replace("\n", "\\n")
                    raise ValueError(f"JSON 解析失败，文件可能损坏。片段: {snippet}") from None
                buffer += chunk
                continue

            yield obj
            buffer = buffer[idx:]


def extract_main_path(mapping: Dict[str, Any], current_node: Optional[str]) -> List[Dict[str, Any]]:
    """根据 current_node 回溯 parent，得到主链节点（从根到叶）。"""
    if not isinstance(mapping, dict) or not current_node:
        return []

    path: List[Dict[str, Any]] = []
    node_id: Optional[str] = str(current_node)
    visited: Set[str] = set()

    while node_id:
        if node_id in visited:
            break
        visited.add(node_id)
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            break
        path.append(node)
        parent = node.get("parent")
        node_id = str(parent) if parent else None

    path.reverse()
    return path


def _extract_text_and_images(content: Any) -> Tuple[str, List[str], str]:
    if not isinstance(content, dict):
        text = "" if content is None else str(content)
        return text, [], ""

    content_type = str(content.get("content_type") or "")
    pieces: List[str] = []
    images: List[str] = []

    text_field = content.get("text")
    if isinstance(text_field, str) and text_field.strip():
        pieces.append(text_field)

    parts = content.get("parts")
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, str):
                if part.strip():
                    pieces.append(part)
                continue
            if isinstance(part, dict):
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text.strip():
                    pieces.append(part_text)

                pointer = part.get("asset_pointer") or part.get("asset") or part.get("file_id") or part.get("asset_id")
                images.extend(_coerce_asset_pointer(pointer))
                continue

    images.extend(_coerce_asset_pointer(content.get("asset_pointer")))
    text = "\n".join(pieces).strip()
    return text, images, content_type


def _coerce_asset_pointer(pointer: Any) -> List[str]:
    if isinstance(pointer, str) and pointer.strip():
        return [pointer.strip()]
    if isinstance(pointer, dict):
        for key in ("file_id", "asset_id", "id"):
            v = pointer.get(key)
            if isinstance(v, str) and v.strip():
                return [v.strip()]
    return []


def normalize_conversation(raw: Dict[str, Any]) -> NormalizedConversation:
    conversation_id = str(raw.get("id") or raw.get("conversation_id") or "")
    model = str(raw.get("default_model_slug") or raw.get("model") or "")
    create_time = raw.get("create_time")
    update_time = raw.get("update_time")

    mapping = raw.get("mapping") or {}
    current_node = raw.get("current_node")
    nodes = extract_main_path(mapping, current_node)

    messages: List[NormalizedMessage] = []
    images: List[str] = []
    user_context_parts: List[str] = []

    for node in nodes:
        msg = node.get("message")
        if not isinstance(msg, dict):
            continue

        author = msg.get("author") or {}
        role = str(author.get("role") or "").strip().lower()
        if not role:
            continue

        content_text, content_images, content_type = _extract_text_and_images(msg.get("content"))
        images.extend(content_images)

        source_id = msg.get("id") or node.get("id")
        norm = NormalizedMessage(
            role=role,
            content=content_text,
            content_type=content_type,
            create_time=msg.get("create_time"),
            source_id=str(source_id) if source_id else None,
        )

        if content_type == "user_editable_context":
            if content_text.strip():
                user_context_parts.append(content_text.strip())
            continue

        messages.append(norm)

    user_context = "\n".join(user_context_parts).strip() or None
    return NormalizedConversation(
        conversation_id=conversation_id,
        model=model,
        create_time=float(create_time) if isinstance(create_time, (int, float)) else None,
        update_time=float(update_time) if isinstance(update_time, (int, float)) else None,
        messages=messages,
        user_context=user_context,
        images=images or None,
    )


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _PRIVATE_USE_RE.sub("", text)
    text = _SEDIMENT_RE.sub("", text)
    text = _MULTI_BLANK_LINES_RE.sub("\n\n\n", text)
    return text.strip()


def _mask_pii(text: str) -> Tuple[str, int]:
    hits = 0

    def repl(pattern: re.Pattern[str], replacement: str, s: str) -> str:
        nonlocal hits
        matches = list(pattern.finditer(s))
        if not matches:
            return s
        hits += len(matches)
        return pattern.sub(replacement, s)

    text = repl(_API_KEY_RE, "[API_KEY]", text)
    text = repl(_EMAIL_RE, "[EMAIL]", text)
    text = repl(_CN_MOBILE_RE, "[MOBILE]", text)
    text = repl(_ID_CARD_RE, "[ID_CARD]", text)
    text = repl(_CREDIT_CARD_RE, "[CARD]", text)
    return text, hits


def _apply_limits(messages: List[Dict[str, str]], options: DistillOptions) -> List[Dict[str, str]]:
    if options.max_messages > 0 and len(messages) > options.max_messages:
        messages = messages[-options.max_messages :]

    if options.max_chars > 0:
        total = sum(len(m.get("content", "")) for m in messages)
        while total > options.max_chars and len(messages) > 2:
            removed = messages.pop(0)
            total -= len(removed.get("content", ""))

    # 避免从 assistant 开头（通常对训练不友好）
    while messages and messages[0].get("role") == "assistant":
        messages.pop(0)
    return messages


def build_sft_sample(conv: NormalizedConversation, options: DistillOptions, stats: DistillStats) -> Optional[Dict[str, Any]]:
    allowed_roles = {"user", "assistant"}
    if options.keep_system:
        allowed_roles.add("system")
    if options.keep_tool:
        allowed_roles.add("tool")

    kept: List[Dict[str, str]] = []
    pii_hits_total = 0

    for msg in conv.messages:
        if msg.role not in allowed_roles:
            continue
        if not options.keep_code and msg.content_type == "code":
            continue
        if not options.keep_tool and msg.role == "tool":
            continue

        cleaned = _clean_text(msg.content or "")
        if not cleaned:
            continue

        if options.pii_policy not in {"mask", "drop", "keep"}:
            raise ValueError("pii_policy 仅支持: mask / drop / keep")

        if options.pii_policy == "drop":
            _, hits = _mask_pii(cleaned)
            if hits:
                stats.drop("pii_drop")
                return None
        elif options.pii_policy == "mask":
            cleaned, hits = _mask_pii(cleaned)
            pii_hits_total += hits

        kept.append({"role": msg.role, "content": cleaned})

    # 末尾必须是 assistant（否则移除尾部 user/tool 等）
    while kept and kept[-1]["role"] != "assistant":
        kept.pop()

    if not any(m["role"] == "user" for m in kept) or not any(m["role"] == "assistant" for m in kept):
        stats.drop("no_qa_pair")
        return None

    kept = _apply_limits(kept, options)
    if not kept:
        stats.drop("empty_after_limits")
        return None

    if pii_hits_total > 0:
        stats.masked_pii_conversations += 1
        stats.masked_pii_hits += pii_hits_total

    stats.kept_messages += len(kept)
    return {"messages": kept}


def distill_openai_export(
    input_path: Path,
    output_root: Path,
    run_id: str,
    options: DistillOptions,
) -> DistillStats:
    logger = get_logger("OpenAIExportDistill")
    output_root.mkdir(parents=True, exist_ok=True)

    sources = discover_openai_export_sources(input_path)
    if not sources:
        raise ValueError(f"未发现可用的 conversations.json: {input_path}")

    normalized_dir = output_root / "normalized"
    sft_dir = output_root / "sft"
    stats_dir = output_root / "stats"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = normalized_dir / "conversations.jsonl"
    sft_path = sft_dir / "text.jsonl"
    stats_path = stats_dir / "summary.json"
    manifest_path = output_root / "manifest.json"

    source_hashes = [sha256sum(s.conversations_path) for s in sources]
    input_hash = (
        source_hashes[0]
        if len(source_hashes) == 1
        else hashlib.sha256("\n".join(sorted(source_hashes)).encode("utf-8")).hexdigest()
    )
    started_at = datetime.now().isoformat(timespec="seconds")

    stats = DistillStats()
    allow_models = {m.strip() for m in options.allow_models if str(m).strip()}

    if len(sources) > 1:
        logger.info(f"检测到 {len(sources)} 份导出，将合并蒸馏到同一份输出")

    with open(normalized_path, "w", encoding="utf-8") as f_norm, open(sft_path, "w", encoding="utf-8") as f_sft:
        for source in sources:
            for raw in iter_json_array_items(source.conversations_path):
                if not isinstance(raw, dict):
                    continue

                stats.total_conversations += 1
                try:
                    conv = normalize_conversation(raw)
                except Exception:
                    stats.drop("normalize_error")
                    continue

                if allow_models and conv.model and conv.model not in allow_models:
                    stats.drop("model_filtered")
                    continue

                if options.cutoff_ts is not None and conv.create_time is not None and conv.create_time < options.cutoff_ts:
                    stats.drop("cutoff_ts")
                    continue

                stats.normalized_messages += len(conv.messages)
                f_norm.write(json.dumps(asdict(conv), ensure_ascii=False) + "\n")

                sample = build_sft_sample(conv, options, stats)
                if not sample:
                    continue

                f_sft.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats.kept_conversations += 1

    payload = {
        "pipeline": "openai-distill",
        "run_id": run_id,
        "created_at": started_at,
        "input_path": str(input_path.as_posix()),
        "input_hash": input_hash,
        "inputs": [
            {
                "conversations_path": str(s.conversations_path.as_posix()),
                "sha256": source_hashes[idx],
                "export_root": str(s.export_root.as_posix()),
                "chat_html_path": str(s.chat_html_path.as_posix()) if s.chat_html_path else None,
            }
            for idx, s in enumerate(sources)
        ],
        "filters": {
            "allow_models": sorted(list(allow_models)),
            "cutoff_ts": options.cutoff_ts,
            "pii_policy": options.pii_policy,
            "keep_system": options.keep_system,
            "keep_code": options.keep_code,
            "keep_tool": options.keep_tool,
            "max_chars": options.max_chars,
            "max_messages": options.max_messages,
        },
        "outputs": {
            "normalized": str(normalized_path.as_posix()),
            "sft_text": str(sft_path.as_posix()),
            "stats": str(stats_path.as_posix()),
        },
        "counts": asdict(stats),
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, ensure_ascii=False, indent=2)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"完成 OpenAI 导出蒸馏: {output_root}")
    logger.info(f"SFT 输出: {sft_path}")
    return stats
