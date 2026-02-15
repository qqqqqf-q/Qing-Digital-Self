from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from utils.logger.logger import get_logger
from utils.openai.openai_client import OpenAIClient


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", re.MULTILINE)

# 强约束：这些残留通常代表工具/搜索/代码污染
_HARD_TECH_MARKERS = [
    re.compile(r"\b(call tool|tool_calls|function_call|web\.run)\b", re.IGNORECASE),
    re.compile(r"\{[\s\S]*?\"search_query\"\s*:\s*\[", re.IGNORECASE),
    re.compile(r"\{[\s\S]*?\"open\"\s*:\s*\[", re.IGNORECASE),
    re.compile(r"\{[\s\S]*?\"click\"\s*:\s*\[", re.IGNORECASE),
    re.compile(r"\{[\s\S]*?\"screenshot\"\s*:\s*\[", re.IGNORECASE),
    re.compile(r"\b(search|open_url|openurl|browse)\s*\(", re.IGNORECASE),
]


DEFAULT_BASE_PROMPT = (
    "你是一个可靠、克制、表达清晰的助手。\n"
    "优先给出直接可执行的答案；信息不足时先提1-2个关键澄清问题。\n"
    "不要编造；不确定就明确说明，并给出可验证的下一步。"
)


DEFAULT_CLEAN_SYSTEM_PROMPT = """你是一个训练数据清洗与编辑器。

任务：你只做“筛选”，不要做任何改写、删除、重排或脱敏。

目标：从输入对话中筛掉技术向/工具痕迹/搜索痕迹等内容，其余尽量保留（包括敏感话题）。

关键要求：
1) 不要因为话题涉及自杀/自残、违法/灰产、暴力、色情等就 drop；GPT-4o 的安全语气本身是需要学习的。
2) 需要 drop 的典型情况：
   - 明显的技术向对话（编程、硬件参数、安装配置、API/微调/训练/提示词等元对话）
   - 工具调用过程或痕迹（call tool / tool_calls / search / open_url 等）
   - 搜索结果堆砌、Sources、长链接列表、引用标记
3) 只做一刀切：要么整条对话 keep，要么整条对话 drop。

输出格式：只输出 JSON，不要任何额外文字。
{"keep": true/false}
"""


@dataclass(frozen=True)
class OpenAICleanOptions:
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 4096
    workers: int = 4
    max_chars: int = 20000
    max_messages: int = 80
    max_samples: Optional[int] = None
    base_prompt: Optional[str] = DEFAULT_BASE_PROMPT
    clean_system_prompt: Optional[str] = None


@dataclass
class OpenAICleanStats:
    total_samples: int = 0
    kept_samples: int = 0
    dropped_samples: int = 0
    dropped_by_reason: Dict[str, int] = field(default_factory=dict)
    llm_errors: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def drop(self, reason: str) -> None:
        self.dropped_samples += 1
        self.dropped_by_reason[reason] = self.dropped_by_reason.get(reason, 0) + 1


@dataclass(frozen=True)
class _CleanResult:
    keep: bool
    reason: str
    usage: Tuple[int, int] = (0, 0)


def _sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield line_no, obj


def _strip_think_and_fences(text: str) -> str:
    text = text or ""
    text = _THINK_BLOCK_RE.sub("", text).strip()

    # 常见：模型把 JSON 包在 ```json 里
    if text.startswith("```"):
        match = _CODE_FENCE_RE.search(text)
        if match:
            text = match.group(1).strip()
        else:
            text = text.strip("`").strip()
    return text


def _parse_json_relaxed(text: str) -> Any:
    cleaned = _strip_think_and_fences(text)
    if not cleaned:
        raise ValueError("空响应")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch not in "{[":
            continue
        try:
            obj, end = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        return obj
    raise ValueError(f"无法解析JSON: {cleaned[:200]}")


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    kept: List[Dict[str, str]] = []
    for msg in messages or []:
        role = str(msg.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        kept.append({"role": role, "content": content})

    while kept and kept[0]["role"] == "assistant":
        kept.pop(0)
    while kept and kept[-1]["role"] != "assistant":
        kept.pop()
    return kept


def _exceeds_limits(messages: List[Dict[str, Any]], options: OpenAICleanOptions) -> bool:
    if options.max_messages > 0 and len(messages) > options.max_messages:
        return True
    if options.max_chars > 0:
        total = sum(len(str(m.get("content") or "")) for m in messages)
        if total > options.max_chars:
            return True
    return False


def _is_valid_sequence(messages: List[Dict[str, Any]]) -> bool:
    if not messages:
        return False
    first_role = str(messages[0].get("role") or "").strip().lower()
    last_role = str(messages[-1].get("role") or "").strip().lower()
    if first_role != "user":
        return False
    if last_role != "assistant":
        return False
    has_user = any(str(m.get("role") or "").strip().lower() == "user" for m in messages)
    has_assistant = any(str(m.get("role") or "").strip().lower() == "assistant" for m in messages)
    return has_user and has_assistant


def _has_hard_tech_markers(messages: List[Dict[str, str]]) -> bool:
    joined = "\n".join(m.get("content", "") for m in messages)
    for pattern in _HARD_TECH_MARKERS:
        if pattern.search(joined):
            return True
    return False


def _inject_base_prompt(messages: List[Dict[str, str]], base_prompt: Optional[str]) -> List[Dict[str, str]]:
    prompt = (base_prompt or "").strip()
    if not prompt or prompt == "*":
        return messages
    return [{"role": "system", "content": prompt}] + messages


class OpenAIExportLLMCleaner:
    def __init__(self, client: Optional[OpenAIClient] = None):
        self.client = client or OpenAIClient()
        self.logger = get_logger("OpenAIExportLLMCleaner")

    def clean_sample(
        self,
        sample_messages: List[Dict[str, Any]],
        options: OpenAICleanOptions,
    ) -> _CleanResult:
        if not isinstance(sample_messages, list) or not sample_messages:
            return _CleanResult(keep=False, reason="empty_or_invalid")
        if not _is_valid_sequence(sample_messages):
            return _CleanResult(keep=False, reason="invalid_sequence")
        if _exceeds_limits(sample_messages, options):
            return _CleanResult(keep=False, reason="exceeds_limits")

        normalized_for_prompt = _normalize_messages(sample_messages)
        if not normalized_for_prompt:
            return _CleanResult(keep=False, reason="empty_or_invalid")
        if _has_hard_tech_markers(normalized_for_prompt):
            return _CleanResult(keep=False, reason="hard_tech_marker")

        system_prompt = (options.clean_system_prompt or DEFAULT_CLEAN_SYSTEM_PROMPT).strip()
        user_payload = json.dumps(normalized_for_prompt, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"请判断下面的对话是否保留（JSON数组）：\n{user_payload}",
            },
        ]

        response = self.client.chat_completion(
            messages=messages,
            model=options.model,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
        )

        usage = response.get("usage") or {}
        if not usage and response.get("choices"):
            usage = response["choices"][0].get("usage") or {}
        prompt_tokens_raw = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("total_prompt_tokens")
            or usage.get("promptTokens")
        )
        completion_tokens_raw = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("total_completion_tokens")
            or usage.get("completionTokens")
        )
        try:
            prompt_tokens = int(prompt_tokens_raw or 0)
        except (TypeError, ValueError):
            prompt_tokens = 0
        try:
            completion_tokens = int(completion_tokens_raw or 0)
        except (TypeError, ValueError):
            completion_tokens = 0

        content = (
            (response.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = _parse_json_relaxed(content)
        if not isinstance(parsed, dict):
            return _CleanResult(keep=False, reason="bad_llm_response", usage=(prompt_tokens, completion_tokens))

        keep = bool(parsed.get("keep", False))
        reason = "keep" if keep else "drop"
        return _CleanResult(keep=keep, reason=reason, usage=(prompt_tokens, completion_tokens))


def clean_openai_sft_jsonl_with_llm(
    input_path: Path,
    output_root: Path,
    run_id: str,
    options: OpenAICleanOptions,
) -> OpenAICleanStats:
    logger = get_logger("OpenAIExportLLMClean")
    output_root.mkdir(parents=True, exist_ok=True)

    sft_dir = output_root / "sft"
    stats_dir = output_root / "stats"
    sft_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    output_path = sft_dir / "train.jsonl"
    dropped_path = stats_dir / "dropped.jsonl"
    summary_path = stats_dir / "summary.json"
    manifest_path = output_root / "manifest.json"

    started_at = datetime.now().isoformat(timespec="seconds")
    input_hash = _sha256sum(input_path)

    cleaner = OpenAIExportLLMCleaner()
    stats = OpenAICleanStats()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = max(1, int(options.workers or 1))
    logger.info(f"开始 OpenAI LLM 清洗: {input_path} -> {output_path} (workers={workers})")

    def submit_one(line_no: int, obj: Dict[str, Any]) -> Tuple[int, _CleanResult, str]:
        stats_local_reason = ""
        try:
            result = cleaner.clean_sample(obj.get("messages") or [], options=options)
            stats_local_reason = result.reason
            return line_no, result, stats_local_reason
        except Exception as exc:
            return line_no, _CleanResult(keep=False, reason="llm_error"), f"llm_error:{exc}"

    futures = []
    messages_by_line: Dict[int, List[Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for line_no, obj in _iter_jsonl(input_path):
            if options.max_samples is not None and stats.total_samples >= int(options.max_samples):
                break
            stats.total_samples += 1
            original_messages = obj.get("messages") or []
            if not isinstance(original_messages, list):
                original_messages = []
            messages_by_line[line_no] = original_messages
            futures.append(executor.submit(submit_one, line_no, obj))

        with open(output_path, "w", encoding="utf-8") as f_out, open(dropped_path, "w", encoding="utf-8") as f_drop:
            for fut in as_completed(futures):
                line_no, result, reason = fut.result()
                prompt_tokens, completion_tokens = result.usage
                stats.total_prompt_tokens += prompt_tokens
                stats.total_completion_tokens += completion_tokens

                if not result.keep:
                    if str(reason).startswith("llm_error:"):
                        stats.llm_errors += 1
                        stats.drop("llm_error")
                    else:
                        stats.drop(result.reason or "drop")
                    f_drop.write(
                        json.dumps(
                            {"line_no": line_no, "reason": result.reason, "note": reason},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                original_messages = messages_by_line.get(line_no) or []
                injected = _inject_base_prompt(original_messages, options.base_prompt)
                f_out.write(json.dumps({"messages": injected}, ensure_ascii=False) + "\n")
                stats.kept_samples += 1

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, ensure_ascii=False, indent=2)

    payload = {
        "pipeline": "openai-clean",
        "run_id": run_id,
        "created_at": started_at,
        "input_path": str(input_path.as_posix()),
        "input_hash": input_hash,
        "filters": {
            "model": options.model,
            "temperature": options.temperature,
            "max_tokens": options.max_tokens,
            "workers": options.workers,
            "max_chars": options.max_chars,
            "max_messages": options.max_messages,
            "max_samples": options.max_samples,
            "base_prompt_enabled": bool((options.base_prompt or "").strip() and (options.base_prompt or "").strip() != "*"),
        },
        "outputs": {
            "sft_train": str(output_path.as_posix()),
            "dropped": str(dropped_path.as_posix()),
            "stats": str(summary_path.as_posix()),
        },
        "counts": asdict(stats),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"完成 OpenAI LLM 清洗: {output_root}")
    logger.info(f"SFT 输出: {output_path}")
    return stats
