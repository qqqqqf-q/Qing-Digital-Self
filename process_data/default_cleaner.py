import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Type

import pandas as pd
import requests

from utils.logger.logger import get_logger


TEXT_TYPES = {"文本", "text"}
IMAGE_TYPES = {"图片", "image"}
CUT_TYPES = {
    "cut",
    "图片",
    "视频",
    "合并转发的聊天记录",
    "语音",
    "(分享)音乐",
    "(分享)卡片式链接",
    "(分享)笔记",
    "(分享)小程序",
    "(分享)收藏夹",
    "(分享)视频号名片",
    "(分享)视频号视频",
    "粘贴的文本",
    "未知",
    "image",
    "video",
    "merged forward chat records",
    "voice",
    "(share) music",
    "(share) card link",
    "(share) note",
    "(share) mini program",
    "(share) favorites",
    "(share) video account card",
    "(share) video account video",
    "pasted text",
    "unknown",
}
SKIP_TYPES = {
    "添加好友",
    "推荐公众号",
    "动画表情",
    "位置",
    "文件",
    "位置共享",
    "引用回复",
    "群公告",
    "转账",
    "语音通话",
    "系统通知",
    "消息撤回",
    "拍一拍",
    "邀请加群",
    "add friend",
    "recommend official account",
    "sticker",
    "location",
    "file",
    "location sharing",
    "reply with quote",
    "group announcement",
    "transfer",
    "voice call",
    "system notification",
    "message recall",
    "pat pat",
    "invite to group",
}
TEXT_TYPES_LOWER = {t.lower() for t in TEXT_TYPES}
IMAGE_TYPES_LOWER = {t.lower() for t in IMAGE_TYPES}
CUT_TYPES_LOWER = {t.lower() for t in CUT_TYPES}
SKIP_TYPES_LOWER = {t.lower() for t in SKIP_TYPES}
PII_PATTERNS = [
    re.compile(r"\b1[3-9]\d{9}\b"),
    re.compile(r"\b\d{15,18}[0-9Xx]\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{16,19}\b"),
]


@dataclass
class ChatMessage:
    id: Any
    msg_svr_id: str
    type_name: str
    is_sender: int
    talker: str
    msg: str
    src: Any
    create_time: pd.Timestamp
    room_name: Optional[str]
    is_forward: bool
    modality: Optional[str] = None


@dataclass
class CutMessage:
    is_sender: int
    cut_type: str
    create_time: pd.Timestamp


class TimeWindowStrategy:
    def __init__(self, time_window_seconds: int):
        self.time_window = max(0, time_window_seconds)

    def is_same_conversation(
        self, history: Sequence[ChatMessage], current: ChatMessage
    ) -> bool:
        if not history:
            return True
        last = history[-1]
        delta = abs((current.create_time - last.create_time).total_seconds())
        return delta <= self.time_window


class VisionImageDescriber:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str,
        max_workers: int,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.max_workers = max(1, max_workers)
        self.logger = get_logger("VisionImageDescriber")
        self.prompt = (
            "请用不超过100字描述图片里的关键内容，包含界面、文字或场景信息。"
        )

    def apply(self, qa_pairs: List[Any], media_dir: Path) -> List[Any]:
        tasks: List[Path] = []
        for qa in qa_pairs:
            images = qa.images or []
            for rel_path in images:
                path = media_dir / rel_path
                tasks.append(path)
        if not tasks:
            return qa_pairs
        descriptions = self._describe_many(tasks)
        desc_iter = iter(descriptions)
        for qa in qa_pairs:
            images = qa.images or []
            if not images:
                continue
            for message in qa.messages:
                while "<image>" in message.content:
                    try:
                        desc = next(desc_iter)
                    except StopIteration:
                        desc = "[图片描述缺失]"
                    message.content = message.content.replace(
                        "<image>", f"\n[图片描述: {desc}]\n", 1
                    )
            qa.images = []
        return qa_pairs

    def _describe_many(self, tasks: List[Path]) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self._describe_single, tasks))

    def _describe_single(self, image_path: Path) -> str:
        if not image_path.exists():
            self.logger.warning(f"图片不存在: {image_path}")
            return "[图片不存在]"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{self._image_format(image_path)};base64"
                                f",{self._encode_image(image_path)}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0.2,
        }
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                return "[图片描述失败]"
            return choices[0]["message"]["content"].strip()
        except Exception as exc:
            self.logger.error(f"图片识别失败: {image_path}, error: {exc}")
            return "[图片描述失败]"

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        with open(image_path, "rb") as fp:
            return base64.b64encode(fp.read()).decode("utf-8")

    @staticmethod
    def _image_format(image_path: Path) -> str:
        suffix = image_path.suffix.lower().strip(".")
        return "jpeg" if suffix == "jpg" else suffix or "png"


class DefaultLLMCleaner:
    """默认的结构化 CSV→QA 构建逻辑"""

    def __init__(
        self,
        config: Any,
        qa_cls: Type[Any],
        message_cls: Type[Any],
        system_prompt: str,
    ):
        self.logger = get_logger("DefaultLLMCleaner")
        self.config = config
        self.QaPair = qa_cls
        self.Message = message_cls
        self.system_prompt = system_prompt or ""
        self.include_type = {str(t).lower() for t in config.get("include_type", ["text"])}
        self.blocked_words = self._load_blocked_words(config.get("blocked_words", []))
        self.single_strategy = TimeWindowStrategy(
            int(config.get("single_combine_time_window", 2) * 60)
        )
        self.qa_strategy = TimeWindowStrategy(
            int(config.get("qa_match_time_window", 5) * 60)
        )
        self.combine_msg_max_length = int(config.get("combine_msg_max_length", 2048))
        self.messages_max_length = int(config.get("messages_max_length", 2048))
        self.max_image_num = int(config.get("max_image_num", 2))
        self.add_time_to_system = bool(config.get("add_time_to_system", False))
        self.media_dir = Path(config.get("media_dir", "./dataset/media"))
        self.images_dir = self.media_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.vision_processor: Optional[VisionImageDescriber] = None
        if config.get("vision_api_enable") and config.get("vision_api_key"):
            api_url = config.get("vision_api_url") or ""
            api_key = config.get("vision_api_key") or ""
            model_name = config.get("vision_api_model") or ""
            if api_url and api_key and model_name:
                self.vision_processor = VisionImageDescriber(
                    api_url=api_url,
                    api_key=api_key,
                    model_name=model_name,
                    max_workers=int(config.get("vision_api_max_workers", 4)),
                )

    def build_qa_pairs(self, input_path: str) -> List[Any]:
        csv_files = self._collect_csv_files(Path(input_path))
        if not csv_files:
            self.logger.error(f"未找到CSV数据: {input_path}")
            return []
        all_messages: List[ChatMessage] = []
        for csv_file in csv_files:
            messages = self._load_csv(csv_file)
            all_messages.extend(messages)
        all_messages.sort(key=lambda msg: msg.create_time)
        grouped_messages = self._group_consecutive_messages(all_messages)
        qa_pairs = self._match_qa(grouped_messages)
        if self.vision_processor:
            qa_pairs = self.vision_processor.apply(qa_pairs, self.media_dir)
        self.logger.info(f"结构化清洗得到 {len(qa_pairs)} 条候选QA")
        return qa_pairs

    def _collect_csv_files(self, base_path: Path) -> List[Path]:
        if base_path.is_file() and base_path.suffix.lower() == ".csv":
            return [base_path]
        if not base_path.exists():
            return []
        csv_files: List[Path] = []
        for folder in sorted(base_path.iterdir()):
            if folder.is_dir():
                for csv_path in sorted(folder.glob("*.csv")):
                    csv_files.append(csv_path)
        csv_files.sort(key=lambda p: self._extract_sequence(p.name))
        return csv_files

    @staticmethod
    def _extract_sequence(name: str) -> int:
        match = re.search(r"_(\d+)_\d+\.csv$", name)
        return int(match.group(1)) if match else 0

    def _load_csv(self, file_path: Path) -> List[ChatMessage]:
        try:
            df = pd.read_csv(
                file_path,
                encoding="utf-8",
                dtype=str,
                keep_default_na=False,
            )
        except Exception as exc:
            self.logger.error(f"读取CSV失败 {file_path}: {exc}")
            return []
        if "CreateTime" not in df.columns:
            self.logger.warning(f"文件缺少 CreateTime 列: {file_path}")
            return []
        df["CreateTime"] = self._parse_create_time_column(df["CreateTime"])
        messages: List[ChatMessage] = []
        for _, row in df.iterrows():
            type_name = str(row.get("type_name", "")).strip()
            type_lower = type_name.lower()
            if type_lower in SKIP_TYPES_LOWER:
                continue
            is_sender = self._safe_int(row.get("is_sender"), 0)
            is_forward = self._to_bool(row.get("is_forward"))
            if is_sender == 1 and is_forward:
                continue
            msg_content = str(row.get("msg", "")).replace("\n", "").strip()
            modality = None

            if type_lower in IMAGE_TYPES_LOWER:
                modality = "image"
                if is_sender == 0:
                    normalized_src = self._resolve_image_path(row.get("src"))
                    if normalized_src:
                        msg_content = "<image>"
                        row["src"] = normalized_src
                    else:
                        type_name = "Cut"
                        type_lower = "cut"
                else:
                    type_name = "Cut"
                    type_lower = "cut"
            elif type_lower not in TEXT_TYPES_LOWER:
                row["msg"] = ""
                msg_content = ""

            if type_lower in TEXT_TYPES_LOWER:
                if not msg_content or self._contains_blocked_word(msg_content):
                    continue
                if self._contains_pii(msg_content):
                    continue

            timestamp = pd.to_datetime(row.get("CreateTime"), errors="coerce")
            if pd.isna(timestamp):
                continue
            chat_message = ChatMessage(
                id=row.get("id") or row.get("MsgSvrID"),
                msg_svr_id=str(row.get("MsgSvrID", "")),
                type_name=type_name or "",
                is_sender=is_sender,
                talker=str(row.get("talker", "")),
                msg=msg_content,
                src=row.get("src", ""),
                create_time=timestamp,
                room_name=row.get("room_name") or None,
                is_forward=is_forward,
                modality=modality,
            )
            messages.append(chat_message)
        return messages

    def _parse_create_time_column(self, series: pd.Series) -> pd.Series:
        """向量化解析 CreateTime，避免逐行转时间造成的巨大开销"""
        parsed = pd.to_datetime(
            series,
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
            cache=True,
        )
        missing_mask = parsed.isna()
        if missing_mask.any():
            parsed.loc[missing_mask] = pd.to_datetime(
                series[missing_mask],
                errors="coerce",
                infer_datetime_format=True,
                cache=True,
            )
        return parsed

    def _group_consecutive_messages(
        self, messages: List[ChatMessage]
    ) -> List[Any]:
        if not messages:
            return []
        grouped: List[Any] = []
        current_group: List[ChatMessage] = []
        for msg in messages:
            if self._is_cut(msg):
                if current_group:
                    grouped.append(self._combine_group(current_group))
                    current_group = []
                grouped.append(
                    CutMessage(
                        is_sender=msg.is_sender,
                        cut_type=msg.type_name,
                        create_time=msg.create_time,
                    )
                )
                continue
            if not current_group:
                current_group = [msg]
                continue
            last = current_group[-1]
            if (
                msg.is_sender == last.is_sender
                and msg.talker == last.talker
                and self.single_strategy.is_same_conversation([last], msg)
            ):
                current_group.append(msg)
            else:
                grouped.append(self._combine_group(current_group))
                current_group = [msg]
        if current_group:
            grouped.append(self._combine_group(current_group))
        return grouped

    def _match_qa(self, messages: List[Any]) -> List[Any]:
        WAITING_INSTRUCTION = "waiting_instruction"
        WAITING_RESPONSE = "waiting_response"
        state = WAITING_INSTRUCTION
        qa_list: List[Any] = []
        conversation_messages: List[Any] = []
        conversation_images: List[str] = []
        current_instruction: Optional[ChatMessage] = None
        last_message: Optional[ChatMessage] = None
        qa_id = 0

        def flush():
            nonlocal qa_id, conversation_messages, conversation_images
            if conversation_messages and last_message:
                qa_id = self._save_pair(
                    qa_list,
                    qa_id,
                    last_message.create_time,
                    conversation_messages,
                    conversation_images,
                )
                conversation_messages = []
                conversation_images = []

        for msg in messages:
            if isinstance(msg, CutMessage):
                flush()
                state = WAITING_INSTRUCTION
                current_instruction = None
                last_message = None
                continue
            if state == WAITING_INSTRUCTION:
                if msg.is_sender == 0:
                    current_instruction = msg
                    last_message = msg
                    state = WAITING_RESPONSE
                else:
                    last_message = msg
                continue
            if state == WAITING_RESPONSE:
                if msg.is_sender == 0:
                    if (
                        last_message
                        and not self.qa_strategy.is_same_conversation([last_message], msg)
                    ):
                        flush()
                    current_instruction = msg
                    last_message = msg
                else:
                    if (
                        last_message
                        and current_instruction
                        and self.qa_strategy.is_same_conversation([last_message], msg)
                    ):
                        conversation_messages.append(
                            self.Message(role="user", content=current_instruction.msg)
                        )
                        conversation_messages.append(
                            self.Message(role="assistant", content=msg.msg)
                        )
                        self._collect_images(current_instruction, conversation_images)
                        last_message = msg
                    state = WAITING_INSTRUCTION
                    current_instruction = None
        flush()
        return qa_list

    def _save_pair(
        self,
        qa_list: List[Any],
        qa_id: int,
        timestamp: pd.Timestamp,
        conversation_messages: List[Any],
        images: List[str],
    ) -> int:
        total_length = sum(len(msg.content) for msg in conversation_messages)
        if total_length > self.messages_max_length:
            self.logger.debug(
                f"对话长度超限({total_length}>{self.messages_max_length})，跳过"
            )
            return qa_id
        if len(images) > self.max_image_num:
            self.logger.debug("图片数量超限，跳过该QA")
            return qa_id
        system_prompt = self._compose_system_prompt(timestamp)
        qa = self.QaPair(
            id=qa_id,
            messages=list(conversation_messages),
            score=0,
            images=list(images),
            system_prompt=system_prompt,
        )
        qa_list.append(qa)
        return qa_id + 1

    def _compose_system_prompt(self, timestamp: pd.Timestamp) -> Optional[str]:
        base_prompt = (
            self.system_prompt if self.system_prompt.strip() != "*" else ""
        )
        if self.add_time_to_system and not pd.isna(timestamp):
            time_info = timestamp.strftime("%m-%d %H:%M:%S")
            if base_prompt:
                return f"{base_prompt} Current datetime: {time_info}"
            return f"Current datetime: {time_info}"
        return base_prompt or None

    def _collect_images(
        self, instruction: ChatMessage, conversation_images: List[str]
    ) -> None:
        src = instruction.src
        if isinstance(src, list):
            conversation_images.extend([s for s in src if s])
        elif isinstance(src, str) and src:
            conversation_images.append(src)

    def _combine_group(self, group: List[ChatMessage]) -> ChatMessage:
        if len(group) == 1:
            return group[0]
        base = group[0]
        contents = [msg.msg for msg in group if msg.msg]
        combined_content = "\n".join(contents)
        if len(combined_content) > self.combine_msg_max_length:
            combined_content = combined_content[: self.combine_msg_max_length]
        image_list: List[str] = []
        for msg in group:
            if msg.modality == "image" and msg.src:
                image_list.append(msg.src)
        src_value: Any = image_list if image_list else base.src
        return ChatMessage(
            id=base.id,
            msg_svr_id=base.msg_svr_id,
            type_name=base.type_name,
            is_sender=base.is_sender,
            talker=base.talker,
            msg=combined_content,
            src=src_value,
            create_time=group[-1].create_time,
            room_name=base.room_name,
            is_forward=base.is_forward,
            modality=base.modality,
        )

    def _is_cut(self, message: ChatMessage) -> bool:
        type_lower = message.type_name.lower()
        if type_lower in CUT_TYPES_LOWER:
            return True
        return message.modality == "image" and message.is_sender == 1

    def _load_blocked_words(self, base: Iterable[str]) -> List[str]:
        words = {str(word).strip() for word in base if str(word).strip()}
        extra_path = Path("dataset/blocked_words.json")
        if extra_path.exists():
            try:
                with open(extra_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    extra = data.get("blocked_words", [])
                    words.update(str(word).strip() for word in extra if str(word).strip())
            except (json.JSONDecodeError, OSError) as exc:
                self.logger.warning(f"加载额外屏蔽词失败: {exc}")
        return [word for word in words if word]

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(str(value).strip())
        except Exception:
            return default

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        return text in {"1", "true", "yes", "y"}

    def _contains_blocked_word(self, text: str) -> bool:
        lowered = text.lower()
        for word in self.blocked_words:
            if not word:
                continue
            if word.lower() in lowered:
                return True
        return False

    def _contains_pii(self, text: str) -> bool:
        for pattern in PII_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _resolve_image_path(self, raw_src: Any) -> Optional[str]:
        src_str = str(raw_src or "").strip()
        if not src_str:
            return None
        candidate = Path(src_str)
        if candidate.exists():
            try:
                return str(candidate.relative_to(self.media_dir))
            except ValueError:
                return str(candidate)
        stem = Path(src_str).stem
        matched = list(self.images_dir.glob(f"{stem}.*"))
        if matched:
            return str(matched[0].relative_to(self.media_dir))
        return None
