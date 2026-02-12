#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QQ群聊数据解析器（group_msg_table）

输出格式与QQParser(c2c_msg_table)保持一致，便于复用后续清洗/转换链路。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.config.config import get_config
from utils.logger.logger import get_logger
from utils.database.db_connector import DatabaseConnector

from .qq_parser import QQParser

config = get_config()
logger = get_logger("QQGroupParser")


class QQGroupParser(QQParser):
    """QQ群聊解析器：读取 group_msg_table 并输出统一CSV"""

    def __init__(
        self,
        db_path: str,
        output_dir: str = "./dataset/csv/",
        qq_number_ai: Optional[str] = None,
        group_focus_ai: Optional[bool] = None,
        group_context_before: Optional[int] = None,
        group_context_after: Optional[int] = None,
    ):
        super().__init__(db_path=db_path, output_dir=output_dir, qq_number_ai=qq_number_ai)
        self.db_connector = DatabaseConnector(db_path)
        self.group_focus_ai = group_focus_ai
        self.group_context_before = group_context_before
        self.group_context_after = group_context_after

    def _format_timestamp_safe(self, timestamp: int) -> str:
        """兼容秒/毫秒时间戳"""
        if not timestamp:
            return ""
        try:
            if timestamp > 10_000_000_000:  # 约2286年，通常说明是毫秒
                timestamp = int(timestamp / 1000)
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            return ""

    def _get_ai_qq_int(self) -> Optional[int]:
        ai_qq = self.qq_number_ai or config.get("qq_number_ai")
        try:
            return int(ai_qq) if ai_qq is not None else None
        except (ValueError, TypeError):
            logger.warning(f"AI QQ号配置无效: {ai_qq}")
            return None

    def _should_skip_by_meta(
        self,
        msg_type: Optional[int],
        sub_msg_type: Optional[int],
        send_status: Optional[int],
        send_type: Optional[int],
    ) -> bool:
        """基于元信息快速过滤，减少后续LLM清洗成本"""
        if send_status is not None and int(send_status) != 2:
            return True

        if send_type is not None and int(send_type) == 5:
            return True

        if msg_type is None:
            return True

        msg_type = int(msg_type)
        sub_msg_type = int(sub_msg_type or 0)

        # 空/损坏/空白
        if msg_type in (0, 1):
            return True

        # 系统/文件/语音/视频/红包/应用/表情包/合并转发
        if msg_type in (3, 5, 6, 7, 8, 10, 11, 17):
            return True

        # 文本类：剔除明显的非文本
        if msg_type == 2 and sub_msg_type in (2, 3, 16, 65, 577, 4096):
            return True

        # 回复类里带图/复杂卡片的，优先丢弃（文本回复仍可通过protobuf提取）
        if msg_type == 9 and sub_msg_type in (34, 35, 49, 51):
            return True

        return False

    def _get_group_context_config(self) -> Tuple[bool, int, int]:
        focus_ai = self.group_focus_ai
        if focus_ai is None:
            focus_ai = bool(config.get("qq_group_focus_ai", True))

        before = self.group_context_before
        if before is None:
            before = config.get("qq_group_context_before", 6)
        after = self.group_context_after
        if after is None:
            after = config.get("qq_group_context_after", 2)

        try:
            before_int = max(0, int(before))
        except (ValueError, TypeError):
            before_int = 6
        try:
            after_int = max(0, int(after))
        except (ValueError, TypeError):
            after_int = 2

        return bool(focus_ai), before_int, after_int

    def get_all_groups(self, ai_qq_int: Optional[int], focus_ai: bool) -> List[int]:
        """获取群号（可选：仅返回AI发过言的群）"""
        try:
            self.db_connector.connect()
            if focus_ai:
                if ai_qq_int is not None:
                    rows = self.db_connector.query(
                        "SELECT DISTINCT `40030` FROM group_msg_table WHERE `40030` IS NOT NULL AND `40010` = 2 AND `40033` = ?",
                        (ai_qq_int,),
                    )
                else:
                    rows = self.db_connector.query(
                        "SELECT DISTINCT `40030` FROM group_msg_table WHERE `40030` IS NOT NULL AND `40010` = 2 AND `40013` IN (1,2)"
                    )
            else:
                rows = self.db_connector.query(
                    "SELECT DISTINCT `40030` FROM group_msg_table WHERE `40030` IS NOT NULL AND `40010` = 2"
                )
            return [int(row[0]) for row in rows if row and row[0] is not None]
        except Exception as e:
            logger.error(f"获取群号失败: {e}")
            return []
        finally:
            if self.db_connector.conn:
                self.db_connector.conn.close()

    def _get_overall_stats(self, ai_qq_int: Optional[int]) -> Dict[str, int]:
        """获取整体统计信息（用于解释导出数量）"""
        stats = {
            "total_rows": 0,
            "total_groups": 0,
            "ai_msgs": 0,
            "ai_text_msgs": 0,
            "self_msgs_by_send_type": 0,
        }

        try:
            self.db_connector.connect()
            stats["total_rows"] = int(self.db_connector.query("SELECT COUNT(*) FROM group_msg_table")[0][0])
            stats["total_groups"] = int(
                self.db_connector.query("SELECT COUNT(DISTINCT `40030`) FROM group_msg_table WHERE `40030` IS NOT NULL")[0][0]
            )
            stats["self_msgs_by_send_type"] = int(
                self.db_connector.query("SELECT COUNT(*) FROM group_msg_table WHERE `40013` IN (1,2)")[0][0]
            )

            if ai_qq_int is not None:
                stats["ai_msgs"] = int(
                    self.db_connector.query("SELECT COUNT(*) FROM group_msg_table WHERE `40033` = ?", (ai_qq_int,))[0][0]
                )
                stats["ai_text_msgs"] = int(
                    self.db_connector.query(
                        "SELECT COUNT(*) FROM group_msg_table WHERE `40033` = ? AND `40011` = 2 AND `40012` = 1",
                        (ai_qq_int,),
                    )[0][0]
                )
        except Exception as e:
            logger.warning(f"统计信息获取失败: {e}")
        finally:
            if self.db_connector.conn:
                self.db_connector.conn.close()

        return stats

    def parse_group_messages(self, group_qq: int, ai_qq_int: Optional[int]) -> List[Dict[str, Any]]:
        """解析指定群的所有消息"""
        messages: List[Dict[str, Any]] = []
        focus_ai, before_count, after_count = self._get_group_context_config()

        try:
            self.db_connector.connect()

            seen_msg_ids = set()
            msg_index = 0

            try:
                if focus_ai:
                    if ai_qq_int is not None:
                        seq_rows = self.db_connector.query(
                            """
                            SELECT `40003`
                            FROM group_msg_table
                            WHERE `40030` = ? AND `40010` = 2 AND `40033` = ? AND `40041` = 2 AND `40003` IS NOT NULL
                            ORDER BY `40003` ASC
                            """,
                            (group_qq, ai_qq_int),
                        )
                    else:
                        seq_rows = self.db_connector.query(
                            """
                            SELECT `40003`
                            FROM group_msg_table
                            WHERE `40030` = ? AND `40010` = 2 AND `40013` IN (1,2) AND `40041` = 2 AND `40003` IS NOT NULL
                            ORDER BY `40003` ASC
                            """,
                            (group_qq,),
                        )

                    anchor_seqs = [int(r[0]) for r in seq_rows if r and r[0] is not None]
                else:
                    anchor_seqs = []
            except Exception:
                anchor_seqs = []

            if focus_ai and anchor_seqs:
                ranges = []
                for seq in anchor_seqs:
                    start = max(0, seq - before_count)
                    end = seq + after_count
                    ranges.append((start, end))

                ranges.sort(key=lambda x: (x[0], x[1]))
                merged = []
                for start, end in ranges:
                    if not merged or start > merged[-1][1] + 1:
                        merged.append([start, end])
                    else:
                        merged[-1][1] = max(merged[-1][1], end)

                for start, end in merged:
                    rows = self.db_connector.query(
                        """
                        SELECT
                            `40001`, `40050`, `40033`, `40011`, `40012`, `40013`, `40041`, `40800`, `40003`
                        FROM group_msg_table
                        WHERE `40030` = ? AND `40010` = 2 AND `40003` BETWEEN ? AND ? AND `40800` IS NOT NULL
                        ORDER BY `40003` ASC, `40050` ASC, `40001` ASC
                        """,
                        (group_qq, start, end),
                    )

                    for (
                        msg_id,
                        msg_time,
                        sender_qq,
                        msg_type,
                        sub_msg_type,
                        send_type,
                        send_status,
                        blob_data,
                        _msg_seq,
                    ) in rows:
                        if msg_id is not None and msg_id in seen_msg_ids:
                            continue
                        if msg_id is not None:
                            seen_msg_ids.add(msg_id)

                        if not blob_data:
                            continue

                        if self._should_skip_by_meta(msg_type, sub_msg_type, send_status, send_type):
                            continue

                        if isinstance(blob_data, memoryview):
                            blob_data = blob_data.tobytes()

                        content = self.extract_text_content(blob_data)
                        if not content:
                            continue

                        content = self.normalize_multiple_messages(content.strip())
                        if not self.is_enhanced_valid_text(content):
                            continue

                        if self.is_media_content(content, blob_data):
                            continue

                        if ai_qq_int is not None:
                            is_sender = 1 if sender_qq == ai_qq_int else 0
                        else:
                            is_sender = 1 if int(send_type or 0) in (1, 2) else 0

                        message_type = self.determine_message_type(content, blob_data)
                        msg_index += 1
                        messages.append(
                            {
                                "id": msg_index,
                                "MsgSvrID": str(msg_id or f"{msg_time}_{sender_qq}"),
                                "type_name": message_type,
                                "is_sender": is_sender,
                                "talker": str(sender_qq or ""),
                                "msg": content,
                                "src": "",
                                "CreateTime": self._format_timestamp_safe(int(msg_time or 0)),
                                "room_name": f"QQ_GROUP_{group_qq}",
                                "is_forward": 0,
                            }
                        )

                return messages

            # 回退：全量读取（仍带基础元信息过滤）
            try:
                rows = self.db_connector.query(
                    """
                    SELECT
                        `40001`, `40050`, `40033`, `40011`, `40012`, `40013`, `40041`, `40800`
                    FROM group_msg_table
                    WHERE `40030` = ? AND `40010` = 2 AND `40800` IS NOT NULL
                    ORDER BY `40050` ASC, `40003` ASC, `40001` ASC
                    """,
                    (group_qq,),
                )
            except Exception:
                rows = self.db_connector.query(
                    """
                    SELECT
                        `40001`, `40050`, `40033`, `40011`, `40012`, `40013`, `40041`, `40800`
                    FROM group_msg_table
                    WHERE `40030` = ? AND `40010` = 2 AND `40800` IS NOT NULL
                    ORDER BY `40050` ASC, `40001` ASC
                    """,
                    (group_qq,),
                )

            for idx, (msg_id, msg_time, sender_qq, msg_type, sub_msg_type, send_type, send_status, blob_data) in enumerate(rows):
                if not blob_data:
                    continue

                if self._should_skip_by_meta(msg_type, sub_msg_type, send_status, send_type):
                    continue

                if isinstance(blob_data, memoryview):
                    blob_data = blob_data.tobytes()

                content = self.extract_text_content(blob_data)
                if not content:
                    continue

                content = self.normalize_multiple_messages(content.strip())
                if not self.is_enhanced_valid_text(content):
                    continue

                if self.is_media_content(content, blob_data):
                    continue

                if ai_qq_int is not None:
                    is_sender = 1 if sender_qq == ai_qq_int else 0
                else:
                    is_sender = 1 if int(send_type or 0) in (1, 2) else 0

                message_type = self.determine_message_type(content, blob_data)

                messages.append(
                    {
                        "id": idx + 1,
                        "MsgSvrID": str(msg_id or f"{msg_time}_{sender_qq}"),
                        "type_name": message_type,
                        "is_sender": is_sender,
                        "talker": str(sender_qq or ""),
                        "msg": content,
                        "src": "",
                        "CreateTime": self._format_timestamp_safe(int(msg_time or 0)),
                        "room_name": f"QQ_GROUP_{group_qq}",
                        "is_forward": 0,
                    }
                )

        except Exception as e:
            logger.error(f"解析群 {group_qq} 消息失败: {e}")
        finally:
            if self.db_connector.conn:
                self.db_connector.conn.close()

        return messages

    def save_to_csv(self, messages: List[Dict[str, Any]], group_qq: int) -> None:
        """保存消息到CSV文件"""
        if not messages:
            return

        group_dir = Path(self.output_dir) / f"QQ_GROUP_{group_qq}"
        group_dir.mkdir(parents=True, exist_ok=True)

        csv_file = group_dir / f"QQ_GROUP_{group_qq}_chat.csv"
        fieldnames = [
            "id",
            "MsgSvrID",
            "type_name",
            "is_sender",
            "talker",
            "msg",
            "src",
            "CreateTime",
            "room_name",
            "is_forward",
        ]

        try:
            import csv

            with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(messages)
            logger.info(f"成功保存 {len(messages)} 条群消息到 {csv_file}")
        except Exception as e:
            logger.error(f"保存群CSV失败: {e}")

    def parse_all(self) -> None:
        """解析所有群聊数据并转换为CSV格式"""
        logger.info("开始解析QQ群聊数据(group_msg_table)...")

        focus_ai, before_count, after_count = self._get_group_context_config()
        ai_qq_int = self._get_ai_qq_int()

        if focus_ai:
            logger.info(f"启用群聊上下文抽样: before={before_count}, after={after_count}")

        overall = self._get_overall_stats(ai_qq_int)
        if overall.get("total_rows"):
            logger.info(
                f"群聊库统计: 总消息 {overall.get('total_rows')} 条，群数 {overall.get('total_groups')} 个，"
                f"本机发送(sendType=1/2) {overall.get('self_msgs_by_send_type')} 条"
            )
        if ai_qq_int is not None:
            logger.info(
                f"AI QQ({ai_qq_int}) 群聊消息: 总计 {overall.get('ai_msgs')} 条，其中纯文本(40011=2,40012=1) {overall.get('ai_text_msgs')} 条"
            )

        groups = self.get_all_groups(ai_qq_int, focus_ai)
        if not groups:
            logger.warning("未找到任何群聊数据")
            return

        if ai_qq_int is None:
            logger.warning("AI QQ号未配置或无效，将使用sendType推断发送方")

        total_messages = 0
        for group_qq in groups:
            logger.info(f"正在处理群号: {group_qq}")
            messages = self.parse_group_messages(group_qq, ai_qq_int)
            if messages:
                self.save_to_csv(messages, group_qq)
                total_messages += len(messages)
            else:
                logger.warning(f"群号 {group_qq} 没有有效消息")

        logger.info(f"群聊解析完成，总共处理了 {total_messages} 条消息")
