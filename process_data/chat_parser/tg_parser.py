#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import os
import pandas as pd
import re
import argparse
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
from pathlib import Path
import sys

# 添加项目根目录到路径以导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.config.config import get_config
from utils.logger.logger import get_logger

config = get_config()
logger = get_logger('TGParser')

class TGParser:
    """
    Telegram 聊天数据解析器
    将 Telegram 导出的 JSON 格式转换为统一的 CSV 格式
    """
    
    def __init__(self, data_dir: str = "./dataset/original/", output_dir: str = "./dataset/csv/", telegram_chat_id: Optional[str] = None):
        """
        初始化 Telegram 解析器
        
        Args:
            data_dir: Telegram 数据目录路径
            output_dir: CSV 输出目录
            telegram_chat_id: AI的聊天名称，用于区分发送者
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.telegram_chat_id = telegram_chat_id
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 消息类型映射
        self.message_types = {
            "message": "text",
            "photo": "image",
            "video": "video", 
            "audio": "voice",
            "voice": "voice",
            "document": "file",
            "animation": "image",
            "sticker": "image",
            "video_message": "video",
            "location": "location",
            "contact": "contact",
            "poll": "poll",
            "service": "system"
        }
        
        # 媒体类型标识符
        self.media_fields = [
            'photo', 'video', 'audio', 'voice', 'document', 
            'animation', 'sticker', 'video_message', 'location',
            'contact', 'poll', 'file'
        ]
    
    def find_result_json_files(self) -> List[Path]:
        """查找所有有效的 result.json 文件"""
        json_files = []
        
        if not self.data_dir.exists():
            logger.error(f"数据目录不存在: {self.data_dir}")
            return json_files
        
        # 遍历数据目录下的所有子目录
        for item in self.data_dir.iterdir():
            if item.is_dir():
                result_file = item / "result.json"
                if result_file.exists():
                    # 验证 JSON 文件格式
                    if self.validate_json_format(result_file):
                        json_files.append(result_file)
                        logger.info(f"找到有效的 result.json: {result_file}")
                    else:
                        logger.warning(f"跳过无效的 result.json: {result_file}")
        
        return json_files
    
    def validate_json_format(self, json_file: Path) -> bool:
        """验证 JSON 文件格式是否符合 Telegram 导出格式"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查必要字段
            required_fields = ['name', 'type', 'id', 'messages']
            for field in required_fields:
                if field not in data:
                    logger.error(f"JSON 文件缺少必要字段 '{field}': {json_file}")
                    return False
            
            # 检查是否有消息
            if not isinstance(data['messages'], list):
                logger.error(f"messages 字段格式错误: {json_file}")
                return False
            
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 文件格式错误: {json_file}, 错误: {e}")
            return False
        except Exception as e:
            logger.error(f"验证 JSON 文件时出错: {json_file}, 错误: {e}")
            return False
    
    def extract_text_content(self, message: Dict[str, Any]) -> str:
        """提取消息的文本内容"""
        text_content = ""
        
        # 直接文本字段
        if 'text' in message:
            text_field = message['text']
            
            if isinstance(text_field, str):
                text_content = text_field
            elif isinstance(text_field, list):
                # 处理复杂的文本实体结构
                text_parts = []
                for item in text_field:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                text_content = ''.join(text_parts)
        
        # 清理和验证文本
        text_content = text_content.strip()
        
        # 规范化多次发送的消息（将连续空格替换为换行符）
        text_content = self.normalize_multiple_messages(text_content)
        
        return text_content if self.is_valid_text(text_content) else ""
    
    def normalize_multiple_messages(self, text: str) -> str:
        """规范化多次发送的消息：将连续空格替换为换行符"""
        if not text:
            return text
        
        # 将连续的多个空格（2个或以上）替换为换行符
        # 这样可以保持原有的消息分段结构
        import re
        # 匹配2个或以上连续空格，替换为换行符
        normalized = re.sub(r' {2,}', '\n', text)
        return normalized
    
    def is_valid_text(self, text: str) -> bool:
        """验证文本是否有效"""
        if not text or len(text.strip()) < 1:
            return False
        
        text_clean = text.strip()
        
        # 检查是否为空或无效内容
        invalid_content = ['', 'null', 'undefined', 'None']
        if text_clean.lower() in invalid_content:
            return False
        
        # 检查是否包含有意义的字符（中文、英文、数字）
        meaningful_chars = sum(1 for c in text_clean if c.isalpha() or c.isdigit() or '\u4e00' <= c <= '\u9fff')
        if meaningful_chars == 0:
            return False
        
        # 检查可读性：有意义字符占比不能太低
        if len(text_clean) > 0 and meaningful_chars / len(text_clean) < 0.3:
            return False
        
        return True
    
    def determine_message_type(self, message: Dict[str, Any]) -> str:
        """判断消息类型"""
        # 检查媒体字段
        for media_field in self.media_fields:
            if media_field in message:
                return self.message_types.get(media_field, "media")
        
        # 检查消息类型字段
        msg_type = message.get('type', 'message')
        if msg_type in self.message_types:
            return self.message_types[msg_type]
        
        # 检查是否有文本内容
        if 'text' in message and message['text']:
            return "text"
        
        # 默认类型
        return "other"
    
    def get_media_info(self, message: Dict[str, Any]) -> str:
        """获取媒体信息"""
        media_info_parts = []
        
        # 检查各种媒体字段
        for media_field in self.media_fields:
            if media_field in message:
                if media_field == 'photo':
                    media_info_parts.append("[图片]")
                elif media_field == 'video':
                    media_info_parts.append("[视频]")
                elif media_field == 'audio' or media_field == 'voice':
                    media_info_parts.append("[语音]")
                elif media_field == 'document':
                    file_name = message.get('file_name', '未知文件')
                    media_info_parts.append(f"[文件: {file_name}]")
                elif media_field == 'sticker':
                    emoji = message.get('sticker_emoji', '')
                    media_info_parts.append(f"[表情包{emoji}]")
                elif media_field == 'animation':
                    media_info_parts.append("[动图]")
                elif media_field == 'location':
                    media_info_parts.append("[位置]")
                elif media_field == 'contact':
                    media_info_parts.append("[联系人]")
                elif media_field == 'poll':
                    media_info_parts.append("[投票]")
                else:
                    media_info_parts.append(f"[{media_field}]")
        
        return ' '.join(media_info_parts) if media_info_parts else ""
    
    def format_timestamp(self, date_str: str, unixtime_str: str = None) -> str:
        """格式化时间戳"""
        try:
            # 优先使用 unixtime
            if unixtime_str:
                timestamp = int(unixtime_str)
                return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            # 使用 ISO 格式日期
            if date_str:
                # Telegram 导出格式: "2025-05-28T19:06:37"
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d %H:%M:%S')
        
        except (ValueError, TypeError) as e:
            logger.debug(f"时间格式化失败: {date_str}, {unixtime_str}, 错误: {e}")
            return ""
        
        return ""
    
    def determine_sender(self, message: Dict[str, Any], chat_info: Dict[str, Any]) -> Tuple[int, str]:
        """判断消息发送者
        
        Returns:
            Tuple[int, str]: (is_sender, talker)
        """
        # 获取发送者信息
        from_info = message.get('from', '')
        from_id = message.get('from_id', '')
        
        # 处理 from_id 格式 (如 "user5884743199")
        sender_id = ""
        if from_id and isinstance(from_id, str):
            if from_id.startswith('user'):
                sender_id = from_id[4:]  # 去掉 "user" 前缀
            else:
                sender_id = from_id
        
        # 判断是否是 AI 发送的消息
        is_sender = 0
        if self.telegram_chat_id:
            # 如果配置了 AI 的聊天名称，使用聊天名称进行匹配
            if from_info == self.telegram_chat_id:
                is_sender = 1
        
        # 使用发送者名字作为 talker，如果没有则使用 ID
        talker = from_info or sender_id or "unknown"
        
        return is_sender, talker
    
    def parse_chat_file(self, json_file: Path) -> List[Dict[str, Any]]:
        """解析单个聊天文件"""
        messages = []
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chat_info = {
                'name': data.get('name', ''),
                'type': data.get('type', ''),
                'id': data.get('id', ''),
            }
            
            logger.info(f"正在解析聊天: {chat_info['name']} (ID: {chat_info['id']})")
            
            for idx, message in enumerate(data.get('messages', [])):
                try:
                    # 跳过服务消息（如群组创建、成员加入等）
                    if message.get('type') == 'service':
                        continue
                    
                    # 提取文本内容
                    text_content = self.extract_text_content(message)
                    
                    # 获取媒体信息
                    media_info = self.get_media_info(message)
                    
                    # 组合最终消息内容
                    final_content = ""
                    if text_content and media_info:
                        final_content = f"{media_info} {text_content}"
                    elif text_content:
                        final_content = text_content
                    elif media_info:
                        final_content = media_info
                    
                    # 如果没有任何有效内容，跳过
                    if not final_content:
                        continue
                    
                    # 判断消息类型
                    message_type = self.determine_message_type(message)
                    
                    # 判断发送者
                    is_sender, talker = self.determine_sender(message, chat_info)
                    
                    # 格式化时间
                    create_time = self.format_timestamp(
                        message.get('date', ''),
                        message.get('date_unixtime', '')
                    )
                    
                    # 构建消息对象
                    parsed_message = {
                        'id': idx + 1,
                        'MsgSvrID': f"{message.get('id', idx)}_{chat_info['id']}",
                        'type_name': message_type,
                        'is_sender': is_sender,
                        'talker': talker,
                        'msg': final_content,
                        'src': '',  # Telegram 媒体文件路径需要单独处理
                        'CreateTime': create_time,
                        'room_name': f"TG_{chat_info['name']}_{chat_info['id']}",
                        'is_forward': 1 if 'forwarded_from' in message else 0
                    }
                    
                    messages.append(parsed_message)
                
                except Exception as e:
                    logger.error(f"解析消息时出错 (消息ID: {message.get('id', idx)}): {e}")
                    continue
        
        except Exception as e:
            logger.error(f"解析文件失败: {json_file}, 错误: {e}")
        
        return messages
    
    def save_to_csv(self, messages: List[Dict[str, Any]], chat_info: Dict[str, str]):
        """保存消息到 CSV 文件"""
        if not messages:
            logger.warning(f"聊天 {chat_info.get('name', 'unknown')} 没有有效消息")
            return
        
        # 创建聊天目录
        chat_name = chat_info.get('name', 'unknown').replace('/', '_').replace('\\', '_')
        chat_id = chat_info.get('id', 'unknown')
        chat_dir = self.output_dir / f"TG_{chat_name}_{chat_id}"
        chat_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV 文件路径
        csv_file = chat_dir / f"TG_{chat_name}_{chat_id}_chat.csv"
        
        # CSV 字段
        fieldnames = [
            'id', 'MsgSvrID', 'type_name', 'is_sender', 'talker',
            'msg', 'src', 'CreateTime', 'room_name', 'is_forward'
        ]
        
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(messages)
            
            logger.info(f"成功保存 {len(messages)} 条消息到 {csv_file}")
        
        except Exception as e:
            logger.error(f"保存 CSV 文件失败: {e}")
    
    def parse_all(self):
        """解析所有 Telegram 聊天数据并转换为 CSV 格式"""
        logger.info("开始解析 Telegram 聊天数据...")
        
        # 查找所有有效的 result.json 文件
        json_files = self.find_result_json_files()
        
        if not json_files:
            logger.warning("未找到任何有效的 Telegram 数据文件")
            return
        
        logger.info(f"找到 {len(json_files)} 个 Telegram 聊天文件")
        
        total_messages = 0
        for json_file in json_files:
            logger.info(f"正在处理文件: {json_file}")
            
            # 解析聊天文件
            messages = self.parse_chat_file(json_file)
            
            if messages:
                # 从第一条消息获取聊天信息用于文件名
                chat_info = {
                    'name': json_file.parent.name,  # 使用目录名作为聊天名
                    'id': messages[0]['room_name'].split('_')[-1] if messages else 'unknown'
                }
                
                # 保存到 CSV
                self.save_to_csv(messages, chat_info)
                total_messages += len(messages)
            else:
                logger.warning(f"文件 {json_file} 没有有效消息")
        
        logger.info(f"解析完成，总共处理了 {total_messages} 条消息")

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='Telegram 聊天记录解析器 - 将 Telegram 导出的 JSON 格式转换为 CSV 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python tg_parser.py --data-dir "./dataset/original/" --telegram-chat-id "YourChatName" --output "./dataset/csv/"
  python tg_parser.py  # 使用配置文件中的参数
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=False,
        help='Telegram 数据目录路径 (默认: "./dataset/original/")'
    )
    
    parser.add_argument(
        '--telegram-chat-id',
        type=str,
        required=False,
        help='AI的聊天名称，用于区分发送者 (默认从配置文件获取)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        help='输出目录路径 (默认: "./dataset/csv/")'
    )
    
    return parser

def get_effective_config(args):
    """
    获取有效配置，优先级：命令行参数 > 配置文件 > 默认值
    """
    data_dir = args.data_dir or config.get('tg_data_dir') or "./dataset/original/"
    telegram_chat_id = args.telegram_chat_id or config.get('telegram_chat_id') or None
    output_dir = args.output or config.get('output_dir') or "./dataset/csv/"
    
    return data_dir, telegram_chat_id, output_dir

def main():
    """主函数"""
    # 创建命令行解析器
    parser = create_parser()
    args = parser.parse_args()
    
    # 获取有效配置
    data_dir, telegram_chat_id, output_dir = get_effective_config(args)
    
    # 验证数据目录
    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        return 1
    
    if not telegram_chat_id:
        logger.warning("未指定 AI 聊天名称，所有消息将被标记为 is_sender=0")
    
    # 输出配置信息
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"AI 聊天名称: {telegram_chat_id or '未设置'}")
    logger.info(f"输出目录: {output_dir}")
    
    # 创建解析器并执行解析
    tg_parser = TGParser(data_dir, output_dir, telegram_chat_id)
    tg_parser.parse_all()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)