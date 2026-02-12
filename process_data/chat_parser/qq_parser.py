#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
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
from utils.database.db_connector import DatabaseConnector

config = get_config()
logger = get_logger('QQParser')

class QQParser:
    """
    QQ 聊天数据解析器
    将 QQ SQLite 数据库格式转换为统一的 CSV 格式
    """
    
    def __init__(self, db_path: str, output_dir: str = "./dataset/csv/", qq_number_ai: Optional[str] = None):
        """
        初始化 QQ 解析器
        
        Args:
            db_path: QQ 数据库文件路径
            output_dir: CSV 输出目录
            qq_number_ai: AI的QQ号码，用于区分发送者
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.qq_number_ai = qq_number_ai
        self.db_connector = DatabaseConnector(db_path)
        # 增强过滤器的已知乱码模式库
        self.known_garbage_patterns = [
            'tAB>b)Z)L',
            'c/Pخ𝅘𝅥Xo',  
            'ړ~!ufX1L',
            'Rd;kj\ncd',
            'HR=    ί0\\5O',
            'E|gQf',
        ]
        
        # 正则表达式模式
        self.regex_patterns = [
            r'^[A-Z]{1,3}[><!|=~]{1,3}',  # 大写字母+特殊符号开头
            r'[)><!;=]{2,}',              # 连续特殊符号
            r'[A-Za-z]{1,3}[^\w\s\u4e00-\u9fff]{2,}',  # 字母后跟多个特殊符号
            r'^[A-Z][a-z]?[^\w\s]+',      # 单个大写字母开头的乱码
            r'[\\][0-9]',                 # 反斜杠+数字
        ]
        
        # Unicode字符范围检查
        self.problematic_unicode_ranges = [
            (0x0600, 0x06FF),  # 阿拉伯文
            (0x0400, 0x04FF),  # 西里尔文  
            (0x0370, 0x03FF),  # 希腊文
            (0xE000, 0xF8FF),  # 私用区
        ]
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 消息类型映射
        self.message_types = {
            0: "text",
            1: "image", 
            2: "voice",
            3: "video",
            4: "file",
            5: "location",
            6: "link",
            7: "system",
            8: "sticker",
            9: "audio",
            10: "other"
        }
    
    def decode_varint(self, data: bytes, pos: int = 0) -> Tuple[int, int]:
        """从指定位置解码一个 Varint"""
        result = 0
        shift = 0
        original_pos = pos
        while True:
            if pos >= len(data):
                raise IndexError(f"Varint解码失败: 在位置{original_pos}意外结束")
            b = data[pos]
            result |= (b & 0x7F) << shift
            pos += 1
            if not (b & 0x80):
                break
            shift += 7
            if shift >= 64:
                raise ValueError("Varint过长或无效")
        return result, pos
    
    def parse_protobuf_fields(self, data: bytes) -> List[Tuple[int, int, bytes]]:
        """解析protobuf字段"""
        pos = 0
        fields = []
        while pos < len(data):
            try:
                tag, pos = self.decode_varint(data, pos)
                field_num = tag >> 3
                wire_type = tag & 0x07
                
                if wire_type == 0:  # varint
                    value, pos = self.decode_varint(data, pos)
                    fields.append((field_num, wire_type, value.to_bytes((value.bit_length() + 7) // 8, 'little')))
                elif wire_type == 1:  # 64-bit
                    if pos + 8 > len(data):
                        break
                    value_bytes = data[pos:pos+8]
                    pos += 8
                    fields.append((field_num, wire_type, value_bytes))
                elif wire_type == 2:  # length-delimited
                    length, pos = self.decode_varint(data, pos)
                    if pos + length > len(data):
                        break
                    value_bytes = data[pos:pos+length]
                    pos += length
                    fields.append((field_num, wire_type, value_bytes))
                elif wire_type == 5:  # 32-bit
                    if pos + 4 > len(data):
                        break
                    value_bytes = data[pos:pos+4]
                    pos += 4
                    fields.append((field_num, wire_type, value_bytes))
                else:
                    break
            except (IndexError, ValueError):
                break
        return fields
    
    def find_strings_recursively(self, fields: List[Tuple[int, int, bytes]]) -> List[str]:
        """递归查找字符串"""
        strings = []
        for field_num, wire_type, data in fields:
            if wire_type == 2:  # length-delimited
                try:
                    text = data.decode('utf-8', errors='ignore')
                    if text and self.is_valid_text(text):
                        strings.append(text)
                except UnicodeDecodeError:
                    pass
                
                # 递归解析嵌套字段
                try:
                    nested_fields = self.parse_protobuf_fields(data)
                    strings.extend(self.find_strings_recursively(nested_fields))
                except:
                    pass
        return strings
    
    def contains_sensitive_info(self, text: str) -> bool:
        """检测是否包含敏感信息需要过滤"""
        if not text:
            return False
        
        sensitive_patterns = [
            # 用户名和路径
            r'Users\\\w+',  # Windows用户名
            r'C:\\Users',     # 系统路径
            # QQ相关路径
            r'Tencent Files',
            r'NTOS.*::', 
            # 电话号码模式
            r'\b1[3-9]\d{9}\b',
            # 身份证号
            r'\b\d{17}[\dX]\b',
            # IP地址
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ]
        
        import re
        return any(re.search(pattern, text) for pattern in sensitive_patterns)
    
    def contains_garbled_chars(self, text: str) -> bool:
        """检测是否包含乱码字符"""
        if not text:
            return False
        
        # 常见的乱码字符模式
        garbled_patterns = [
            # 非常用Unicode字符
            lambda c: 0x0080 <= ord(c) <= 0x00FF and c not in ' ¡¢£¤¥¦§¨©ª«¬­®¯',
            # 控制字符
            lambda c: 0x0000 <= ord(c) <= 0x001F and c not in '\t\n\r',
            # 高Unicode区间的特殊字符
            lambda c: ord(c) > 0xFFFF,
            # 私用区字符
            lambda c: 0xE000 <= ord(c) <= 0xF8FF,
        ]
        
        garbled_count = sum(1 for c in text for pattern in garbled_patterns if pattern(c))
        
        # 如果乱码字符比例太高，认为是乱码文本
        if len(text) > 0 and garbled_count / len(text) > 0.2:
            return True
        
        # 检查是否包含太多非可打印字符
        non_printable = sum(1 for c in text if not c.isprintable())
        if len(text) > 0 and non_printable / len(text) > 0.3:
            return True
        
        return False
    
    def is_meaningful_text(self, text: str) -> bool:
        """判断是否为有意义的文本内容"""
        if not text or len(text.strip()) < 1:
            return False
        
        text = text.strip()
        
        # 检查是否包含乱码字符
        if self.contains_garbled_chars(text):
            return False
        
        # 检查是否包含有意义的字符（中文、英文字母、数字）
        meaningful_chars = sum(1 for c in text if c.isalpha() or c.isdigit() or '一' <= c <= '鿿')
        if meaningful_chars == 0:
            return False
        
        # 检查可读性：有意义字符占比不能太低
        total_chars = len(text)
        if total_chars > 0 and meaningful_chars / total_chars < 0.5:  # 提高阈值
            return False
        
        # 检查是否为常见的有意义词语
        # 如果包含常用词汇，认为是有意义的
        common_words = [
            # 中文常用词
            '你', '我', '他', '她', '是', '不', '有', '的', '了', '在', '都', '可以', '也',
            '怎么', '什么', '为什么', '对', '好', '行', '的确', '真假', '这', '那',
            '太', '点', '了', '猜', '都', '心动', '堆料', '猿',
            # 英文常用词
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'how', 'what', 'why', 'when', 'where', 'who', 'yes', 'no', 'ok', 'good', 'bad',
            # 品牌名称
            'Neo', 'iPhone', 'Android', 'Windows', 'Mac'
        ]
        
        text_lower = text.lower()
        contains_common_word = any(word.lower() in text_lower for word in common_words)
        
        # 如果包含常用词汇，则认为是有意义的
        if contains_common_word:
            return True
        
        # 否则需要更严格的验证
        # 至少包含3个有意义字符，且不是纯数字或符号
        if meaningful_chars >= 3 and not text.replace(' ', '').isdigit():
            # 进一步检查：不能全部是特殊字符
            special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
            return special_chars / len(text) < 0.5
        
        return False
    
    def is_valid_text(self, text: str) -> bool:
        """验证文本是否有效"""
        if not text or len(text) < 1:
            return False
        
        # 检查是否为空字符串或undefined
        text_clean = text.strip().lower()
        if not text_clean or text_clean in ['undefined', 'null', 'none', '']:
            return False
        
        # 检查是否为媒体内容
        if self.is_media_content(text):
            return False
        
        # 检查是否包含敏感信息
        if self.contains_sensitive_info(text):
            return False
        
        # 检查是否为有意义的文本
        if not self.is_meaningful_text(text):
            return False
            
        return True
    
    def clean_text_content(self, text: str) -> str:
        """清理文本内容，去除乱码和不可见字符"""
        if not text:
            return ""
        
        # 去除控制字符和双向文本控制符
        bidi_controls = {'\u202a','\u202b','\u202c','\u202d','\u202e','\u2066','\u2067','\u2068','\u2069'}
        control_chars = set(chr(i) for i in range(32)) - {'\t', '\n', '\r'}
        unwanted_chars = bidi_controls | control_chars
        
        # 清理字符
        cleaned = ''.join(ch for ch in text if ch not in unwanted_chars)
        
        # 去除开头和结尾的非字母数字字符（但保留中文等）
        cleaned = cleaned.strip()
        
        # 如果开头或结尾有明显的乱码模式，尝试移除
        while cleaned and not (cleaned[0].isalnum() or ord(cleaned[0]) > 127):
            cleaned = cleaned[1:]
        
        while cleaned and not (cleaned[-1].isalnum() or ord(cleaned[-1]) > 127 or cleaned[-1] in '.!?。！？,，;:；：'):
            cleaned = cleaned[:-1]
        
        return cleaned.strip()
    
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
    
    def calculate_text_quality(self, text: str) -> float:
        """计算文本质量评分"""
        if not text:
            return 0.0
        
        score = 0.0
        total_chars = len(text)
        
        # 中文字符加分
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        score += (chinese_chars / total_chars) * 2.0
        
        # 英文字母加分
        ascii_letters = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        score += (ascii_letters / total_chars) * 1.5
        
        # 数字适量加分
        digits = sum(1 for c in text if c.isdigit())
        score += min((digits / total_chars) * 0.5, 0.3)
        
        # 标点符号适量加分
        punctuation = sum(1 for c in text if c in '.,!?;:。，！？；：')
        score += min((punctuation / total_chars) * 0.3, 0.2)
        
        # 长度奖励（但不要太长）
        if 5 <= total_chars <= 200:
            score += 0.5
        elif total_chars > 200:
            score += 0.2
        
        return score
    
    def extract_text_content(self, data: bytes) -> str:
        """从protobuf数据中提取纯文本内容"""
        try:
            fields = self.parse_protobuf_fields(data)
            strings = self.find_strings_recursively(fields)
            
            # 查找最合适的文本内容
            candidates = []
            for text in strings:
                if not text:
                    continue
                
                cleaned = self.clean_text_content(text)
                if cleaned and len(cleaned) >= 1:
                    # 计算文本质量评分
                    score = self.calculate_text_quality(cleaned)
                    candidates.append((cleaned, score, len(cleaned)))
            
            if not candidates:
                return ""
            
            # 按质量评分和长度排序，选择最佳候选
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            return candidates[0][0]
            
        except Exception:
            return ""
    
    def is_encoded_data(self, text: str) -> bool:
        """检测是否为编码数据（QQ的图片/文件编码）"""
        if not text or len(text) < 8:
            return False
        
        text = text.strip()
        
        # QQ常见的编码数据模式
        qq_encoded_patterns = [
            # 十进制数字开头的编码（通常是图片ID）
            lambda s: len(s) > 20 and s.startswith(('10P', '20P', '30P')) and any(c.isalnum() for c in s[3:]),
            # base64类似的编码
            lambda s: len(s) > 20 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-' for c in s),
            # 纯数字ID
            lambda s: len(s) > 15 and s.isdigit(),
            # 十六进制编码
            lambda s: len(s) > 16 and all(c in '0123456789abcdefABCDEF' for c in s),
            # 混合字母数字无意义字符串
            lambda s: len(s) > 12 and sum(1 for c in s if c.isalnum()) / len(s) > 0.8 and not any(word in s.lower() for word in ['neo', 'phone', 'user', 'file', 'path'])
        ]
        
        return any(pattern(text) for pattern in qq_encoded_patterns)
    
    def contains_file_path(self, text: str) -> bool:
        """检测是否包含文件路径"""
        if not text:
            return False
        
        path_indicators = [
            'C:\\', 'D:\\', 'E:\\', 'F:\\',  # Windows路径
            '/home/', '/usr/', '/var/', '/tmp/',  # Linux路径 
            'Documents', 'Desktop', 'Downloads',  # 常见文件夹
            'Tencent Files', 'QQ',  # QQ相关路径
            'NTOS', 'Full::'  # QQ系统标识符
        ]
        
        return any(indicator in text for indicator in path_indicators)
    
    def is_qq_download_link(self, text: str) -> bool:
        """检测是否为QQ下载链接或文件传输链接"""
        if not text:
            return False
        
        # QQ文件下载链接模式
        qq_download_patterns = [
            # 标准下载链接
            'download?appid=',
            'fileid=',
            '&spec=',
            # 其他QQ文件相关模式
            'qq.com/file/',
            'qzone.qq.com/',
            'weiyun.com/',
            # 文件传输相关
            'appid=1406',  # QQ常用应用ID
            'EhR', 'EhQ',   # QQ文件ID常用前缀
        ]
        
        # 检查是否匹配下载链接模式
        if any(pattern in text for pattern in qq_download_patterns):
            return True
        
        # 检查是否为URL模式（但不是正常网址）
        if ('?' in text and '&' in text and '=' in text and 
            len(text) > 50 and 
            not any(domain in text.lower() for domain in ['http://', 'https://', 'www.'])):
            return True
        
        return False
    
    def is_media_content(self, text: str, blob_data: bytes = None) -> bool:
        """检测是否为媒体内容（图片、表情包、语音等）"""
        if not text:
            return True
        
        # 常见的媒体内容标识符
        media_indicators = [
            '[图片]', '[表情]', '[Pic]', '[图]', '[Image]',
            '[语音]', '[Voice]', '[音频]', '[Audio]',
            '[视频]', '[Video]', '[文件]', '[File]',
            '[动画表情]', '[Sticker]', '[贴纸]',
            'data:image/', '.jpg', '.png', '.gif', '.jpeg', '.webp',
            '.mp3', '.wav', '.mp4', '.avi'
        ]
        
        # 检查文本中是否包含媒体标识符
        if any(indicator in text for indicator in media_indicators):
            return True
        
        # 检查是否为QQ下载链接
        if self.is_qq_download_link(text):
            return True
        
        # 检查是否为编码数据
        if self.is_encoded_data(text):
            return True
        
        # 检查是否包含文件路径
        if self.contains_file_path(text):
            return True
        
        # 检查是否包含大量非可读字符（可能是编码后的媒体数据）
        non_printable = sum(1 for c in text if not c.isprintable() or ord(c) < 32)
        if len(text) > 10 and non_printable / len(text) > 0.3:
            return True
        
        return False
    
    def determine_message_type(self, content: str, blob_data: bytes) -> str:
        """判断消息类型"""
        if not content:
            return "system"
        
        # 检查是否为QQ下载链接（文件传输）
        if self.is_qq_download_link(content):
            return "file"
        
        # 检查是否为语音消息
        if any(indicator in content for indicator in ['[语音]', '[Voice]', '[音频]', '[Audio]']):
            return "voice"
        
        # 检查是否为视频消息
        if any(indicator in content for indicator in ['[视频]', '[Video]']):
            return "video"
        
        # 检查是否为文件消息
        if any(indicator in content for indicator in ['[文件]', '[File]']):
            return "file"
        
        # 检查是否为图片或表情
        if any(indicator in content for indicator in ['[图片]', '[表情]', '[Pic]', '[图]', '[Image]', '[动画表情]', '[Sticker]', '[贴纸]']):
            return "image"
        
        # 检查是否为媒体内容
        if self.is_media_content(content, blob_data):
            return "media"
        
        # 默认为文本消息
        return "text"
    
    def format_timestamp(self, timestamp: int) -> str:
        """格式化时间戳"""
        try:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, OSError):
            return ""
    
    def get_all_peers(self) -> List[int]:
        """获取所有对话对象的QQ号"""
        try:
            self.db_connector.connect()
            peers = self.db_connector.query("SELECT DISTINCT `40030` FROM c2c_msg_table WHERE `40030` IS NOT NULL")
            return [peer[0] for peer in peers]
        except Exception as e:
            logger.error(f"获取对话对象失败: {e}")
            return []
        finally:
            if self.db_connector.conn:
                self.db_connector.conn.close()
    
    def parse_peer_messages(self, peer_qq: int) -> List[Dict[str, Any]]:
        """解析指定对象的所有消息"""
        messages = []
        
        try:
            self.db_connector.connect()
            
            # 查询该QQ号的所有消息
            rows = self.db_connector.query("""
                SELECT `40050`, `40030`, `40033`, `40800`
                FROM c2c_msg_table
                WHERE `40030` = ? AND `40800` IS NOT NULL
                ORDER BY `40050` ASC
            """, (peer_qq,))
            
            # 优先使用实例参数，其次使用配置文件
            ai_qq = self.qq_number_ai or config.get("qq_number_ai")
            try:
                ai_qq_int = int(ai_qq) if ai_qq is not None else None
            except (ValueError, TypeError):
                logger.warning(f"AI QQ号配置无效: {ai_qq}")
                ai_qq_int = None
            
            if ai_qq_int is None:
                logger.warning(
                    f"处理QQ号{peer_qq}时，AI QQ号未配置或无效，所有消息将被标记为is_sender=0"
                )
            else:
                logger.info(f"处理QQ号{peer_qq}，使用的AI QQ号: {ai_qq_int}")
            
            for idx, (timestamp, sender_30, sender_33, blob_data) in enumerate(rows):
                if not blob_data:
                    continue
                
                # 提取文本内容
                content = self.extract_text_content(blob_data)
                
                # 使用集成的增强过滤器进行验证
                if not content:
                    continue
                    
                # 去除首尾空格
                content = content.strip()
                
                # 规范化多次发送的消息（将连续空格替换为换行符）
                content = self.normalize_multiple_messages(content)
                
                # 使用集成的增强过滤器检查文本有效性
                if not self.is_enhanced_valid_text(content):
                    logger.debug(f"跳过无效或乱码文本: {repr(content[:50])}...")
                    continue
                
                # 额外检查媒体内容
                if self.is_media_content(content, blob_data):
                    logger.debug(f"跳过媒体内容: {content[:50]}...")
                    continue
                
                # 判断发送者
                # sender_33 是消息发送者的QQ号
                # 如果 sender_33 == ai_qq_int，说明AI发送的消息， is_sender = 1
                # 否则是对方发送的消息， is_sender = 0
                is_sender = 1 if (ai_qq_int is not None and sender_33 == ai_qq_int) else 0
                talker = str(sender_33)
                
                # 调试信息：输出判断逻辑
                if ai_qq_int is not None:
                    logger.debug(f"发送者QQ: {sender_33}, AI_QQ: {ai_qq_int}, is_sender: {is_sender}")
                
                # 判断消息类型
                message_type = self.determine_message_type(content, blob_data)
                
                message = {
                    'id': idx + 1,
                    'MsgSvrID': f"{timestamp}_{sender_33}",
                    'type_name': message_type,
                    'is_sender': is_sender,
                    'talker': talker,
                    'msg': content,
                    'src': '',  # QQ数据库中媒体文件路径需要单独处理
                    'CreateTime': self.format_timestamp(timestamp),
                    'room_name': f'QQ_{peer_qq}',
                    'is_forward': 0  # 转发消息检测需要更复杂的逻辑
                }
                
                messages.append(message)
        
        except Exception as e:
            logger.error(f"解析QQ号{peer_qq}的消息失败: {e}")
        
        finally:
            if self.db_connector.conn:
                self.db_connector.conn.close()
        
        return messages
    
    def save_to_csv(self, messages: List[Dict[str, Any]], peer_qq: int):
        """保存消息到CSV文件"""
        if not messages:
            return
        
        # 创建对话对象目录
        peer_dir = self.output_dir / f"QQ_{peer_qq}"
        peer_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV文件路径
        csv_file = peer_dir / f"QQ_{peer_qq}_chat.csv"
        
        # CSV字段
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
            logger.error(f"保存CSV文件失败: {e}")
    
    def parse_all(self):
        """解析所有对话数据并转换为CSV格式"""
        logger.info("开始解析QQ聊天数据...")
        
        # 获取所有对话对象
        peers = self.get_all_peers()
        if not peers:
            logger.warning("未找到任何对话数据")
            return
        
        logger.info(f"找到 {len(peers)} 个对话对象")
        
        total_messages = 0
        for peer_qq in peers:
            logger.info(f"正在处理QQ号: {peer_qq}")
            
            # 解析该对象的消息
            messages = self.parse_peer_messages(peer_qq)
            
            if messages:
                # 保存到CSV
                self.save_to_csv(messages, peer_qq)
                total_messages += len(messages)
            else:
                logger.warning(f"QQ号 {peer_qq} 没有有效消息")
        
        logger.info(f"解析完成，总共处理了 {total_messages} 条消息")
    
    # ========== 增强文本过滤器方法 ==========
    
    def contains_problematic_unicode(self, text: str) -> bool:
        """检查是否包含问题Unicode字符"""
        for char in text:
            char_code = ord(char)
            for start, end in self.problematic_unicode_ranges:
                if start <= char_code <= end:
                    return True
        return False
    
    def matches_garbage_pattern(self, text: str) -> bool:
        """检查是否匹配已知的乱码模式"""
        # 直接字符串匹配
        for pattern in self.known_garbage_patterns:
            if pattern in text:
                return True
        
        # 正则表达式匹配  
        for pattern in self.regex_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def has_excessive_special_chars(self, text: str) -> bool:
        """检查特殊字符比例是否过高"""
        if len(text) <= 2:
            return False
            
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace() and ord(c) < 256)
        return special_chars / len(text) > 0.4
    
    def is_enhanced_garbled_text(self, text: str) -> bool:
        """增强的乱码检测"""
        if not text or not text.strip():
            return True
        
        text = text.strip()
        
        # 检查已知乱码模式
        if self.matches_garbage_pattern(text):
            return True
        
        # 检查问题Unicode字符
        if self.contains_problematic_unicode(text):
            return True
        
        # 检查非可打印字符
        non_printable = sum(1 for c in text if not c.isprintable())
        if non_printable > 0:
            return True
        
        # 检查特殊字符比例
        if self.has_excessive_special_chars(text):
            return True
        
        # 对于短字符串，更严格的验证
        if len(text) <= 10:
            meaningful_chars = sum(1 for c in text if c.isalpha() or c.isdigit() or '\u4e00' <= c <= '\u9fff')
            if meaningful_chars / len(text) < 0.8:
                return True
        
        return False
    
    def is_enhanced_valid_text(self, text: str) -> bool:
        """增强的文本有效性验证"""
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        # 首先检查是否为乱码
        if self.is_enhanced_garbled_text(text):
            return False
        
        # 检查是否包含有意义的字符
        meaningful_chars = sum(1 for c in text if c.isalpha() or c.isdigit() or '\u4e00' <= c <= '\u9fff')
        if meaningful_chars == 0:
            return False
        
        # 棄查常用词汇
        common_words = [
            # 中文常用词（扩展版）
            '\u4f60', '\u6211', '\u4ed6', '\u5979', '\u662f', '\u4e0d', '\u6709', '\u7684', '\u4e86', '\u5728', '\u90fd', '\u53ef\u4ee5', '\u4e5f',
            '\u600e\u4e48', '\u4ec0\u4e48', '\u4e3a\u4ec0\u4e48', '\u5bf9', '\u597d', '\u884c', '\u7684\u786e', '\u771f\u5047', '\u8fd9', '\u90a3',
            '\u592a', '\u70b9', '\u4e86', '\u90a3\u4e2a', '\u8bd5\u8bd5', '\u80fd\u4e0d\u80fd', '\u54e6\u4e0d\u5bf9', '\u6ca1\u4e8b', '\u53ef\u4ee5',
            '\u5f00\u6e90', '\u547d\u4ee4\u884c', '\u52a8\u753b', '\u54c6', '\u8fd9\u4e48\u5927', '\u56e0\u4e3a', '\u5199', '\u7684\u786e',
            # 英文常用词
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'how', 'what', 'why', 'when', 'where', 'who', 'yes', 'no', 'ok', 'good', 'bad',
            'idk', 'sketch', 
            # 技术相关
            'qt5', 'qt', 'pyqt', 'neo', 'flux'
        ]
        
        text_lower = text.lower()
        contains_common_word = any(word.lower() in text_lower for word in common_words)
        
        # 如果包含常用词汇，则认为是有意义的
        if contains_common_word:
            return True
        
        # 最后的验证：至少3个有意义字符且特殊字符比例不高
        if meaningful_chars >= 3:
            special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
            return special_chars / len(text) < 0.3
        
        return False

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='QQ聊天记录解析器 - 将QQ SQLite数据库转换为CSV格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python qq_parser.py --qq-db-path "/path/to/qq.db" --qq-number-ai "123456789" --output "./output/"
  python qq_parser.py  # 使用配置文件中的参数
        """
    )
    
    parser.add_argument(
        '--qq-c2c-db-path',
        dest='qq_c2c_db_path',
        type=str,
        required=False,
        help='QQ私聊(c2c_msg_table)数据库文件路径 (默认从配置文件获取)'
    )

    parser.add_argument(
        '--qq-db-path',
        dest='qq_c2c_db_path',
        type=str,
        required=False,
        help='QQ数据库文件路径 (兼容旧参数，等同于--qq-c2c-db-path)'
    )
    
    parser.add_argument(
        '--qq-number-ai',
        type=str,
        required=False,
        help='AI的QQ号码,用于区分发送者 (默认从配置文件获取)'
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
    # 从命令行参数、配置文件和默认值中获取参数
    qq_db_path = (
        getattr(args, 'qq_c2c_db_path', None)
        or config.get('qq_c2c_db_path')
        or config.get('qq_db_path')
        or None
    )
    qq_number_ai = args.qq_number_ai or config.get('qq_number_ai') or None
    output_dir = args.output or config.get('output_dir') or "./dataset/csv/"
    
    return qq_db_path, qq_number_ai, output_dir

def main():
    """主函数"""
    # 创建命令行解析器
    parser = create_parser()
    args = parser.parse_args()
    
    # 获取有效配置
    qq_db_path, qq_number_ai, output_dir = get_effective_config(args)
    
    # 验证必要参数
    if not qq_db_path:
        logger.error("QQ私聊数据库路径未指定，请使用 --qq-c2c-db-path 参数或在配置文件中设置 qq_c2c_db_path")
        return 1
        
    if not os.path.exists(qq_db_path):
        logger.error(f"QQ数据库文件不存在: {qq_db_path}")
        return 1
    
    if not qq_number_ai:
        logger.warning("未指定AI QQ号，所有消息将被标记为 is_sender=0")
    
    # 输出配置信息
    logger.info(f"QQ数据库路径: {qq_db_path}")
    logger.info(f"AI QQ号: {qq_number_ai or '未设置'}")
    logger.info(f"输出目录: {output_dir}")
    
    # 创建解析器并执行解析
    qq_parser = QQParser(qq_db_path, output_dir, qq_number_ai)
    qq_parser.parse_all()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
