#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import os
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Tuple, Dict
from database.db_connector import DatabaseConnector
import datetime
from collections import defaultdict
from config.config import get_config
from logger.logger import get_logger
import threading
import math

# 导入LM Studio支持
from openai.openai_client import LLMDataCleaner

# 创建全局清洗器实例
llm_cleaner = LLMDataCleaner()
# 获取配置实例
config = get_config()
logger = get_logger('Generate_training_data')

# 全局中断标志
interrupt_flag = threading.Event()

def signal_handler(signum, frame):
    """处理Ctrl+C中断信号"""
    logger.info("接收到中断信号，正在优雅退出...")
    interrupt_flag.set()

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

# ---------- 解析相关：微优化 ----------

def decode_varint(data: bytes, pos: int = 0) -> tuple[int, int]:
    """从指定位置解码一个 Varint"""
    result = 0
    shift = 0
    original_pos = pos
    while True:
        if pos >= len(data):
            raise IndexError(f"Varint decoding failed: buffer ended unexpectedly at position {original_pos}.")
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            break
        shift += 7
        if shift >= 64:
            raise ValueError("Varint is too long or invalid.")
    return result, pos

def parse_protobuf_fields(data: bytes) -> list[tuple[int, int, bytes]]:
    """解析protobuf字段"""
    pos = 0
    fields = []
    ln = len(data)
    while pos < ln:
        try:
            tag, pos = decode_varint(data, pos)
            field_number = tag >> 3
            wire_type = tag & 0x7

            if wire_type == 0:  # Varint
                value, pos = decode_varint(data, pos)
                # 避免频繁 to_bytes 转换，保留为最小表示
                v = value.to_bytes((value.bit_length() + 7) // 8, 'little') if value else b'\x00'
                fields.append((field_number, wire_type, v))
            elif wire_type == 1:  # 64-bit
                if pos + 8 > ln:
                    raise IndexError(f"Field {field_number}: Not enough data for 64-bit value.")
                value = data[pos:pos+8]
                pos += 8
                fields.append((field_number, wire_type, value))
            elif wire_type == 2:  # Length-delimited
                length, pos = decode_varint(data, pos)
                end = pos + length
                if end > ln:
                    raise IndexError(f"Field {field_number}: Declared length {length} exceeds buffer size.")
                value = data[pos:end]
                pos = end
                fields.append((field_number, wire_type, value))
            elif wire_type == 5:  # 32-bit
                if pos + 4 > ln:
                    raise IndexError(f"Field {field_number}: Not enough data for 32-bit value.")
                value = data[pos:pos+4]
                pos += 4
                fields.append((field_number, wire_type, value))
            else:
                # 跳过未知类型
                break
        except (ValueError, IndexError):
            break
    return fields


def find_strings_recursively(fields: list[tuple[int, int, bytes]]) -> list[str]:
    """递归查找UTF-8字符串，直接返回字符串列表，减少元组与拷贝"""
    out: List[str] = []
    for _, wire_type, value in fields:
        if wire_type == 2:
            try:
                text = value.decode('utf-8')
                # 先快速校验
                if text and any(c.isprintable() for c in text):
                    out.append(text)
            except UnicodeDecodeError:
                nested_fields = parse_protobuf_fields(value)
                if nested_fields:
                    out.extend(find_strings_recursively(nested_fields))
    return out


def calculate_entropy(text: str) -> float:
    """计算字符串的熵值，用于检测无序字符串"""
    if not text:
        return 0.0
    
    # 统计每个字符的出现频率
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # 计算熵值
    entropy = 0.0
    text_len = len(text)
    for count in char_counts.values():
        probability = count / text_len
        entropy -= probability * math.log2(probability)
    
    return entropy

# 预处理图片/非文本指示器，降小写并用集合加速查询
_IMAGE_INDICATORS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
    '.mp4', '.amr', '.mp3', '.avi', '.mov', '.wmv',  # 扩展媒体扩展
    'offpic_new', 'thumb', 'ori', 'pic', 'bigpic', 'smallpic',
    '[动画表情]', '[表情]', '[图片]', '[闪照]', '[表情]', '[动画]', '[贴图]',
    '[语音]', '[视频]', '[文件]', '[红包]', '[转账]', '[位置]', '[名片]',  # 常见IM占位
    '[系统消息]', '[QQ消息]', '[群消息]', '[好友消息]',
    'ntosfull', 'documents\\tencent files', 'tencent files',
    'term=', 'is_origin=', '_198.jpg', '_720.jpg', '_198_', '_720_',
    '/1684773595-', '/3259379048-', '/thumb/', '/ori/', '/pic/',
    '-05d65b12ef0481a7337c63775bac3ffd', 'b0ddba64f40a8d68eec61b26eabf7750',
    'u_sp2m7izf1zlnjvtcwqx0sg', 'u_ovchmti-zvlwu_0zm_ulgw',
    'multimedia.nt.qq.com.cn', 'qq.com', 'tencent.com', 'gtimg.cn',
    'mqqapi://', 'https://', 'http://', 'ftp://', 'file://',
    'undefined', 'undefine', 'null', 'none',
    '请使用新版手机qq查看', '连续互发消息中断', '恋爱之钥',
    '点击恢复', 'er0s', 'show_pslcard', 'pslcard',
    'interactive_new', 'club.vip.qq.com', 'vip.qq.com',
    'downv6.qq.com', 'qzone_love', 'ti.qq.com',
    # QQ互动标签和系统消息
    'gtip align=', 'qq uin=', 'uin=', 'col=', 'nm=', 'jp=', 'nor txt=',
    '踢了踢', '摸摸头', '完成了', '获得', '查看详情', '互动标识',
    # JSON结构和URL参数
    '{"', '"}', '"align"', '"items"', '"type"', '"txt"', '"col"', '"url"', '"img"',
    'local_jp', 'param', 'url=', 'rkey=', 'target_uin=', 'mutualmark_id=',
    # 系统提示和状态消息
    '你们互发消息，小水花开心极了，恭喜重新点亮初泛涟漪',
    '语音通话', '视频通话', '通话时长', '通话结束',
    # 情侣和互动标识
    '情侣空间', '情侣', '神仙眷侣', '知己浪花', '累计互发消息超过', '爱意永恒', '恭喜升级', '互发消息', '畅聊之火',
    '友谊的小船', '友谊的巨轮', '聊天气泡', '互动标识', '好友互动',
    # 可疑短串和特殊标识
    'd42ae9486fdbc4b7', '305102010004363034020100020445d1',
    # 添加对类似 u_2_Tv579rOiOIDBW9sUrPBA 这种格式的过滤
    'u_',
    # 其他需要过滤的内容
    'qqweb', 'res', 'mutualmark', 'nudgeaction', 'expression.jpg',
    'nudgemall', 'actionid=', 'interactive', 'new/index',
    # 文件路径相关
    'c:\\users\\', 'c:\\', '\\nt_qq\\', '\\nt_data\\', '\\pic\\', '\\thumb\\',
    '我现在有事不在'
}

def is_image_content(text: str) -> bool:
    """检测是否为图片/非文本内容，用于LLM清洗前的标记"""
    if not text:
        return True  # 空内容视为非文本
    
    text_lower = text.lower().strip()
    
    # 检查是否在指示器集合中
    if any(indicator in text_lower for indicator in _IMAGE_INDICATORS):
        return True
    
    # 检查是否为纯JSON结构（包含大括号、引号、冒号等）
    if text.startswith('{') and text.endswith('}'):
        return True
    if '"type"' in text_lower and '"txt"' in text_lower:
        return True
    if '"align"' in text_lower and '"items"' in text_lower:
        return True
    
    # 检查是否为HTML/XML标签格式
    if '<gtip' in text_lower or '</gtip>' in text_lower:
        return True
    if '<qq ' in text_lower or '<img ' in text_lower:
        return True
    if 'align="center"' in text_lower:
        return True
    
    # 检查是否为纯URL或文件路径
    if text.startswith(('http://', 'https://', 'ftp://', 'file://')):
        return True
    if 'c:\\' in text_lower and ('.jpg' in text_lower or '.png' in text_lower):
        return True
    
    # 检查是否为系统消息格式
    if 'qq uin=' in text_lower and 'col=' in text_lower:
        return True
    
    # 检查是否包含特殊编码格式
    if '305102010004' in text or 'ntosfull' in text_lower:
        return True
    
    # 检查是否为纯数字或特殊字符
    if text.isdigit() or len(text.strip()) < 2:
        return True
    
    # 过滤类似 u_2_Tv579rOiOIDBW9sUrPBA 这种格式的随机字符串
    if text.startswith('u_') and '_' in text and len(text) > 20:
        parts = text.split('_')
        if len(parts) >= 3 and all(part.isalnum() for part in parts[1:]):
            # 检查是否有足够的字符混合（大小写+数字）
            alnum = sum(1 for c in text if c.isalnum())
            if alnum / len(text) > 0.8:
                has_upper = any(c.isupper() for c in text)
                has_lower = any(c.islower() for c in text)
                has_digit = any(c.isdigit() for c in text)
                if (has_upper + has_lower + has_digit) >= 2:
                    return True
        
    # 高熵检测：过滤掉无序字符串
    entropy = calculate_entropy(text)
    # 如果熵值超过阈值（例如5.0），则认为是无序字符串
    if entropy > 5.0:
        return True
        
    return False

def extract_text_content(data: bytes) -> str:
    """从protobuf数据中提取纯文本内容，保留检测功能但不进行过滤"""
    try:
        fields = parse_protobuf_fields(data)
        strings = find_strings_recursively(fields)
        max_text = ""
        max_len = 0
        for text in strings:
            if not text:
                continue
            # 只去除控制字符，保留所有内容
            bidi_controls = {'\u202a','\u202b','\u202c','\u202d','\u202e','\u2066','\u2067','\u2068','\u2069'}
            cleaned = ''.join(ch for ch in text if (ch >= ' ' or ch in '\t\n\r') and ch not in bidi_controls).strip()
            if cleaned:
                l = len(cleaned)
                if l > max_len:
                    max_len = l
                    max_text = cleaned
        return max_text
    except Exception:
        return ""

# ---------- 会话整形（聚合为 messages 格式） ----------

SYSTEM_PROMPT = config.get("system_prompt") or "你是一个专业的数据清洗专家，需要对一整天的对话记录进行过滤和整理。请严格按照以下步骤执行：\n\n请按照以下要求：\n1. 删除规则  \n   a. **系统提示**：所有以系统或平台自动生成的提示性文字（如\"开始通话\"\"加载中\"\"自动补齐\"等）  \n   b. **垃圾消息**：无意义字符、乱码、广告或重复发送的同一内容  \n   c. **格式错误**：残缺不全的 JSON、乱码截断的文字  \n   d. **语音/通话记录**：如\"语音通话 00:30\"之类的通话时长记录  \n   e. **社交水印**：包含\"情侣空间\"、\"神仙眷侣\"、\"知己浪花\"、\"累计互发消息超过\"、\"小水花开心极了\"、\"摸摸头\"、\"获得\"、\"互动标识\"等表达社交关系或互动量的句子  \n   f. **图片/媒体消息**：包含URL链接、文件路径、图片标签等明显非对话内容\n   g. **高熵无序字符串**：如随机字符序列、编码数据等无意义内容\n2. 保留规则  \n   - 完全保留所有有意义的用户和 AI 间的对话，无论长短；  \n   - 原文不改动，保留所有 Emoji 及`[图片]`等占位；  \n   - 严格保持原始对话顺序。\n3. 返回需要保留的消息索引列表\n4.保留所有的emoji,或者以中括号[]包裹的表情包和内容,保留\"[图片]\"\n请回复一个JSON格式:\n{\n    \"retained_indices\": [0, 1, 3, 5, ...],\n    \"removed_count\": 5,\n    \"reason\": \"删除了系统提示和垃圾消息\"\n}"

def flush_dialog(f, dialog: List[dict], pretty_f=None):
    """将当前聚合的对话写为一条 jsonl，保持user和assistant交替出现
    ChatML 规范保障：
    - messages 以 system 开头
    - role 仅限 system/user/assistant
    - 对话必须至少包含一对 user -> assistant
      * 若最后一条不是 assistant，自动补 assistant
      * 若整段仅有单 user，则补一个默认 assistant 回复
      * 若整段仅有单 assistant，则在前面补一个默认 user 提示
    """
    if not dialog:
        return
    
    # 不再合并连续相同角色的消息，保持原始交替顺序
    # 只保留 user/assistant 角色，按时间顺序排列
    messages = []
    for msg in dialog:
        r = msg.get("role")
        c = msg.get("content", "").strip()
        if r in ("user", "assistant") and c:
            messages.append({"role": r, "content": c})

    # 若对话为空，则不输出
    if not messages:
        return

    # 确保user和assistant交替出现
    # 处理连续相同角色的消息
    alternating_messages = []
    for msg in messages:
        if not alternating_messages:
            # 第一条消息
            alternating_messages.append(msg)
        else:
            # 如果当前消息角色与上一条相同，合并内容
            if alternating_messages[-1]["role"] == msg["role"]:
                alternating_messages[-1]["content"] += "\n" + msg["content"]
            else:
                alternating_messages.append(msg)

    # 处理"只有单 user 或单 assistant"的场景
    roles = {m["role"] for m in alternating_messages}
    if roles == {"user"}:
        alternating_messages.append({
            "role": "assistant",
            "content": ""
        })
    elif roles == {"assistant"}:
        alternating_messages.insert(0, {
            "role": "user",
            "content": ""
        })

    # 再次确保最后一条为 assistant
    if alternating_messages[-1]["role"] != "assistant":
        alternating_messages.append({
            "role": "assistant",
            "content": ""
        })

    payload = {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + alternating_messages}

    # 机器可读（单行）
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # 人类可读（缩进）
    if pretty_f is not None:
        pretty_f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")



# ---------- 行处理与顺序扫描（加入去重） ----------

def process_row(row: Tuple[int, int, int, bytes]) -> Optional[dict]:
    """
    处理一行数据，提取 role、content 及对端 peer（40030）；无效返回 None。
    过滤掉图片/非文本内容，确保只保留有意义的文本对话
    """
    timestamp, sender_30, sender_33, blob_data = row
    if not blob_data:
        return None
    text_content = extract_text_content(blob_data)
    if not text_content:
        return None
    
    # 直接过滤掉图片/非文本内容
    if is_image_content(text_content):
        return None
    
    # 获取配置中的AI QQ号
    ai_qq = config.get("qq_number_ai")
    
    # 判断角色：
    # sender_30 是对话的对方QQ号（peer）
    # sender_33 是消息发送者的QQ号
    # 注意：确保类型一致，将配置值转换为int
    try:
        ai_qq_int = int(ai_qq) if ai_qq is not None else None
    except (ValueError, TypeError):
        ai_qq_int = None
    
    if ai_qq_int is None:
        logger.warning("AI QQ号未配置，所有消息将被标记为user角色")
        role = "user"
    elif sender_33 == ai_qq_int:  # AI发送的消息
        role = "assistant"
    else:  # 非AI发送的消息（用户/好友）
        role = "user"
    
    return {
        "timestamp": timestamp,
        "peer": sender_30,
        "role": role,
        "content": text_content,
        "is_media": False  # 已经过滤，这里标记为False
    }

# 线程安全的写入锁
write_lock = threading.Lock()

def process_single_peer(peer: int, db_path: str, output_file: str) -> int:
    """
    处理单个QQ号的所有消息，支持中断和实时写入
    """
    if interrupt_flag.is_set():
        return 0
        
    # 为每个线程创建独立的数据库连接
    db = DatabaseConnector(db_path)
    try:
        db.connect()
        
        # 获取该QQ号的所有消息
        rows = db.query("""
            SELECT `40050`, `40030`, `40033`, `40800`
            FROM c2c_msg_table
            WHERE `40030` = ? AND `40800` IS NOT NULL
            ORDER BY `40050` ASC
        """, (peer,))
        
        if not rows:
            return 0

        # 按日期分组的消息
        daily_messages = defaultdict(list)
        
        # 处理该QQ号的所有消息
        for (timestamp, sender_30, sender_33, blob_data) in rows:
            if interrupt_flag.is_set():
                logger.info(f"检测到中断信号，停止处理QQ号 {peer}")
                break
                
            res = process_row((timestamp, sender_30, sender_33, blob_data))
            if res is None:
                continue
            
            # 计算UTC+12时区的日期
            utc_time = datetime.datetime.fromtimestamp(timestamp, datetime.UTC)
            utc_plus_12_time = utc_time + datetime.timedelta(hours=12)
            message_date = utc_plus_12_time.strftime('%Y-%m-%d')
            
            daily_messages[message_date].append(res)

        # 处理每个日期的对话
        written_dialogs = 0
        
        for date, messages in daily_messages.items():
            if interrupt_flag.is_set():
                break
                
            if not messages or len(messages) <= 1:
                continue
            
            # 使用LLM进行按天清洗
            try:
                logger.info(f"使用LLM清洗 {date} 的对话 ({len(messages)}条消息)")
                llm_messages = [
                    {
                        "role": msg["role"], 
                        "content": msg["content"], 
                        "timestamp": msg["timestamp"],
                        "is_media": msg.get("is_media", False)
                    }
                    for msg in messages
                ]
                cleaned_messages = llm_cleaner.clean_daily_conversation(llm_messages, date)
                
                # 只保留必要字段
                daily_dialog = [{"role": msg["role"], "content": msg["content"]} for msg in cleaned_messages]
                logger.info(f"LLM清洗完成: {date} 保留 {len(daily_dialog)}/{len(messages)} 条消息")
                
            except Exception as e:
                logger.error(f"LLM清洗失败 {date}: {e}，保留原始数据")
                # 如果LLM失败，保留所有原始消息
                daily_dialog = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

            if daily_dialog:
                # 实时写入文件
                with write_lock:
                    with open(output_file, 'a', encoding='utf-8', buffering=1024 * 1024) as f:
                        flush_dialog(f, daily_dialog)
                        f.flush()  # 立即刷新到磁盘
                        os.fsync(f.fileno())
                written_dialogs += 1
        
        return written_dialogs
        
    except Exception as e:
        logger.error(f"处理QQ号 {peer} 时出错: {e}")
        return 0
    finally:
        db.close()

def main():
    # 输出文件名（保持原有），自动生成 pretty 版本
    output_file = "training_data.jsonl"
    db = DatabaseConnector(config.get('db_path'))
    
    # 并发配置
    max_workers = config.get('max_workers', 4)  # 默认4个并发线程
    
    try:
        db.connect()
        logger.info(f"开始生成训练数据，输出到: {output_file}")
        logger.info("按 Ctrl+C 可随时中断程序，已处理的数据不会丢失")

        # 获取所有唯一的QQ号（peer）
        peers = db.query("SELECT DISTINCT `40030` FROM c2c_msg_table WHERE `40030` IS NOT NULL")
        all_peers = [peer[0] for peer in peers]
        total_peers = len(all_peers)
        logger.info(f"找到 {total_peers} 个QQ号，使用 {max_workers} 个并发线程")

        # 清空输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            pass

        # 使用线程池并行处理QQ号
        total_written = 0
        processed_peers = 0
        db_path = config.get('db_path')
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_peer = {
                executor.submit(process_single_peer, peer, db_path, output_file): peer
                for peer in all_peers
            }
            
            # 收集结果
            for future in as_completed(future_to_peer):
                if interrupt_flag.is_set():
                    logger.info("检测到中断信号，停止提交新任务...")
                    # 取消未开始的任务
                    for f in future_to_peer:
                        if not f.done():
                            f.cancel()
                    break
                    
                peer = future_to_peer[future]
                try:
                    written_count = future.result()
                    total_written += written_count
                    processed_peers += 1
                    
                    # 实时进度显示
                    progress = (processed_peers / total_peers) * 100
                    logger.info(f"进度: {processed_peers}/{total_peers} ({progress:.1f}%) - "
                              f"QQ号 {peer} 处理完成，写入 {written_count} 段对话，"
                              f"总计: {total_written} 段")
                    
                    # 每处理10个QQ号就显示一次文件大小
                    if processed_peers % 10 == 0:
                        try:
                            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                            logger.info(f"当前输出文件大小: {file_size:.2f} MB")
                        except:
                            pass
                            
                except Exception as e:
                    processed_peers += 1
                    logger.error(f"QQ号 {peer} 处理失败: {e}")

        if interrupt_flag.is_set():
            logger.info(f"程序被中断! 已处理 {processed_peers}/{total_peers} 个QQ号，写出 {total_written} 段对话")
        else:
            logger.info(f"完成! 总共写出 {total_written} 段对话到 {output_file}")

    except Exception as e:
        logger.error(f"错误: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()