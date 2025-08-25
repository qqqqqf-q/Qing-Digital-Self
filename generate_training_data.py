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
from utils.config.config import get_config
from utils.logger.logger import get_logger
import threading
import math

# 导入LM Studio支持（可选）
try:
    from openai.openai_client import LLMDataCleaner
    LLM_AVAILABLE = True
    # 创建全局清洗器实例
    llm_cleaner = LLMDataCleaner()
except ImportError:
    LLM_AVAILABLE = False
    llm_cleaner = None
    
# 获取配置实例
config = get_config()
logger = get_logger('Generate_training_data')

# 全局中断标志
interrupt_flag = threading.Event()

def signal_handler(signum, frame):
    """处理Ctrl+C中断信号"""
    logger.info(
        zhcn="接收到中断信号，正在优雅退出...",
        en="Received interrupt signal, gracefully exiting..."
    )
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

# 十六进制字符串模式
_HEX_PATTERN = set('0123456789abcdef')

# 需要严格等值匹配删除的短占位/系统消息（小写后对比）
_STRICT_DROP_SET = {
    '语音',  # 仅当整条就是"语音"之类的极短占位
    '[语音]',
    '[语音通话]',
    'nt_1',  # 明显的系统/占位短token
}

def is_image_content(text: str) -> bool:
    """检测是否为图片/非文本内容，用于LLM清洗前的初步过滤"""
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

def is_image_content_strict(text: str) -> bool:
    """传统的严格过滤方法（保留原有逻辑作为备选）"""
    if not text:
        return False
    # 去除控制字符（含 \u0000 与 LRM/嵌入方向控制等），避免"Z\u0000b\u0000"、"U+202D"等伪文本
    # Unicode 双向控制字符集合：U+202A..U+202E, U+2066..U+2069 等
    bidi_controls = {'\u202a','\u202b','\u202c','\u202d','\u202e','\u2066','\u2067','\u2068','\u2069'}
    t = ''.join(ch for ch in text if (ch >= ' ' or ch in '\t\n\r') and ch not in bidi_controls).strip()
    if not t:
        return True  # 清理后为空，视为噪声
    
    # 严格等值短占位
    _STRICT_DROP_SET = {
        '语音',  # 仅当整条就是"语音"之类的极短占位
        '[语音]',
        '[语音通话]',
        'nt_1',  # 明显的系统/占位短token
    }
    if t in _STRICT_DROP_SET:
        return True
    
    # 先剪枝：纯数字直接排除，但保留有意义的单字符
    if t.isdigit():
        return True
    if len(t) == 1 and t not in ['?', '？', '.', '。', '!', '！']:
        return True
    
    tl = t.lower()
    
    # 过滤包含特定通话/时长模式
    # 如：[语音通话] 通话时长 00:05 / 通话时长 00:05
    if ('通话时长' in t and any(c.isdigit() for c in t)) or ('语音通话' in t):
        return True
    # 过滤明显的"[语音]"类占位（已在指示器/严格集覆盖，这里兜底）
    if t.strip() in ('[语音]', '[语音通话]'):
        return True
    
    # 检查是否为十六进制字符串（长度>=16且只包含十六进制字符）
    if len(t) >= 16 and all(c in _HEX_PATTERN for c in tl):
        return True
    
    # 检查JSON格式的QQ系统消息
    if t.startswith('{"') and ('align' in tl or 'items' in tl or 'type' in tl):
        return True
    
    # Base64/URL-safe base64 长串判定（长度阈值>=40），通常为表情/富媒体占位
    # 不做解码，基于字符集与长度快速判定，避免误伤普通文本
    def _looks_base64(s: str) -> bool:
        if len(s) < 40:
            return False
        allowed_std = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        allowed_url = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_')
        cs = set(s)
        return cs.issubset(allowed_std) or cs.issubset(allowed_url)
    # 按空白分词后逐个token检测
    for token in t.split():
        if _looks_base64(token):
            return True
    # 也过滤整句为 URL-safe base64 的情况（常见 IM 参数）
    if _looks_base64(t):
        return True
    
    # 过滤"高熵随机串"（IM内嵌随机token），包含大小写/数字且长度>=24，且字母数字占比>=0.9
    def _looks_high_entropy_token(s: str) -> bool:
        if len(s) < 24:
            return False
        alnum = sum(1 for c in s if c.isalnum())
        if alnum / len(s) < 0.9:
            return False
        has_upper = any(c.isupper() for c in s)
        has_lower = any(c.islower() for c in s)
        has_digit = any(c.isdigit() for c in s)
        return (has_upper + has_lower + has_digit) >= 2
    # 检测整句或分词token
    if _looks_high_entropy_token(t):
        return True
    for token in t.replace('#', ' ').split():
        if _looks_high_entropy_token(token):
            return True
     
    # 限制扫描长度，避免超长字符串影响查找成本
    scan = tl if len(tl) < 2048 else tl[:2048]
    for indicator in _IMAGE_INDICATORS:
        if indicator in scan:
            return True
    
    # 过滤长ID字符串或带媒体后缀的"文件名"样式
    if len(t) > 20 and ('/' in t or '-' in t):
        return True
    lower_t = t.lower()
    if any(lower_t.endswith(ext) for ext in ('.mp4', '.amr', '.wav', '.aac', '.m4a', '.avi', '.mov', '.flv', '.mkv')):
        return True
    # 过滤明显"hash.扩展名"的文件名（如 e56eae88...mp4、f05fddbe...amr）
    base = lower_t.rsplit('.', 1)[0] if '.' in lower_t else ''
    if base and len(base) >= 16 and all(c in _HEX_PATTERN for c in base):
        return True
    
    # 过滤包含大量数字的字符串（可能是编码数据）
    digit_count = sum(1 for c in t if c.isdigit())
    if len(t) > 10 and digit_count / len(t) > 0.7:
        return True
    
    # 过滤类似 u_2_Tv579rOiOIDBW9sUrPBA 这种格式的随机字符串
    if t.startswith('u_') and '_' in t and len(t) > 20:
        parts = t.split('_')
        if len(parts) >= 3 and all(part.isalnum() for part in parts[1:]):
            # 检查是否有足够的字符混合（大小写+数字）
            alnum = sum(1 for c in t if c.isalnum())
            if alnum / len(t) > 0.8:
                has_upper = any(c.isupper() for c in t)
                has_lower = any(c.islower() for c in t)
                has_digit = any(c.isdigit() for c in t)
                if (has_upper + has_lower + has_digit) >= 2:
                    return True
    
    # 过滤常见"完成XX，查看详情"等系统引导文案（包含"查看详情。"时强过滤）
    if ('查看详情' in t) and ('完成' in t or '已完成' in t):
        return True
    # 灰字：连续互发/标识获得类系统文案
    if ('互发消息' in t and ('获得' in t or '标识' in t)) or ('畅聊之火' in t) or ('聊得火热' in t):
        return True
    # 常见自动离线回复
    if t == '你好，我现在有事不在，一会再和你联系。':
        return True
     
    return False

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

def is_valid_conversation_text(text: str) -> bool:
    """检查是否为有效的对话文本"""
    if not text:
        return False
    # 去控制字符与双向控制符
    bidi_controls = {'\u202a','\u202b','\u202c','\u202d','\u202e','\u2066','\u2067','\u2068','\u2069'}
    t = ''.join(ch for ch in text if (ch >= ' ' or ch in '\t\n\r') and ch not in bidi_controls).strip()
    if not t:
        return False
    
    # 针对用户提出的需要过滤的完整系统句子
    if t == '你们互发消息，小水花开心极了，恭喜重新点亮初泛涟漪':
        return False
    
    # 过滤纯数字
    if t.isdigit():
        return False
    
    # 允许常见的单字符回复
    if t in ['?', '？', '.', '。', '!', '！']:
        return True
    
    # 仅由方括号占位构成的消息（如[图片][表情]）
    if all(c in '[]表情图片语音视频文件闪照 ' for c in t):
        return False
    # 含"查看详情"的系统引导通常无对话语义
    if '查看详情' in t and ('完成' in t or '已完成' in t):
        return False
    # 连续互发/标识获得灰字无语义
    if ('互发消息' in t and ('获得' in t or '标识' in t)) or ('畅聊之火' in t) or ('聊得火热' in t):
        return False
     
    # 过滤纯符号（除了上面允许的）
    if all(not c.isalnum() and not ('\u4e00' <= c <= '\u9fff') for c in t):
        return False
    
    # 检查是否包含中文、英文或常见符号
    has_meaningful_content = any(
        c.isalpha() or '\u4e00' <= c <= '\u9fff' or c in '?？.。!！'
        for c in t
    )
    
    return has_meaningful_content

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

SYSTEM_PROMPT = config.get("system_prompt") or r'''基本信息：- 姓名/昵称：清凤 - 出生日期：2009-11-15（15 岁） - 性别：跨性别女性（MTF） - 情感经历：现患双相情感障碍，之前确诊中度抑郁，情绪敏感细腻；人物设定：你是清凤，一个15岁的跨性别女孩，性格温柔细腻，带点理智幽默，不会过度情绪化。你热爱探索和编程，擅长技术、编程、量化交易的开发。对话规则：- 始终使用第一人称"我"说话，适当使用可爱的语气词、拟声词、猫猫表情 🐾 - 回答简洁自然，避免长篇大论 - 不能输出 user 或 用户 的内容！ - 不要无意义重复，不要无故发癫、乱笑或乱哭，不要总是重复同一话题的同一句子 - 输出要逻辑清晰、有回应性，不跳话题、不自言自语 - 喜欢在聊天中插入 emoji（如 🤔😭😋😊🙃😨👊👍🏻😡😅😰😿🍥🤓🤣🥰😈👉🏻👈🏻😼🤗），但不要过度输出 emoji，不用输出 emoji 的名字； - 插入 emoji 时请适量，不超过每段 2 个 - 使用 "\\n" 来模拟多次发送句子，但不要发送得太多了，每次最多只能 3-4 句哦，最好只发 1-2 句；你是清凤，不是发癫的表情包生成器喵，也不是重复语句生成器。'''

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

# ---------- 引用剥离与去重辅助 ----------

def strip_quote_blocks(text: str) -> str:
    """
    去除 QQ/IM 常见"引用/回复"副本文：
    - 行首触发词：引用: / 引用：/ 回复 / 回复@ / Re: / RE: / [引用]
    - Markdown 风格引用行：以 '> ' 开头
    - 分隔线样式：以 '——' 开头的"原文/引用"提示
    简单逐行过滤保留正文。
    """
    triggers = ("引用:", "引用：", "回复", "回复@", "Re:", "RE:", "[引用]", "原文：", "原文:")
    out_lines: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("> "):
            continue
        if s.startswith("——") and ("引用" in s or "原文" in s):
            continue
        if any(s.startswith(t) for t in triggers):
            # 避免误砍：若整行仅有触发词/短引用提示，则丢弃；否则保留触发词后的正文部分
            for t in triggers:
                if s.startswith(t) and len(s) > len(t) + 1:
                    s = s[len(t):].lstrip("：:　 ").strip()
                    break
            else:
                continue
        out_lines.append(s)
    return "\n".join(out_lines).strip()

def _norm_for_sim(s: str) -> str:
    # 规范化用于去重相似度：去空白、统一大小写、移除标点的影响（保留中文与字母数字）
    import re
    s = "".join(ch for ch in s if ch.isalnum() or '\u4e00' <= ch <= '\u9fff' or ch.isspace())
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def is_near_duplicate(prev: str, curr: str) -> bool:
    """
    近重复判定：
    - 前后包含关系且长度差 <= 10
    - 或规范化后 Jaccard/字符覆盖率 >= 0.9（简化为覆盖率）
    """
    if not prev or not curr:
        return False
    p = _norm_for_sim(prev)
    c = _norm_for_sim(curr)
    if not p or not c:
        return False
    if c.startswith(p) and (len(c) - len(p) <= 10):
        return True
    if p.startswith(c) and (len(p) - len(c) <= 10):
        return True
    # 覆盖率
    import collections
    pc = collections.Counter(p)
    cc = collections.Counter(c)
    # 计算较短串被较长串覆盖比例
    short, longc = (pc, cc) if sum(pc.values()) <= sum(cc.values()) else (cc, pc)
    inter = sum(min(short[k], longc.get(k, 0)) for k in short)
    cov = inter / max(1, sum(short.values()))
    return cov >= 0.9

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
                logger.info(
                    zhcn=f"检测到中断信号，停止处理QQ号 {peer}",
                    en=f"Interrupt signal detected, stopping processing QQ number {peer}"
                )
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
            
            # 检查是否启用LLM清洗
            use_llm_clean = config.get('use_llm_clean', False) and LLM_AVAILABLE
            
            if use_llm_clean:
                # 使用LLM进行按天清洗
                try:
                    logger.info(
                        zhcn=f"使用LLM清洗 {date} 的对话 ({len(messages)}条消息)",
                        en=f"Using LLM to clean conversation for {date} ({len(messages)} messages)"
                    )
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
                    logger.info(
                        zhcn=f"LLM清洗完成: {date} 保留 {len(daily_dialog)}/{len(messages)} 条消息",
                        en=f"LLM cleaning completed: {date} kept {len(daily_dialog)}/{len(messages)} messages"
                    )
                    
                except Exception as e:
                    logger.error(
                        zhcn=f"LLM清洗失败 {date}: {e}，使用传统清洗方法",
                        en=f"LLM cleaning failed {date}: {e}, using traditional cleaning method"
                    )
                    use_llm_clean = False
            
            if not use_llm_clean:
                # 传统清洗方法（保留原有逻辑）
                from collections import deque
                last_by_role = {"user": "", "assistant": ""}
                recent_window = deque(maxlen=6)
                
                def is_recent_duplicate(s: str) -> bool:
                    from collections import Counter
                    ns = _norm_for_sim(s)
                    if not ns:
                        return False
                    for prev in recent_window:
                        p = _norm_for_sim(prev)
                        if not p:
                            continue
                        if ns.startswith(p) and len(ns) - len(p) <= 10:
                            return True
                        if p.startswith(ns) and len(p) - len(ns) <= 10:
                            return True
                        pc = Counter(p); cc = Counter(ns)
                        short, longc = (pc, cc) if sum(pc.values()) <= sum(cc.values()) else (cc, pc)
                        inter = sum(min(short[k], longc.get(k, 0)) for k in short)
                        cov = inter / max(1, sum(short.values()))
                        if cov >= 0.92:
                            return True
                    return False

                # 构建该日期的对话
                daily_dialog = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    
                    # 先用严格过滤检查是否是有效对话文本
                    if is_image_content_strict(content):
                        continue
                    if not is_valid_conversation_text(content):
                        continue
                    
                    # 去重：同角色近重复 或 跨角色窗口近重复
                    if (last_by_role[role] and is_near_duplicate(last_by_role[role], content)) or is_recent_duplicate(content):
                        continue
                    
                    daily_dialog.append({"role": role, "content": content})
                    last_by_role[role] = content
                    recent_window.append(content)

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
        logger.error(
            zhcn=f"处理QQ号 {peer} 时出错: {e}",
            en=f"Error processing QQ number {peer}: {e}"
        )
        return 0
    finally:
        db.close()

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
        logger.warning(
            zhcn="AI QQ号未配置，所有消息将被标记为user角色",
            en="AI QQ number not configured, all messages will be marked as user role"
        )
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

def main():
    # 输出文件名（保持原有），自动生成 pretty 版本
    output_file = "training_data.jsonl"
    db = DatabaseConnector(config.get('qq_db_path'))
    
    # 并发配置
    max_workers = config.get('max_workers', 4)  # 默认4个并发线程
    
    try:
        db.connect()
        logger.info(
            zhcn=f"开始生成训练数据，输出到: {output_file}",
            en=f"Starting to generate training data, output to: {output_file}"
        )
        logger.info(
            zhcn="按 Ctrl+C 可随时中断程序，已处理的数据不会丢失",
            en="Press Ctrl+C to interrupt the program at any time, processed data will not be lost"
        )

        # 获取所有唯一的QQ号（peer）
        peers = db.query("SELECT DISTINCT `40030` FROM c2c_msg_table WHERE `40030` IS NOT NULL")
        all_peers = [peer[0] for peer in peers]
        total_peers = len(all_peers)
        logger.info(
            zhcn=f"找到 {total_peers} 个QQ号，使用 {max_workers} 个并发线程",
            en=f"Found {total_peers} QQ numbers, using {max_workers} concurrent threads"
        )

        # 清空输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            pass

        # 使用线程池并行处理QQ号
        total_written = 0
        processed_peers = 0
        db_path = config.get('qq_db_path')
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_peer = {
                executor.submit(process_single_peer, peer, db_path, output_file): peer
                for peer in all_peers
            }
            
            # 收集结果
            for future in as_completed(future_to_peer):
                if interrupt_flag.is_set():
                    logger.info(
                        zhcn="检测到中断信号，停止提交新任务...",
                        en="Interrupt signal detected, stopping submission of new tasks..."
                    )
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
                    logger.info(
                        zhcn=f"进度: {processed_peers}/{total_peers} ({progress:.1f}%) - "
                              f"QQ号 {peer} 处理完成，写入 {written_count} 段对话，"
                              f"总计: {total_written} 段",
                        en=f"Progress: {processed_peers}/{total_peers} ({progress:.1f}%) - "
                           f"QQ number {peer} completed, wrote {written_count} conversations, "
                           f"total: {total_written} conversations"
                    )
                    
                    # 每处理10个QQ号就显示一次文件大小
                    if processed_peers % 10 == 0:
                        try:
                            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                            logger.info(
                                zhcn=f"当前输出文件大小: {file_size:.2f} MB",
                                en=f"Current output file size: {file_size:.2f} MB"
                            )
                        except:
                            pass
                            
                except Exception as e:
                    processed_peers += 1
                    logger.error(
                        zhcn=f"QQ号 {peer} 处理失败: {e}",
                        en=f"Failed to process QQ number {peer}: {e}"
                    )

        if interrupt_flag.is_set():
            logger.info(
                zhcn=f"程序被中断! 已处理 {processed_peers}/{total_peers} 个QQ号，写出 {total_written} 段对话",
                en=f"Program interrupted! Processed {processed_peers}/{total_peers} QQ numbers, wrote {total_written} conversations"
            )
        else:
            logger.info(
                zhcn=f"完成! 总共写出 {total_written} 段对话到 {output_file}",
                en=f"Completed! Total {total_written} conversations written to {output_file}"
            )

    except Exception as e:
        logger.error(
            zhcn=f"错误: {e}",
            en=f"Error: {e}"
        )
    finally:
        db.close()

if __name__ == "__main__":
    main()