#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple, Dict
from database.db_connector import DatabaseConnector
import datetime
from collections import defaultdict
from config.config import get_config
from logger.logger import get_logger

# 导入LM Studio支持（可选）
try:
    from openai.openai_client import llm_cleaner
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    llm_cleaner = None
# 获取配置实例
config = get_config()
logger = get_logger('Generate_training_data')

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
    '.mp4', '.amr',  # 媒体扩展
    'offpic_new', 'thumb', 'ori', 'pic\\',
    '[动画表情]', '[表情]', '[图片]', '[闪照]',
    '[语音]', '[视频]', '[文件]',  # 常见IM占位
    'ntosfull', 'documents\\tencent files',
    'term=', 'is_origin=', '_198.jpg', '_720.jpg',
    '/1684773595-', '/3259379048-',
    '-05d65b12ef0481a7337c63775bac3ffd',
    'u_sp2m7izf1zlnjvtcwqx0sg', 'u_ovchmti-zvlwu_0zm_ulgw',
    'multimedia.nt.qq.com.cn', 'qq.com',
    'mqqapi://', 'https://', 'http://',
    'undefined', 'undefine',
    '请使用新版手机qq查看', '连续互发消息中断', '恋爱之钥',
    '点击恢复', 'er0s', 'show_pslcard',
    'interactive_new', 'club.vip.qq.com',
    'downv6.qq.com', 'qzone_love',
    # 新增需过滤的系统提示/灰字
    '你们互发消息，小水花开心极了，恭喜重新点亮初泛涟漪',
    '语音通话',  # 与时长模式配合过滤
    # 也过滤你提到的可疑短串前缀片段（非唯一，仅用于命中）
    'd42ae9486fdbc4b7',
    # 添加对类似 u_2_Tv579rOiOIDBW9sUrPBA 这种格式的过滤
    'u_',
    # IM链接参数常见噪声
    '&rkey=',
    '情侣空间','情侣','神仙眷侣','知己浪花','累计互发消息超过'
}

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
    """判断是否为图片/系统/非文本相关内容（需过滤）"""
    if not text:
        return False
    # 去除控制字符（含 \u0000 与 LRM/嵌入方向控制等），避免"Z\u0000b\u0000"、"U+202D"等伪文本
    # Unicode 双向控制字符集合：U+202A..U+202E, U+2066..U+2069 等
    bidi_controls = {'\u202a','\u202b','\u202c','\u202d','\u202e','\u2066','\u2067','\u2068','\u2069'}
    t = ''.join(ch for ch in text if (ch >= ' ' or ch in '\t\n\r') and ch not in bidi_controls).strip()
    if not t:
        return True  # 清理后为空，视为噪声
    
    # 严格等值短占位
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
    """从protobuf数据中提取纯文本内容（即时选最大，避免收集全部）"""
    try:
        fields = parse_protobuf_fields(data)
        strings = find_strings_recursively(fields)
        max_text = ""
        max_len = 0
        for text in strings:
            if not text:
                continue
            # 先做强清洗（控制字符+双向控制符）
            bidi_controls = {'\u202a','\u202b','\u202c','\u202d','\u202e','\u2066','\u2067','\u2068','\u2069'}
            cleaned = ''.join(ch for ch in text if (ch >= ' ' or ch in '\t\n\r') and ch not in bidi_controls).strip()
            if not cleaned:
                continue
            if is_image_content(cleaned):
                continue
            if not is_valid_conversation_text(cleaned):
                continue
            l = len(cleaned)
            if l > max_len:
                max_len = l
                max_text = cleaned
        return max_text
    except Exception:
        return ""

# ---------- 会话整形（聚合为 messages 格式） ----------

SYSTEM_PROMPT = config.get("system_prompt")

def flush_dialog(f, dialog: List[dict], pretty_f=None):
    """将当前聚合的对话写为一条 jsonl，按角色合并内容用 \\n 连接；可选同步写入可读版
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
    # 合并连续相同 role 的消息，只保留 user/assistant，system 统一注入
    merged: List[dict] = []
    for msg in dialog:
        r = msg.get("role")
        c = msg.get("content", "")
        if r not in ("user", "assistant"):
            continue
        if merged and merged[-1]["role"] == r:
            merged[-1]["content"] += "\\n" + c
        else:
            merged.append({"role": r, "content": c})

    # 若对话为空，则不输出
    if not merged:
        return

    # 处理"只有单 user 或单 assistant"的场景
    roles = {m["role"] for m in merged}
    if roles == {"user"}:
        merged.append({
            "role": "assistant",
            "content": ""
        })
    elif roles == {"assistant"}:
        merged.insert(0, {
            "role": "user",
            "content": ""
        })

    # 再次确保最后一条为 assistant
    if merged[-1]["role"] != "assistant":
        merged.append({
            "role": "assistant",
            "content": ""
        })

    payload = {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + merged}

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

# ---------- 行处理与顺序扫描（加入去重） ----------

def process_row(row: Tuple[int, int, int, bytes]) -> Optional[dict]:
    """
    处理一行数据，提取 role、content 及对端 peer（40030）；无效返回 None。
    """
    timestamp, sender_30, sender_33, blob_data = row
    if not blob_data:
        return None
    text_content = extract_text_content(blob_data)
    if not text_content:
        return None
    if sender_33 == config.get("qq_number_ai"):  # AI
        role = "assistant"
    elif sender_33 == sender_30:  # 用户/好友
        role = "user"
    else:
        return None
    return {"timestamp": timestamp, "peer": sender_30, "role": role, "content": text_content}

def main():
    # 输出文件名（保持原有），自动生成 pretty 版本
    output_file = "training_data.jsonl"
    db = DatabaseConnector(config.get('db_path'))

    try:
        db.connect()
        logger.info(
            zhcn=f"开始生成训练数据，输出到: {output_file}",
            en=f"Starting to generate training data, output to: {output_file}"
        )

        # 获取所有唯一的QQ号（peer）
        peers = db.query("SELECT DISTINCT `40030` FROM c2c_msg_table WHERE `40030` IS NOT NULL")
        all_peers = [peer[0] for peer in peers]
        logger.info(
            zhcn=f"找到 {len(all_peers)} 个QQ号",
            en=f"Found {len(all_peers)} QQ numbers"
        )

        with open(output_file, 'w', encoding='utf-8', buffering=1024 * 1024) as f:
            
            written_dialogs = 0
            processed = 0

            # 对每个QQ号进行处理
            for peer in all_peers:
                logger.info(
                    zhcn=f"处理QQ号: {peer}",
                    en=f"Processing QQ number: {peer}"
                )
                
                # 获取该QQ号的所有消息
                rows = db.query("""
                    SELECT `40050`, `40030`, `40033`, `40800`
                    FROM c2c_msg_table
                    WHERE `40030` = ? AND `40800` IS NOT NULL
                    ORDER BY `40050` ASC
                """, (peer,))
                
                if not rows:
                    continue

                # 按日期分组的消息
                daily_messages = defaultdict(list)
                
                # 处理该QQ号的所有消息
                for (timestamp, sender_30, sender_33, blob_data) in rows:
                    processed += 1
                    
                    res = process_row((timestamp, sender_30, sender_33, blob_data))
                    if res is None:
                        continue
                    
                    # 计算UTC+12时区的日期
                    utc_time = datetime.datetime.fromtimestamp(timestamp, datetime.UTC)
                    utc_plus_12_time = utc_time + datetime.timedelta(hours=12)
                    message_date = utc_plus_12_time.strftime('%Y-%m-%d')
                    
                    daily_messages[message_date].append(res)

                # 对每个日期生成一个对话
                for date, messages in daily_messages.items():
                    if not messages or len(messages) <= 1:
                        continue
                    
                    # 检查是否启用LLM清洗
                    use_llm_clean = config.get('use_llm_clean', False) and LLM_AVAILABLE
                    
                    if use_llm_clean:
                        # 使用LLM进行按天清洗
                        try:
                            logger.info(
                                zhcn=f"使用LLM清洗 {date} 的对话 ({len(messages)}条消息)",
                                en=f"Using LLM to clean {date} conversation ({len(messages)} messages)"
                            )
                            llm_messages = [{"role": msg["role"], "content": msg["content"], "timestamp": msg["timestamp"]}
                                          for msg in messages]
                            cleaned_messages = llm_cleaner.clean_daily_conversation(llm_messages, date)
                            
                            # 只保留必要字段
                            daily_dialog = [{"role": msg["role"], "content": msg["content"]}
                                          for msg in cleaned_messages]
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
                            
                            # 去重：同角色近重复 或 跨角色窗口近重复
                            if (last_by_role[role] and is_near_duplicate(last_by_role[role], content)) or is_recent_duplicate(content):
                                continue
                            
                            daily_dialog.append({"role": role, "content": content})
                            last_by_role[role] = content
                            recent_window.append(content)

                    # 写入该日期的对话
                    if daily_dialog:
                        flush_dialog(f, daily_dialog)
                        written_dialogs += 1

                if processed % 5000 == 0:
                    logger.info(
                        zhcn=f"已处理消息: {processed}，已写入对话段: {written_dialogs}",
                        en=f"Processed messages: {processed}, written dialogs: {written_dialogs}"
                    )

        logger.info(
            zhcn=f"完成! 处理 {processed} 条消息，写出 {written_dialogs} 段对话到 {output_file}",
            en=f"Completed! Processed {processed} messages, wrote {written_dialogs} dialogs to {output_file}"
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