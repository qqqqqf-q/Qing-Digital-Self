#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM数据清洗模块

"""

import os
import json
import logging
import argparse
import re
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.config.config import get_config
from utils.logger.logger import get_logger
from utils.openai.openai_client import OpenAIClient, LLMDataCleaner

# 获取配置和日志
config = get_config()
logger = get_logger('LLM_Data_Cleaner')


@dataclass
class Message:
    """消息数据结构"""
    role: str
    content: str


@dataclass
class QaPair:
    """问答对数据结构"""
    id: int
    messages: List[Message]
    score: Optional[int] = None
    valid_segments: Optional[List[int]] = None
    images: Optional[List[str]] = None


@dataclass
class QaPairScore:
    """LLM打分结果"""
    score: int  # 1-5分
    reason: Optional[str] = None


class LLMScoringStrategy:
    
    def __init__(self, client: Optional[OpenAIClient] = None, accept_score: int = 2, batch_size: int = None, workers: int = None):
        """
        初始化LLM打分策略
        
        Args:
            client: OpenAI客户端实例
            accept_score: 可接受的最低分数阈值
            batch_size: 批处理大小
            workers: 工作线程数
        """
        self.client = client or OpenAIClient()
        self.accept_score = accept_score
        self.model = config.get('OpenAI_Model', 'default')
        self.batch_size = batch_size
        self.workers = workers
        
    def judge(self, qa_pairs: List[QaPair]) -> None:
        """
        对问答对进行LLM打分
        
        Args:
            qa_pairs: 问答对列表，会直接修改其score属性
        """
        logger.info(f"开始LLM打分，共{len(qa_pairs)}个问答对")
        
        # 构建打分提示词
        scoring_prompt = self._build_scoring_prompt()
        
        # 批量处理配置
        clean_set_args = config.get('clean_set_args', {})
        openai_api = clean_set_args.get('openai_api', {})
        batch_size = self.batch_size if self.batch_size is not None else openai_api.get('clean_batch_size', 10)
        workers = self.workers if self.workers is not None else openai_api.get('clean_workers', 4)
        
        logger.info(f"实际使用批处理大小: {batch_size}")
        logger.info(f"实际使用工作线程数: {workers}")
        
        # 分批处理
        batches = []
        for i in range(0, len(qa_pairs), batch_size):
            batches.append(qa_pairs[i:i + batch_size])
        
        # 多线程处理批次
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            future_to_batch = {
                executor.submit(self._score_batch, batch, scoring_prompt): batch 
                for batch in batches
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(batches), desc="LLM打分进度") as pbar:
                for future in as_completed(future_to_batch):
                    try:
                        future.result()  # 获取结果，如果有异常会抛出
                    except Exception as e:
                        batch = future_to_batch[future]
                        logger.error(f"批次处理失败: {e}")
                        # 给失败的批次默认分数
                        for qa in batch:
                            if qa.score is None:
                                qa.score = 0
                    finally:
                        pbar.update(1)
    
    def _build_scoring_prompt(self) -> str:
        """构建LLM打分提示词"""
        return """# 角色
你是一个数据质量评估员。

# 任务
你的任务是评估下面提供的聊天记录的**逻辑性**、**相关性**以及**风格代表性**。目标是识别并过滤掉那些回答与问题**明显不匹配**、**逻辑严重混乱**的样本，筛选出具有人类聊天风格独特性与辨识度的样本。

# 评分标准
请根据以下标准给出1到5的整数评分：

**5分（优秀）**：
- 问答逻辑清晰，回答与问题高度相关
- 体现出独特的个人风格和语言习惯
- 对话自然流畅，具有很好的代表性

**4分（良好）**：
- 问答基本合理，逻辑较为清晰
- 有一定的个人风格特征
- 对话较为自然

**3分（一般）**：
- 问答基本相关，但可能存在轻微逻辑问题
- 风格特征不够明显
- 对话尚可接受

**2分（较差）**：
- 问答相关性较弱，存在明显逻辑问题
- 缺乏个人风格特征
- 对话质量较低

**1分（很差）**：
- 问答完全不匹配或逻辑严重混乱
- 无个人风格可言
- 对话质量极差，不适合训练

**重要考量:**
1. **简短回答的有效性:** 诸如"好的"、"是的"、"收到"、"嗯"、"知道了"等简短的肯定、确认或应答，在合适的语境下是完全**有逻辑且相关的**。**不要仅仅因为回答简短就将其评为低分。**
2. **处理错别字和自我纠正:** 聊天记录中可能包含常见的打字错误（错别字）或用户先打错字随后又自行纠正的情况。在评估时，请**聚焦于用户想要表达的最终意图**，而不是纠结于表面的错误。

# 输出要求
请严格按照以下JSON格式输出，包含输入数据的id和你给出的1到5的整数评分score，不要包含任何其他文字、解释或标签。
{{"id": "{id}","score": <这里填入1到5的整数评分>}}"""
    
    def _score_batch(self, batch: List[QaPair], scoring_prompt: str) -> None:
        """
        批量打分
        
        Args:
            batch: 问答对批次
            scoring_prompt: 打分提示词
        """
        # 构建批量请求
        qa_list = []
        for qa in batch:
            if qa.images:  # 包含图片的直接给高分
                qa.score = 6
                continue
                
            messages_str = ""
            for msg in qa.messages:
                if msg.role == "user":
                    messages_str += f"Q: {msg.content}\n"
                elif msg.role == "assistant":
                    messages_str += f"A: {msg.content}\n"
            
            qa_list.append({
                "id": qa.id,
                "Q": next((msg.content for msg in qa.messages if msg.role == "user"), ""),
                "A": next((msg.content for msg in qa.messages if msg.role == "assistant"), "")
            })
        
        if not qa_list:
            return
            
        qa_list_json = json.dumps(qa_list, ensure_ascii=False)
        prompt_text = f"{scoring_prompt}\n\n请评估以下问答对：\n{qa_list_json}"
        
        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": scoring_prompt},
                    {"role": "user", "content": f"请评估以下问答对：\n{qa_list_json}"}
                ],
                model=self.model,
                temperature=0,
                max_tokens=2000
            )
            
            # 解析响应
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            self._parse_scores(batch, content)
            
        except Exception as e:
            logger.error(f"LLM打分失败: {e}")
            # 失败时给默认分数
            for qa in batch:
                if qa.score is None:
                    qa.score = 0
    
    def _parse_scores(self, batch: List[QaPair], content: str) -> None:
        """
        解析LLM返回的分数
        
        Args:
            batch: 问答对批次
            content: LLM返回的内容
        """
        try:
            # 清理内容，移除多余的空白字符
            content = content.strip()
            
            # 尝试解析JSON数组格式
            if content.startswith('['):
                scores = json.loads(content)
                for score_data in scores:
                    qa_id = score_data.get('id')
                    score = score_data.get('score', 0)
                    for qa in batch:
                        if qa.id == qa_id:
                            qa.score = score
                            break
            else:
                # 处理多个JSON对象连在一起的情况
                # 按行分割并尝试解析每一行
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                
                for line in lines:
                    # 跳过非JSON行
                    if not (line.startswith('{') and line.endswith('}')):
                        continue
                        
                    try:
                        score_data = json.loads(line)
                        qa_id = score_data.get('id')
                        score = score_data.get('score', 0)
                        for qa in batch:
                            if qa.id == qa_id:
                                qa.score = score
                                break
                    except json.JSONDecodeError:
                        # 单行解析失败，继续处理下一行
                        continue
                        
        except json.JSONDecodeError:
            logger.warning(f"无法解析LLM响应: \n {content}")
            # 解析失败时给默认分数
            for qa in batch:
                if qa.score is None:
                    qa.score = 0
    
    def filter_by_score(self, qa_pairs: List[QaPair]) -> List[QaPair]:
        """
        根据分数过滤问答对
        
        Args:
            qa_pairs: 问答对列表
            
        Returns:
            过滤后的问答对列表
        """
        filtered = [qa for qa in qa_pairs if qa.score and qa.score >= self.accept_score]
        
        logger.info(f"LLM打分过滤完成: {len(filtered)}/{len(qa_pairs)} 个问答对通过筛选")
        
        # 统计分数分布
        scores = [qa.score for qa in qa_pairs if qa.score is not None]
        if scores:
            score_series = pd.Series(scores)
            score_counts = score_series.value_counts().sort_index()
            logger.info(f"分数分布: {dict(score_counts)}")
        
        return filtered


class SegmentStrategy:
    """generate_training_data风格的可用句段策略"""
    
    def __init__(self, client: Optional[OpenAIClient] = None):
        """
        初始化句段策略
        
        Args:
            client: OpenAI客户端实例
        """
        self.client = client or OpenAIClient()
        self.llm_cleaner = LLMDataCleaner(self.client)
        self.model = config.get('OpenAI_Model', 'default')
    
    def process_conversation(self, messages: List[Dict[str, Any]], date: Optional[str] = None) -> List[Dict[str, str]]:
        """
        处理对话，返回可用句段
        
        Args:
            messages: 消息列表
            date: 日期标识
            
        Returns:
            处理后的消息列表
        """
        try:
            # 使用现有的LLM清洗功能
            cleaned_messages = self.llm_cleaner.clean_daily_conversation(messages, date)
            
            logger.info(f"句段策略处理完成: 保留 {len(cleaned_messages)}/{len(messages)} 条消息")
            
            return cleaned_messages
            
        except Exception as e:
            logger.error(f"句段策略处理失败: {e}")
            # 失败时返回原始消息（去除无效内容）
            return self._fallback_process(messages)
    
    def _fallback_process(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        备用处理方法
        
        Args:
            messages: 消息列表
            
        Returns:
            处理后的消息列表
        """
        from generate_training_data import is_valid_conversation_text, is_image_content_strict
        
        valid_messages = []
        for msg in messages:
            content = msg.get('content', '').strip()
            if content and not is_image_content_strict(content) and is_valid_conversation_text(content):
                valid_messages.append({
                    'role': msg.get('role', 'user'),
                    'content': content
                })
        
        return valid_messages


class LLMDataProcessor:
    """LLM数据处理器主类"""
    
    def __init__(self, parser: str = 'scoring', **kwargs):
        """
        初始化数据处理器
        
        Args:
            parser: 处理策略类型 ('scoring' 或 'segment')
            **kwargs: 其他参数
        """
        self.parser = parser
        self.client = OpenAIClient()
        
        # 从配置中读取system prompt
        self.system_prompt = config.get('system_prompt', '')
        
        if parser == 'scoring':
            accept_score = kwargs.get('accept_score', config.get('accept_score', 2))
            batch_size = kwargs.get('batch_size', None)
            workers = kwargs.get('workers', None)
            self.strategy = LLMScoringStrategy(self.client, accept_score, batch_size, workers)
        elif parser == 'segment':
            self.strategy = SegmentStrategy(self.client)
        else:
            raise ValueError(f"不支持的处理策略: {parser}")
    
    def process_file(self, input_path: str, output_path: str, **kwargs) -> int:
        """
        处理文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            **kwargs: 其他参数
            
        Returns:
            处理结果状态码
        """
        try:
            logger.info(f"开始处理文件: {input_path} -> {output_path}")
            logger.info(f"使用策略: {self.parser}")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if self.parser == 'scoring':
                return self._process_with_scoring(input_path, output_path, **kwargs)
            elif self.parser == 'segment':
                return self._process_with_segment(input_path, output_path, **kwargs)
            
        except Exception as e:
            logger.error(f"文件处理失败: {e}")
            return 1
    
    def _process_with_scoring(self, input_path: str, output_path: str, **kwargs) -> int:
        """
        使用打分策略处理
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            **kwargs: 其他参数
            
        Returns:
            处理结果状态码
        """
        try:
            # 读取输入数据
            qa_pairs = self._load_qa_pairs(input_path)
            
            if not qa_pairs:
                logger.warning("没有找到有效的问答对")
                return 1
            
            # LLM打分
            self.strategy.judge(qa_pairs)
            
            # 根据分数过滤
            filtered_pairs = self.strategy.filter_by_score(qa_pairs)
            
            # 保存结果
            self._save_qa_pairs(filtered_pairs, output_path)
            
            logger.info(f"打分策略处理完成: {output_path}")
            return 0
            
        except Exception as e:
            logger.error(f"打分策略处理失败: {e}")
            return 1
    
    def _process_with_segment(self, input_path: str, output_path: str, **kwargs) -> int:
        """
        使用句段策略处理
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            **kwargs: 其他参数
            
        Returns:
            处理结果状态码
        """
        try:
            # 读取输入数据
            conversations = self._load_conversations(input_path)
            
            if not conversations:
                logger.warning("没有找到有效的对话数据")
                return 1
            
            # 处理对话
            processed_conversations = []
            for i, conv in enumerate(tqdm(conversations, desc="处理对话")):
                try:
                    processed = self.strategy.process_conversation(conv.get('messages', []), f"conversation_{i}")
                    if processed:
                        processed_conversations.append({
                            'messages': processed
                        })
                except Exception as e:
                    logger.warning(f"处理对话 {i} 失败: {e}")
                    continue
            
            # 保存结果
            self._save_conversations(processed_conversations, output_path)
            
            logger.info(f"句段策略处理完成: {output_path}")
            return 0
            
        except Exception as e:
            logger.error(f"句段策略处理失败: {e}")
            return 1
    
    def _load_qa_pairs(self, input_path: str) -> List[QaPair]:
        """
        加载问答对数据
        
        Args:
            input_path: 输入文件路径或目录路径
            
        Returns:
            问答对列表
        """
        qa_pairs = []
        
        try:
            if os.path.isfile(input_path):
                # 单个文件处理
                qa_pairs.extend(self._load_qa_pairs_from_file(input_path))
            elif os.path.isdir(input_path):
                # 目录处理，遍历所有CSV文件
                csv_files = []
                for root, dirs, files in os.walk(input_path):
                    for file in files:
                        if file.endswith('.csv'):
                            csv_files.append(os.path.join(root, file))
                
                logger.info(f"找到 {len(csv_files)} 个CSV文件")
                
                for csv_file in tqdm(csv_files, desc="加载CSV文件"):
                    try:
                        pairs = self._load_qa_pairs_from_csv(csv_file)
                        qa_pairs.extend(pairs)
                    except Exception as e:
                        logger.warning(f"加载CSV文件失败 {csv_file}: {e}")
                        continue
            else:
                logger.error(f"输入路径不存在: {input_path}")
                return []
        
        except Exception as e:
            logger.error(f"加载问答对失败: {e}")
            return []
        
        logger.info(f"总共加载了 {len(qa_pairs)} 个问答对")
        return qa_pairs
    
    def _load_qa_pairs_from_file(self, file_path: str) -> List[QaPair]:
        """
        从单个JSONL文件加载问答对
        
        Args:
            file_path: 文件路径
            
        Returns:
            问答对列表
        """
        qa_pairs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        messages = []
                        
                        # 解析消息格式
                        if 'messages' in data:
                            for msg in data['messages']:
                                messages.append(Message(
                                    role=msg.get('role', 'user'),
                                    content=msg.get('content', '')
                                ))
                        elif 'conversations' in data:
                            # 兼容其他格式
                            for msg in data['conversations']:
                                messages.append(Message(
                                    role=msg.get('from', 'user'),
                                    content=msg.get('value', '')
                                ))
                        
                        if messages:
                            qa_pairs.append(QaPair(
                                id=f"{os.path.basename(file_path)}_{line_no}",
                                messages=messages,
                                images=data.get('images', [])
                            ))
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过无效JSON行 {line_no}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            return []
        
        return qa_pairs
    
    def _load_qa_pairs_from_csv(self, csv_file: str) -> List[QaPair]:
        """
        从CSV文件加载问答对
        
        Args:
            csv_file: CSV文件路径
            
        Returns:
            问答对列表
        """
        qa_pairs = []
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # 按时间分组对话
            conversations = []
            current_conversation = []
            
            for _, row in df.iterrows():
                # 检查必要字段
                if pd.isna(row.get('msg', '')) or pd.isna(row.get('is_sender', '')):
                    continue
                
                content = str(row['msg']).strip()
                is_sender = row.get('is_sender', 0)
                
                if not content:
                    continue
                
                # 判断角色：is_sender=1表示发送者（助手），is_sender=0表示接收者（用户）
                role = 'assistant' if is_sender == 1 else 'user'
                
                current_conversation.append(Message(role=role, content=content))
                
                # 简单的对话分割逻辑：当遇到用户消息且当前对话不为空时，结束当前对话
                if role == 'user' and len(current_conversation) > 1:
                    if current_conversation[-2].role == 'assistant':
                        # 保存当前对话（去掉最后一条用户消息）
                        if len(current_conversation) > 1:
                            conversations.append(current_conversation[:-1])
                        # 开始新对话
                        current_conversation = [current_conversation[-1]]
            
            # 添加最后一个对话
            if current_conversation:
                conversations.append(current_conversation)
            
            # 转换为QaPair格式
            for i, conv in enumerate(conversations):
                if len(conv) >= 2:  # 至少要有一问一答
                    qa_pairs.append(QaPair(
                        id=f"{os.path.basename(csv_file)}_{i}",
                        messages=conv
                    ))
        
        except Exception as e:
            logger.error(f"处理CSV文件失败 {csv_file}: {e}")
            return []
        
        return qa_pairs
    
    def _load_conversations(self, input_path: str) -> List[Dict[str, Any]]:
        """
        加载对话数据
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            对话列表
        """
        conversations = []
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if 'messages' in data:
                            conversations.append(data)
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过无效JSON行 {line_no}: {e}")
                        continue
        
        except FileNotFoundError:
            logger.error(f"输入文件不存在: {input_path}")
            return []
        except Exception as e:
            logger.error(f"加载对话失败: {e}")
            return []
        
        logger.info(f"加载了 {len(conversations)} 个对话")
        return conversations
    
    def _save_qa_pairs(self, qa_pairs: List[QaPair], output_path: str) -> None:
        """
        保存问答对数据

        Args:
            qa_pairs: 问答对列表
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for qa in qa_pairs:
                    # 构建完整的messages列表，包含system prompt
                    messages = []
                    
                    # 添加系统提示（如果有且不为空）
                    system_prompt = getattr(self, 'system_prompt', None)
                    if system_prompt and system_prompt.strip() and system_prompt != "*":
                        messages.append({"role": "system", "content": system_prompt})
                    
                    # 合并连续的user消息并处理空格符
                    processed_messages = self._process_messages(qa.messages)
                    
                    # 添加处理后的对话消息
                    messages.extend([
                        {'role': msg.role, 'content': msg.content}
                        for msg in processed_messages
                    ])
                    
                    data = {
                        'messages': messages
                    }
                    if qa.images:
                        data['images'] = qa.images
                    # 移除score字段，不保存到输出
                    
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            logger.info(f"保存了 {len(qa_pairs)} 个问答对到 {output_path}")
            
        except Exception as e:
            logger.error(f"保存问答对失败: {e}")
            raise
    
    def _process_messages(self, messages: List[Message]) -> List[Message]:
        """
        处理消息：合并连续的相同角色消息并替换空格符为\n
        
        Args:
            messages: 原始消息列表
            
        Returns:
            处理后的消息列表
        """
        if not messages:
            return []
        
        processed_messages = []
        current_role_content = []
        current_role = None
        
        for msg in messages:
            cleaned_content = self._replace_spaces_with_newlines(msg.content)
            
            if msg.role == current_role:
                # 收集连续的相同角色消息内容
                current_role_content.append(cleaned_content)
            else:
                # 处理之前收集的相同角色消息
                if current_role_content and current_role:
                    combined_content = "\n".join(current_role_content)
                    processed_messages.append(Message(role=current_role, content=combined_content))
                
                # 开始收集新角色的消息
                current_role = msg.role
                current_role_content = [cleaned_content]
        
        # 处理最后剩余的相同角色消息
        if current_role_content and current_role:
            combined_content = "\n".join(current_role_content)
            processed_messages.append(Message(role=current_role, content=combined_content))
        
        return processed_messages
    
    def _replace_spaces_with_newlines(self, content: str) -> str:
        """
        将句子中的空格符替换为\n
        
        Args:
            content: 原始内容
            
        Returns:
            处理后的内容
        """
        if not content:
            return content
        
        # 将连续的空格替换为单个换行符
        # 保留标点符号后的空格，避免破坏句子结构
        content = re.sub(r'[。！？；，、：] +', lambda m: m.group(0)[0] + '\n', content)
        # 处理普通空格
        content = re.sub(r' +', '\n', content)
        
        return content.strip()
    
    def _save_conversations(self, conversations: List[Dict[str, Any]], output_path: str) -> None:
        """
        保存对话数据

        Args:
            conversations: 对话列表
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for conv in conversations:
                    # 处理对话中的消息
                    if 'messages' in conv:
                        messages = [Message(role=msg.get('role', 'user'), content=msg.get('content', ''))
                                   for msg in conv['messages']]
                        processed_messages = self._process_messages(messages)
                        
                        # 转换为字典格式
                        conv_data = {
                            'messages': [
                                {'role': msg.role, 'content': msg.content}
                                for msg in processed_messages
                            ]
                        }
                        f.write(json.dumps(conv_data, ensure_ascii=False) + '\n')
                    else:
                        f.write(json.dumps(conv, ensure_ascii=False) + '\n')
            
            logger.info(f"保存了 {len(conversations)} 个对话到 {output_path}")
            
        except Exception as e:
            logger.error(f"保存对话失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLM数据清洗工具')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    parser.add_argument('--parser', '-p', choices=['scoring', 'segment'], default='scoring',
                       help='处理策略: scoring(打分策略) 或 segment(句段策略)')
    parser.add_argument('--accept-score', type=int, default=2,
                       help='可接受的最低分数阈值(仅用于scoring策略)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='批处理大小')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = LLMDataProcessor(
        parser=args.parser,
        accept_score=args.accept_score,
        batch_size=args.batch_size
    )
    
    # 处理文件
    result = processor.process_file(args.input, args.output)
    
    if result == 0:
        logger.info("处理完成")
    else:
        logger.error("处理失败")
    
    return result


if __name__ == '__main__':
    sys.exit(main())