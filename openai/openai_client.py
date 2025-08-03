#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time
from typing import Dict, List, Optional, Any
from config.config import get_config
from logger.logger import get_logger

logger = get_logger('OpenAI_Client')

class OpenAIClient:
    """OpenAI API客户端，用于与本地LM Studio服务器交互"""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        初始化LM Studio客户端
        
        Args:
            base_url: LM Studio服务器地址，如果为None则从配置中读取
        """
        self.config = get_config()
        self.base_url = base_url or self.config.get('OpenAI_URL', 'http://localhost:1234')
        self.timeout = self.config.get('OpenAI_timeout', 30)
        self.max_retries = self.config.get('OpenAI_max_retries', 3)
        self.retry_delay = self.config.get('OpenAI_retry_delay', 1.0)
        
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """发送HTTP请求到LM Studio服务器"""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    json=data,
                    headers={'Content-Type': 'application/json'},
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                else:
                    logger.error(f"所有重试都失败: {e}")
                    raise
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       stream: bool = False) -> Dict[str, Any]:
        """
        发送聊天补全请求
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            model: 模型名称，如果为None则使用默认模型
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否流式返回
            
        Returns:
            LLM响应结果
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if model:
            payload["model"] = model
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        return self._make_request("v1/chat/completions", payload)
    
    def list_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []
    
    def health_check(self) -> bool:
        """检查LM Studio服务器是否健康"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


class LLMDataCleaner:
    """基于LLM的数据清洗器"""
    
    def __init__(self, client: Optional[OpenAIClient] = None):
        """
        初始化数据清洗器
        
        Args:
            client: LM Studio客户端实例，如果为None则创建新实例
        """
        self.client = client or OpenAIClient()
        self.config = get_config()
        self.model = self.config.get('OpenAI_Model', 'default')
        
    def clean_text(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        使用LLM清洗单个文本
        
        Args:
            text: 需要清洗的文本
            context: 可选的上下文信息
            
        Returns:
            包含清洗结果的字典
        """
        system_prompt = """你是一个数据清洗专家。请分析给定的文本，判断其是否为有效的对话内容，并提供清洗建议。
/no_think
请按照以下格式回复：
{
    "is_valid": true/false,
    "cleaned_text": "清洗后的文本",
    "reason": "判断原因",
    "confidence": 0.0-1.0
}

判断标准：
- 是否为有意义的对话内容
- 是否包含垃圾信息或系统消息
- 文本是否过于简短或无意义
- 是否有明显的格式问题"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请分析以下文本：\n\n{text}\n\n /no_think 上下文：{context or '无'} "}
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.1,
                max_tokens=500
            )
            
            # 验证响应结构
            choices = response.get("choices", [])
            if not choices:
                logger.error("LLM响应中没有choices字段")
                return {"is_valid": False, "cleaned_text": text, "reason": "LLM响应中没有choices", "confidence": 0.0}
            
            first_choice = choices[0]
            message = first_choice.get("message")
            if not message:
                logger.error("LLM响应的choice中没有message字段")
                return {"is_valid": False, "cleaned_text": text, "reason": "LLM响应的choice中没有message", "confidence": 0.0}
            
            content = message.get("content", "")
            if not content:
                logger.error("LLM响应的message中没有content字段")
                return {"is_valid": False, "cleaned_text": text, "reason": "LLM响应的message中没有content", "confidence": 0.0}

            # 尝试解析JSON格式的响应
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"LLM响应JSON解析失败: {e}，内容：{content}")
                return {
                    "is_valid": False,
                    "cleaned_text": text,
                    "reason": f"LLM响应格式异常，无法解析JSON: {str(e)}",
                    "confidence": 0.3
                }

            # 验证必要字段
            required_fields = ["is_valid", "cleaned_text", "reason", "confidence"]
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                logger.warning(f"LLM响应缺少必要字段: {missing_fields}")
                # 设置缺失字段的默认值
                for field in missing_fields:
                    if field == "is_valid":
                        result["is_valid"] = False
                    elif field == "cleaned_text":
                        result["cleaned_text"] = text
                    elif field == "reason":
                        result["reason"] = f"缺少必要字段: {missing_fields}"
                    elif field == "confidence":
                        result["confidence"] = 0.5
                result["reason"] += "（已设置默认值）"

            return result
                
        except Exception as e:
            logger.error(f"LLM清洗失败: {e}")
            return {
                "is_valid": True,
                "cleaned_text": text,
                "reason": f"LLM调用失败: {str(e)}",
                "confidence": 0.0
            }
    
    def batch_clean(self, texts: List[str], contexts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        批量清洗文本
        
        Args:
            texts: 文本列表
            contexts: 上下文列表，与texts一一对应
            
        Returns:
            清洗结果列表
        """
        if contexts is None:
            contexts = [None] * len(texts)
            
        results = []
        for text, context in zip(texts, contexts):
            result = self.clean_text(text, context)
            results.append(result)
            
        return results
    
    def clean_daily_conversation(self, messages: List[Dict[str, Any]], date: Optional[str] = None) -> List[Dict[str, str]]:
        """
        按天清洗一整天的对话
        
        Args:
            messages: 当天的对话消息列表，格式为 [{"role": "user/assistant", "content": "...", "timestamp": timestamp, "is_media": bool}]
            date: 日期字符串，用于标识
            
        Returns:
            清洗后的对话消息列表，保持原始语气和顺序
        """
        if not messages:
            return []
        
        # 构建完整的对话文本
        conversation_text = ""
        messages_by_index = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if content:
                role_name = "用户" if role == "user" else "AI"
                conversation_text += f"[{i}] {role_name}: {content}\n"
                messages_by_index.append({
                    "index": i,
                    "role": role,
                    "content": content,
                    "original": msg
                })
        
        if not conversation_text.strip():
            return []
        
        system_prompt = f"""你是一个专业的数据清洗专家，需要对一整天的对话记录进行过滤和整理。请严格按照以下步骤执行：

请严格按照以下要求：
1. 删除规则  
   a. **系统提示**：所有以系统或平台自动生成的提示性文字（如"开始通话""加载中""自动补齐"等）  
   b. **垃圾消息**：无意义字符、乱码、广告或重复发送的同一内容  
   c. **格式错误**：残缺不全的 JSON、乱码截断的文字  
   d. **语音/通话记录**：如"语音通话 00:30"之类的通话时长记录  
   e. **社交水印**：包含"情侣空间"、"神仙眷侣"、"知己浪花"、"累计互发消息超过"、"小水花开心极了"、"摸摸头"、"获得"、"互动标识"等表达社交关系或互动量的句子  
   f. **图片/媒体消息**：包含URL链接、文件路径、图片标签等明显非对话内容
2. 保留规则  
   - 完全保留所有有意义的用户和 AI 间的对话，无论长短；  
   - 原文不改动，保留所有 Emoji 及`[图片]`等占位；  
   - 严格保持原始对话顺序。
3. 返回需要保留的消息索引列表
4.保留所有的emoji,或者以中括号[]包裹的表情包和内容,保留"[图片]"
请回复一个JSON格式:
{{
    "retained_indices": [0, 1, 3, 5, ...],
    "removed_count": 5,
    "reason": "删除了系统提示和垃圾消息"
}}"""

        messages_llm = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{date}的对话记录如下：{conversation_text}请严格遵循system_prompt,分析输入数据,并返回需要保留的消息索引/no_think"}
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages_llm,
                model=self.model,
                temperature=0.1,
                max_tokens=1000
            )
            
            # 验证响应结构
            choices = response.get("choices", [])
            if not choices:
                logger.error("LLM响应中没有choices字段")
                return [{"role": msg.get("role"), "content": msg.get("content"), "timestamp": msg.get("timestamp")}
                       for msg in messages]
            
            first_choice = choices[0]
            message = first_choice.get("message")
            if not message:
                logger.error("LLM响应的choice中没有message字段")
                return [{"role": msg.get("role"), "content": msg.get("content"), "timestamp": msg.get("timestamp")}
                       for msg in messages]
            
            content = message.get("content", "")
            if not content:
                logger.error("LLM响应的message中没有content字段")
                return [{"role": msg.get("role"), "content": msg.get("content"), "timestamp": msg.get("timestamp")}
                       for msg in messages]
            
            # 移除<think>标签内容（如果存在）
            if "<think>" in content and "</think>" in content:
                start = content.find("<think>")
                end = content.find("</think>") + 8
                content = content[:start] + content[end:]
                content = content.strip()
            
            # 尝试解析JSON格式的响应
            try:
                result = json.loads(content)
                
                # 验证必要字段
                if "retained_indices" not in result:
                    logger.error("LLM响应缺少retained_indices字段")
                    return [{"role": msg.get("role"), "content": msg.get("content"), "timestamp": msg.get("timestamp")}
                           for msg in messages]
                
                retained_indices = result.get("retained_indices", [])
                
                # 根据索引返回保留的消息
                cleaned_messages = []
                for idx in retained_indices:
                    if 0 <= idx < len(messages):
                        original_msg = messages[idx]
                        cleaned_messages.append({
                            "role": original_msg.get("role"),
                            "content": original_msg.get("content"),
                            "timestamp": original_msg.get("timestamp")
                        })
                
                logger.info(f"LLM清洗完成: 原始{len(messages)}条 -> 保留{len(cleaned_messages)}条")
                return cleaned_messages
                
            except json.JSONDecodeError as e:
                logger.error(f"LLM响应JSON解析失败: {e}，内容：{content}")
                return [{"role": msg.get("role"), "content": msg.get("content"), "timestamp": msg.get("timestamp")}
                       for msg in messages]
                
        except Exception as e:
            logger.error(f"LLM清洗失败: {e}")
            # 如果LLM调用失败，返回原始消息
            return [{"role": msg.get("role"), "content": msg.get("content"), "timestamp": msg.get("timestamp")}
                   for msg in messages]

    def clean_conversation(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        兼容旧接口，调用新的按天清洗方法
        
        Args:
            messages: 对话消息列表
            
        Returns:
            清洗后的对话消息列表
        """
        return self.clean_daily_conversation(messages)


# 全局客户端实例
OpenAI_client = OpenAIClient()
llm_cleaner = LLMDataCleaner(OpenAI_client)
