#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LM Studio集成测试脚本
"""

import sys
import os
from typing import List, Dict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.openai.openai_client import OpenAIClient, LLMDataCleaner
from utils.config.config import get_config
from utils.logger.logger import get_logger

logger = get_logger('Test_OpenAI')

def test_OpenAI_health():
    """测试OpenAI API服务器连接"""
    print("=" * 50)
    print("测试OpenAI API服务器连接...")
    
    client = OpenAIClient()
    
    if client.health_check():
        print("[成功] OpenAI API服务器连接成功")
        
        # 获取可用模型
        models = client.list_models()
        if models:
            print(f"[成功] 可用模型: {', '.join(models)}")
        else:
            print("[警告] 无法获取模型列表")
        return True
    else:
        print("[错误] OpenAI API服务器连接失败")
        print("请确保: 1. OpenAI API服务正在运行 2. 服务器地址正确")
        return False

def test_daily_cleaning():
    """测试按天清洗功能"""
    print("\n" + "=" * 50)
    print("测试按天清洗功能...")
    
    # 模拟一天的对话数据
    test_messages = [
        {"role": "user", "content": "早上好", "timestamp": 1704067200},
        {"role": "assistant", "content": "早上好！今天感觉怎么样？", "timestamp": 1704067210},
        {"role": "user", "content": "还行吧", "timestamp": 1704067300},
        {"role": "user", "content": "[图片]", "timestamp": 1704067350},
        {"role": "assistant", "content": "看起来不错！", "timestamp": 1704067360},
        {"role": "user", "content": "语音通话 00:30", "timestamp": 1704067400},
        {"role": "user", "content": "今天天气真好", "timestamp": 1704067500},
        {"role": "assistant", "content": "是啊，阳光明媚", "timestamp": 1704067510},
        {"role": "user", "content": "你们互发消息，小水花开心极了", "timestamp": 1704067520},
        {"role": "user", "content": "晚上吃什么？", "timestamp": 1704070800},
        {"role": "assistant", "content": "想吃火锅", "timestamp": 1704070810},
        {"role": "user", "content": "自动补齐", "timestamp": 1704070860},
        {"role": "assistant", "content": "阿巴阿巴啊啊啊不不不", "timestamp": 1704070890},
    ]
    
    cleaner = LLMDataCleaner()
    
    print("原始消息:")
    for i, msg in enumerate(test_messages):
        print(f"  {i}. [{msg['role']}] {msg['content']}")
    
    try:
        cleaned = cleaner.clean_daily_conversation(test_messages, "2024-01-01")
        
        print(f"\n清洗后消息 ({len(cleaned)}/{len(test_messages)}条):")
        for i, msg in enumerate(cleaned):
            print(f"  {i}. [{msg['role']}] {msg['content']}")
        
        return True
        
    except Exception as e:
        print(f"[错误] 清洗测试失败: {e}")
        return False

def test_config():
    """测试配置"""
    print("\n" + "=" * 50)
    print("测试配置...")
    
    config = get_config()
    
    print(f"OpenAI API URL: {config.get('OpenAI_URL')}")
    print(f"OpenAI API Model: {config.get('OpenAI_Model')}")
    print(f"Use LLM Clean: {config.get('use_llm_clean')}")
    print(f"Timeout: {config.get('OpenAI_timeout')}s")
    
    return True

def main():
    """运行所有测试"""
    print("OpenAI API集成测试")
    print("=" * 50)
    
    # 测试配置
    test_config()
    
    # 测试连接
    if not test_OpenAI_health():
        print("\n❌ 连接测试失败，跳过清洗测试")
        return
    
    # 测试清洗功能
    test_daily_cleaning()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    
    print("\n使用说明:")
    print("1. 启动OpenAI API并加载模型")
    print("2. 在seeting.jsonc中设置清洗配置启用LLM清洗")
    print("3. 运行 generate_training_data.py 进行数据清洗")

if __name__ == "__main__":
    main()