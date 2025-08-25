#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将qa_final.json转换为新的训练数据格式（使用instruction/input/output结构，输出为chatml格式）并随机插入到training_data.jsonl中
保留system prompt，移除原有格式
"""

import json
import random
import os
import sys
import argparse
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger.logger import get_logger

# 创建 logger 实例
logger = get_logger('MergeDataRuozhiba')

# 新的system prompt
NEW_SYSTEM_PROMPT = """你是一位专业的AI助手，具备广泛的知识和技能。你能够以清晰、准确的方式回答各种问题，并提供有用的建议和信息。

你的特点：
- 知识渊博，涵盖多个领域
- 回答简洁明了，逻辑清晰
- 善于分析问题并提供实用解决方案
- 保持客观中立的态度
- 使用通俗易懂的语言

请始终以助手的身份为用户提供帮助。"""


def load_qa_data(qa_file: str) -> List[Dict[str, Any]]:
    """加载QA数据，支持.json和.jsonl格式"""
    data = []
    if qa_file.endswith('.jsonl'):
        # 处理JSONL格式
        with open(qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        # 处理JSON格式
        with open(qa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return data


def load_training_data(training_file: str) -> List[Dict[str, Any]]:
    """加载training_data.jsonl数据"""
    data = []
    if os.path.exists(training_file):
        with open(training_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def convert_qa_to_new_format(qa_data: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, Any]]:
    """将qa数据转换为新的格式，使用instruction/input/output结构，输出为chatml格式"""
    converted_data = []
    
    for qa_item in qa_data:
        # 支持两种格式：turns数组格式和instruction/input/output格式
        if 'turns' in qa_item:
            # 原始turns格式
            turns = qa_item['turns']
            if len(turns) < 2:
                continue
                
            # 提取用户输入作为instruction，AI回复作为output
            user_turns = [turn for turn in turns if turn.get('role') == 'user']
            ai_turns = [turn for turn in turns if turn.get('role') == 'ai']
            
            if not user_turns or not ai_turns:
                continue
                
            instruction = user_turns[0].get('text', '')
            output = ai_turns[0].get('text', '')
        elif 'instruction' in qa_item and 'output' in qa_item:
            # instruction/input/output格式
            instruction = qa_item.get('instruction', '')
            input_text = qa_item.get('input', '')
            output = qa_item.get('output', '')
            
            # 如果有input，将其附加到instruction后面
            if input_text:
                instruction = f"{instruction}\n\n{input_text}".strip()
        else:
            continue
            
        # 构建chatml格式的消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        
        converted_data.append({"messages": messages})
    
    return converted_data


def merge_data_randomly(original_data: List[Dict[str, Any]], new_data: List[Dict[str, Any]], random_seed: int = 42) -> List[Dict[str, Any]]:
    """随机合并数据"""
    random.seed(random_seed)
    
    # 合并数据
    merged_data = original_data + new_data
    
    # 随机打乱
    random.shuffle(merged_data)
    
    return merged_data


def merge_data_by_percentage(original_data: List[Dict[str, Any]], new_data: List[Dict[str, Any]], percentage: float, seed: int = 42):
    """按百分比合并数据"""
    random.seed(seed)
    
    # 计算要插入的行数
    total_original = len(original_data)
    insert_count = max(1, int(total_original * percentage / 100))
    
    # 如果新数据不足，使用所有新数据
    if len(new_data) < insert_count:
        insert_count = len(new_data)
    
    # 随机选择要插入的新数据
    selected_new_data = random.sample(new_data, insert_count)
    
    # 合并数据
    merged_data = original_data + selected_new_data
    
    # 随机打乱
    random.shuffle(merged_data)
    
    return merged_data, insert_count


def save_training_data(data: List[Dict[str, Any]], output_file: str):
    """保存训练数据到jsonl文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + '\n')


def main():
    parser = argparse.ArgumentParser(description='将qa_final.json转换为training_data.jsonl格式并按百分比插入')
    parser.add_argument('--qa-file', default='qa_final.json', help='QA数据文件路径')
    parser.add_argument('--training-file', default='training_data.jsonl', help='训练数据文件路径')
    parser.add_argument('--output-file', default='training_data_merged.jsonl', help='输出文件路径')
    parser.add_argument('--percentage', type=float, default=100.0, help='插入百分比 (0-100)')
    parser.add_argument('--use-new-prompt', action='store_true', help='使用新的system prompt')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 选择system prompt
    if args.use_new_prompt:
        system_prompt = NEW_SYSTEM_PROMPT
    else:
        try:
            import sys
            sys.path.append('.')
            from utils.config.config import get_config
            config = get_config()
            system_prompt = config.get("system_prompt")
        except:
            system_prompt = NEW_SYSTEM_PROMPT
    
    logger.info("开始转换数据")
    print("开始转换数据...")
    
    # 加载数据
    qa_data = load_qa_data(args.qa_file)
    training_data = load_training_data(args.training_file)
    
    logger.info(f"原始训练数据: {len(training_data)} 条")
    logger.info(f"QA数据: {len(qa_data)} 条")
    print(f"原始training_data.jsonl有 {len(training_data)} 条数据")
    print(f"qa_final.json有 {len(qa_data)} 条对话")
    
    # 转换QA数据格式
    converted_qa_data = convert_qa_to_new_format(qa_data, system_prompt)
    
    logger.info(f"转换后QA数据: {len(converted_qa_data)} 条")
    print(f"转换后的QA数据有 {len(converted_qa_data)} 条")
    
    # 按百分比合并数据
    merged_data, insert_count = merge_data_by_percentage(
        training_data, converted_qa_data, args.percentage, args.seed
    )
    
    logger.info(f"合并后数据: {len(merged_data)} 条")
    print(f"合并后的数据有 {len(merged_data)} 条")
    
    # 保存合并后的数据
    save_training_data(merged_data, args.output_file)
    
    logger.info(f"数据合并完成，保存到: {args.output_file}")
    logger.info(f"统计：原始{len(training_data)}条，插入{insert_count}条，总计{len(merged_data)}条")
    print(f"数据合并完成！已保存到 {args.output_file}")
    print("统计信息:")
    print(f"- 原始数据: {len(training_data)} 条")
    print(f"- 计划插入: {int(len(training_data) * args.percentage / 100)} 条 ({args.percentage}%)")
    print(f"- 实际插入: {insert_count} 条")
    print(f"- 总数据量: {len(merged_data)} 条")


if __name__ == "__main__":
    main()