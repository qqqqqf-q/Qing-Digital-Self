#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将qa_final.json转换为training_data.jsonl格式并随机插入到training_data.jsonl中
"""

import json
import random
import os
import argparse
from typing import List, Dict, Any

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
    """加载qa_final.json数据"""
    with open(qa_file, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def convert_qa_to_chatml_format(qa_data: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, Any]]:
    """将qa数据转换为chatml格式"""
    converted_data = []
    
    for qa_item in qa_data:
        if 'turns' not in qa_item:
            continue
            
        messages = [{"role": "system", "content": system_prompt}]
        
        for turn in qa_item['turns']:
            role = turn.get('role', 'user')
            text = turn.get('text', '')
            
            # 映射角色到chatml格式
            if role == 'ai':
                role = 'assistant'
            elif role == 'user':
                role = 'user'
            
            messages.append({
                "role": role,
                "content": text
            })
        
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
            from config.config import get_config
            config = get_config()
            system_prompt = config.get("system_prompt")
        except:
            system_prompt = NEW_SYSTEM_PROMPT
    
    print("开始转换数据...")
    
    # 加载数据
    qa_data = load_qa_data(args.qa_file)
    training_data = load_training_data(args.training_file)
    
    print(f"原始training_data.jsonl有 {len(training_data)} 条数据")
    print(f"qa_final.json有 {len(qa_data)} 条对话")
    
    # 转换QA数据格式
    converted_qa_data = convert_qa_to_chatml_format(qa_data, system_prompt)
    
    print(f"转换后的QA数据有 {len(converted_qa_data)} 条")
    
    # 按百分比合并数据
    merged_data, insert_count = merge_data_by_percentage(
        training_data, converted_qa_data, args.percentage, args.seed
    )
    
    print(f"合并后的数据有 {len(merged_data)} 条")
    
    # 保存合并后的数据
    save_training_data(merged_data, args.output_file)
    
    print(f"数据合并完成！已保存到 {args.output_file}")
    print("统计信息:")
    print(f"- 原始数据: {len(training_data)} 条")
    print(f"- 计划插入: {int(len(training_data) * args.percentage / 100)} 条 ({args.percentage}%)")
    print(f"- 实际插入: {insert_count} 条")
    print(f"- 总数据量: {len(merged_data)} 条")


if __name__ == "__main__":
    main()