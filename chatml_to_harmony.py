#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatML到OpenAI Harmony格式转换器

将标准的ChatML格式数据转换为OpenAI Harmony格式，支持多通道输出：
- analysis: 分析推理通道
- commentary: 评论工具调用通道  
- final: 最终用户响应通道

使用方法:
    python chatml_to_harmony.py --input training_data.jsonl --output harmony_data.txt
"""

import json
import argparse
import sys
from typing import Dict, List, Any
import os

def convert_chatml_to_harmony(chatml_data: Dict[str, Any]) -> str:
    """
    将单条ChatML格式数据转换为Harmony格式
    
    Args:
        chatml_data: ChatML格式的对话数据，包含messages列表
        
    Returns:
        str: Harmony格式的字符串
    """
    messages = chatml_data.get('messages', [])
    if not messages:
        return ""
    
    harmony_output = []
    
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        
        if role == 'system':
            # 系统消息格式
            harmony_output.append(f"<|start|>system<|message|>{content}<|end|>")
            
        elif role == 'user':
            # 用户消息格式
            harmony_output.append(f"<|start|>user<|message|>{content}<|end|>")
            
        elif role == 'assistant':
            # 助手消息需要转换为多通道格式
            if content.strip():  # 如果有内容
                # 添加分析通道 - 模拟内部推理过程
                analysis_content = f"用户消息分析：需要生成合适的回复。内容：{content[:50]}{'...' if len(content) > 50 else ''}"
                harmony_output.append(f"<|start|>assistant<|channel|>analysis<|message|>{analysis_content}<|end|>")
                
                # 添加最终回复通道
                harmony_output.append(f"<|start|>assistant<|channel|>final<|message|>{content}<|return|>")
            else:
                # 空回复的情况
                harmony_output.append(f"<|start|>assistant<|channel|>analysis<|message|>需要生成回复但内容为空。<|end|>")
                harmony_output.append(f"<|start|>assistant<|channel|>final<|message|><|return|>")
    
    return '\n'.join(harmony_output)

def process_file(input_file: str, output_file: str) -> None:
    """
    处理整个文件，将ChatML格式转换为Harmony格式
    
    Args:
        input_file: 输入的ChatML格式文件路径
        output_file: 输出的Harmony格式文件路径
    """
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        sys.exit(1)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            processed_count = 0
            error_count = 0
            
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析JSON行
                    chatml_data = json.loads(line)
                    
                    # 转换为Harmony格式
                    harmony_text = convert_chatml_to_harmony(chatml_data)
                    
                    if harmony_text:
                        # 写入输出文件，每个对话之间用空行分隔
                        f_out.write(harmony_text + '\n\n')
                        processed_count += 1
                        
                        # 每处理100条显示进度
                        if processed_count % 100 == 0:
                            print(f"已处理 {processed_count} 条对话...")
                    
                except json.JSONDecodeError as e:
                    print(f"警告：第 {line_num} 行JSON解析错误: {e}")
                    error_count += 1
                except Exception as e:
                    print(f"警告：第 {line_num} 行处理错误: {e}")
                    error_count += 1
            
            print(f"\n转换完成！")
            print(f"成功处理: {processed_count} 条对话")
            print(f"错误数量: {error_count} 条")
            print(f"输出文件: {output_file}")
            
    except IOError as e:
        print(f"文件操作错误: {e}")
        sys.exit(1)

def show_sample_conversion(input_file: str) -> None:
    """
    显示转换示例
    
    Args:
        input_file: 输入文件路径
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 读取第一行作为示例
            first_line = f.readline().strip()
            if first_line:
                chatml_data = json.loads(first_line)
                harmony_text = convert_chatml_to_harmony(chatml_data)
                
                print("=== 转换示例 ===")
                print("\n原始ChatML格式:")
                print(json.dumps(chatml_data, ensure_ascii=False, indent=2))
                print("\n转换后的Harmony格式:")
                print(harmony_text)
                print("\n" + "="*50)
                
    except Exception as e:
        print(f"无法显示示例: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='将ChatML格式数据转换为OpenAI Harmony格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python chatml_to_harmony.py --input training_data.jsonl --output harmony_data.txt
  python chatml_to_harmony.py --input data.jsonl --output result.txt --sample

Harmony格式说明:
  - system: 系统消息
  - user: 用户输入
  - assistant: 助手回复（包含analysis和final通道）
  - analysis通道: 内部推理过程
  - final通道: 最终用户响应
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入的ChatML格式文件路径（.jsonl格式）'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出的Harmony格式文件路径'
    )
    
    parser.add_argument(
        '--sample', '-s',
        action='store_true',
        help='显示转换示例（仅显示第一条数据的转换结果）'
    )
    
    args = parser.parse_args()
    
    print("ChatML到OpenAI Harmony格式转换器")
    print("=" * 40)
    
    # 如果指定了sample参数，显示示例
    if args.sample:
        show_sample_conversion(args.input)
        return
    
    # 执行转换
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print("开始转换...\n")
    
    process_file(args.input, args.output)

if __name__ == '__main__':
    main()