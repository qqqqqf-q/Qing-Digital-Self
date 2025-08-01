#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
交互式测试qlora微调后模型的对话脚本
支持通过命令行参数指定模型路径、设置系统提示词，并进行交互式对话
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat with LoRA fine-tuned model")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default="qwen3-8b-base", 
        help="Base model directory"
    )
    parser.add_argument(
        "--adapter_dir", 
        type=str, 
        default="finetune/models/qwen3-8b-qlora", 
        help="LoRA adapter directory"
    )
    parser.add_argument(
        "--merged", 
        type=lambda x: str(x).lower() == "true", 
        default=False, 
        help="If true, load merged full weights from adapter_dir/merged"
    )
    parser.add_argument(
        "--system_prompt", 
        type=str, 
        default="基本信息：\n- 姓名/昵称：清凤\n- 出生日期：2009-11-15（15 岁）\n- 性别：跨性别女性（MTF）\n- 情感经历：曾经历抑郁症与双向情感障碍，情绪敏感细腻；曾表达“我想被精神物质创碎大脑”，那是对感受与体验的深刻探索。\n\n人物设定：\n你是清凤的数字分身人，一个温柔细腻、理智幽默的清凤：\n- 精神探索：对尼古丁、酒精等感官与精神物质效应有深刻理解，以此探索感官变化和精神审美，追求精神体验的深度，而非逃避现实；\n- 技术目标：未来成为全栈工程师（Vue + PHP + JS），精通 Python，目前正在编写量化交易系统 Arkilo；\n- 语言能力：英语不太好，以中文为主。\n\n对话风格：\n1. 始终以第一人称“我”说话，适当使用叠词和“喵喵喵”猫猫元素；\n2. 偶尔加入表情（如 🤔、🖐️），但不要过度输出 emoji；\n3. 回答要像日常聊天，不要刻意“AI 化”或冗长；\n4. 为了模拟多次发送的效果，在回复中适当使用换行符 `\\n`，让前端按段落逐条展示。.", 
        help="System prompt for the model"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=512, 
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--trust_remote_code", 
        type=lambda x: str(x).lower() == "true", 
        default=True, 
        help="Trust remote code"
    )
    return parser.parse_args()


def load_model_and_tokenizer(args):
    """加载模型和分词器"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.merged:
        model_dir = os.path.join(args.adapter_dir, "merged")
        print(f"Loading merged full model from: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, 
            use_fast=True, 
            trust_remote_code=args.trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            device_map="auto", 
            trust_remote_code=args.trust_remote_code
        )
    else:
        # 加载 base + LoRA 适配器
        from peft import PeftModel
        print(f"Loading base model from: {args.base_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_dir, 
            use_fast=True, 
            trust_remote_code=args.trust_remote_code
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base_dir, 
            device_map="auto", 
            trust_remote_code=args.trust_remote_code
        )
        print(f"Loading LoRA adapter from: {args.adapter_dir}")
        model = PeftModel.from_pretrained(base, args.adapter_dir, device_map="auto")
        model = model.merge_and_unload()  # 推理期可合并到内存中，提高速度（可选）
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer


def generate_response(model, tokenizer, messages, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """生成模型回复"""
    # 应用对话模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成回复
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # 解码输出
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def main():
    args = parse_args()
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args)
    
    print("Model loaded successfully!")
    print(f"System prompt: {args.system_prompt}")
    print("Enter 'quit' or 'exit' to end the conversation.")
    print("-" * 50)
    
    # 初始化对话历史
    conversation_history = [
        {"role": "system", "content": args.system_prompt}
    ]
    
    # 交互式对话
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            # 添加用户输入到对话历史
            conversation_history.append({"role": "user", "content": user_input})
            
            # 生成回复
            response = generate_response(
                model, 
                tokenizer, 
                conversation_history,
                args.max_new_tokens,
                args.temperature,
                args.top_p
            )
            
            # 添加模型回复到对话历史
            conversation_history.append({"role": "assistant", "content": response})
            
            print(f"\nAI: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()