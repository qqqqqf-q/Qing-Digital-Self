#!/usr/bin/env python3
"""
将微调后的模型转换为GGUF格式并支持自定义量化

使用方法:
    python convert_to_gguf.py --model_path ./finetune/models/qwen3-8b-qlora --output_dir ./gguf_models --quantization q4_k_m

支持的量化类型:
    q4_0: 4-bit quantization
    q4_1: 4-bit quantization
    q5_0: 5-bit quantization
    q5_1: 5-bit quantization
    q8_0: 8-bit quantization
    q2_k: 2-bit quantization
    q3_k: 3-bit quantization
    q4_k: 4-bit quantization
    q5_k: 5-bit quantization
    q6_k: 6-bit quantization
    q8_k: 8-bit quantization
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import shutil
import json

def check_llama_cpp():
    """检查llama.cpp是否安装"""
    try:
        subprocess.run(["python", "-m", "llama_cpp", "--version"], 
                      capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def install_llama_cpp():
    """安装llama-cpp-python"""
    print("正在安装llama-cpp-python...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"], 
                      check=True)
        print("llama-cpp-python安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"安装失败: {e}")
        return False

def convert_to_gguf(model_path, output_dir, quantization="q4_k_m", use_gpu=False):
    """将模型转换为GGUF格式"""
    
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否是Hugging Face格式的模型
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"在{model_path}中找不到config.json，可能不是有效的Hugging Face模型")
    
    # 获取模型名称
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        model_name = config.get("_name_or_path", "unknown_model")
        if model_name == "unknown_model":
            model_name = model_path.name
    
    # 输出文件名
    output_filename = f"{model_name}_{quantization}.gguf"
    output_path = output_dir / output_filename
    
    print(f"正在转换模型: {model_path}")
    print(f"输出路径: {output_path}")
    print(f"量化类型: {quantization}")
    
    # 使用llama.cpp的转换脚本
    try:
        # 首先尝试使用convert-hf-to-gguf.py
        convert_script = "convert-hf-to-gguf.py"
        
        # 检查convert脚本是否存在
        script_path = None
        possible_paths = [
            "llama.cpp/convert-hf-to-gguf.py",
            "convert-hf-to-gguf.py",
            str(Path.home() / ".local/bin/convert-hf-to-gguf.py")
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                script_path = path
                break
        
        if script_path is None:
            # 如果没有找到脚本，尝试使用llama-cpp-python的转换
            print("未找到convert-hf-to-gguf.py，尝试使用llama-cpp-python...")
            return convert_with_llama_cpp_python(model_path, output_path, quantization, use_gpu)
        
        # 构建转换命令
        cmd = [
            "python", script_path,
            str(model_path),
            "--outtype", quantization,
            "--outfile", str(output_path)
        ]
        
        if use_gpu:
            cmd.extend(["--use-gpu"])
        
        print(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"转换失败: {result.stderr}")
            return False
        
        print(f"转换成功: {output_path}")
        return True
        
    except Exception as e:
        print(f"转换过程中出错: {e}")
        return False

def convert_with_llama_cpp_python(model_path, output_path, quantization, use_gpu):
    """使用llama-cpp-python进行转换"""
    try:
        from llama_cpp import Llama
        
        # 加载模型
        llm = Llama.from_pretrained(
            repo_id=str(model_path),
            filename="*",
            verbose=True
        )
        
        # 保存为GGUF格式
        llm.save_model(str(output_path))
        
        print(f"使用llama-cpp-python转换成功: {output_path}")
        return True
        
    except ImportError:
        print("llama-cpp-python未安装，请先安装")
        return False
    except Exception as e:
        print(f"llama-cpp-python转换失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="将微调后的模型转换为GGUF格式")
    parser.add_argument("--model_path", type=str, required=True,
                        help="微调后的模型路径")
    parser.add_argument("--output_dir", type=str, default="./gguf_models",
                        help="输出目录")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", 
                                "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k", "q4_k_m"],
                        help="量化类型")
    parser.add_argument("--use_gpu", action="store_true",
                        help="使用GPU加速")
    parser.add_argument("--install_deps", action="store_true",
                        help="自动安装依赖")
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_llama_cpp():
        if args.install_deps:
            if not install_llama_cpp():
                print("无法安装llama-cpp-python，请手动安装")
                return 1
        else:
            print("请先安装llama-cpp-python: pip install llama-cpp-python")
            print("或使用 --install_deps 自动安装")
            return 1
    
    try:
        success = convert_to_gguf(
            args.model_path,
            args.output_dir,
            args.quantization,
            args.use_gpu
        )
        
        if success:
            print("转换完成！")
            return 0
        else:
            print("转换失败")
            return 1
            
    except Exception as e:
        print(f"错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())