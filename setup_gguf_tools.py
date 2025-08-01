#!/usr/bin/env python3
"""
GGUF转换工具安装脚本
自动安装llama.cpp和相关依赖
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform

def run_command(cmd, cwd=None):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, 
                              capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_git():
    """检查是否安装git"""
    success, _, _ = run_command("git --version")
    return success

def install_llama_cpp():
    """安装llama.cpp"""
    print("正在安装llama.cpp...")
    
    # 克隆llama.cpp仓库
    if not os.path.exists("llama.cpp"):
        print("正在克隆llama.cpp...")
        success, stdout, stderr = run_command("git clone https://github.com/ggerganov/llama.cpp.git")
        if not success:
            print(f"克隆失败: {stderr}")
            return False
    
    # 进入目录
    llama_dir = Path("llama.cpp")
    
    # 根据操作系统选择编译方式
    system = platform.system()
    
    if system == "Windows":
        print("检测到Windows系统，使用MSVC编译...")
        
        # 检查是否有MSVC
        success, _, _ = run_command("cl")
        if not success:
            print("警告: 未检测到MSVC编译器，尝试使用make...")
            # 尝试使用make
            success, _, _ = run_command("make")
            if not success:
                print("错误: 需要安装Visual Studio或MinGW才能编译")
                return False
        
        # 编译
        print("正在编译llama.cpp...")
        success, stdout, stderr = run_command("make", cwd=llama_dir)
        
    else:  # Linux/macOS
        print("正在编译llama.cpp...")
        success, stdout, stderr = run_command("make", cwd=llama_dir)
    
    if not success:
        print(f"编译失败: {stderr}")
        return False
    
    print("llama.cpp编译成功！")
    return True

def install_python_requirements():
    """安装Python依赖"""
    print("正在安装Python依赖...")
    
    packages = [
        "torch",
        "transformers",
        "sentencepiece",
        "protobuf",
        "numpy"
    ]
    
    for package in packages:
        print(f"正在安装 {package}...")
        success, _, _ = run_command(f"{sys.executable} -m pip install {package}")
        if not success:
            print(f"警告: 安装 {package} 失败")
    
    return True

def create_convert_script():
    """创建转换脚本"""
    script_content = '''#!/usr/bin/env python3
"""
高级GGUF转换脚本
使用llama.cpp官方工具进行转换
"""

import argparse
import os
import sys
from pathlib import Path
import json
import subprocess

def convert_model(model_path, output_dir, quantization="q4_k_m", use_gpu=False):
    """转换模型到GGUF格式"""
    
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    
    if not model_path.exists():
        print(f"错误: 模型路径不存在: {model_path}")
        return False
    
    # 检查llama.cpp
    llama_dir = Path("llama.cpp")
    if not llama_dir.exists():
        print("错误: llama.cpp未找到，请先运行setup_gguf_tools.py")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取模型名称
    try:
        with open(model_path / "config.json", 'r') as f:
            config = json.load(f)
            model_name = config.get("_name_or_path", model_path.name)
    except:
        model_name = model_path.name
    
    # 转换HF到GGUF
    convert_script = llama_dir / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        print("错误: convert-hf-to-gguf.py未找到")
        return False
    
    # 基础GGUF文件
    base_gguf = output_dir / f"{model_name}.gguf"
    
    print("正在转换为GGUF格式...")
    cmd = [
        sys.executable, str(convert_script),
        str(model_path),
        "--outfile", str(base_gguf)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"转换失败: {result.stderr}")
        return False
    
    print(f"基础GGUF文件已创建: {base_gguf}")
    
    # 量化
    if quantization != "f16":
        print(f"正在应用量化: {quantization}...")
        quantized_gguf = output_dir / f"{model_name}_{quantization}.gguf"
        
        quantize_exe = llama_dir / "quantize"
        if not quantize_exe.exists():
            quantize_exe = llama_dir / "quantize.exe"
        
        if not quantize_exe.exists():
            print("错误: quantize程序未找到")
            return False
        
        cmd = [str(quantize_exe), str(base_gguf), str(quantized_gguf), quantization]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"量化失败: {result.stderr}")
            return False
        
        # 删除未量化的文件
        base_gguf.unlink()
        
        print(f"量化完成: {quantized_gguf}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="转换模型到GGUF格式")
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--output_dir", default="./gguf_models", help="输出目录")
    parser.add_argument("--quantization", default="q4_k_m", 
                       choices=["f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", 
                               "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k", "q4_k_m"],
                       help="量化类型")
    
    args = parser.parse_args()
    
    success = convert_model(args.model_path, args.output_dir, args.quantization)
    
    if success:
        print("转换完成！")
        return 0
    else:
        print("转换失败")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
'''
    
    with open("convert_hf_to_gguf.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    if platform.system() != "Windows":
        os.chmod("convert_hf_to_gguf.py", 0o755)

def main():
    print("Qing-Agent GGUF工具安装器")
    print("=" * 30)
    print()
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("错误: 需要Python 3.8或更高版本")
        return 1
    
    # 检查Git
    if not check_git():
        print("错误: 未安装git，请先安装git")
        return 1
    
    # 安装llama.cpp
    if not install_llama_cpp():
        return 1
    
    # 安装Python依赖
    install_python_requirements()
    
    # 创建转换脚本
    create_convert_script()
    
    print()
    print("安装完成！")
    print()
    print("使用方法:")
    print("  python convert_hf_to_gguf.py --model_path ./finetune/models/qwen3-8b-qlora --quantization q4_k_m")
    print()
    print("支持的量化类型:")
    print("  f16, q4_0, q4_1, q5_0, q5_1, q8_0")
    print("  q2_k, q3_k, q4_k, q5_k, q6_k, q8_k, q4_k_m")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())