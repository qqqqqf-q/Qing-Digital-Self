# Qing-Agent GGUF模型转换指南

## 概述

本指南将帮助您将微调后的Qwen3-8B模型转换为GGUF格式，并支持自定义量化级别。

## 快速开始

### 方法1: 一键安装（推荐）

1. 运行安装脚本：
```bash
python setup_gguf_tools.py
```

2. 转换模型：
```bash
python convert_hf_to_gguf.py --model_path ./finetune/models/qwen3-8b-qlora --quantization q4_k_m
```

### 方法2: 使用批处理文件（Windows）

1. 双击运行 `convert_to_gguf.bat`
2. 按提示输入参数

### 方法3: 使用基础脚本

```bash
python convert_to_gguf.py --model_path ./finetune/models/qwen3-8b-qlora --quantization q4_k_m --output_dir ./gguf_models
```

## 支持的量化类型

| 量化类型 | 描述 | 文件大小 | 推理速度 | 质量 |
|----------|------|----------|----------|------|
| `f16` | 16位浮点 | ~15GB | 快 | 最好 |
| `q8_0` | 8位量化 | ~8GB | 快 | 很好 |
| `q5_k` | 5位量化 | ~5.5GB | 快 | 好 |
| `q4_k_m` | 4位量化（推荐） | ~4.5GB | 快 | 很好 |
| `q4_0` | 4位量化 | ~4GB | 快 | 好 |
| `q3_k` | 3位量化 | ~3.5GB | 中等 | 一般 |
| `q2_k` | 2位量化 | ~2.5GB | 慢 | 较差 |

## 使用示例

### 转换微调后的模型

```bash
# 转换默认微调模型
python convert_hf_to_gguf.py --model_path ./finetune/models/qwen3-8b-qlora --quantization q4_k_m

# 转换4bit模型
python convert_hf_to_gguf.py --model_path ./finetune/models/qwen3-8b-qlora-4bit --quantization q4_k_m

# 使用高质量量化
python convert_hf_to_gguf.py --model_path ./finetune/models/qwen3-8b-qlora --quantization q5_k

# 指定输出目录
python convert_hf_to_gguf.py --model_path ./finetune/models/qwen3-8b-qlora --output_dir ./my_models --quantization q4_k_m
```

### 批量转换

创建 `batch_convert.py`：

```python
import os
import subprocess

models = [
    "./finetune/models/qwen3-8b-qlora",
    "./finetune/models/qwen3-8b-qlora-4bit"
]

quantizations = ["q4_k_m", "q5_k", "q8_0"]

for model in models:
    for quant in quantizations:
        cmd = f"python convert_hf_to_gguf.py --model_path {model} --quantization {quant}"
        print(f"执行: {cmd}")
        subprocess.run(cmd, shell=True)
```

## 验证转换结果

### 检查文件

转换完成后，检查输出目录：

```bash
dir gguf_models
```

应该看到类似：
- `qwen3-8b-qlora_q4_k_m.gguf`
- `qwen3-8b-qlora_q5_k.gguf`

### 测试模型

使用llama.cpp测试：

```bash
# Windows
llama.cpp/main.exe -m gguf_models/qwen3-8b-qlora_q4_k_m.gguf -p "你好"

# Linux/macOS
./llama.cpp/main -m gguf_models/qwen3-8b-qlora_q4_k_m.gguf -p "你好"
```

## 常见问题

### 1. 找不到convert-hf-to-gguf.py

运行安装脚本：
```bash
python setup_gguf_tools.py
```

### 2. 编译错误

**Windows:**
- 安装Visual Studio Build Tools
- 或安装MinGW

**Linux:**
```bash
sudo apt-get update
sudo apt-get install build-essential
```

### 3. 内存不足

- 使用更小的量化类型（如q3_k）
- 增加系统交换空间
- 使用CPU转换（添加--use-cpu参数）

### 4. 模型加载失败

确保模型目录包含：
- config.json
- pytorch_model.bin 或 model.safetensors
- tokenizer.json
- tokenizer_config.json

## 性能对比

| 量化类型 | 加载时间 | 推理速度 | 显存占用 | 适用场景 |
|----------|----------|----------|----------|----------|
| f16 | 15s | 100% | 16GB | 服务器 |
| q8_0 | 10s | 95% | 8GB | 高性能 |
| q4_k_m | 8s | 85% | 4.5GB | 平衡选择 |
| q3_k | 6s | 70% | 3.5GB | 低资源 |

## 高级用法

### 自定义转换参数

编辑 `convert_hf_to_gguf.py` 中的转换命令：

```python
# 添加自定义参数
cmd = [
    "python", str(convert_script),
    str(model_path),
    "--outfile", str(output_path),
    "--use-temp-file",  # 使用临时文件
    "--verbose"         # 详细输出
]
```

### 多GPU支持

对于大模型，可以使用多GPU：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python convert_hf_to_gguf.py --model_path ./model --use-gpu
```

## 文件结构

转换后的目录结构：

```
Qing-Agent/
├── finetune/models/
│   ├── qwen3-8b-qlora/
│   └── qwen3-8b-qlora-4bit/
├── gguf_models/          # 转换后的GGUF文件
│   ├── qwen3-8b-qlora_q4_k_m.gguf
│   └── qwen3-8b-qlora_q5_k.gguf
├── llama.cpp/            # llama.cpp工具
├── convert_hf_to_gguf.py # 高级转换脚本
├── convert_to_gguf.py    # 基础转换脚本
└── setup_gguf_tools.py   # 安装脚本
```

## 技术支持

如果遇到问题：

1. 检查Python版本 >= 3.8
2. 确保有足够的磁盘空间（至少20GB）
3. 查看转换日志文件
4. 尝试不同的量化类型
5. 重启后重试

## 下一步

转换完成后，您可以使用：
- llama.cpp进行本地推理
- Ollama运行模型
- 部署到生产环境
- 集成到应用程序中