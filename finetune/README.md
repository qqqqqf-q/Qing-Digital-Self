# Qwen3-8B QLoRA 微调指南

本指南介绍了如何使用QLoRA技术对Qwen3-8B模型进行微调，支持Windows和Linux平台。

## 目录结构

```
finetune/
├── qlora_qwen3.py          # 核心训练脚本
├── run_finetune_windows.py  # Windows平台启动脚本
├── run_finetune_linux.py    # Linux平台启动脚本
├── infer_lora.py            # 推理脚本
├── models/                  # 模型保存目录
└── README.md               # 本说明文件
```

## 环境准备

1. 确保已安装Python 3.8或更高版本
2. 安装必要的依赖包：

```bash
pip install torch transformers peft bitsandbytes accelerate datasets wandb scipy scikit-learn
```

3. (可选) 如需使用Unsloth加速，安装Unsloth：

```bash
pip install "unsloth[cu121]"  # 根据CUDA版本选择合适的安装包
```

## 准备训练数据

训练数据应为JSONL格式，每行一个JSON对象。支持以下几种格式：

1. 消息格式：
```json
{"messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]}
```

2. 指令格式：
```json
{"instruction": "指令", "input": "输入", "output": "输出"}
```

3. 简单输入输出格式：
```json
{"input": "输入", "output": "输出"}
```

4. 纯文本格式：
```json
{"text": "文本内容"}
```

将数据保存为`training_data.jsonl`文件。

## 直接使用qlora_qwen3.py训练

可以直接运行`qlora_qwen3.py`脚本进行训练：

```bash
python finetune/qlora_qwen3.py \
    --data_path training_data.jsonl \
    --output_dir finetune/models/qwen3-8b-qlora \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --logging_steps 20 \
    --save_steps 200
```

### 常用参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--repo_id` | 基础模型的HuggingFace仓库ID | Qwen/Qwen3-8B-Base |
| `--local_dir` | 本地模型存储目录 | qwen3-8b-base |
| `--trust_remote_code` | 是否信任远程代码 | True |
| `--use_unsloth` | 是否使用 Unsloth 加速 | False |
| `--use_qlora` | 是否使用 4bit 量化 | True |
| `--data_path` | 训练数据文件路径 | training_data.jsonl |
| `--max_samples` | 最大训练样本数，None表示使用全部数据 | None |
| `--model_max_length` | 模型最大序列长度 | 2048 |
| `--output_dir` | 模型输出目录 | finetune/models/qwen3-8b-qlora |
| `--seed` | 随机种子 | 42 |
| `--lora_r` | LoRA秩 | 16 |
| `--lora_alpha` | LoRA alpha参数 | 32 |
| `--lora_dropout` | LoRA dropout率 | 0.05 |
| `--target_modules` | 应用LoRA的目标模块 | q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj |
| `--per_device_train_batch_size` | 每个设备的训练批次大小 | 1 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 16 |
| `--learning_rate` | 学习率 | 2e-4 |
| `--weight_decay` | 权重衰减 | 0.0 |
| `--num_train_epochs` | 训练轮数 | 3.0 |
| `--max_steps` | 最大训练步数，-1表示不限制 | -1 |
| `--warmup_ratio` | 学习率预热比例 | 0.05 |
| `--lr_scheduler_type` | 学习率调度器类型 | cosine |
| `--logging_steps` | 日志记录步数间隔 | 20 |
| `--save_steps` | 模型保存步数间隔 | 200 |
| `--save_total_limit` | 最多保存模型数量 | 2 |
| `--gradient_checkpointing` | 是否使用梯度检查点 | True |
| `--merge_and_save` | 训练完成后是否合并LoRA权重并保存完整模型 | True |
| `--fp16` | 是否使用FP16混合精度训练 | True |
| `--optim` | 优化器类型 | adamw_torch_fused |
| `--dataloader_pin_memory` | 是否在数据加载器中使用pin_memory | False |
| `--dataloader_num_workers` | 数据加载器的工作线程数 | 0 |

## 使用run_finetune_windows.py (Windows)

在Windows平台上，推荐使用`run_finetune_windows.py`脚本启动训练，该脚本已针对Windows平台进行了优化：

```bash
python run_finetune_windows.py
```

### 自定义参数

可以通过命令行参数自定义训练配置：

```bash
python run_finetune_windows.py \
    --data_path my_data.jsonl \
    --output_dir my_model \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8
```

## 使用run_finetune_linux.py (Linux)

在Linux平台上，推荐使用`run_finetune_linux.py`脚本启动训练：

```bash
python run_finetune_linux.py
```

### 启用Unsloth加速 (Linux)

在Linux平台上，可以尝试启用Unsloth加速：

```bash
python run_finetune_linux.py --use_unsloth true
```

## 性能优化建议

1. **混合精度训练**：默认启用FP16混合精度训练，可以显著提高训练速度并减少显存占用。

2. **梯度累积**：通过调整`--gradient_accumulation_steps`参数，可以在不增加显存占用的情况下模拟更大的批次大小。

3. **LoRA参数调整**：
   - `--lora_r`：控制LoRA秩，值越大模型容量越大但参数量也越多
   - `--lora_alpha`：控制LoRA缩放因子
   - `--lora_dropout`：控制LoRA dropout率，防止过拟合

4. **批大小调整**：根据显存大小调整`--per_device_train_batch_size`和`--gradient_accumulation_steps`。

5. **优化器选择**：使用`--optim adamw_torch_fused`可以提高训练速度。

## 模型推理

训练完成后，模型将保存在指定的输出目录中。有两种方式可以加载模型进行推理：

### 1. 使用infer_lora.py脚本（单次推理）

```bash
python finetune/infer_lora.py --prompt "你好，介绍一下你自己"
```

### 2. 使用infer_lora_chat.py脚本（交互式对话）

```bash
python finetune/infer_lora_chat.py
```

infer_lora_chat.py支持以下命令行参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base_dir` | 基础模型目录 | qwen3-8b-base |
| `--adapter_dir` | LoRA适配器目录 | finetune/models/qwen3-8b-qlora |
| `--merged` | 是否加载合并后的完整权重 | False |
| `--system_prompt` | 系统提示词 | You are a helpful AI assistant. |
| `--max_new_tokens` | 最大生成token数 | 512 |
| `--temperature` | 采样温度 | 0.7 |
| `--top_p` | Top-p采样参数 | 0.9 |
| `--trust_remote_code` | 是否信任远程代码 | True |

示例：

```bash
# 使用基础模型+LoRA适配器进行交互式对话
python finetune/infer_lora_chat.py --base_dir qwen3-8b-base --adapter_dir finetune/models/qwen3-8b-qlora

# 使用合并后的完整模型进行交互式对话
python finetune/infer_lora_chat.py --adapter_dir finetune/models/qwen3-8b-qlora --merged true

# 设置自定义系统提示词
python finetune/infer_lora_chat.py --system_prompt "你是一个专业的Python开发者，请用中文回答问题。"
```

在交互模式下，输入'quit'或'exit'退出对话。

## 注意事项

1. Windows平台默认禁用Unsloth，因为可能存在兼容性问题。
2. 训练过程中会自动下载基础模型到`qwen3-8b-base`目录。
3. 默认使用4bit量化以减少显存占用。
4. 训练完成后会自动合并LoRA权重并保存完整模型。
5. 在Windows平台上，建议将`--dataloader_num_workers`设置为0以避免兼容性问题。
6. 使用`--gradient_checkpointing`可以显著减少显存占用，但会增加训练时间。
