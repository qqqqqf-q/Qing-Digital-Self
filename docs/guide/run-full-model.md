## 3.5 .(不建议)微调后直接运行全量模型(建议直接看第4,5,6,7步,等转换为guff并量化完再跑)
### 指定自定义模型路径
```bash
python infer_lora_chat.py --base_dir my-base-model --adapter_dir my-lora-adapter
```

### 使用合并后的模型
```bash
python infer_lora_chat.py --merged true --adapter_dir my-lora-adapter
```

### 调整生成参数
```bash
python infer_lora_chat.py --temperature 0.9 --top_p 0.95 --max_new_tokens 1024
```

### 使用自定义系统提示词
```bash
python infer_lora_chat.py --system_prompt "你是一个乐于助人的AI助手。"
```
### 命令行参数说明
| 参数名称 | 类型 | 默认值 | 描述 |
|---------|------|-------|------|
| `--base_dir` | str | `qwen3-8b-base` | 基础模型目录 |
| `--adapter_dir` | str | `finetune/models/qwen3-8b-qlora` | LoRA适配器目录 |
| `--merged` | bool | `False` | 如果为True，则从adapter_dir/merged加载合并后的完整权重 |
| `--system_prompt` | str | 清凤数字分身人设 | 模型的系统提示词 |
| `--max_new_tokens` | int | `512` | 生成的最大新token数量 |
| `--temperature` | float | `0.7` | 采样温度 |
| `--top_p` | float | `0.9` | Top-p采样参数 |
| `--trust_remote_code` | bool | `True` | 是否信任远程代码 |