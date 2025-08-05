## 3.5 (Not Recommended) Run Full Model Directly After Fine-tuning (Recommend going directly to steps 4,5,6,7, wait until converted to GGUF and quantized before running)
### Specify Custom Model Path
```bash
python infer_lora_chat.py --base_dir my-base-model --adapter_dir my-lora-adapter
```

### Use Merged Model
```bash
python infer_lora_chat.py --merged true --adapter_dir my-lora-adapter
```

### Adjust Generation Parameters
```bash
python infer_lora_chat.py --temperature 0.9 --top_p 0.95 --max_new_tokens 1024
```

### Use Custom System Prompt
```bash
python infer_lora_chat.py --system_prompt "You are a helpful AI assistant."
```
### Command Line Parameters
| Parameter Name | Type | Default Value | Description |
|----------------|------|---------------|-------------|
| `--base_dir` | str | `qwen3-8b-base` | Base model directory |
| `--adapter_dir` | str | `finetune/models/qwen3-8b-qlora` | LoRA adapter directory |
| `--merged` | bool | `False` | If True, load merged full weights from adapter_dir/merged |
| `--system_prompt` | str | Qing's digital avatar persona | Model's system prompt |
| `--max_new_tokens` | int | `512` | Maximum number of new tokens to generate |
| `--temperature` | float | `0.7` | Sampling temperature |
| `--top_p` | float | `0.9` | Top-p sampling parameter |
| `--trust_remote_code` | bool | `True` | Whether to trust remote code |