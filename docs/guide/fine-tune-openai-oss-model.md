# 番外篇: 微调OpenAI OSS模型
## 此篇内容未经过实机测试(我的3080显存太小跑不动)
### 欢迎能测试的朋友进行实机测试并将结果或Bug发在Issues上
### ~~能直接PR修复那就更棒了~~

---

> 因为OSS发布的时间,似乎微调OSS和Qwen的并不能通用  
> 并且最好使用新的`unsloth` `torch` `transformers` 等库  

[这是Unsloth提供的OSS微调经验](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb#scrollTo=WQSmUBxXx2r-)

## 以下是快速微调指南
> 建议使用新的虚拟环境  
> 和Qwen的微调环境分离  

请确保以下的库:
`torch>=2.8.0` `triton>=3.4.0`
**并且!请保证Unsloth和Unsloth_zoo的最新版**
> 原requirements.txt的unsloth只支持到2025.8.1版本,并不能微调oss  

```bash
pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" "unsloth[base] @ git+https://github.com/unslothai/unsloth" torchvision bitsandbytes git+https://github.com/huggingface/transformers git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
```

## 开始微调

```bash
python3 run_finetune_oss.py
```
---
| 参数                              | 类型         | 默认值                                                                                         | 可选值                                 | 说明                   |
| ------------------------------- | ---------- | ------------------------------------------------------------------------------------------- | ----------------------------------- | -------------------- |
| `--repo_id`                     | str        | `unsloth/gpt-oss-20b-unsloth-bnb-4bit`                                                      | -                                   | HF 仓库ID              |
| `--local_dir`                   | str        | `gpt-oss-20b-unsloth-bnb-4bit`                                                              | -                                   | 本地模型目录               |
| `--use_unsloth`                 | str        | `false`                                                                                     | `true`, `false`                     | 是否使用unsloth          |
| `--use_qlora`                   | str        | `true`                                                                                      | `true`, `false`                     | 是否使用QLoRA            |
| `--data_path`                   | str        | `training_data.jsonl`                                                                       | -                                   | 训练数据路径               |
| `--eval_data_path`              | str / None | None                                                                                        | -                                   | 验证数据路径               |
| `--max_samples`                 | str / None | None                                                                                        | -                                   | 最大训练样本数              |
| `--max_eval_samples`            | str / None | None                                                                                        | -                                   | 最大验证样本数              |
| `--model_max_length`            | str        | `2048`                                                                                      | -                                   | 最大序列长度               |
| `--output_dir`                  | str        | `finetune/models/qwen3-30b-a3b-qlora`                                                       | -                                   | 输出目录                 |
| `--seed`                        | str        | `42`                                                                                        | -                                   | 随机种子                 |
| `--per_device_train_batch_size` | str        | `1`                                                                                         | -                                   | 每设备训练批次大小            |
| `--per_device_eval_batch_size`  | str        | `1`                                                                                         | -                                   | 每设备验证批次大小            |
| `--gradient_accumulation_steps` | str        | `16`                                                                                        | -                                   | 梯度累积步数               |
| `--learning_rate`               | str        | `2e-4`                                                                                      | -                                   | 学习率                  |
| `--num_train_epochs`            | str        | `3`                                                                                         | -                                   | 训练轮数                 |
| `--max_steps`                   | str        | `-1`                                                                                        | -                                   | 最大步数（-1为不限）          |
| `--lora_r`                      | str        | `16`                                                                                        | -                                   | LoRA 秩               |
| `--lora_alpha`                  | str        | `32`                                                                                        | -                                   | LoRA alpha           |
| `--lora_dropout`                | str        | `0.05`                                                                                      | -                                   | LoRA dropout率        |
| `--target_modules`              | str        | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`                                   | -                                   | LoRA 目标模块            |
| `--weight_decay`                | str        | `0.0`                                                                                       | -                                   | 权重衰减                 |
| `--moe_enable`                  | str        | `false`                                                                                     | `true`, `false`                     | 是否启用 MoE             |
| `--moe_lora_scope`              | str        | `expert_only`                                                                               | `expert_only`, `router_only`, `all` | LoRA 注入范围            |
| `--moe_expert_patterns`         | str        | `experts.ffn.(gate_proj\|up_proj\|down_proj),layers.[0-9]+.mlp.experts.[0-9]+.(w1\|w2\|w3)` | -                                   | 专家线性层模式（正则）          |
| `--moe_router_patterns`         | str        | `router.(gate\|dense)`                                                                      | -                                   | 路由/门控层模式（正则）         |
| `--moe_max_experts_lora`        | str        | `-1`                                                                                        | -                                   | 每层最多注入 LoRA 的专家数     |
| `--moe_dry_run`                 | str        | `false`                                                                                     | `true`, `false`                     | 仅打印匹配模块并退出           |
| `--load_precision`              | str        | `fp16`                                                                                      | `int8`, `int4`, `fp16`              | 模型加载精度               |
| `--use_flash_attention_2`       | str        | `false`                                                                                     | `true`, `false`                     | 是否启用 FlashAttention2 |
| `--logging_steps`               | str        | `1`                                                                                         | -                                   | 日志记录步数               |
| `--eval_steps`                  | str        | `50`                                                                                        | -                                   | 验证间隔步数               |
| `--save_steps`                  | str        | `200`                                                                                       | -                                   | 保存模型步数               |
| `--save_total_limit`            | str        | `2`                                                                                         | -                                   | 最多保存数                |
| `--warmup_ratio`                | str        | `0.05`                                                                                      | -                                   | 预热比例                 |
| `--lr_scheduler_type`           | str        | `cosine`                                                                                    | -                                   | 学习率调度器类型             |
| `--resume_from_checkpoint`      | str / None | None                                                                                        | -                                   | 从检查点恢复               |
| `--no-gradient_checkpointing`   | flag       | False                                                                                       | -                                   | 不使用梯度检查点             |
| `--no-merge_and_save`           | flag       | False                                                                                       | -                                   | 不合并并保存模型             |
| `--fp16`                        | str        | `true`                                                                                      | `true`, `false`                     | 是否使用fp16             |
| `--optim`                       | str        | `adamw_torch_fused`                                                                         | -                                   | 优化器                  |
| `--dataloader_pin_memory`       | str        | `false`                                                                                     | `true`, `false`                     | 是否固定数据加载器内存          |
| `--dataloader_num_workers`      | str        | `0`                                                                                         | -                                   | DataLoader 线程数       |
| `--dataloader_prefetch_factor`  | str        | `2`                                                                                         | -                                   | DataLoader 预取因子      |
| `--use_gradient_checkpointing`  | str        | `true`                                                                                      | `true`, `false`, `unsloth`          | 梯度检查点设置              |
| `--full_finetuning`             | str        | `false`                                                                                     | `true`, `false`                     | 是否全量微调               |
---

> 下面是一个4090微调`gpt-oss-20b-unsloth-bnb-4bit`的范例
```bash
python3 run_finetune_oss.py --output_dir /root/autodl-fs/gpt-oss-20b-unsloth-bnb-4bit --local_dir gpt-oss-20b-bnb-4bit --data_path ./training_data_ruozhi.jsonl --eval_data_path ./training_data_ruozhi_eval.jsonl --use_qlora true --lora_dropout 0.05 --num_train_epochs 8 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 2e-5 --lr_scheduler cosine --logging_steps 5 --eval_steps 40 --save_steps 200 --warmup_ratio 0.05 --dataloader_num_workers 16 --fp16 true --use_unsloth true --no-gradient_checkpointing --dataloader_prefetch_factor 4 --load_precision int4
```

