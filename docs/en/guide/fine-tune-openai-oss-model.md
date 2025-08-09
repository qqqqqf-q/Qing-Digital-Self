# Extra Chapter: Fine-tuning the OpenAI OSS Model

## This guide has **not** been tested in practice (my 3080 doesn’t have enough VRAM to run it)

### If you can test it, please share your results or bugs in the Issues section

### ~~Even better if you can submit a PR fix directly~~

---

> Due to the OSS release timing, fine-tuning OSS and Qwen may not be interchangeable.
> It’s also best to use the latest versions of `unsloth`, `torch`, `transformers`, and related libraries.

[Here’s Unsloth’s OSS fine-tuning tutorial](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-%2820B%29-Fine-tuning.ipynb#scrollTo=WQSmUBxXx2r-)

## Quick Fine-tuning Guide

> It’s recommended to use a **new virtual environment**,
> separate from your Qwen fine-tuning environment.

Make sure you have:
`torch>=2.8.0` `triton>=3.4.0`
**And importantly — ensure `unsloth` and `unsloth_zoo` are up to date.**

> The original `requirements.txt` for unsloth only supports up to version 2025.8.1, which does not work for OSS fine-tuning.

```bash
pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" "unsloth[base] @ git+https://github.com/unslothai/unsloth" torchvision bitsandbytes git+https://github.com/huggingface/transformers git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
```

## Start Fine-tuning

```bash
python3 run_finetune_oss.py
```

---

| Parameter                       | Type       | Default                                                                                     | Options                             | Description                          |
| ------------------------------- | ---------- | ------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------ |
| `--repo_id`                     | str        | `unsloth/gpt-oss-20b-unsloth-bnb-4bit`                                                      | -                                   | Hugging Face repository ID           |
| `--local_dir`                   | str        | `gpt-oss-20b-unsloth-bnb-4bit`                                                              | -                                   | Local model directory                |
| `--use_unsloth`                 | str        | `false`                                                                                     | `true`, `false`                     | Whether to use Unsloth               |
| `--use_qlora`                   | str        | `true`                                                                                      | `true`, `false`                     | Whether to use QLoRA                 |
| `--data_path`                   | str        | `training_data.jsonl`                                                                       | -                                   | Training data path                   |
| `--eval_data_path`              | str / None | None                                                                                        | -                                   | Evaluation data path                 |
| `--max_samples`                 | str / None | None                                                                                        | -                                   | Maximum number of training samples   |
| `--max_eval_samples`            | str / None | None                                                                                        | -                                   | Maximum number of evaluation samples |
| `--model_max_length`            | str        | `2048`                                                                                      | -                                   | Max sequence length                  |
| `--output_dir`                  | str        | `finetune/models/qwen3-30b-a3b-qlora`                                                       | -                                   | Output directory                     |
| `--seed`                        | str        | `42`                                                                                        | -                                   | Random seed                          |
| `--per_device_train_batch_size` | str        | `1`                                                                                         | -                                   | Training batch size per device       |
| `--per_device_eval_batch_size`  | str        | `1`                                                                                         | -                                   | Evaluation batch size per device     |
| `--gradient_accumulation_steps` | str        | `16`                                                                                        | -                                   | Gradient accumulation steps          |
| `--learning_rate`               | str        | `2e-4`                                                                                      | -                                   | Learning rate                        |
| `--num_train_epochs`            | str        | `3`                                                                                         | -                                   | Number of training epochs            |
| `--max_steps`                   | str        | `-1`                                                                                        | -                                   | Max steps (-1 for unlimited)         |
| `--lora_r`                      | str        | `16`                                                                                        | -                                   | LoRA rank                            |
| `--lora_alpha`                  | str        | `32`                                                                                        | -                                   | LoRA alpha                           |
| `--lora_dropout`                | str        | `0.05`                                                                                      | -                                   | LoRA dropout rate                    |
| `--target_modules`              | str        | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`                                   | -                                   | LoRA target modules                  |
| `--weight_decay`                | str        | `0.0`                                                                                       | -                                   | Weight decay                         |
| `--moe_enable`                  | str        | `false`                                                                                     | `true`, `false`                     | Enable MoE                           |
| `--moe_lora_scope`              | str        | `expert_only`                                                                               | `expert_only`, `router_only`, `all` | LoRA injection scope                 |
| `--moe_expert_patterns`         | str        | `experts.ffn.(gate_proj\|up_proj\|down_proj),layers.[0-9]+.mlp.experts.[0-9]+.(w1\|w2\|w3)` | -                                   | Expert linear layer regex patterns   |
| `--moe_router_patterns`         | str        | `router.(gate\|dense)`                                                                      | -                                   | Router/gating layer regex patterns   |
| `--moe_max_experts_lora`        | str        | `-1`                                                                                        | -                                   | Max LoRA-injected experts per layer  |
| `--moe_dry_run`                 | str        | `false`                                                                                     | `true`, `false`                     | Print matched modules and exit       |
| `--load_precision`              | str        | `fp16`                                                                                      | `int8`, `int4`, `fp16`              | Model load precision                 |
| `--use_flash_attention_2`       | str        | `false`                                                                                     | `true`, `false`                     | Enable FlashAttention 2              |
| `--logging_steps`               | str        | `1`                                                                                         | -                                   | Logging interval in steps            |
| `--eval_steps`                  | str        | `50`                                                                                        | -                                   | Evaluation interval in steps         |
| `--save_steps`                  | str        | `200`                                                                                       | -                                   | Model save interval in steps         |
| `--save_total_limit`            | str        | `2`                                                                                         | -                                   | Max number of saved models           |
| `--warmup_ratio`                | str        | `0.05`                                                                                      | -                                   | Warmup ratio                         |
| `--lr_scheduler_type`           | str        | `cosine`                                                                                    | -                                   | Learning rate scheduler type         |
| `--resume_from_checkpoint`      | str / None | None                                                                                        | -                                   | Resume from checkpoint               |
| `--no-gradient_checkpointing`   | flag       | False                                                                                       | -                                   | Disable gradient checkpointing       |
| `--no-merge_and_save`           | flag       | False                                                                                       | -                                   | Skip merge and save                  |
| `--fp16`                        | str        | `true`                                                                                      | `true`, `false`                     | Use fp16                             |
| `--optim`                       | str        | `adamw_torch_fused`                                                                         | -                                   | Optimizer                            |
| `--dataloader_pin_memory`       | str        | `false`                                                                                     | `true`, `false`                     | Pin DataLoader memory                |
| `--dataloader_num_workers`      | str        | `0`                                                                                         | -                                   | DataLoader workers                   |
| `--dataloader_prefetch_factor`  | str        | `2`                                                                                         | -                                   | DataLoader prefetch factor           |
| `--use_gradient_checkpointing`  | str        | `true`                                                                                      | `true`, `false`, `unsloth`          | Gradient checkpointing mode          |
| `--full_finetuning`             | str        | `false`                                                                                     | `true`, `false`                     | Enable full fine-tuning              |

---

> Example: Fine-tuning `gpt-oss-20b-unsloth-bnb-4bit` on an RTX 4090:

```bash
python3 run_finetune_oss.py --output_dir /root/autodl-fs/gpt-oss-20b-unsloth-bnb-4bit --local_dir gpt-oss-20b-bnb-4bit --data_path ./training_data_ruozhi.jsonl --eval_data_path ./training_data_ruozhi_eval.jsonl --use_qlora true --lora_dropout 0.05 --num_train_epochs 8 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 2e-5 --lr_scheduler cosine --logging_steps 5 --eval_steps 40 --save_steps 200 --warmup_ratio 0.05 --dataloader_num_workers 16 --fp16 true --use_unsloth true --no-gradient_checkpointing --dataloader_prefetch_factor 4 --load_precision int4
```

