# Extra: Fine-tuning OpenAI OSS Model

## This content has been tested on real hardware (vgpu 32g)

### If you find a bug, please open an issue or contact the author

### ~~It would be even better if you could directly submit a PR fix~~

---

# Please Note

> During testing, using `unsloth/gpt-oss-20b-unsloth-bnb-4bit` seemed to fine-tune and merge successfully.
> However, conversion to GGUF failed.
> When testing `unsloth/gpt-oss-20b`, encountered the error `'GptOssTopKRouter' object has no attribute 'weight'`.
> This appears to be a widespread issue; I’ve found many others encountering it during fine-tuning as well.
> Please give the Unsloth and OpenAI teams some time — they’ll fix it.
> Once updated, I will immediately update this document and the code.

---

### Fine-tuning the OSS Model

> Due to the release timing of OSS, fine-tuning methods for OSS and Qwen do not seem interchangeable.
> Also, it’s best to use the latest versions of `unsloth`, `torch`, `transformers`, etc.

[Here’s Unsloth’s OSS fine-tuning experience](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-%2820B%29-Fine-tuning.ipynb#scrollTo=WQSmUBxXx2r-)

## Quick Fine-tuning Guide

> It’s recommended to use a new virtual environment
> Separate it from your Qwen fine-tuning environment

Please ensure you have:
`torch>=2.8.0` `triton>=3.4.0`
**And! Make sure `unsloth` and `unsloth_zoo` are the latest versions**

> The original requirements.txt only supports unsloth up to version 2025.8.1, which cannot fine-tune OSS.

### Run the following before installing dependencies

```bash
pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" "unsloth[base] @ git+https://github.com/unslothai/unsloth" torchvision bitsandbytes git+https://github.com/huggingface/transformers git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
```

### Install dependencies

```bash
pip install -r requirements_oss.txt
```

## **Note: This model requires OpenAI Harmony format training data for fine-tuning**

Use `chatml_to_harmony.py` to convert ChatML format training data to Harmony format:

```bash
python3 chatml_to_harmony.py --input training_data.jsonl --output training_data_harmony.txt
```

### Download the model

```bash
huggingface-cli download unsloth/gpt-oss-20b-BF16 --local-dir gpt-oss-20b
```

> If you don’t have huggingface-cli, install it first:

```bash
pip install huggingface-hub
```

> If you need a mirror site, run:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

> During testing, using `unsloth/gpt-oss-20b-unsloth-bnb-4bit` seemed to fine-tune and merge successfully.
> However, conversion to GGUF failed.
> When testing `unsloth/gpt-oss-20b`, encountered the error `'GptOssTopKRouter' object has no attribute 'weight'`.
> This appears to be a widespread issue; I’ve found many others encountering it during fine-tuning as well.
> Please give the Unsloth and OpenAI teams some time — they’ll fix it.
> Once updated, I will immediately update this document and the code.

## Start Fine-tuning

```bash
python3 run_finetune_oss.py
```

---

| Parameter                       | Type       | Default Value                                                                               | Optional Values                     | Description                          |
| ------------------------------- | ---------- | ------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------ |
| `--repo_id`                     | str        | `unsloth/gpt-oss-20b-unsloth-bnb-4bit`                                                      | -                                   | HF repository ID                     |
| `--local_dir`                   | str        | `gpt-oss-20b-unsloth-bnb-4bit`                                                              | -                                   | Local model directory                |
| `--use_unsloth`                 | str        | `false`                                                                                     | `true`, `false`                     | Whether to use unsloth               |
| `--use_qlora`                   | str        | `true`                                                                                      | `true`, `false`                     | Whether to use QLoRA                 |
| `--data_path`                   | str        | `training_data.jsonl`                                                                       | -                                   | Training data path                   |
| `--eval_data_path`              | str / None | None                                                                                        | -                                   | Evaluation data path                 |
| `--max_samples`                 | str / None | None                                                                                        | -                                   | Max number of training samples       |
| `--max_eval_samples`            | str / None | None                                                                                        | -                                   | Max number of evaluation samples     |
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
| `--moe_lora_scope`              | str        | `expert_only`                                                                               | `expert_only`, `router_only`, `all` | LoRA injection scope for MoE         |
| `--moe_expert_patterns`         | str        | `experts.ffn.(gate_proj\|up_proj\|down_proj),layers.[0-9]+.mlp.experts.[0-9]+.(w1\|w2\|w3)` | -                                   | Expert linear layer patterns (regex) |
| `--moe_router_patterns`         | str        | `router.(gate\|dense)`                                                                      | -                                   | Router/gating layer patterns (regex) |
| `--moe_max_experts_lora`        | str        | `-1`                                                                                        | -                                   | Max LoRA experts per layer           |
| `--moe_dry_run`                 | str        | `false`                                                                                     | `true`, `false`                     | Only print matched modules and exit  |
| `--load_precision`              | str        | `fp16`                                                                                      | `int8`, `int4`, `fp16`              | Model load precision                 |
| `--use_flash_attention_2`       | str        | `false`                                                                                     | `true`, `false`                     | Enable FlashAttention2               |
| `--logging_steps`               | str        | `1`                                                                                         | -                                   | Logging step interval                |
| `--eval_steps`                  | str        | `50`                                                                                        | -                                   | Evaluation step interval             |
| `--save_steps`                  | str        | `200`                                                                                       | -                                   | Model save step interval             |
| `--save_total_limit`            | str        | `2`                                                                                         | -                                   | Max saved model count                |
| `--warmup_ratio`                | str        | `0.05`                                                                                      | -                                   | Warmup ratio                         |
| `--lr_scheduler_type`           | str        | `cosine`                                                                                    | -                                   | LR scheduler type                    |
| `--resume_from_checkpoint`      | str / None | None                                                                                        | -                                   | Resume from checkpoint               |
| `--no-gradient_checkpointing`   | flag       | False                                                                                       | -                                   | Disable gradient checkpointing       |
| `--no-merge_and_save`           | flag       | False                                                                                       | -                                   | Do not merge and save model          |
| `--fp16`                        | str        | `true`                                                                                      | `true`, `false`                     | Use fp16                             |
| `--optim`                       | str        | `adamw_torch_fused`                                                                         | -                                   | Optimizer                            |
| `--dataloader_pin_memory`       | str        | `false`                                                                                     | `true`, `false`                     | Pin dataloader memory                |
| `--dataloader_num_workers`      | str        | `0`                                                                                         | -                                   | Number of dataloader workers         |
| `--dataloader_prefetch_factor`  | str        | `2`                                                                                         | -                                   | Dataloader prefetch factor           |
| `--use_gradient_checkpointing`  | str        | `true`                                                                                      | `true`, `false`, `unsloth`          | Gradient checkpointing setting       |
| `--full_finetuning`             | str        | `false`                                                                                     | `true`, `false`                     | Enable full fine-tuning              |
| `--data_format`                 | str        | `harmony`                                                                                   | `harmony`, `jsonl`                  | Data format                          |

---

> Below is an example for fine-tuning `gpt-oss-20b-unsloth-bnb-4bit`:

```bash
python3 run_finetune_oss.py --output_dir /root/autodl-fs/gpt-oss-20b-unsloth-bnb-4bit --local_dir gpt-oss-20b-4bit --data_path ./harmony_small.txt --eval_data_path ./harmony_small_eval.txt --use_qlora true --lora_dropout 0.05 --num_train_epochs 8 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 2e-5 --lr_scheduler cosine --logging_steps 5 --eval_steps 40 --save_steps 200 --warmup_ratio 0.05 --dataloader_num_workers 16 --fp16 true --use_unsloth true --no-gradient_checkpointing --dataloader_prefetch_factor 4 --load_precision int4 --data_format harmony
```
