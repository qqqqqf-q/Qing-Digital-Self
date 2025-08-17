# Extra: Fine-tuning OpenAI OSS Model

## This has been tested on real hardware (vgpu 32G)

### If you find bugs, please open an issue or contact the author

### ~~Even better if you submit a PR fix~~

---

# Do not attempt to fine-tune OSS models for now

# Based on testing, there are still many fine-tuning bugs

# ~~If you’re a brave soul, ignore this warning~~

<img src="https://cdn.nodeimage.com/i/yHMIuFusDfJkDupyVyEdKNE1fUBiDy4C.png" alt="yHMIuFusDfJkDupyVyEdKNE1fUBiDy4C.png">

> After long testing, still unresolved. Contributions welcome: submit PRs to this project or open issues with Unsloth.

---

### Fine-tuning OSS Models

> Because of OSS’s release timing, fine-tuning OSS is not interchangeable with Qwen.
> It’s best to use the latest versions of `unsloth`, `torch`, `transformers`, etc.

[Here is Unsloth’s OSS fine-tuning notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-%2820B%29-Fine-tuning.ipynb)

---

## Environment Setup

> It’s recommended to create a fresh virtual environment,
> separated from your Qwen fine-tuning environment.
> The `unsloth` in the original requirements.txt only supports up to 2025.8.1 and cannot fine-tune OSS.

### Run this command before installing dependencies

```bash
pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" "unsloth[base] @ git+https://github.com/unslothai/unsloth" torchvision bitsandbytes git+https://github.com/huggingface/transformers git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
```

### Install dependencies

```bash
pip install -r requirements_oss.txt
```

---

## **Important: This model requires training data in OpenAI Harmony format**

Convert your ChatML-format training data into Harmony format using:

```bash
python3 chatml_to_harmony.py --input training_data.jsonl --output training_data_harmony.txt
```

---

### Download the model

```bash
huggingface-cli download unsloth/gpt-oss-20b-unsloth-bnb-4bit --local-dir gpt-oss-20b
```

> If you don’t have `huggingface-cli`, install it:

```bash
pip install huggingface-hub
```

> For mirror sites:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Start Fine-tuning

```bash
python3 run_finetune_oss.py
```

---

| Parameter                       | Type     | Default Value                                                                               | Options                             | Description                    |
| ------------------------------- | -------- | ------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------ |
| `--repo_id`                     | str      | `unsloth/gpt-oss-20b-unsloth-bnb-4bit`                                                      | -                                   | HF repo ID                     |
| `--local_dir`                   | str      | `gpt-oss-20b-unsloth-bnb-4bit`                                                              | -                                   | Local model directory          |
| `--use_unsloth`                 | str      | `false`                                                                                     | `true`, `false`                     | Whether to use Unsloth         |
| `--use_qlora`                   | str      | `true`                                                                                      | `true`, `false`                     | Whether to use QLoRA           |
| `--data_path`                   | str      | `training_data.jsonl`                                                                       | -                                   | Training data path             |
| `--eval_data_path`              | str/None | None                                                                                        | -                                   | Evaluation data path           |
| `--max_samples`                 | str/None | None                                                                                        | -                                   | Maximum training samples       |
| `--max_eval_samples`            | str/None | None                                                                                        | -                                   | Maximum evaluation samples     |
| `--model_max_length`            | str      | `2048`                                                                                      | -                                   | Max sequence length            |
| `--output_dir`                  | str      | `finetune/models/qwen3-30b-a3b-qlora`                                                       | -                                   | Output directory               |
| `--seed`                        | str      | `42`                                                                                        | -                                   | Random seed                    |
| `--per_device_train_batch_size` | str      | `1`                                                                                         | -                                   | Train batch size per device    |
| `--per_device_eval_batch_size`  | str      | `1`                                                                                         | -                                   | Eval batch size per device     |
| `--gradient_accumulation_steps` | str      | `16`                                                                                        | -                                   | Gradient accumulation steps    |
| `--learning_rate`               | str      | `2e-4`                                                                                      | -                                   | Learning rate                  |
| `--num_train_epochs`            | str      | `3`                                                                                         | -                                   | Number of training epochs      |
| `--max_steps`                   | str      | `-1`                                                                                        | -                                   | Max steps (-1 = unlimited)     |
| `--lora_r`                      | str      | `16`                                                                                        | -                                   | LoRA rank                      |
| `--lora_alpha`                  | str      | `32`                                                                                        | -                                   | LoRA alpha                     |
| `--lora_dropout`                | str      | `0.05`                                                                                      | -                                   | LoRA dropout rate              |
| `--target_modules`              | str      | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`                                   | -                                   | LoRA target modules            |
| `--weight_decay`                | str      | `0.0`                                                                                       | -                                   | Weight decay                   |
| `--moe_enable`                  | str      | `false`                                                                                     | `true`, `false`                     | Enable MoE                     |
| `--moe_lora_scope`              | str      | `expert_only`                                                                               | `expert_only`, `router_only`, `all` | LoRA injection scope           |
| `--moe_expert_patterns`         | str      | `experts.ffn.(gate_proj\|up_proj\|down_proj),layers.[0-9]+.mlp.experts.[0-9]+.(w1\|w2\|w3)` | -                                   | Expert linear layer regex      |
| `--moe_router_patterns`         | str      | `router.(gate\|dense)`                                                                      | -                                   | Router/gate layer regex        |
| `--moe_max_experts_lora`        | str      | `-1`                                                                                        | -                                   | Max LoRA experts per layer     |
| `--moe_dry_run`                 | str      | `false`                                                                                     | `true`, `false`                     | Dry-run (only print matches)   |
| `--load_precision`              | str      | `fp16`                                                                                      | `int8`, `int4`, `fp16`              | Model load precision           |
| `--use_flash_attention_2`       | str      | `false`                                                                                     | `true`, `false`                     | Enable FlashAttention2         |
| `--logging_steps`               | str      | `1`                                                                                         | -                                   | Logging steps                  |
| `--eval_steps`                  | str      | `50`                                                                                        | -                                   | Eval interval                  |
| `--save_steps`                  | str      | `200`                                                                                       | -                                   | Save checkpoint steps          |
| `--save_total_limit`            | str      | `2`                                                                                         | -                                   | Max checkpoints to keep        |
| `--warmup_ratio`                | str      | `0.05`                                                                                      | -                                   | Warmup ratio                   |
| `--lr_scheduler_type`           | str      | `cosine`                                                                                    | -                                   | LR scheduler type              |
| `--resume_from_checkpoint`      | str/None | None                                                                                        | -                                   | Resume from checkpoint         |
| `--no-gradient_checkpointing`   | flag     | False                                                                                       | -                                   | Disable gradient checkpointing |
| `--no-merge_and_save`           | flag     | False                                                                                       | -                                   | Skip merge and save            |
| `--fp16`                        | str      | `true`                                                                                      | `true`, `false`                     | Use FP16                       |
| `--optim`                       | str      | `adamw_torch_fused`                                                                         | -                                   | Optimizer                      |
| `--dataloader_pin_memory`       | str      | `false`                                                                                     | `true`, `false`                     | Pin dataloader memory          |
| `--dataloader_num_workers`      | str      | `0`                                                                                         | -                                   | Dataloader workers             |
| `--dataloader_prefetch_factor`  | str      | `2`                                                                                         | -                                   | Dataloader prefetch factor     |
| `--use_gradient_checkpointing`  | str      | `true`                                                                                      | `true`, `false`, `unsloth`          | Gradient checkpointing mode    |
| `--full_finetuning`             | str      | `false`                                                                                     | `true`, `false`                     | Full model fine-tuning         |
| `--data_format`                 | str      | `harmony`                                                                                   | `harmony`, `jsonl`                  | Data format                    |

---

> Example fine-tuning run for `gpt-oss-20b-unsloth-bnb-4bit` (adjust as needed):

```bash
python3 run_finetune_oss.py --output_dir /root/autodl-fs/gpt-oss-20b --local_dir /root/autodl-tmp/gpt-oss-20b --data_path ./harmony_small.txt --eval_data_path ./harmony_small_eval.txt --use_qlora true --lora_dropout 0.05 --num_train_epochs 2 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 2e-5 --lr_scheduler cosine --logging_steps 5 --eval_steps 40 --save_steps 200 --warmup_ratio 0.05 --dataloader_num_workers 16 --fp16 true --use_unsloth true --dataloader_prefetch_factor 4 --load_precision int4 --data_format harmony --save_gguf true --gguf_quantization f16
```
