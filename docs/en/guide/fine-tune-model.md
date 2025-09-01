## Fine-tuning the Model

> It is recommended to use a CPU with high single-core performance for fine-tuning.
> Otherwise, there may be a CPU bottleneck (cause not yet identified — PRs to fix are welcome).

## Before You Begin — Environment Setup

> It's simple, don't worry.

```bash
git clone https://github.com/qqqqqf-q/Qing-Digital-Self.git --depth 1
```

Or use a mirror (China mainland acceleration):

```bash
git clone https://hk.gh-proxy.com/https://github.com/qqqqqf-q/Qing-Digital-Self.git  --depth 1
```

# Configure the Environment

```bash
python3 environment/setup_env.py --install
```

Just follow the default process.
Installation includes built-in checks.
You can also use:

```bash
python3 environment/setup_env.py --check
```

To check the environment.

If you encounter issues with Unsloth installation, please install it manually.
First run the following command:

```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```

It will output a pip command — copy it and run it in your shell.
For example:

```bash
pip install --upgrade pip && pip install "unsloth[cu126-ampere-torch270] @ git+https://github.com/unslothai/unsloth.git"
```

If you encounter issues with flash attention installation,
You can try visiting [this GitHub repository](https://github.com/Dao-AILab/flash-attention/releases/)
To find the offline installation package you need (this doesn't require compilation and will be much faster).
Commands are similar to:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'

pip install flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

---

# Now the Actual Fine-tuning

> Parameters can actually be left empty during testing, as defaults are provided.
> By default, it seems to use 8-bit quantization (this needs modification).

Run the fine-tuning script:

```bash
python run_finetune.py
```

### Model-related Parameters (4 columns, scroll to view all)

| Parameter Name                  | Type | Default Value                            | Description                                                 |
| ------------------------------- | ---- | ---------------------------------------- | ----------------------------------------------------------- |
| `--repo_id`                     | str  | `'Qwen/Qwen3-30B-A3B-Instruct-2507'`     | HF repository ID                                            |
| `--local_dir`                   | str  | `'qwen3-30b-a3b-instruct'`               | Local model directory                                       |
| `--use_unsloth`                 | str  | `'false'`                                | Whether to use Unsloth                                      |
| `--use_qlora`                   | str  | `'true'`                                 | Whether to use QLoRA                                        |
| `--data_path`                   | str  | `'training_data.jsonl'`                  | Training data path                                          |
| `--eval_data_path`              | str  | `None`                                   | Evaluation data path                                        |
| `--max_samples`                 | str  | `None`                                   | Maximum number of training samples                          |
| `--max_eval_samples`            | str  | `None`                                   | Maximum number of evaluation samples                        |
| `--model_max_length`            | str  | `'2048'`                                 | Maximum sequence length                                     |
| `--output_dir`                  | str  | `'finetune/models/qwen3-30b-a3b-qlora'`  | Output directory                                            |
| `--seed`                        | str  | `'42'`                                   | Random seed                                                 |
| `--per_device_train_batch_size` | str  | `'1'`                                    | Per-device training batch size                              |
| `--per_device_eval_batch_size`  | str  | `'1'`                                    | Per-device evaluation batch size                            |
| `--gradient_accumulation_steps` | str  | `'16'`                                   | Gradient accumulation steps                                 |
| `--learning_rate`               | str  | `'2e-4'`                                 | Learning rate                                               |
| `--num_train_epochs`            | str  | `'3'`                                    | Number of training epochs                                   |
| `--max_steps`                   | str  | `'-1'`                                   | Maximum steps (-1 means unlimited)                          |
| `--lora_r`                      | str  | `'16'`                                   | LoRA rank                                                   |
| `--lora_alpha`                  | str  | `'32'`                                   | LoRA alpha                                                  |
| `--lora_dropout`                | str  | `'0.05'`                                 | LoRA dropout rate                                           |
| `--target_modules`              | str  | `'too long, check file'`                 | LoRA target modules                                         |
| `--weight_decay`                | str  | `'0.0'`                                  | Weight decay                                                |
| `--moe_enable`                  | str  | `'false'`                                | Whether to enable MoE injection logic                       |
| `--moe_lora_scope`              | str  | `'expert_only'`                          | LoRA injection scope                                        |
| `--moe_expert_patterns`         | str  | `'too long to include here, check file'` | Expert linear layer patterns                                |
| `--moe_router_patterns`         | str  | `'markdown would parse it, check file'`  | Router/gating linear layer patterns                         |
| `--moe_max_experts_lora`        | str  | `'-1'`                                   | Max number of LoRA experts per layer                        |
| `--moe_dry_run`                 | str  | `'false'`                                | Whether to do a dry run only                                |
| `--load_precision`              | str  | `'fp16'`                                 | Model load precision: `int8` / `int4` / `fp16`              |
| `--logging_steps`               | str  | `'1'`                                    | Logging interval (steps)                                    |
| `--eval_steps`                  | str  | `'50'`                                   | Evaluation interval (steps)                                 |
| `--save_steps`                  | str  | `'200'`                                  | Model save interval (steps)                                 |
| `--save_total_limit`            | str  | `'2'`                                    | Maximum number of saved models                              |
| `--warmup_ratio`                | str  | `'0.05'`                                 | Learning rate warmup ratio                                  |
| `--lr_scheduler_type`           | str  | `'cosine'`                               | Learning rate scheduler type                                |
| `--resume_from_checkpoint`      | str  | `None`                                   | Path to resume training from checkpoint                     |
| `--no-gradient_checkpointing`   | flag | `False`                                  | Disable gradient checkpointing (enable by adding this flag) |
| `--no-merge_and_save`           | flag | `False`                                  | Do not merge and save model (enable by adding this flag)    |
| `--fp16`                        | str  | `'true'`                                 | Whether to use fp16                                         |
| `--optim`                       | str  | `'adamw_torch_fused'`                    | Optimizer name                                              |
| `--dataloader_pin_memory`       | str  | `'false'`                                | Whether to pin DataLoader memory                            |
| `--dataloader_num_workers`      | str  | `'0'`                                    | Number of DataLoader workers                                |
| `--dataloader_prefetch_factor`  | str  | `'2'`                                    | DataLoader prefetch factor                                  |
| `--use_flash_attention_2`       | str  | `'false'`                                | Use FlashAttention2 (not effective for Unsloth)             |

---

> The parameters are still quite complex — it’s best to consult an AI for help.
> Below is an example of fine-tuning `qwen2.5-7b-instruct` on an RTX 4090:

```bash
python3 run_finetune.py --output_dir /root/autodl-fs/qwen2.5-7b-qing-v1 --local_dir ./model/Qwen2.5-7B-Instruct --data_path ./dataset/sft.jsonl --use_qlora true --lora_dropout 0.1 --num_train_epochs 8 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 2e-5 --lr_scheduler cosine --logging_steps 5 --eval_steps 40 --save_steps 200 --warmup_ratio 0.05 --dataloader_num_workers 16 --fp16 true --use_unsloth true --no-gradient_checkpointing  --load_precision int8
```

### Evaluation Set Not Working

* Check that `--eval_data_path` is correct.
* Ensure evaluation data format matches training data.
* Look for console output saying “no evaluation data path provided”.

### GPU Out of Memory

* Reduce `--per_device_eval_batch_size`.
* Reduce `--max_eval_samples`.
* Increase the `--eval_steps` interval.

### Dev Notes
```bash
python3 cli.py train start
```
This parameter still seems unusable, with many bugs that need fixing.
