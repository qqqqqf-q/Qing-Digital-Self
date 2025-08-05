## 3. Fine-tune Model

## Before this, you need to configure the environment
> Very simple, don't worry
```bash
git clone https://github.com/qqqqqf-q/Qing-Digital-Self.git --depth 1
```

Activate virtual environment:

* On Linux/Mac:

  ```bash
  source venv/bin/activate
  ```
* On Windows:

  ```bash
  .\venv\Scripts\activate
  ```

* Install dependencies

```bash
pip install -r requirements.txt
```

> PS: ~~I tested for a long time on the dependency step, I don't know why there were a bunch of strange problems before()~~
> But this requirements is the version I tested myself, ~~should be stable~~

---

### If you need the unsloth+torch version provided by Unsloth, please run the following command
```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```
It will output a pip command, please copy it and run it in shell
For example
```bash
pip install --upgrade pip && pip install "unsloth[cu126-ampere-torch270] @ git+https://github.com/unslothai/unsloth.git"
```

# The following is the real fine-tuning

>Parameters actually don't need to be filled during testing, they all have default values

> Seems to default to 8bit quantization, needs modification
* Run fine-tuning script:

  ```bash
  python run_finetune.py
  ```
### Model Related Parameters (table has four columns, please scroll to view)

| Parameter Name                  | Type | Default Value                                | Description                           |
| ------------------------------- | ---- | -------------------------------------------- | ------------------------------------- |
| `--repo_id`                     | str  | `'Qwen/Qwen3-30B-A3B-Instruct-2507'`       | HF repository ID                      |
| `--local_dir`                   | str  | `'qwen3-30b-a3b-instruct'`                  | Local model directory                 |
| `--use_unsloth`                 | str  | `'false'`                                   | Whether to use unsloth                |
| `--use_qlora`                   | str  | `'true'`                                    | Whether to use qlora                  |
| `--data_path`                   | str  | `'training_data.jsonl'`                     | Training data path                    |
| `--eval_data_path`              | str  | `None`                                      | Validation data file path             |
| `--max_samples`                 | str  | `None`                                      | Maximum training samples              |
| `--max_eval_samples`            | str  | `None`                                      | Maximum validation samples            |
| `--model_max_length`            | str  | `'2048'`                                    | Maximum sequence length               |
| `--output_dir`                  | str  | `'finetune/models/qwen3-30b-a3b-qlora'`     | Output directory                      |
| `--seed`                        | str  | `'42'`                                      | Random seed                           |
| `--per_device_train_batch_size` | str  | `'1'`                                       | Training batch size per device        |
| `--per_device_eval_batch_size`  | str  | `'1'`                                       | Validation batch size per device      |
| `--gradient_accumulation_steps` | str  | `'16'`                                      | Gradient accumulation steps           |
| `--learning_rate`               | str  | `'2e-4'`                                    | Learning rate                         |
| `--num_train_epochs`            | str  | `'3'`                                       | Number of training epochs             |
| `--max_steps`                   | str  | `'-1'`                                      | Maximum steps, -1 means no limit     |
| `--lora_r`                      | str  | `'16'`                                      | LoRA rank                             |
| `--lora_alpha`                  | str  | `'32'`                                      | LoRA alpha value                      |
| `--lora_dropout`                | str  | `'0.05'`                                    | LoRA dropout rate                     |
| `--target_modules`              | str  | `'Too long, please check in file'`          | LoRA target modules                   |
| `--weight_decay`                | str  | `'0.0'`                                     | Weight decay                          |
| `--moe_enable`                  | str  | `'false'`                                   | Whether to enable MoE injection logic|
| `--moe_lora_scope`              | str  | `'expert_only'`                             | LoRA injection scope                  |
| `--moe_expert_patterns`         | str  | `'Too long, check in file'`                 | Expert linear layer patterns          |
| `--moe_router_patterns`         | str  | `'Markdown converts, check in file'`        | Router/gate linear layer patterns     |
| `--moe_max_experts_lora`        | str  | `'-1'`                                      | Max experts per layer for LoRA       |
| `--moe_dry_run`                 | str  | `'false'`                                   | Whether it's a Dry-Run                |
| `--load_precision`              | str  | `'fp16'`                                    | Model loading precision: `int8`/`int4`/`fp16` |
| `--logging_steps`               | str  | `'1'`                                       | Logging steps                         |
| `--eval_steps`                  | str  | `'50'`                                      | Evaluation interval steps             |
| `--save_steps`                  | str  | `'200'`                                     | Model saving steps                    |
| `--save_total_limit`            | str  | `'2'`                                       | Maximum number of saved models        |
| `--warmup_ratio`                | str  | `'0.05'`                                    | Learning rate warmup ratio            |
| `--lr_scheduler_type`           | str  | `'cosine'`                                  | Learning rate scheduler type          |
| `--resume_from_checkpoint`      | str  | `None`                                      | Checkpoint path to resume training    |
| `--no-gradient_checkpointing`   | flag | `False`                                     | Don't use gradient checkpointing      |
| `--no-merge_and_save`           | flag | `False`                                     | Don't merge and save model            |
| `--fp16`                        | str  | `'true'`                                    | Whether to use fp16                   |
| `--optim`                       | str  | `'adamw_torch_fused'`                       | Optimizer name                        |
| `--dataloader_pin_memory`       | str  | `'false'`                                   | Whether to pin DataLoader memory      |
| `--dataloader_num_workers`      | str  | `'0'`                                       | DataLoader worker threads             |
| `--dataloader_prefetch_factor`  | str  | `'2'`                                       | DataLoader prefetch factor            |
| `--use_flash_attention_2`       | str  | `'false'`                                   | Whether to use FlashAttention2        |

---
> Parameters are still too complex, suggest asking AI
> Below is an example of fine-tuning `qwen3-8b-base` on 4090
```bash
python3 run_finetune.py --output_dir /root/autodl-fs/qwen3-8b-qing-v4 --local_dir qwen3-8b-base --data_path ./training_data_ruozhi.jsonl --eval_data_path ./training_data_ruozhi_eval.jsonl --use_qlora true --lora_dropout 0.05 --num_train_epochs 8 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 2e-5 --lr_scheduler cosine --logging_steps 5 --eval_steps 40 --save_steps 200 --warmup_ratio 0.05 --dataloader_num_workers 16 --fp16 true --use_unsloth true --no-gradient_checkpointing --dataloader_prefetch_factor 4
```
### Validation Set Not Working
- Check if `--eval_data_path` path is correct
- Confirm validation data file format matches training data
- Check console output for "validation data path not provided" prompt

### GPU Memory Insufficient
- Reduce `--per_device_eval_batch_size`
- Reduce `--max_eval_samples`
- Increase `--eval_steps` interval