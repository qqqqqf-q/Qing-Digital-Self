## 3. 微调模型

> Windows 上 Unsloth 兼容性不好，Linux 上代码有 bug，所以用 `no_unsloth` 版本。<br>
> ~~其实是unsloth版本没写完~~

>参数在测试时其实可以不填,都是有默认值的  

> 似乎是默认8bit量化,有待修改  
* 运行微调脚本：

  ```bash
  python run_finetune.py
  ```
###  模型相关参数

| 参数名                   | 类型   | 默认值                    | 说明                  |
| --------------------- | ---- | ---------------------- | ------------------- |
| `--repo_id`           | str  | `"Qwen/Qwen3-8B-Base"` | 基础模型或 MoE 模型的仓库ID   |
| `--local_dir`         | str  | `"qwen3-8b-base"`      | 本地模型存储目录            |
| `--trust_remote_code` | bool | `True`                 | 是否信任远程代码            |
| `--use_unsloth`       | bool | `False`                | 是否使用 Unsloth 加速     |
| `--use_qlora`         | bool | `True`                 | 是否使用 8bit 量化（QLoRA） |

---

### 数据相关参数

| 参数名                  | 类型         | 默认值                     | 说明                     |
| -------------------- | ---------- | ----------------------- | ---------------------- |
| `--data_path`        | str        | `"training_data.jsonl"` | 训练数据文件路径               |
| `--eval_data_path`   | str / None | `None`                  | 验证数据文件路径，None 表示不使用验证集 |
| `--max_samples`      | int / None | `None`                  | 最大训练样本数，None 表示用全部     |
| `--max_eval_samples` | int / None | `None`                  | 最大验证样本数，None 表示用全部     |
| `--model_max_length` | int        | `2048`                  | 最大序列长度                 |

---

###  训练相关参数

| 参数名            | 类型  | 默认值                                | 说明   |
| -------------- | --- | ---------------------------------- | ---- |
| `--output_dir` | str | `"finetune/models/qwen3-8b-qlora"` | 输出目录 |
| `--seed`       | int | `42`                               | 随机种子 |

---

###  LoRA 参数

| 参数名                | 类型    | 默认值                                                         | 说明                |
| ------------------ | ----- | ----------------------------------------------------------- | ----------------- |
| `--lora_r`         | int   | `16`                                                        | LoRA 秩            |
| `--lora_alpha`     | int   | `32`                                                        | LoRA alpha        |
| `--lora_dropout`   | float | `0.05`                                                      | LoRA dropout      |
| `--target_modules` | str   | `"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"` | LoRA 目标模块列表（逗号分隔） |

---

###  MoE 参数

| 参数名                      | 类型   | 默认值                                                                                         | 说明                                          |            |
| ------------------------ | ---- | ------------------------------------------------------------------------------------------- | ------------------------------------------- | ---------- |
| `--moe_enable`           | bool | `False`                                                                                     | 是否启用 MoE 注入逻辑                               |            |
| `--moe_lora_scope`       | str  | `"expert_only"`                                                                             | LoRA 注入范围：`expert_only`、`router_only`、`all` |            |
| `--moe_expert_patterns`  | str  | `experts.ffn.(gate_proj\|up_proj\|down_proj),layers.[0-9]+.mlp.experts.[0-9]+.(w1\|w2\|w3)` | 专家线性层匹配正则（兼容 Qwen-MoE、Mixtral）              |            |
| `--moe_router_patterns`  | str  | \`"router.(gate                                                                             | dense)"\`                                   | 路由/门控层匹配模式 |
| `--moe_max_experts_lora` | int  | `-1`                                                                                        | 每层最多注入 LoRA 的专家数，`-1` 表示全部                  |            |
| `--moe_dry_run`          | bool | `False`                                                                                     | 仅打印匹配模块，不执行训练                               |            |

---

###  训练超参数

| 参数名                             | 类型    | 默认值        | 说明              |
| ------------------------------- | ----- | ---------- | --------------- |
| `--per_device_train_batch_size` | int   | `1`        | 每卡训练 batch size |
| `--per_device_eval_batch_size`  | int   | `1`        | 每卡验证 batch size |
| `--gradient_accumulation_steps` | int   | `16`       | 梯度累积步数          |
| `--learning_rate`               | float | `2e-4`     | 学习率             |
| `--weight_decay`                | float | `0.0`      | 权重衰减            |
| `--num_train_epochs`            | float | `3.0`      | 训练轮数            |
| `--max_steps`                   | int   | `-1`       | 最大步数，`-1` 表示不限制 |
| `--warmup_ratio`                | float | `0.05`     | 学习率预热比例         |
| `--lr_scheduler_type`           | str   | `"cosine"` | 学习率调度器类型        |

---

###  其他参数

| 参数名                        | 类型   | 默认值                   | 说明                          |
| -------------------------- | ---- | --------------------- | --------------------------- |
| `--logging_steps`          | int  | `1`                   | 日志输出间隔（步）                   |
| `--eval_steps`             | int  | `50`                  | 验证间隔步数                      |
| `--save_steps`             | int  | `200`                 | 模型保存间隔                      |
| `--save_total_limit`       | int  | `2`                   | 最多保存多少个检查点                  |
| `--gradient_checkpointing` | bool | `True`                | 是否使用梯度检查点                   |
| `--merge_and_save`         | bool | `True`                | 是否合并 LoRA 并保存完整模型           |
| `--fp16`                   | bool | `True`                | 是否使用 FP16                   |
| `--optim`                  | str  | `"adamw_torch_fused"` | 优化器                         |
| `--dataloader_pin_memory`  | bool | `False`               | DataLoader 是否使用 pin\_memory |
| `--dataloader_num_workers` | int  | `0`                   | DataLoader 工作线程数            |

---

### 验证集未生效
- 检查`--eval_data_path`路径是否正确
- 确认验证数据文件格式与训练数据一致
- 查看控制台输出是否有"未提供验证数据路径"的提示

### GPU显存不足
- 减小`--per_device_eval_batch_size`
- 减小`--max_eval_samples`
- 增加`--eval_steps`间隔