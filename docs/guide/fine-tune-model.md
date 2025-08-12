##  微调模型
> 建议使用高单核的CPU进行微调  
> 不然可能存在CPU瓶颈(暂时没找到问题所在,欢迎PR修复)  
## 在此之前,你需要配置环境
> 很简单,不必担心
```bash
git clone https://github.com/qqqqqf-q/Qing-Digital-Self.git --depth 1
```

创建虚拟环境
```bash
python3 -m venv venv
```


  激活虚拟环境：

  * 在Linux/Mac上：

    ```bash
    source venv/bin/activate
    ```
  * 在Windows上：

    ```bash
    .\venv\Scripts\activate
    ```

* 安装依赖

```bash
pip install -r requirements.txt
```

> PS:~~我在依赖这一步测试了很久很久,之前不知道为什么就是有一堆奇怪的问题()~~  
> 但是这个requirements是我自己测试出来的版本,~~应该是稳定的吧~~

---

### 如果你需要Unsloth提供的unsloth+torch的版本,请运行以下命令
```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```
它会输出一个pip命令,请复制下来并在shell里运行
例如
```bash
pip install --upgrade pip && pip install "unsloth[cu126-ampere-torch270] @ git+https://github.com/unslothai/unsloth.git"
```

# 以下才是真正的微调

>参数在测试时其实可以不填,都是有默认值的  

> 似乎是默认8bit量化,有待修改  
* 运行微调脚本：

  ```bash
  python run_finetune.py
  ```
###  模型相关参数(列表有四列,请滚动查看)

| 参数名                             | 类型   | 默认值                                                         | 说明                              |            
| ------------------------------- | ---- | --------------------------------------------- | ------------------------------- | 
| `--repo_id`                     | str  | `'Qwen/Qwen3-30B-A3B-Instruct-2507'`                        | HF 仓库ID                         |            
| `--local_dir`                   | str  | `'qwen3-30b-a3b-instruct'`                                  | 本地模型目录                          |            
| `--use_unsloth`                 | str  | `'false'`                                                   | 是否使用 unsloth                    |            
| `--use_qlora`                   | str  | `'true'`                                                    | 是否使用 qlora                      |            
| `--data_path`                   | str  | `'training_data.jsonl'`                                     | 训练数据路径                          |            
| `--eval_data_path`              | str  | `None`                                                      | 验证数据文件路径                        |            
| `--max_samples`                 | str  | `None`                                                      | 最大训练样本数                         |            
| `--max_eval_samples`            | str  | `None`                                                      | 最大验证样本数                         |            
| `--model_max_length`            | str  | `'2048'`                                                    | 最大序列长度                          |            
| `--output_dir`                  | str  | `'finetune/models/qwen3-30b-a3b-qlora'`                     | 输出目录                            |            
| `--seed`                        | str  | `'42'`                                                      | 随机种子                            |            
| `--per_device_train_batch_size` | str  | `'1'`                                                       | 每设备训练批次大小                       |           
| `--per_device_eval_batch_size`  | str  | `'1'`                                                       | 每设备验证批次大小                       |            
| `--gradient_accumulation_steps` | str  | `'16'`                                                      | 梯度累积步数                          |            
| `--learning_rate`               | str  | `'2e-4'`                                                    | 学习率                             |            
| `--num_train_epochs`            | str  | `'3'`                                                       | 训练轮数                            |            
| `--max_steps`                   | str  | `'-1'`                                                      | 最大步数，-1表示不限制                    |            
| `--lora_r`                      | str  | `'16'`                                                      | LoRA 秩                          |            
| `--lora_alpha`                  | str  | `'32'`                                                      | LoRA alpha 值                    |           
| `--lora_dropout`                | str  | `'0.05'`                                                    | LoRA dropout 率                  |            
| `--target_modules`              | str  | `'太长了请进文件查看'`                                         | LoRA 目标模块                       |            
| `--weight_decay`                | str  | `'0.0'`                                                     | 权重衰减                            |            
| `--moe_enable`                  | str  | `'false'`                                                   | 是否启用 MoE 注入逻辑                   |            
| `--moe_lora_scope`              | str  | `'expert_only'`                                             | LoRA 注入范围                       |            
| `--moe_expert_patterns`         | str  | `'太长了写不下,去文件里看'`                                     | 专家线性层模式                         |            
| `--moe_router_patterns`         | str  | `'markdown会转译,也去文件里看'`                                | 路由/门控线性层模式                     |             
| `--moe_max_experts_lora`        | str  | `'-1'`                                                      | 每层注入 LoRA 的专家数上限                |            
| `--moe_dry_run`                 | str  | `'false'`                                                   | 是否为 Dry-Run                     |            
| `--load_precision`              | str  | `'fp16'`                                                    | 模型加载精度：`int8` / `int4` / `fp16` |            
| `--logging_steps`               | str  | `'1'`                                                       | 日志记录步数                          |            
| `--eval_steps`                  | str  | `'50'`                                                      | 验证间隔步数                          |            
| `--save_steps`                  | str  | `'200'`                                                     | 保存模型步数                          |            
| `--save_total_limit`            | str  | `'2'`                                                       | 最多保存模型数量                        |           
| `--warmup_ratio`                | str  | `'0.05'`                                                    | 学习率预热比例                         |            
| `--lr_scheduler_type`           | str  | `'cosine'`                                                  | 学习率调度器类型                        |            
| `--resume_from_checkpoint`      | str  | `None`                                                      | 恢复训练的检查点路径                      |            
| `--no-gradient_checkpointing`   | flag | `False`                                                     | 不使用梯度检查点（使用时加此参数）               |            
| `--no-merge_and_save`           | flag | `False`                                                     | 不合并并保存模型（使用时加此参数）               |            
| `--fp16`                        | str  | `'true'`                                                    | 是否使用 fp16                       |            
| `--optim`                       | str  | `'adamw_torch_fused'`                                       | 优化器名称                           |            
| `--dataloader_pin_memory`       | str  | `'false'`                                                   | 是否固定 DataLoader 内存              |            
| `--dataloader_num_workers`      | str  | `'0'`                                                       | DataLoader 工作线程数                |            
| `--dataloader_prefetch_factor`  | str  | `'2'`                                                       | DataLoader 预取因子                 |            
| `--use_flash_attention_2`       | str  | `'false'`                                                   | 是否使用 FlashAttention2(对unsloth无效) （使用时加此参数）           |            

---
> 参数还是太复杂了,建议询问AI  
> 下面是一个4090微调`qwen3-8b-base`的范例
```bash
python3 run_finetune.py --output_dir /root/autodl-fs/qwen3-8b-qing-v4 --local_dir qwen3-8b-base --data_path ./training_data_ruozhi.jsonl --eval_data_path ./training_data_ruozhi_eval.jsonl --use_qlora true --lora_dropout 0.05 --num_train_epochs 8 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 2e-5 --lr_scheduler cosine --logging_steps 5 --eval_steps 40 --save_steps 200 --warmup_ratio 0.05 --dataloader_num_workers 16 --fp16 true --use_unsloth true --no-gradient_checkpointing --dataloader_prefetch_factor 4
```
### 验证集未生效
- 检查`--eval_data_path`路径是否正确
- 确认验证数据文件格式与训练数据一致
- 查看控制台输出是否有"未提供验证数据路径"的提示

### GPU显存不足
- 减小`--per_device_eval_batch_size`
- 减小`--max_eval_samples`
- 增加`--eval_steps`间隔