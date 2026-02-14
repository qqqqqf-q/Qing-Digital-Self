# 微调模型
## 克隆镜像
```bash
git clone https://github.com/qqqqqf-q/MirrorFlow.git --depth 1
```
或使用镜像(中国大陆加速)
```bash
git clone https://hk.gh-proxy.com/https://github.com/qqqqqf-q/MirrorFlow.git  --depth 1
```

## 配置环境
### 使用`Screen`防止终端中断
```bash
# 创建会话
screen -S finetune
# 回到会话
screen -r finetune
# 查看所有screen会话
screen -ls

# 如果screen未安装请运行
sudo apt install screen
```

### 虚拟环境
```bash
pip install uv
uv venv
source .venv/bin/activate
```
### 安装项目依赖
```bash
uv pip install -r "requirements.txt"
```


### 安装`Llama Factory`
```bash
uv pip install -U llamafactory
```
正常情况应自动安装`CUDA版本Torch`
若未安装请使用以下命令
```bash
uv pip install torch torchvision
```
> 照常应该会自动下载CUDA加速版本,如果显示为CPU版请前往[torch官网](https://pytorch.org/get-started/locally/)寻找下载命令  

`Llama Factory`完成安装后  
可以通过使用` llamafactory-cli version `来快速校验安装是否成功  
若等待一段时间后有llamafactory字样则为安装成功 

### 若有多卡训练需求  
请安装`DeepSpeed`
```bash
uv pip install deepspeed
```

### 若需更多需求  
请前往[Llama Factory 文档](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html#extra-dependency)安装对应依赖

---

# 准备模型和配置
### 参数配置
```bash
cp setting_template.jsonc setting.jsonc
```
使用`nano`或`vim`编辑器编辑`setting.jsonc`

---
在设置完`setting.jsonc`中的模型参数后使用CLI的download不需要附加`--model-repo`等  参数

> 支持从ModelScope和HuggingFace下载模型：

> 如果需要Huggingface镜像站请先运行`export HF_ENDPOINT=https://hf-mirror.com`
### 推荐使用 CLI 模型管理命令

```bash
# 下载模型 - 使用配置文件中的设置（推荐）
python3 cli.py model download

# 下载模型 - 指定参数
python3 cli.py model download \
  --model-repo Qwen/Qwen-3-8B-Base \
  --model-path ./model/Qwen-3-8B-Base \
  --download-source modelscope
# download-source 可选 modelscope 或者 huggingface

# 列出已下载的模型
python3 cli.py model list

# 查看所有模型的详细信息
python3 cli.py model info

# 查看指定模型的详细信息
python3 cli.py model info ./model/Qwen2.5-7B-Instruct
```

---

## 微调

* 运行Llama Factory WebUI：

```bash
python cli.py train webui start --share --no-browser
```

### 模型参数解释请参考[Llama Factory 文档](https://llamafactory.readthedocs.io/zh-cn/latest/advanced/arguments.html)  

---
程序运行后  
默认应会有一个`public URL`  
请复制到浏览器打开
> 若出现`frpc`报错请根据指引安装`frpc`  
---
进入网页后请往下翻找到`配置路径`,选择`finetune-config.yaml`  
并点击`载入训练参数`  
将会加载你的`setting.jsonc`配置

剩下的模型训练参数均可在`Web UI`中配置

---

## 补充
可以在`pip`命令后增加这些内地源加速
```bash
-i https://mirrors.cloud.tencent.com/pypi/simple/
-i https://repo.huaweicloud.com/repository/pypi/simple/
-i http://mirrors.aliyun.com/pypi/simple/
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

---
---
---
# 以下是旧版本(使用自制脚本进行Qlora+Unsloth)微调的指南
`````
##  微调模型
> 建议使用高单核的CPU进行微调  
> 不然可能存在CPU瓶颈(暂时没找到问题所在,欢迎PR修复)  
## 在此之前,你需要配置环境
> 很简单,不必担心
```bash
git clone https://github.com/qqqqqf-q/MirrorFlow.git --depth 1
```
或使用镜像(中国大陆加速)
```bash
git clone https://hk.gh-proxy.com/https://github.com/qqqqqf-q/MirrorFlow.git  --depth 1
```

# 配置环境
```bash
python3 environment/setup_env.py --install 
```
默认跟着流程走就好
安装完自带检查
也可以使用
```bash
python3 environment/setup_env.py --check
``` 
来进行检查环境
如果遇到unsloth无法安装请自行安装
先运行以下命令
```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```
它会输出一个pip命令,请复制下来并在shell里运行 例如

```bash
pip install --upgrade pip && pip install "unsloth[cu126-ampere-torch270] @ git+https://github.com/unslothai/unsloth.git"
```
如果遇到flah attn安装问题
可以尝试前往[此Github仓库](https://github.com/Dao-AILab/flash-attention/releases/)
来查看你需要的离线安装包(这个不需要编译,会快非常多)
命令类似:
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'

pip install flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
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
> 下面是一个4090微调`qwen2.5-7b-instruct`的范例
```bash
python3 run_finetune.py --output_dir /root/autodl-fs/qwen2.5-7b-qing-v1 --local_dir ./model/Qwen2.5-7B-Instruct --data_path ./runs/chat/<run_id>/sft/train.jsonl --use_qlora true --lora_dropout 0.1 --num_train_epochs 8 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 2e-5 --lr_scheduler cosine --logging_steps 5 --eval_steps 40 --save_steps 200 --warmup_ratio 0.05 --dataloader_num_workers 16 --fp16 true --use_unsloth true --no-gradient_checkpointing  --load_precision int8
```
### 验证集未生效
- 检查`--eval_data_path`路径是否正确
- 确认验证数据文件格式与训练数据一致
- 查看控制台输出是否有"未提供验证数据路径"的提示

### GPU显存不足
- 减小`--per_device_eval_batch_size`
- 减小`--max_eval_samples`
- 增加`--eval_steps`间隔

### Dev注
```bash
python3 cli.py train start
```
此参数似乎还是不太能使用的样子,有很多Bug,有待修改
`````
