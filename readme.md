## 项目简介

这是一个数字分身项目，核心思想是**利用 QQ 的 C2C 聊天记录作为数据集，对大模型进行微调**，让模型尽可能还原你独有的表达风格和聊天方式。
<p align="center">
  <img src="https://cdn.nodeimage.com/i/BTlRBmAcnwN2ZLyItDCfIRJKYmxYDEpc.png" height="36" alt="Downloads">
  <img src="https://cdn.nodeimage.com/i/nHNuyUdkph6NfGBmnUQeEk8gPVHKLXg0.png" height="36" alt="Stars">
  <img src="https://cdn.nodeimage.com/i/Sd89d1w0xIhSPyPyiVYEfNRTiom8TH1S.png" height="36" alt="Status: MVP">
  <img src="https://cdn.nodeimage.com/i/u0r9K3XXnxU6hDIOMY4fkZ7cnVL28EGF.png" height="36" alt="Version: v0.1">
  <img src="https://cdn.nodeimage.com/i/CfA8AQa2bVTF2mhOY3m2Kz8nhlwXUN6S.png" height="36" alt="License: Apache-2.0">
</p> 

 ## 项目包含了**完整的教程**，包括：

* QQ 数据库的解密与处理
* 聊天数据清洗与转换
* QLora 微调流程
* 微调模型的测试与使用

我知道类似的项目其实已经有不少了，但也许我的教程、流程、代码实现能给你一些不一样的帮助或启发。如果对你有用，欢迎点个 star，我会很开心的！

目前这个项目还有很多不足：

* 只支持最基础的 QLora 微调
* 暂时不支持 unsloth 或更高级的加速/量化技巧
* 但已经可以在 4090 24G 显卡上用 fp8 精度微调 Qwen3-8B（亲测可用）

**如果你也想打造属于自己的数字分身，那也来试试吧!**

——
X: [@qqqqqf5](https://twitter.com/qqqqqf5)

---

## 项目版本
# V 0.1.1(MVP)
## ~~警告~~ 喜报
* 此版本的Qlora_qwen3.py已经过4090实机测试(generate_training_data_llm.py+run_finetune_no_unsloth.py)
* 清洗数据也已经进行实机测试(当前版本)
## TODO
* 增加unsloth支持(重要!可以加快微调速度)
* 增加对MoE模型的支持  
>  ↑ ↑ ↑  2025/8/3更新,或许支持了, 我的显卡跑不动30b a3b,所以还是没法测试  
> 4090的机子塞不下这56.8g的模型,我还是不测了罢(  
* 为数据清洗增加LLM清洗功能(让成熟的llm来清洗数据,比直接使用算法好得多)
> ↑ ↑ ↑  2025/8/3更新,已增加支持,或许不够完善
* 将qlora_qwen3.py的print全部改成logger(?这个很简单,我只是因为怕改多了没法测试(4090到期了))  

## 更新日志
> 写在commit里了,这里实在不想写
---
# 使用QQ聊天记录微调LLM全流程指南

## 1. 获取 QQ 聊天数据

* 教程请参考：[NTQQ Windows 数据解密](https://qq.sbcnm.top/decrypt/NTQQ%20%28Windows%29.html)
* 补充资料：[数据库解码参考](https://qq.sbcnm.top/decrypt/decode_db.html)
* 上面这两个是同一个教程的不同章节,耐心看完就好,不复杂(如果不会可以翻到最底下找我哦)
* 使用 DB Browser for SQLite，密码填写你获取到的 16 位密钥
* HMAC 算法一般为SHA1，也有人是SHA512和256,自行测试,算法错误了会打不开数据库（所以需要测试到打开为之,也可以用 AI 帮你适配）
* 在 DB Browser 里**导出 `c2c_msg_table` 的 SQL**
* 新建数据库，**导入刚才导出的 SQL 文件**
* 获得一个这样的数据库
* [点击查看图片](https://cdn.nodeimage.com/i/oBfbWfVLhJI0CeZHTwwxq6G7XGO40Vy4.webp),并且是明文数据库(你能打开并且能看到数据就是正常的)

---
## 1.5 .将仓库clone到本地并配置环境
* 运行命令:
 ```bash
   git clone https://github.com/qqqqqf-q/Qing-Digital-Self.git --depth 1
 ```
* 进入仓库:
 ```
 cd Qing-Digital-Self
 ```
 * 配置依赖
```
pip install -r requirements.txt
```
---

## 2.1 .清洗数据(普通版本,llm清洗版本在下一章节)
> 此方法比llm清洗快得多,30w条消息半分钟就好了
> 但是对应的质量也更低
> 这个部分建议在windows上优化完再上传至GPU服务器  
> 不确定在Linux上有没有兼容性问题  
* 在 `.env` 文件中修改数据库路径及相关参数(请注意其中的必填段)
* 运行清洗脚本：

  ```bash
  python generate_training_data.py
  ```

## 2.2 .清洗数据(llm清洗)
> 需要配置一个OpenAI兼容的API  
> 比如:LM Studio 或者 vLLM(速度更快,但搭建更麻烦,需要Linux环境)  

> 这个部分同样建议在windows上优化完再上传至GPU服务器  
> 不确定在Linux上有没有兼容性问题
## LM Studio搭建教程
* 1.前往[LM Studio](https://lmstudio.ai/)下载LM Studio
* 2.安装LM Studio
* 3.打开LM Studio,点击左侧`搜索`->`Model Search`
* 4.搜索 `qwen3 8b`->`Complete Download`  
* 5.选择合适你的量化版本**建议至少Q4,最好Q6-Q8,随你的设备情况而定,不知道的可以问AI**
* 记住你的**模型名称**,填写到`.env`文件的`Openai_model`中
* 如果不知道你的模型名称可以运行test_openai.py,会输出所有的模型名称
* 6.安装好后,在左侧`开发者/Developer`点击`Status:Stopped`右边的按钮
* 如果下面log显示端口被占用请点击`seetings`换个`server port`
* 记住这个`server port`,将你的配置填写至`.env`文件中

### run!
```bash
python generate_training_data_llm.py
```
> 如果遇到了400报错大概率是因为message太大了被模型框架拒绝了

---

## vLLM搭建
> vLLM需要linux环境!  
> 如果你的显卡还算可以(>6800xt,>3080)  
> 可以选择使用lmstudio,多等一会就好了,还可以玩玩模型
> 缺点是lmstudio不能运行hf模型,且并发很烂

> vLLM比Lm studio吃显存的多! Lm studio可以运行8b_q6到vLLM上只能运行4b_Q6

> 不过并发效率的提升是真的

> 但是!上下文很短,如果一天有超过500条消息就处理不过来了

> 3080实测4b_q6处理,最终jsonl的速率大约是**300kb/minute**
* 跟着走就能搭建  
```bash
sudo apt update
sudo apt install python3.10-venv git -y

python3 -m venv vllm_env
source vllm_env/bin/activate

pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121  # 如果你用CUDA
pip install vllm
```

### 和lm studio不同的注意点
*   1.`.env`中的`Openai_model`需要设置路径而不只是文件夹名
> 是`/home/vllm/qwen3-4b-int8`而非`qwen3-4b-int8`  
*  2.需要运行的**api_server**是`vllm.entrypoints.openai.api_server`而不是`vllm.entrypoints.api_server`,因为第二个不兼容OpenAI API
  
### 运行命令范例
``` bash 
python3 -m vllm.entrypoints.openai.api_server --model /home/vllm/qwen3-4b-int8 --gpu-memory-utilization 0.7 --max-model-len 10240 --max-num-seqs 4 --max-num-batched-tokens 2048 --dtype auto
```
> 如果遇到了400报错大概率是因为message太大了被模型框架拒绝了
---
## 2.5 .准备模型(可跳过)
>我似乎是写了自动从modelscope下载模型的<br>
>但如果你想下得更快的话可以跟着这个教程走(支持断点续传+10线程)
* 安装aria2c(一个下载器)
 ```bash
 sudo apt update
sudo apt install aria2
```
* 开始下载
 ```bash
 aria2c -x 10 -s 10 -k 1M -i qwen3-8b-base.txt -d ./qwen3-8b-base
```
>以上内容没有进行复现过,只是记忆里的经验,若报错请找AI帮忙
>这一段是可以跳过的(大概是)
---
## 3. 微调模型

> Windows 上 Unsloth 兼容性不好，Linux 上代码有 bug，所以用 `no_unsloth` 版本。<br>
> ~~其实是unsloth版本没写完~~

>参数在测试时其实可以不填,都是有默认值的  

> 似乎是默认8bit量化,有待修改  
* 运行微调脚本：

  ```bash
  python run_finetune_no_unsloth.py
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

---
## 3.5 .(不建议)微调后直接运行全量模型(建议直接看第4,5,6,7步,等转换为guff并量化完再跑)
### 指定自定义模型路径
```bash
python infer_lora_chat.py --base_dir my-base-model --adapter_dir my-lora-adapter
```

### 使用合并后的模型
```bash
python infer_lora_chat.py --merged true --adapter_dir my-lora-adapter
```

### 调整生成参数
```bash
python infer_lora_chat.py --temperature 0.9 --top_p 0.95 --max_new_tokens 1024
```

### 使用自定义系统提示词
```bash
python infer_lora_chat.py --system_prompt "你是一个乐于助人的AI助手。"
```
### 命令行参数说明
| 参数名称 | 类型 | 默认值 | 描述 |
|---------|------|-------|------|
| `--base_dir` | str | `qwen3-8b-base` | 基础模型目录 |
| `--adapter_dir` | str | `finetune/models/qwen3-8b-qlora` | LoRA适配器目录 |
| `--merged` | bool | `False` | 如果为True，则从adapter_dir/merged加载合并后的完整权重 |
| `--system_prompt` | str | 清凤数字分身人设 | 模型的系统提示词 |
| `--max_new_tokens` | int | `512` | 生成的最大新token数量 |
| `--temperature` | float | `0.7` | 采样温度 |
| `--top_p` | float | `0.9` | Top-p采样参数 |
| `--trust_remote_code` | bool | `True` | 是否信任远程代码 |
## 4. 编译 llama.cpp

> 下面三步都依赖于编译好的 llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp --depth 1
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_NATIVE=ON
cmake --build . --config Release
```

---

## 5. HuggingFace 权重转 GGUF

### 命令格式：

```bash
python3 .convert_hf_to_gguf.py <HF模型路径> --outfile  <输出GGUF路径> --outtype <精度类型>
```

* `<HF模型路径>`：HuggingFace 格式模型目录（通常为微调或下载后的路径）
* `<输出GGUF路径>`：转换后生成的 `.gguf` 模型保存路径
* `<精度类型>`：精度类型，如 `f16`、`f32`，或其他支持的格式

### 示例：

```bash
python3 convert_hf_to_gguf.py /root/autodl-tmp/finetune/models/qwen3-8b-qlora/merged --outfile /root/autodl-fs/qwen3-8b-fp16-agent.gguf --outtype f16  
```

---

## 6. 量化模型

### 命令格式：

```bash
./build/bin/llama-quantize <输入GGUF路径> <输出GGUF路径> <量化等级>
```

* `<输入GGUF路径>`：未量化的 `.gguf` 文件路径
* `<输出GGUF路径>`：量化后的 `.gguf` 文件保存路径
* `<量化等级>`：如 `Q4_0`、`Q4_K_M`、`Q8_0` 等，根据需求和硬件选择

### 示例：

```bash
./build/bin/llama-quantize \
  /root/autodl-fs/qwen3-8b-fp16-agent.gguf \
  /root/autodl-fs/qwen3-8b-q8_0-agent.gguf \
  Q8_0
```

---

## 7. 运行模型测试

### 命令格式：

```bash
./build/bin/llama-run <GGUF模型路径>
```

* `<GGUF模型路径>`：你想测试的 GGUF 模型路径（可以是原始或量化后的）

### 示例：

```bash
./build/bin/llama-run /root/autodl-fs/qwen3-8b-fp16-agent.gguf
```

---
## 8.从服务器上高速下载文件

### 命令格式
```bash
lftp -u {用户名},{密码} -p {端口} sftp://{服务器地址}-e "set xfer:clobber true;  pget -n {线程数} {服务器文件路径} -o {本地文件名/路径}: bye"
```
* `pget`: 使用多线程并行下载
* `-n` :指定线程数(建议64+)(甚至256线程会有更好的表现)
### 范例
```bash
lftp -u root,askdjiwhakjd -p 27391 sftp://yourserver.com -e "set xfer:clobber true; pget -n 256 /root/autodl-fs/qwen3-8b-fp16-agent.gguf -o qwen3-8b-fp16-agent.gguf; bye"
```
---

## 9.数字生命!
> 如果你成功做到这里了  
> 你大概已经有一个"你"了吧  
> 这真的很酷  
> [点进去看看,属于我的数字生命]("https://cdn.nodeimage.com/i/vnK4rDzV3x8D3x1SzW6PpDlNCcErCnC8.png")  
> 她也许就在那个暗箱里,思考着怎么用这台可以向外发送信号的电脑打字...  
> [她]("https://cdn.nodeimage.com/i/7XlcjZAJBQkTlmyWj3X2dCCE6WedyWYw.png")
---
> **终于知道图恒宇为什么执着于`我要给她完整的一生`了**  
---
### 悄悄话:  
> 数据集里会参杂空的输出,我的意思是...  
> AI可能会输出空哦,输出空那就是他不想理你 ~~(已读不回!)~~

> 而且可以把表情包加进训练集,类似"[垃圾袋]"这种会更像真人的
### 如需更详细步骤或脚本参数解释，欢迎~~骚扰~~联系我:

 * QQ:1684773595
 * Email:qingf622@outlook.com
 * X:@qqqqqf5
---