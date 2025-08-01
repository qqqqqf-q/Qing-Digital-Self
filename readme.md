## 项目简介

这是一个数字分身项目，核心思想是**利用 QQ 的 C2C 聊天记录作为数据集，对大模型进行微调**，让模型尽可能还原你独有的表达风格和聊天方式。

项目包含了**完整的教程**，包括：

* QQ 数据库的解密与处理
* 聊天数据清洗与转换
* QLora 微调流程
* 微调模型的测试与使用

我知道类似的项目其实已经有不少了，但也许我的教程、流程、代码实现能给你一些不一样的帮助或启发。如果对你有用，欢迎点个 star，我会很开心的！

目前这个项目还有很多不足：

* 只支持最基础的 QLora 微调
* 暂时不支持 unsloth 或更高级的加速/量化技巧
* 但已经可以在 4090 24G 显卡上用 int8 精度微调 Qwen3-8B（亲测可用）

**如果你也想打造属于自己的数字分身，那也来试试吧!**

——
X: [@qqqqqf5](https://twitter.com/qqqqqf5)

---

## 项目版本
# V 0.1(MVP)
## 警告
* 上个测试版本的Qlora_qwen3.py已经过4090实机测试
* 但因为手头显卡显存太小并且租不起4090了遂导致放弃0.1版本测试
* (按理来说应该是能跑的因为我没有改什么东西)
* 清洗数据也已经进行实机测试(当前版本)
## TODO
* 增加unsloth支持(重要!可以加快微调速度)
* 为数据清洗增加LLM清洗功能(让成熟的llm来清洗数据,比直接使用算法好得多)
* 将qlora_qwen3.py的print全部改成logger(?这个很简单,我只是因为怕改多了没法测试(4090到期了))
---
# 使用QQ聊天记录微调LLM全流程指南

## 1. 获取 QQ 聊天数据

* 教程请参考：[NTQQ Windows 数据解密](https://qq.sbcnm.top/decrypt/NTQQ%20%28Windows%29.html)
* 补充资料：[数据库解码参考](https://qq.sbcnm.top/decrypt/decode_db.html)
* 上面这两个是同一个教程,耐心看完就好,不复杂(如果不会可以翻到最底下找我哦)
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

## 2. 清洗数据

* 在 `.env` 文件中修改数据库路径及相关参数(请注意其中的必填段)
* 运行清洗脚本：

  ```bash
  python generate_training_data.py
  ```

---
## 2.5 .准备模型(可跳过)
>我似乎是写了自动从modelspace下载模型的<br>
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
* 运行微调脚本：

  ```bash
  python run_finetune_no_unsloth.py
  ```
| 参数名                         | 类型   | 默认值                   | 可选值         | 说明                        |
| ------------------------------ | ------ | ------------------------ | -------------- | --------------------------- |
| --use_unsloth                  | str    | false                    | true/false     | 是否使用unsloth(请不要填True)             |
| --use_qlora                    | str    | true                     | true/false     | 是否使用qlora               |
| --data_path                    | str    | training_data.jsonl      |                | 训练数据路径                |
| --output_dir                   | str    | finetune/models/qwen3-8b-qlora |            | 输出目录                    |
| --per_device_train_batch_size  | str    | 1                        |                | 每个设备的训练批次大小      |
| --gradient_accumulation_steps  | str    | 16                       |                | 梯度累积步数                |
| --learning_rate                | str    | 2e-4                     |                | 学习率                      |
| --num_train_epochs             | str    | 3                        |                | 训练轮数                    |
| --lora_r                       | str    | 16                       |                | LoRA的秩                    |
| --lora_alpha                   | str    | 32                       |                | LoRA的alpha值               |
| --lora_dropout                 | str    | 0.05                     |                | LoRA的dropout率             |
| --logging_steps                | str    | 20                       |                | 日志记录步数                |
| --local_dir                    | str    | qwen3-8b-base            |                | 本地模型目录                |
| --save_steps                   | str    | 200                      |                | 保存模型步数                |
| --warmup_ratio                 | str    | 0.05                     |                | 预热比例                    |
| --lr_scheduler_type            | str    | cosine                   |                | 学习率调度器类型            |
| --no-gradient_checkpointing    | flag   | false                    |                | 不使用梯度检查点            |
| --no-merge_and_save            | flag   | false                    |                | 不合并并保存模型            |
| --fp16                         | str    | true                     | true/false     | 是否使用fp16                |
| --optim                        | str    | adamw_torch_fused        |                | 优化器                      |
| --dataloader_pin_memory        | str    | false                    | true/false     | 是否固定数据加载器内存      |
| --dataloader_num_workers       | str    | 0                        |                | 数据加载器工作线程数        |

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
python3 ./examples/convert-hf-to-gguf.py <HF模型路径> <输出GGUF路径> --outtype <精度类型>
```

* `<HF模型路径>`：HuggingFace 格式模型目录（通常为微调或下载后的路径）
* `<输出GGUF路径>`：转换后生成的 `.gguf` 模型保存路径
* `<精度类型>`：精度类型，如 `f16`、`f32`，或其他支持的格式

### 示例：

```bash
python3 ./examples/convert-hf-to-gguf.py \
  /root/autodl-tmp/finetune/models/qwen3-8b-qlora/merged \
  /root/autodl-fs/qwen3-8b-fp16-agent.gguf \
  --outtype f16
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

### 如需更详细步骤或脚本参数解释，欢迎~~骚扰~~联系我:

 * QQ:1684773595
 * Email:qingf622@outlook.com
 * X:@qqqqqf5
---