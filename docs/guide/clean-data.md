## 将数据转化为csv
```bash
python cli.py data extract
# 或自定义parser字段
python cli.py data extract --qq-db-path ./data/qq.db --qq-number-ai 1234567890--output ./dataset/csv
```

| 参数 | 说明 | 默认值/备注 |
|------|------|-------------|
| `-h, --help` | 显示帮助信息并退出 | - |
| `--source-type {qq,tg,telegram}` | 指定数据源类型 | 不指定则自动检测 |
| `--data-dir DATA_DIR` | 数据目录路径 | `./dataset/original/` |
| `--output OUTPUT` | 输出目录路径 | `./dataset/csv/` |
| `--qq-db-path QQ_DB_PATH` | QQ 数据库文件路径 | - |
| `--qq-number-ai QQ_NUMBER_AI` | AI 的 QQ 号码（用于区分发送者） | - |
| `--telegram-chat-id TELEGRAM_CHAT_ID` | AI 的 Telegram 聊天名称（用于区分发送者） | - |
| `--tg-data-dir TG_DATA_DIR` | Telegram 数据目录 | 如不指定则使用 `--data-dir` |


## 清洗数据(普通版本,llm清洗版本在下一章节)
> 此方法比llm清洗快得多,30w条消息几秒就好了
> 但是对应的质量也更低
> 这个部分建议在windows上优化完再上传至GPU服务器  

* 在 `setting.jsonc` 文件中修改数据库路径及相关参数(请注意其中的必填段)
* `data_agrs`中的一些字段以及下面的`system prompt`
* 运行清洗脚本：

```bash
python cli.py data clean raw
```
---
## 清洗数据(llm清洗)
> Develop版本正在改进此功能  

> 需要配置一个OpenAI兼容的API  
> 比如:LM Studio 或者 vLLM(速度更快,但搭建更麻烦,需要Linux环境)  

> 这个部分同样建议在windows上优化完再上传至GPU服务器  
> 不确定在Linux上有没有兼容性问题

* 前往`setting.jsonc`文件中修改`clean_set_args`组的`openai_api`字段
* 设置`api_base` `api_key` `model_name`等字段

### run!
```bash
python cli.py data clean llm
```
> 如果遇到了400报错大概率是因为message太大了被模型框架拒绝了

# 以下为不使用云端llm服务清洗使用

## LM Studio搭建教程
* 1.前往[LM Studio](https://lmstudio.ai/)下载LM Studio
* 2.安装LM Studio
* 3.打开LM Studio,点击左侧`搜索`->`Model Search`
* 4.搜索 `qwen2.5-7b-instruct`->`Complete Download`  
* 5.选择合适你的量化版本**建议至少Q4,最好Q6-Q8,随你的设备情况而定,不知道的可以问AI**
* 记住你的**模型名称**,填写到`setting.jsonc`文件的`model_name`中
* 如果不知道你的模型名称可以运行test_openai.py,会输出所有的模型名称
* 6.安装好后,在左侧`开发者/Developer`点击`Status:Stopped`右边的按钮
* 如果下面log显示端口被占用请点击`seetings`换个`server port`
* 记住这个`server port`,将你的配置填写至`setting.jsonc`文件中


---

## vLLM搭建
> vLLM需要linux环境!  
> 如果你的显卡还算可以(>6800xt,>3080)  
> 可以选择使用lmstudio,多等一会就好了,还可以玩玩模型
> 缺点是lmstudio不能运行hf模型,且并发很烂

> vLLM比Lm studio吃显存的多! Lm studio可以运行8b_q6到vLLM上只能运行4b_Q6

> 不过并发效率的提升是真的

> 但是!上下文较短  
> 不过现在应该遇不到那么长的上下文了

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
*   1.`setting.jsonc`中的`model_name`需要设置路径而不只是文件夹名
> 是`/home/vllm/qwen3-4b-int8`而非`qwen3-4b-int8`  
*  2.需要运行的**api_server**是`vllm.entrypoints.openai.api_server`而不是`vllm.entrypoints.api_server`,因为第二个不兼容OpenAI API
  
### 运行命令范例
``` bash v
python3 -m vllm.entrypoints.openai.api_server --model /home/vllm/qwen3-4b-int8 --gpu-memory-utilization 0.7 --max-model-len 10240 --max-num-seqs 4 --max-num-batched-tokens 2048 --dtype auto
```
> 如果遇到了400报错大概率是因为message太大了被模型框架拒绝了

