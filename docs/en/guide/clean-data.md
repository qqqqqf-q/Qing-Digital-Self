## Convert Data to CSV
```bash
python cli.py data extract
# Or customize parser fields
python cli.py data extract --qq-db-path ./data/qq.db --qq-number-ai 1234567890--output ./dataset/csv
```

| Parameter | Description | Default/Notes |
|-----------|-------------|---------------|
| `-h, --help` | Show help information and exit | - |
| `--source-type {qq,tg,telegram}` | Specify data source type | Auto-detect if not specified |
| `--data-dir DATA_DIR` | Data directory path | `./dataset/original/` |
| `--output OUTPUT` | Output directory path | `./dataset/csv/` |
| `--qq-db-path QQ_DB_PATH` | QQ database file path | - |
| `--qq-number-ai QQ_NUMBER_AI` | AI's QQ number (to distinguish sender) | - |
| `--telegram-chat-id TELEGRAM_CHAT_ID` | AI's Telegram chat name (to distinguish sender) | - |
| `--tg-data-dir TG_DATA_DIR` | Telegram data directory | Use `--data-dir` if not specified |


## Clean Data (Regular Version, LLM cleaning version in next section)
> This method is much faster than LLM cleaning, 300k messages done in a few seconds
> But correspondingly the quality is also lower
> This part is recommended to be optimized on Windows first, then uploaded to GPU server

* Modify the database path and related parameters in the `setting.jsonc` file (please note the required fields)
* Some fields in `data_agrs` and the `system prompt` below
* Run the cleaning script:

```bash
python cli.py data clean raw
```
---
## Clean Data (LLM Cleaning)
# Develop version may temporarily not support this feature
# Please prioritize using raw version or wait for updates to new cleaning methods
> Need to configure an OpenAI-compatible API
> For example: LM Studio or vLLM (faster, but more complicated to set up, requires Linux environment)

> This part is also recommended to be optimized on Windows first, then uploaded to GPU server
> Not sure if there are compatibility issues on Linux
## LM Studio Setup Tutorial
* 1. Go to [LM Studio](https://lmstudio.ai/) to download LM Studio
* 2. Install LM Studio
* 3. Open LM Studio, click `Search` -> `Model Search` on the left
* 4. Search for `qwen2.5-7b-instruct` -> `Complete Download`
* 5. Choose a quantization version suitable for you **recommend at least Q4, preferably Q6-Q8, depends on your device situation, ask AI if you don't know**
* Remember your **model name**, fill it in the `Openai_model` field in the `.env` file
* If you don't know your model name, you can run test_openai.py, it will output all model names
* 6. After installation, click the button next to `Status:Stopped` in the left `Developer` section
* If the log below shows port occupied, please click `settings` to change the `server port`
* Remember this `server port`, fill your configuration in the `.env` file

### run!
```bash
python generate_training_data_llm.py
```
> If you encounter 400 error, it's most likely because the message is too large and was rejected by the model framework

## Command Options

```bash
python cli.py data clean llm

# Other parameters
--input   Input CSV directory (default from config)
--output  Output file path (default from config)
--batch-size  Batch size (default from config)
--workers Work processes (default from config)
--resume  Continue from last scored ID
```

---

## vLLM Setup
> vLLM requires Linux environment!
> If your graphics card is decent (>6800xt, >3080)
> You can choose to use LM Studio, just wait a bit longer, and you can also play with the model
> The downside is that LM Studio cannot run HF models, and concurrency is terrible

> vLLM consumes much more VRAM than LM Studio! LM Studio can run 8b_q6 but vLLM can only run 4b_Q6

> However, the improvement in concurrency efficiency is real

> But! The context is very short, if there are more than 500 messages in a day, it cannot handle them

> RTX 3080 tested with 4b_q6 processing, the final jsonl rate is approximately **300kb/minute**
* Follow these steps to set up:
```bash
sudo apt update
sudo apt install python3.10-venv git -y

python3 -m venv vllm_env
source vllm_env/bin/activate

pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121  # If you use CUDA
pip install vllm
```

### Different considerations from LM Studio
* 1. The `model_name` in `setting.jsonc` needs to be set to a path rather than just a folder name
> It should be `/home/vllm/qwen3-4b-int8` instead of `qwen3-4b-int8`
* 2. The **api_server** to run is `vllm.entrypoints.openai.api_server` not `vllm.entrypoints.api_server`, because the second one is not compatible with OpenAI API

### Example run command
```bash
python3 -m vllm.entrypoints.openai.api_server --model /home/vllm/qwen3-4b-int8 --gpu-memory-utilization 0.7 --max-model-len 10240 --max-num-seqs 4 --max-num-batched-tokens 2048 --dtype auto
```
> If you encounter 400 error, it's most likely because the message is too large and was rejected by the model framework


### Dev Notes
> Currently, new LLM processing has not been implemented yet
> The `python cli.py data clean llm` command is available but equivalent to raw
> Also, the `python cli.py data extract` command now only supports QQ Parser, not optimized for multi-parser multi-data sources
> You can add `qq/tg/wx` etc. `metamodel` to support more parsers