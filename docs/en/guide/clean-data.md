## 2.1 Clean Data (Regular Version, LLM cleaning version in next chapter)
> This method is much faster than LLM cleaning, 300k messages done in half a minute
> But correspondingly the quality is also lower
> This part is recommended to be optimized on Windows first, then uploaded to GPU server
> Not sure if there are compatibility issues on Linux
* Modify the database path and related parameters in the `.env` file (please note the required fields)
* Run the cleaning script:

  ```bash
  python generate_training_data.py
  ```

## 2.2 Clean Data (LLM Cleaning)
> Need to configure an OpenAI-compatible API
> For example: LM Studio or vLLM (faster, but more complicated to set up, requires Linux environment)

> This part is also recommended to be optimized on Windows first, then uploaded to GPU server
> Not sure if there are compatibility issues on Linux

## LM Studio Setup Tutorial
* 1. Go to [LM Studio](https://lmstudio.ai/) to download LM Studio
* 2. Install LM Studio
* 3. Open LM Studio, click `Search` -> `Model Search` on the left
* 4. Search for `qwen3 8b` -> `Complete Download`
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
* 1. The `Openai_model` in `.env` needs to be set to a path rather than just a folder name
> It should be `/home/vllm/qwen3-4b-int8` instead of `qwen3-4b-int8`
* 2. The **api_server** to run is `vllm.entrypoints.openai.api_server` not `vllm.entrypoints.api_server`, because the second one is not compatible with OpenAI API

### Example run command
```bash
python3 -m vllm.entrypoints.openai.api_server --model /home/vllm/qwen3-4b-int8 --gpu-memory-utilization 0.7 --max-model-len 10240 --max-num-seqs 4 --max-num-batched-tokens 2048 --dtype auto
```
> If you encounter 400 error, it's most likely because the message is too large and was rejected by the model framework