## Prepare Model (Optional)
>I seem to have written automatic model downloading from ModelScope  
>Of course you can also download manually, because I'm really not sure if there are any bugs there  

# ModelScope Download
### Run the code
```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-14B --local_dir ./qwen3-14b
```
>Yes... it's that simple  
> You need to find the model you need on the ModelScope community yourself

>This section can be skipped (probably)

# HuggingFace Download

### Download model
```bash
huggingface-cli download unsloth/gpt-oss-20b-BF16 --local-dir gpt-oss-20b
```
> If you don't have huggingface-cli, please install it first  
```bash
pip install huggingface-hub
```
> If you need a mirror site, please run first  
```bash
export HF_ENDPOINT=https://hf-mirror.com
```