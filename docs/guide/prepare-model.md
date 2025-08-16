## 准备模型(可跳过)
>我似乎是写了自动从modelscope下载模型的  
>当然也可以手动下载,因为我真的不确定那里有没有Bug  

# ModelScope下载
### 运行代码
```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-14B --local_dir ./qwen3-14b
```
>嗯...对,就是这么简单  
> 需要自己去modelscope社区找你需要的模型

>这一段是可以跳过的(大概是)

# HuggingFace下载

### 下载模型
```bash
huggingface-cli download unsloth/gpt-oss-20b-BF16 --local-dir gpt-oss-20b
```
> 如果没有huggingface-cli,请先安装  
```bash
pip install huggingface-hub
```
> 如果需要镜像站请先运行  
```bash
export HF_ENDPOINT=https://hf-mirror.com
```