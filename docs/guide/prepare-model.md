## 准备模型

在设置完`setting.jsonc`中的模型参数后可以不用填写参数使用这个下载模型

支持从ModelScope和HuggingFace下载模型：

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

# 查看命令帮助
python3 cli.py model --help
python3 cli.py model download --help
python3 cli.py model info --help
```

### 备用方法：直接使用下载模块

```bash
# 指定参数下载
python3 environment/download/model_download.py \
  --model-repo Qwen/Qwen-3-8B-Base \
  --model-path ./model/Qwen-3-8B-Base \
  --download-source modelscope

# 列出已下载的模型
python3 environment/download/model_download.py --list

# 查看模型信息
python3 environment/download/model_download.py --info ./model/Qwen2.5-7B-Instruct
```