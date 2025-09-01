## Prepare Model

After setting up the model parameters in `setting.jsonc`, you can download models without filling in parameters.

Supports downloading models from ModelScope and HuggingFace:

> If you need HuggingFace mirror site, please run `export HF_ENDPOINT=https://hf-mirror.com` first
### Recommended: Use CLI Model Management Commands

```bash
# Download model - using settings from configuration file (recommended)
python3 cli.py model download

# Download model - specify parameters
python3 cli.py model download \
  --model-repo Qwen/Qwen-3-8B-Base \
  --model-path ./model/Qwen-3-8B-Base \
  --download-source modelscope
# download-source can be modelscope or huggingface

# List downloaded models
python3 cli.py model list

# View detailed information of all models
python3 cli.py model info

# View detailed information of specified model
python3 cli.py model info ./model/Qwen2.5-7B-Instruct

# View command help
python3 cli.py model --help
python3 cli.py model download --help
python3 cli.py model info --help
```

### Alternative Method: Direct Use of Download Module

```bash
# Download with specified parameters
python3 environment/download/model_download.py \
  --model-repo Qwen/Qwen-3-8B-Base \
  --model-path ./model/Qwen-3-8B-Base \
  --download-source modelscope

# List downloaded models
python3 environment/download/model_download.py --list

# View model information
python3 environment/download/model_download.py --info ./model/Qwen2.5-7B-Instruct
```