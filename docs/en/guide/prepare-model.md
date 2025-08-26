## Prepare Model (Optional but Recommended)

After setting up the model parameters in `setting.jsonc`, you can download models without additional parameters.

Supports downloading models from ModelScope and HuggingFace:

```bash
# Use settings from configuration file
# Recommended method
python3 environment/download/model_download.py

# Download with specified parameters
python3 environment/download/model_download.py \
  --model-repo Qwen/Qwen-3-8B-Base \
  --model-path ./model/Qwen-3-8B-Base \
  --download-source modelscope
# download-source can be modelscope or huggingface

# List downloaded models
python3 environment/download/model_download.py --list

# View model information
python3 environment/download/model_download.py --info ./model/Qwen2.5-7B-Instruct
```