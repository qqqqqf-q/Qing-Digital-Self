## 准备模型(可跳过但不建议)

在设置完`setting.jsonc`中的模型参数后可以不用填写参数使用这个下载模型

支持从ModelScope和HuggingFace下载模型：

```bash
# 使用配置文件中的设置
# 优先使用此方法
python3 environment/download/model_download.py

# 指定参数下载
python3 environment/download/model_download.py \
  --model-repo Qwen/Qwen-3-8B-Base \
  --model-path ./model/Qwen-3-8B-Base \
  --download-source modelscope
 #download-source 可选modelscope或者huggingface

# 列出已下载的模型
python3 environment/download/model_download.py --list

# 查看模型信息
python3 environment/download/model_download.py --info ./model/Qwen2.5-7B-Instruct
```