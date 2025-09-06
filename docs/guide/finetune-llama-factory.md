# 微调模型
## 克隆镜像
```bash
git clone https://github.com/qqqqqf-q/Qing-Digital-Self.git --depth 1
```
或使用镜像(中国大陆加速)
```bash
git clone https://hk.gh-proxy.com/https://github.com/qqqqqf-q/Qing-Digital-Self.git  --depth 1
```

## 配置环境
### 使用`Screen`防止终端中断
```bash
# 创建会话
screen -S finetune
# 回到会话
screen -r finetune
# 查看所有screen会话
screen -ls

# 如果screen未安装请运行
sudo apt install screen
```

### 虚拟环境
```bash
pip install uv
uv venv
source .venv/bin/activate
```
### 安装项目依赖
```bash
uv pip install -r "requirements.txt"
```
### 安装`torch`
```bash
uv pip install torch torchvision
```
> 照常应该会自动下载CUDA加速版本,如果显示为CPU版请前往[torch官网](https://pytorch.org/get-started/locally/)寻找下载命令

### 安装`Llama Factory`
```bash
uv pip install -U llamafactory
```
完成安装后，可以通过使用` llamafactory-cli version `来快速校验安装是否成功
若等待一段时间后有llamafactory字样则为安装成功

### 若有多卡训练需求  
请安装`DeepSpeed`
```bash
uv pip install deepspeed
```

### 若需更多需求  
请前往[Llama Factory 文档](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html#extra-dependency)安装对应依赖

---

# 准备模型

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
```

---

# 微调

* 运行Llama Factory WebUI：

```bash
python cli.py train webui start --share --no-browser
```

### 模型参数解释请参考[Llama Factory 文档](https://llamafactory.readthedocs.io/zh-cn/latest/advanced/arguments.html)  

---
程序运行后  
默认应会有一个`public URL`  
若出现`frpc`报错请根据指引安装`frpc`  

进入网页后请往下翻找到`配置路径`,选择`finetune-config.yaml`  
并点击`载入训练参数`  
将会加载你的`setting.jsonc`配置

剩下的模型训练参数均可在`Web UI`中配置

---

# 补充
可以在`pip`命令后增加这些内地源加速
```bash
-i https://mirrors.cloud.tencent.com/pypi/simple/
-i https://repo.huaweicloud.com/repository/pypi/simple/
-i http://mirrors.aliyun.com/pypi/simple/
-i https://pypi.tuna.tsinghua.edu.cn/simple
```