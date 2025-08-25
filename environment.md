我们需要一个environment文件夹
存放自动安装环境的脚本
依赖库必须是只有标准库
可以调用config和logger系统因为他们俩也都是只有标准库

检测python3是否大于3.10，没有则提示用户安装python3.12
并提醒过低的python版本可能会存在不兼容的问题
并且不对3.10-的版本做更新
0.新建虚拟环境,使用python3 -m venv qds_env

然后:
source qds_env/bin/activate
或者
.\qds_env\Scripts\activate

1.首先pip install -r requirements.txt
此文件应该包含除机器学习外的所有依赖库
2.安装机器学习依赖
给用户一个选项:
1.cuda auto torch+unsloth
检测用户的cuda版本给出最适合的torch版本
优先pytorch2.8(cu126或128或129)
例如pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
如果没有126128129则使用类似这样的命令
pip install torch==<desired_version>+cu118

再安装unsloth
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
如果timeout则使用镜像站
https://ghfast.top/https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py

它会输出一个pip命令
类似pip install --upgrade pip && pip install "unsloth[cu126-ampere-torch270] @ git+https://github.com/unslothai/unsloth.git"
需要自己运行
再检测peft,pandas_ta,pandas,transformers是否安装
如果没有则使用pip install peft
2.手动安装
再选择
1.
给出所有pytorch版本或用户手动输入版本(类似pip install torch==后将所有版本输出给用户,然后用户选择版本或者install torch==<用户输入版本>,没有则报错)
unsloth,transformers,peft同理
2.
(保持环境激活)
用户自己输入pip或其他命令
如果用户输入的命令是pip install 则直接运行
如果用户输入的命令是其他命令则提示用户是否继续
如果用户输入的命令是exit则退出

最后
检测环境
包括但不限于
import unsloth
import torch
print(torch.__version__)
import transformers
import peft

并且在environment/download下
新建一个model_download.py
支持从modelscoup和huggingface下载模型
默认使用config中提供的model_repo下载到model_path,同时支持parser参数,传入model_repo和model_path(比config高一级,优先使用parser)
此文件主要供给给文件内部使用和cli使用