## 项目简介

## 这是一个数字分身项目，核心思想是**利用 QQ 的 C2C 聊天记录作为数据集，对大模型进行微调**，让模型尽可能还原你独有的表达风格和聊天方式。

## This project is a personal digital twin built by fine-tuning a large language model on your own chat history. The goal is to recreate your unique style of expression and conversational behavior with high fidelity.
<p align="center">
  <img src="https://img.shields.io/badge/Downloads-1-00bfff?style=for-the-badge">
<img src="https://img.shields.io/github/stars/qqqqqf-q/Qing-Digital-Self?style=for-the-badge&color=ff69b4">
<img src="https://img.shields.io/badge/Status-MVP-ff69b4?style=for-the-badge">
<img src="https://img.shields.io/badge/Version-v0.1.2-9370DB?style=for-the-badge">
<img src="https://img.shields.io/github/license/qqqqqf-q/Qing-Digital-Self?style=for-the-badge&color=8A2BE2">

</p>

# The project includes bilingual support.
## 项目包含双语支持

## 中文文档
<a href="https://qqqqqf-q.github.io/Qing-Digital-Self/">
  <img src="https://cdn.nodeimage.com/i/MfTsvmkJD2dQj9c9XZg9XXXS6CYwZBvx.png" alt="快速开始按钮" width="140" style="margin-top: 1em;">
</a>

## English Documents
<a href="https://qqqqqf-q.github.io/Qing-Digital-Self/en/">
  <img src="https://cdn.nodeimage.com/i/wy2ngEGv8sYYpcJ0zdIlMnHgm8lLmbIA.png" alt="快速开始按钮" width="140" style="margin-top: 1em;">
</a>

## 项目包含了**完整的教程**，包括：

* QQ 数据库的解密与处理
* 聊天数据清洗与转换
* QLora 微调流程
* 微调模型的测试与使用
* 使用unsloth加速微调!

我知道类似的项目其实已经有不少了，但也许我的教程、流程、代码实现能给你一些不一样的帮助或启发。如果对你有用，欢迎点个 star，我会很开心的！

目前这个项目还有很多不足：

* 暂时不知道有什么不足
* (如果有问题欢迎开Issues)
* 但已经可以在 4090 24G 显卡上用 fp8 精度微调 Qwen3-8B（亲测可用）

**如果你也想打造属于自己的数字分身，那也来试试吧!**

——
X: [@qqqqqf5](https://twitter.com/qqqqqf5)

---

## 项目版本
# V 0.1.2
## ~~警告~~ 喜报
* 此版本的Qlora_qwen3.py已经过4090实机测试(generate_training_data_llm.py+run_finetune.py)
* 清洗数据也已经进行实机测试(当前版本)
## TODO
* [完成但未测试] 增加对oss模型的支持 (以及MXFP4?这是一个50系的计算,好像我还是没法测试)
> 难点:1.MOE模型 2.非原Qwen系列模型 3.我的3080似乎本地没法测试(无论是微调还是MXFP4)
> 好吧其实一点也不难,只是这几天在写其他项目
* [规划中]增加WebUI支持
> 这真的很重要,微调模型太恶心了,记得加上Frpc支持让大家都可以公网访问  

## 更新日志
> 写在commit里了,这里实在不想写
