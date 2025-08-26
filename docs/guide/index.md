## 项目简介

## 这是一个数字分身项目，核心思想是**利用C2C 聊天记录作为数据集，对大模型进行微调**，让模型尽可能还原你独有的表达风格和聊天方式。

## This project is a personal digital twin built by fine-tuning a large language model on your own chat history. The goal is to recreate your unique style of expression and conversational behavior with high fidelity.
<p align="center">
  <img src="https://img.shields.io/badge/Downloads-1-00bfff?style=for-the-badge">
<img src="https://img.shields.io/github/stars/qqqqqf-q/Qing-Digital-Self?style=for-the-badge&color=ff69b4">
<img src="https://img.shields.io/badge/Status-MVP-ff69b4?style=for-the-badge">
<img src="https://img.shields.io/badge/Version-v0.1.4Dev-9370DB?style=for-the-badge">
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
* 使用unsloth加速训练!

我知道类似的项目其实已经有不少了，但也许我的教程、流程、代码实现能给你一些不一样的帮助或启发。如果对你有用，欢迎点个 star，我会很开心的！

目前这个项目还有很多不足：

* 暂时不知道有什么不足
* (如果有问题欢迎开Issues)
* 但已经可以在 4090 24G 显卡上用 fp8 精度微调 Qwen3-8B（亲测可用）
**"部分代码参考自 Weclone"** 
**如果你也想打造属于自己的数字分身，那也来试试吧!**

——
X: [@qqqqqf5](https://twitter.com/qqqqqf5)
Email: qingf622@outlook.com
Github:[@qqqqqf-q](https://github.com/qqqqqf-q)
---


## 项目版本
# V 0.1.4 Develop

# 项目状态
* 由于0.1.4版本对于代码进行了许多重构
* 所以可能有更多的Bug
* 欢迎各位开发者来提Issues,PR
* 贡献这个小项目
  
# 开发问题
* cli的train,data convert都存在问题,暂时还是只能用老版本调用
* 新的llm清洗功能正在开发(需要包括llm打分,llm输出可用段等内容)
* 支持更多Parser和教程(包括Telegram,Wechat等)
* 微调脚本需要重构(正在思考是继续Qlora+Unsloth还是转向Llama Factory)
* 文档部分由于重构了项目还有一些没有修改的
* 已经被重构的部分没有增加双语支持
* todo1.增加serverapi为webui做准备