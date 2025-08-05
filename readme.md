## 项目简介

## 这是一个数字分身项目，核心思想是**利用 QQ 的 C2C 聊天记录作为数据集，对大模型进行微调**，让模型尽可能还原你独有的表达风格和聊天方式。

## This project is a personal digital twin built by fine-tuning a large language model on your own chat history. The goal is to recreate your unique style of expression and conversational behavior with high fidelity.
<p align="center">
  <img src="https://img.shields.io/badge/Downloads-1-00bfff?style=for-the-badge">
<img src="https://img.shields.io/github/stars/qqqqqf-q/Qing-Digital-Self?style=for-the-badge&color=ff69b4">
<img src="https://img.shields.io/badge/Status-MVP-ff69b4?style=for-the-badge">
<img src="https://img.shields.io/badge/Version-v0.1-9370DB?style=for-the-badge">
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

**如果你也想打造属于自己的数字分身，那也来试试吧!**

——
X: [@qqqqqf5](https://twitter.com/qqqqqf5)

---

## 项目版本
# V 0.1.1(MVP)
## ~~警告~~ 喜报
* 此版本的Qlora_qwen3.py已经过4090实机测试(generate_training_data_llm.py+run_finetune.py)
* 清洗数据也已经进行实机测试(当前版本)
## TODO
* [完成]增加unsloth支持(重要!可以加快微调速度)
> ↑ ↑ ↑ 2025/8/4更新,已支持(实际上之前就能用了,可能是我显卡问题?)
* [完成但未测试]增加对MoE模型的支持  
>  ↑ ↑ ↑  2025/8/3更新,或许支持了, 我的显卡跑不动30b a3b,所以还是没法测试  
> 4090的机子塞不下这56.8g的模型,我还是不测了罢(  
* [完成]为数据清洗增加LLM清洗功能(让成熟的llm来清洗数据,比直接使用算法好得多)
> ↑ ↑ ↑  2025/8/3更新,已增加支持,或许不够完善
* [完成]将qlora_qwen3.py的print全部改成logger(?这个很简单,我只是因为怕改多了没法测试(4090到期了))  
>  ↑ ↑ ↑  2025/8/4更新,已完成

## 更新日志
> 写在commit里了,这里实在不想写
