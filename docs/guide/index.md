## 项目简介

## 这是一个数字分身项目，核心思想是**利用C2C 聊天记录作为数据集，对大模型进行微调**，让模型尽可能还原你独有的表达风格和聊天方式。

<p align="center">
 <img src="https://img.shields.io/github/stars/qqqqqf-q/Qing-Digital-Self?style=for-the-badge&color=ff69b4">
<img src="https://img.shields.io/badge/Status-Develop-ff69b4?style=for-the-badge">
<img src="https://img.shields.io/badge/Version-v0.1.5Dev-9370DB?style=for-the-badge">
<img src="https://img.shields.io/github/license/qqqqqf-q/Qing-Digital-Self?style=for-the-badge&color=8A2BE2">

</p>
## 项目包含了**完整的教程**，包括：

* QQ/WX/TG的数据提取
* 聊天数据清洗与转换
* LlamaFactory 微调流程
* 微调模型的测试与使用


我知道类似的项目其实已经有不少了，但也许我的教程、流程、代码实现能给你一些不一样的帮助或启发。如果对你有用，欢迎点个 star，我会很开心的！

* (如果有问题欢迎开Issues)
* 但已经可以在 4090 24G 显卡上用 fp8 精度微调 Qwen3-8B（亲测可用）
* 并且可以使用ROCm!(使用6800xt+ROCm7.0.2+Ubuntu24.02测试)  
---
**"部分代码参考自 Weclone"** 
**如果你也想打造属于自己的数字分身，那也来试试吧!**


X: [@qqqqqf5](https://twitter.com/qqqqqf5)  
Email: qingf622@outlook.com  
Github:[@qqqqqf-q](https://github.com/qqqqqf-q)
---


## 项目版本
# V 0.1.6 Develop

## 项目状态
* 由于0.1.4版本对于代码进行了许多重构
* 转向`Llama Factory`
* 所以可能有更多的Bug
* 欢迎各位开发者来提Issues,PR
* 贡献这个小项目
  
# 开发问题
* cli的train,data convert都存在问题,暂时还是只能用老版本调用
* 已经被重构的部分没有增加双语支持
* 删掉原来的`run_finetune`和`finetune`脚本
* todo1.增加serverapi为webui做准备
* 代码未优化