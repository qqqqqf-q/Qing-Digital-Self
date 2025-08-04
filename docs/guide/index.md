## 项目简介

这是一个数字分身项目，核心思想是**利用 QQ 的 C2C 聊天记录作为数据集，对大模型进行微调**，让模型尽可能还原你独有的表达风格和聊天方式。
<div align="center">

<a href="#"><img src="https://cdn.nodeimage.com/i/BTlRBmAcnwN2ZLyItDCfIRJKYmxYDEpc.png" alt="Downloads" style="height:36px; display:inline-block; margin-right:8px;" /></a>
<a href="#"><img src="https://cdn.nodeimage.com/i/nHNuyUdkph6NfGBmnUQeEk8gPVHKLXg0.png" alt="Stars" style="height:36px; display:inline-block; margin-right:8px;" /></a>
<a href="#"><img src="https://cdn.nodeimage.com/i/Sd89d1w0xIhSPyPyiVYEfNRTiom8TH1S.png" alt="Status: MVP" style="height:36px; display:inline-block; margin-right:8px;" /></a>
<a href="#"><img src="https://cdn.nodeimage.com/i/u0r9K3XXnxU6hDIOMY4fkZ7cnVL28EGF.png" alt="Version: v0.1" style="height:36px; display:inline-block; margin-right:8px;" /></a>
<a href="#"><img src="https://cdn.nodeimage.com/i/CfA8AQa2bVTF2mhOY3m2Kz8nhlwXUN6S.png" alt="License: Apache-2.0" style="height:36px; display:inline-block;" /></a>

</div>

 ## 项目包含了**完整的教程**，包括：

* QQ 数据库的解密与处理
* 聊天数据清洗与转换
* QLora 微调流程
* 微调模型的测试与使用
* 使用unsloth加速训练!

我知道类似的项目其实已经有不少了，但也许我的教程、流程、代码实现能给你一些不一样的帮助或启发。如果对你有用，欢迎点个 star，我会很开心的！

目前这个项目还有很多不足：

* 暂时不知道有什么不足
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
* [完成]增加对MoE模型的支持  
>  ↑ ↑ ↑  2025/8/3更新,或许支持了, 我的显卡跑不动30b a3b,所以还是没法测试  
> 4090的机子塞不下这56.8g的模型,我还是不测了罢(  
* [完成]为数据清洗增加LLM清洗功能(让成熟的llm来清洗数据,比直接使用算法好得多)
> ↑ ↑ ↑  2025/8/3更新,已增加支持,或许不够完善
* [完成]将qlora_qwen3.py的print全部改成logger(?这个很简单,我只是因为怕改多了没法测试(4090到期了))  
>  ↑ ↑ ↑  2025/8/4更新,已完成

## 更新日志
> 写在commit里了,这里实在不想写

# [快速开始](/guide/quick-start.md)