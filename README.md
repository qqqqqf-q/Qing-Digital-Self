<div align="right">
  <span>[<a href="./README_EN.md">English</a>]</span>
  <span>[<a href="./README.md">简体中文</a>]</span>
</div>

<div align="center">
  <h1>Qing-Digital-Self</h1>
  <p>对话数据到训练闭环：数字分身 + 模型蒸馏</p>
  <div align="center">
    <img src="https://img.shields.io/badge/status-develop-ff69b4" alt="Status" />
    <img src="https://img.shields.io/badge/version-v0.1.6--dev-9370DB" alt="Version" />
    <img src="https://img.shields.io/github/license/qqqqqf-q/Qing-Digital-Self" alt="License" />
    <img src="https://img.shields.io/github/stars/qqqqqf-q/Qing-Digital-Self?style=social" alt="Stars" />
    <img src="https://img.shields.io/github/forks/qqqqqf-q/Qing-Digital-Self?style=social" alt="Forks" />
    <img src="https://img.shields.io/github/last-commit/qqqqqf-q/Qing-Digital-Self" alt="Last Commit" />
    <img src="https://img.shields.io/github/issues/qqqqqf-q/Qing-Digital-Self" alt="Issues" />
  </div>
  <div align="center">
    [<a href="https://qqqqqf-q.github.io/Qing-Digital-Self/">中文文档</a>]
    [<a href="https://qqqqqf-q.github.io/Qing-Digital-Self/en/">Docs</a>]
    [<a href="https://github.com/qqqqqf-q/Qing-Digital-Self/issues">Issues</a>]
    [<a href="https://twitter.com/qqqqqf5">X</a>]
    [<a href="mailto:qingf622@outlook.com">Email</a>]
  </div>
  <hr>
</div>

Qing-Digital-Self 提供一套端到端工具链：

**对话数据 -> 清洗/提取 -> 可训练样本 -> 微调/蒸馏 -> 使用与评测**。

当前主要支持两条路线：

- **数字分身**：用你的聊天记录微调，尽量还原你的表达习惯（请看Readme的下端）
- **GPT-4o 风格对齐**：对齐输出结构、澄清方式、拒答习惯、工具调用行为

## [快速开始](https://qqqqqf-q.github.io/Qing-Digital-Self/)


## KEEP 4o (Distill GPT-4o)

大家很喜欢4o的高情商  
但OpenAI将下架GPT-4o  
我希望通过蒸馏的方式'复刻'4o  
让他'活着'  
- 由于数据的缺少和资金的缺少，再次我希望各位可以通过这些联系方式来联系到我
- 我们需要更多的数据和更多的GPU来`KEEP4o`
- 仅需在OpenAI官网点击`导出数据`并将压缩包发送给我  
X: [@qqqqqf5](https://x.com/qqqqqf5)  
Telegram: [点击此处添加我的双向聊天](https://t.me/NS_qingf_bot)  
---
以下是使用Qwen2.5 1.5b Instruct + 我自己的数据训练的Lora + 修改过的System Prompt的训练结果
### 4o Lora + 仿4o System Prompt
![9SmHj5O98XQQW3UwuSdUYDOsSQjouGBv.webp](https://cdn.nodeimage.com/i/9SmHj5O98XQQW3UwuSdUYDOsSQjouGBv.webp)
### 同上,第二次输出
![S4kxv76frlii26zlX4tmdXXlq1xINXDe.webp](https://cdn.nodeimage.com/i/S4kxv76frlii26zlX4tmdXXlq1xINXDe.webp)
### 仅4o System Prompt
![RRyJRANLgP6v1W2AINNy5uXyz8B4ghb4.webp](https://cdn.nodeimage.com/i/RRyJRANLgP6v1W2AINNy5uXyz8B4ghb4.webp)
### 仅Qwen原模型无SystemPrompt
![OxgLDX78G6ADAZdubwmXP4MUQzQTdS9P.webp](https://cdn.nodeimage.com/i/OxgLDX78G6ADAZdubwmXP4MUQzQTdS9P.webp)

## 开始贡献 / 训练

如果你想参与 Keep4o：

- 贡献数据：在 OpenAI 官网点击`导出数据`，把导出压缩包发给我  
  X: [@qqqqqf5](https://x.com/qqqqqf5)  
  Telegram: [点击此处添加我的双向聊天](https://t.me/NS_qingf_bot)
- 本地训练：先跑 `openai-distill` / `openai-clean` 生成训练集，再按文档里的「快速上手 -> 微调模型」进行微调  
  Docs: <https://qqqqqf-q.github.io/Qing-Digital-Self/>

## 数字分身

仓库也包含一套完整的数字分身教程与流程，包括：

- QQ/TG 数据提取
- 聊天数据清洗与转换
- LlamaFactory 微调流程
- 微调模型的测试与使用

[点击此处快速开始](https://qqqqqf-q.github.io/Qing-Digital-Self/)  
部分代码参考自 Weclone。

## 参与贡献

欢迎通过 Issues/PR 参与贡献。  
若你想贡献数据，请只提交你有权分享的数据，并避免任何隐私/敏感信息。

租 GPU 成本很高。如果你愿意支持算力开销，可以先开 Issue 留言你偏好的方式（我会把它整理成稳定的赞助入口）。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=qqqqqf-q/Qing-Digital-Self&type=date&legend=top-left)](https://www.star-history.com/#qqqqqf-q/Qing-Digital-Self&type=date&legend=top-left)

## License

Apache-2.0
