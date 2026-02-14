# MirrorFlow

<sub>旧名：Qing-Digital-Self（Old Name）</sub>

对话数据到训练闭环：数字分身 + 模型蒸馏。

MirrorFlow 提供一套端到端工具链：

**对话数据 -> 清洗/提取 -> 可训练样本 -> 微调/蒸馏 -> 使用与评测**。

## 两条路线

- **数字分身**：用你的聊天记录微调，尽量还原你的表达习惯
- **GPT-4o 风格对齐（Keep4o）**：对齐输出结构、澄清方式、拒答习惯、工具调用行为

## 从哪里开始

- 数字分身：从左侧「快速上手」第 1 步开始
- Keep4o：从左侧「Keep4o -> 1. 导出 ChatGPT 数据」开始

## 开始贡献 / 训练

- 贡献数据：在 OpenAI 官网点击`导出数据`，把导出压缩包发给我  
  X: [@qqqqqf5](https://x.com/qqqqqf5)  
  Telegram: [点击此处添加我的双向聊天](https://t.me/NS_qingf_bot)
- 本地训练：先跑 `openai-distill` / `openai-clean` 生成训练集，再按「快速上手 -> 微调模型」进行微调
