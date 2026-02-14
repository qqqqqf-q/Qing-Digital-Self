## Keep4o

Keep4o 是一条面向「GPT-4o 风格对齐」的数据流水线：用你自己的 ChatGPT 导出数据，抽取出可训练的 SFT（先做纯文本风格闭环），再用 OpenAI-compatible 的 API 进行二次清洗，最后进入微调/蒸馏。

这条路线的目标是**对齐“输出结构、澄清方式、拒答习惯、表达语气、工具调用倾向”**，而不是做“数字分身”。如果你要训练个人化风格，请走「快速上手」里的 QQ/TG 流程。

---

## 为什么要 Keep4o

很多人喜欢 4o 的表达方式和“对话质感”。Keep4o 希望通过蒸馏与对齐，让这种风格在本地模型上以可复现的方式延续下去（但不追求逐 token 复刻，也不推断任何隐藏 system prompt）。

---

## 你会得到什么

- `runs/openai-distill/<run_id>/sft/text.jsonl`：从 ChatGPT 导出中抽取出来的纯文本 SFT（默认只保留 `gpt-4o,gpt-4-1`）
- `runs/openai-clean/<run_id>/sft/train.jsonl`：LLM 清洗后的可训练数据（去技术/工具/搜索痕迹，注入可选的 4o 风格 system prompt）

---

## 隐私与安全

- ChatGPT 导出数据属于强隐私；请放在 `data/openai-export/`，不要提交到 git
- 默认只保留训练需要的字段；建议开启 PII 策略（默认 `mask`）

---

## 贡献数据（可选）

如果你愿意帮 Keep4o 的数据更丰富（尤其是高质量长对话、复杂澄清、稳定拒答、结构化输出等），可以把你从 ChatGPT 导出的压缩包（或解压后的 `conversations.json`）发给我：

- X：[@qqqqqf5](https://x.com/qqqqqf5)
- Telegram：[点击此处添加我的双向聊天](https://t.me/NS_qingf_bot)

我们需要更多数据与更多 GPU 来 Keep4o。

---

## 当前效果（示例）

以下是使用 `Qwen2.5 1.5B Instruct` + 我自己的数据训练的 LoRA + 仿 4o system prompt 的部分输出示例：

### 4o LoRA + 仿 4o System Prompt

![4o lora + system prompt](https://cdn.nodeimage.com/i/9SmHj5O98XQQW3UwuSdUYDOsSQjouGBv.webp)

### 同上（第二次输出）

![second output](https://cdn.nodeimage.com/i/S4kxv76frlii26zlX4tmdXXlq1xINXDe.webp)

### 仅仿 4o System Prompt

![system prompt only](https://cdn.nodeimage.com/i/RRyJRANLgP6v1W2AINNy5uXyz8B4ghb4.webp)

### 仅 Qwen 原模型（无 System Prompt）

![base model](https://cdn.nodeimage.com/i/OxgLDX78G6ADAZdubwmXP4MUQzQTdS9P.webp)

---

## 下一步

按左侧导航从「导出 ChatGPT 数据」开始走一遍即可。
