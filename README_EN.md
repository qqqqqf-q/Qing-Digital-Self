<div align="right">
  <span>[<a href="./README_EN.md">English</a>]</span>
  <span>[<a href="./README.md">简体中文</a>]</span>
</div>

<div align="center">
  <h1>MirrorFlow</h1>
  <p><sub>旧名：Qing-Digital-Self（Old Name）</sub></p>
  <p>Conversation-to-training pipeline: Digital Twin + Model Distillation</p>
  <div align="center">
    <img src="https://img.shields.io/badge/status-develop-ff69b4" alt="Status" />
    <img src="https://img.shields.io/badge/version-v0.1.6--dev-9370DB" alt="Version" />
    <img src="https://img.shields.io/github/license/qqqqqf-q/MirrorFlow" alt="License" />
    <img src="https://img.shields.io/github/stars/qqqqqf-q/MirrorFlow?style=social" alt="Stars" />
    <img src="https://img.shields.io/github/forks/qqqqqf-q/MirrorFlow?style=social" alt="Forks" />
    <img src="https://img.shields.io/github/last-commit/qqqqqf-q/MirrorFlow" alt="Last Commit" />
    <img src="https://img.shields.io/github/issues/qqqqqf-q/MirrorFlow" alt="Issues" />
  </div>
  <div align="center">
    [<a href="https://qqqqqf-q.github.io/MirrorFlow/">中文文档</a>]
    [<a href="https://qqqqqf-q.github.io/MirrorFlow/en/">Docs</a>]
    [<a href="https://github.com/qqqqqf-q/MirrorFlow/issues">Issues</a>]
    [<a href="https://twitter.com/qqqqqf5">X</a>]
    [<a href="mailto:qingf622@outlook.com">Email</a>]
  </div>
  <hr>
</div>

MirrorFlow provides an end-to-end toolchain:

**conversation data -> cleaning/extraction -> training samples -> fine-tuning/distillation -> usage & evaluation**.

It supports two tracks:

- **Digital Twin**: fine-tune on your own chat history to mimic your personal speaking style (see below)
- **GPT-4o style alignment**: align output structure, clarification habits, refusal behavior, and tool-calling behavior

## [Quick Start](https://qqqqqf-q.github.io/MirrorFlow/en/)

## KEEP 4o (Distill GPT-4o)

Many people love 4o’s “high EQ” style.  
If GPT-4o gets retired, I hope to distill and reproduce the 4o style so it can “stay alive”.

- Due to the lack of data and budget, please reach out via the contacts below.
- We need more data and more GPUs to `KEEP4o`.
- You can export your ChatGPT data from OpenAI and send me the zip archive.

X: [@qqqqqf5](https://x.com/qqqqqf5)  
Telegram: [DM me here](https://t.me/NS_qingf_bot)  
---
Below are some results trained with Qwen2.5 1.5B Instruct + my own data LoRA + a modified system prompt.

### 4o LoRA + 4o-like system prompt
![9SmHj5O98XQQW3UwuSdUYDOsSQjouGBv.webp](https://cdn.nodeimage.com/i/9SmHj5O98XQQW3UwuSdUYDOsSQjouGBv.webp)
### Same setup, second output
![S4kxv76frlii26zlX4tmdXXlq1xINXDe.webp](https://cdn.nodeimage.com/i/S4kxv76frlii26zlX4tmdXXlq1xINXDe.webp)
### 4o-like system prompt only
![RRyJRANLgP6v1W2AINNy5uXyz8B4ghb4.webp](https://cdn.nodeimage.com/i/RRyJRANLgP6v1W2AINNy5uXyz8B4ghb4.webp)
### Qwen base model (no system prompt)
![OxgLDX78G6ADAZdubwmXP4MUQzQTdS9P.webp](https://cdn.nodeimage.com/i/OxgLDX78G6ADAZdubwmXP4MUQzQTdS9P.webp)

## Contribute / Train

If you want to join Keep4o:

- Contribute data: click “Export data” in ChatGPT settings and send me the exported zip archive  
  X: [@qqqqqf5](https://x.com/qqqqqf5)  
  Telegram: [DM me here](https://t.me/NS_qingf_bot)
- Train locally: run `openai-distill` / `openai-clean` to generate the dataset, then follow “Quick Start -> Fine-tune Model”  
  Docs: <https://qqqqqf-q.github.io/MirrorFlow/en/>

## Digital Twin

This repo also includes a full tutorial/pipeline for digital-twin training, including:

- QQ/TG data extraction
- chat cleaning & conversion
- LlamaFactory fine-tuning
- model testing & usage

[Quick start](https://qqqqqf-q.github.io/MirrorFlow/en/)  
Some code is inspired by Weclone.

## Contributing

Contributions are welcome via Issues/PRs.  
If you want to contribute data, please only submit data you have the rights to share, and avoid any private/PII content.

GPU renting is expensive. If you want to support compute cost, open an Issue and tell me your preferred method (I will consolidate it into a stable sponsor entry).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=qqqqqf-q/MirrorFlow&type=date&legend=top-left)](https://www.star-history.com/#qqqqqf-q/MirrorFlow&type=date&legend=top-left)

## License

Apache-2.0
