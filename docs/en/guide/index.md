# Qing-Digital-Self

Conversation-to-training pipeline: Digital Twin + Model Distillation.

Qing-Digital-Self provides an end-to-end toolchain:

**conversation data -> cleaning/extraction -> training samples -> fine-tuning/distillation -> usage & evaluation**.

## Two tracks

- **Digital Twin**: fine-tune on your own chat history to mimic your personal speaking style
- **GPT-4o style alignment (Keep4o)**: align structure, clarification habits, refusal behavior, and tool-calling behavior

## Where to start

- Digital Twin: start from “Quick Start” step 1 in the left sidebar
- Keep4o: start from “Keep4o -> 1. Export ChatGPT Data”

## Contribute / Train

- Contribute data: click “Export data” in ChatGPT settings and send me the exported zip archive  
  X: [@qqqqqf5](https://x.com/qqqqqf5)  
  Telegram: [DM me here](https://t.me/NS_qingf_bot)
- Train locally: run `openai-distill` / `openai-clean` to generate the dataset, then follow “Quick Start -> Fine-tune Model”
