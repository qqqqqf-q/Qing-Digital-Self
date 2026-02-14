## Keep4o

Keep4o is a pipeline for **“GPT-4o style alignment”**: take your own ChatGPT export, distill it into trainable SFT (text-first), optionally run an LLM-based cleanup via an OpenAI-compatible endpoint, then fine-tune/distill.

This route focuses on **output structure, clarification behavior, refusal habits, tone, and tool-call tendencies**. It is not the “digital avatar” route. If you want to fine-tune personal style from QQ/TG data, follow the Quick Start.

---

## Outputs

- `runs/openai-distill/<run_id>/sft/text.jsonl`
- `runs/openai-clean/<run_id>/sft/train.jsonl`

---

## Privacy

- ChatGPT exports are highly private; keep them under `data/openai-export/`
- Do not commit `data/` or `runs/` to git

---

## Next

Start from “Export ChatGPT Data” in the sidebar.
