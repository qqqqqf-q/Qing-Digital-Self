## 2. OpenAI Distill (Generate SFT)

Goal: convert ChatGPT exports under `data/openai-export/` (single or multiple) into trainable SFT (text-first).

---

## Setup

```bash
cp setting_template.jsonc setting.jsonc
pip install -r requirements.txt
```

---

## Run distill

```bash
python cli.py data openai-distill
```

Outputs:

- `runs/openai-distill/<run_id>/sft/text.jsonl`

---

## Common flags

```bash
python cli.py data openai-distill --input ./data/openai-export/user_a/conversations.json
python cli.py data openai-distill --input ./data/openai-export/
python cli.py data openai-distill --allow-models gpt-4o,gpt-4-1
python cli.py data openai-distill --pii-policy mask
python cli.py data openai-distill --keep-code --keep-tool
```

---

## Quick check

```bash
python cli.py data preview --input runs/openai-distill/<run_id>/sft/text.jsonl --count 3
```
