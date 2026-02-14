## 3. OpenAI Clean (LLM Cleanup)

Goal: clean the distilled SFT using an OpenAI-compatible Chat Completions API, to reduce tool/technical/search traces and produce a more stable training dataset.

---

## Configure an OpenAI-compatible endpoint

Edit `setting.jsonc`:

- `data_args.clean_set_args.openai_api.api_base`
- `data_args.clean_set_args.openai_api.api_key`
- `data_args.clean_set_args.openai_api.model_name`
- `data_args.clean_set_args.openai_api.clean_workers`

Note: `api_base` should point to `.../v1/chat/completions`.

---

## Run (defaults to latest distill output)

```bash
python cli.py data openai-clean
```

Outputs:

- `runs/openai-clean/<run_id>/sft/train.jsonl`

---

## Small-scale validation (recommended)

```bash
python cli.py data openai-clean --max-samples 200
```

---

## System prompt injection (optional)

By default, it reads `data_args.openai_sft_system_prompt`.

- Disable: `--no-base-prompt`
- Override: `--base-prompt` / `--base-prompt-file`

