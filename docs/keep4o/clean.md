## 3. OpenAI Clean（LLM 清洗）

目标：用一个 OpenAI-compatible 的 Chat Completions API，对 `openai-distill` 生成的 SFT 做二次清洗：去技术/工具/搜索痕迹，并输出更稳定的训练集。

---

## 配置 OpenAI-compatible API

打开 `setting.jsonc`，修改：

- `data_args.clean_set_args.openai_api.api_base`
- `data_args.clean_set_args.openai_api.api_key`
- `data_args.clean_set_args.openai_api.model_name`
- `data_args.clean_set_args.openai_api.clean_workers`（并发）

注意：`api_base` 需要指向 `.../v1/chat/completions`（模板里就是这个格式）。

---

## 运行清洗（默认清洗最新 distill 产物）

```bash
python cli.py data openai-clean
```

输出会写到：

- `runs/openai-clean/<run_id>/sft/train.jsonl`

---

## 小规模验证（强烈建议先跑）

```bash
python cli.py data openai-clean --max-samples 200
```

---

## system prompt 注入（可选）

默认会读取 `data_args.openai_sft_system_prompt` 作为每条样本的 system prompt。

- 想完全不注入：用 `--no-base-prompt`
- 想临时覆盖：用 `--base-prompt` 或 `--base-prompt-file`

