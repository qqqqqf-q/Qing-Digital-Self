## 2. OpenAI Distill（生成 SFT）

目标：把 `data/openai-export/` 下的 ChatGPT 导出（支持单份或多份）转成可训练的 SFT（先做纯文本）。

---

## 准备环境

```bash
cp setting_template.jsonc setting.jsonc
pip install -r requirements.txt
```

---

## 运行 distill

```bash
python cli.py data openai-distill
```

输出会写到：

- `runs/openai-distill/<run_id>/sft/text.jsonl`

并在命令结束时打印本次 `run_id`。

---

## 常用参数

```bash
# 指定单个 conversations.json
python cli.py data openai-distill --input ./data/openai-export/user_a/conversations.json

# 指定一个目录（会递归发现并合并其中所有 conversations.json）
python cli.py data openai-distill --input ./data/openai-export/

# 只保留指定模型的对话（默认: gpt-4o,gpt-4-1）
python cli.py data openai-distill --allow-models gpt-4o,gpt-4-1

# PII 处理策略：mask / drop / keep（默认: mask）
python cli.py data openai-distill --pii-policy mask

# 默认会丢弃 system/code/tool，避免把工具痕迹混进纯文本训练
# 只有你明确要做工具对齐时才建议打开 keep-code/keep-tool
python cli.py data openai-distill --keep-code --keep-tool
```

---

## 快速检查产物

```bash
python cli.py data preview --input runs/openai-distill/<run_id>/sft/text.jsonl --count 3
```
