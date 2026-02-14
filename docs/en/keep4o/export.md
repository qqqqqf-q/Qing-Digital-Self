## 1. Export ChatGPT Data

Goal: get your ChatGPT export and place `conversations.json` into the project directory layout.

---

## Export from ChatGPT

In ChatGPT settings, find “Export data”, download the export archive and unzip it.

You should typically see:

- `conversations.json`
- `chat.html`
- `assets/` (optional)

---

## Put it into the project (recommended)

```bash
./data/openai-export/conversations.json
```

If you still have the legacy folder `openai_data/`, `openai-distill` can read it and will print a migration tip.

