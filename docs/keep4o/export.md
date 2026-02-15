## 1. 导出 ChatGPT 数据

目标：拿到 ChatGPT 的导出文件，并把 `conversations.json` 放到项目约定的位置。

---

## 从 ChatGPT 导出

在 ChatGPT 的设置里找到“导出数据 / Export data”，下载导出压缩包并解压。

导出内容里通常会包含（以你的实际导出为准）：

- `conversations.json`
- `chat.html`
- `assets/`（可选，图片等附件）

---

## 放到项目目录（推荐）

如果你只有一份导出，把 `conversations.json` 放到：

```bash
./data/openai-export/conversations.json
```

如果你有多份导出（例如两个人的聊天记录），推荐按“每份导出一个子目录”的方式放置（把压缩包解压到不同子目录即可）：

```bash
./data/openai-export/user_a/conversations.json
./data/openai-export/user_a/chat.html
./data/openai-export/user_b/conversations.json
./data/openai-export/user_b/chat.html
```

`openai-distill` 会递归遍历 `data/openai-export/`，自动发现并合并所有 `conversations.json`。

如果你还保留着历史目录 `openai_data/`，也可以先不迁移：`openai-distill` 会自动兼容读取，并在输出里提示你迁移到 `data/openai-export/`。
