# 数据目录结构 V2（Chat + OpenAI-Export）改造说明

更新时间：2026-02-13

## 1. 背景与问题

当前仓库的数据目录更偏“单人本地项目”的自然生长形态：`dataset/` 既承载原始数据，又承载中间产物与最终训练集；同时新增了 `openai_data/`（ChatGPT 导出）这条新数据源，导致：

- **目录职责不清**：同一个目录里混合“输入/中间产物/输出”，外部贡献者很难理解该改哪里、该忽略哪些。
- **多数据源扩展困难**：QQ/TG/WX/OpenAI 导出并列存在时，路径与配置容易互相污染。
- **开源友好度差**：真实数据与密钥路径容易误提交，项目使用说明也会越写越绕。

本次目标不是“重写所有代码”，而是先把**目录职责、迁移路径、配置约定**写清楚，后续按计划逐步落地实现（先做 OpenAI 蒸馏，再回头兼容/迁移 Chat 流水线）。

## 2. 目标与非目标

### 2.1 目标（必须达成）

- **按数据域拆分**：Chat（QQ/TG/WX）与 OpenAI-Export 分开管理，不再挤在同一个 `dataset/` 里。
- **输入与输出分离**：原始输入进 `data/`（隐私、不可提交），所有产物进 `runs/`（可复跑、可审计）。
- **对开源友好**：仓库默认只保留 `dataset/examples/` 这类小样例与格式说明；真实数据和大文件默认被忽略。
- **迁移可控**：允许“先跑通 OpenAI 蒸馏”，并在第二阶段逐步迁移/兼容原 Chat pipeline。

### 2.2 非目标（这次不做）

- 不承诺一次性把所有脚本/CLI 都改到新路径（会分阶段做）。
- 不把任何真实聊天记录、OpenAI 导出、密钥写进仓库（只提供模板与示例）。

## 3. 新目录结构（推荐）

> 原则：`data/` 放“原始输入”，`runs/` 放“每次运行产物”，`dataset/` 只放“公开样例/规范”。

建议的 V2 目录树（示例）：

```text
data/                              # 原始输入（强隐私，默认忽略提交）
  chat/                            # 聊天数据域（QQ/TG/WX）
    qq/
      original/                    # 例如 qq.db / group_msg_table.sql / 导出文件夹
      media/                       # 解析/导出得到的图片等（如需）
    telegram/
      original/
    wechat/
      original/
  openai-export/                   # ChatGPT 导出数据域（强隐私，默认忽略提交）
    conversations.json
    chat.html
    assets/                        # 导出附件（图片等，可选）

runs/                              # 运行产物（默认忽略提交）
  chat/
    20260213_001530/               # run_id（时间戳/可附加短标签）
      csv/                         # 抽取后的结构化结果
      sft/                         # 训练集输出（jsonl）
      stats/                       # 统计与审计
      manifest.json                # 本次运行清单（输入hash/配置/条数等）
  openai-distill/
    20260213_003000/
      normalized/                  # 标准化中间表示（jsonl）
      sft/                         # 最终训练集（text/tool 等）
      stats/
      manifest.json
  openai-clean/
    20260213_010000/
      sft/                         # 清洗后的训练集输出（train.jsonl）
      stats/                       # 清洗统计与丢弃原因
      manifest.json

dataset/                           # 公开数据区（允许提交）
  examples/                        # 小样例（已存在）
  schemas/                         # （建议新增）jsonl字段规范/示例
notes/                             # 设计文档与方案（建议提交）
```

### 3.1 关于“Original 和 CSV 整理进 chat”的解释

这句话建议落地为**“按数据域分到 `chat/` 与 `openai-export/`，再按职责分到 `data/` 与 `runs/`”**：

- `original/` 是原始输入：放 `data/chat/**/original/`
- `csv/` 是运行产物：放 `runs/chat/<run_id>/csv/`

这样既实现了“都在 chat 域里”，又不会把中间产物当成原始数据长期堆积在一个固定目录里。

## 4. 旧目录到新目录的映射（迁移对照）

当前仓库的关键目录（简化）：

- `dataset/original/`（原始输入）
- `dataset/csv/`（中间产物）
- `dataset/media/`（图片等）
- `dataset/sft.jsonl`（训练集输出）
- `openai_data/`（ChatGPT 导出）

建议迁移映射：

- `dataset/original/` → `data/chat/qq/original/`（或按实际数据源拆成 qq/telegram/wechat）
- `dataset/csv/` → `runs/chat/<run_id>/csv/`
- `dataset/media/` → `data/chat/qq/media/`（如属于原始附件）或 `runs/chat/<run_id>/media/`（如属于某次运行）
- `dataset/sft.jsonl` → `runs/chat/<run_id>/sft/train.jsonl`
- `openai_data/` → `data/openai-export/`

> 注：是否“物理移动旧文件夹”可以分阶段做。第一阶段可先让新 pipeline 直接输出到 `runs/`，旧目录当作 legacy 继续可用。

## 5. runs 产物规范（建议统一）

### 5.1 run_id 命名

推荐：`YYYYMMDD_HHMMSS`，可追加短标签（避免空格）：

- `20260213_003000_openai4o`
- `20260213_001530_chat_qq`

### 5.2 manifest.json（强烈建议）

每次运行都输出一个 `manifest.json`，用于可审计与可复现，建议至少包含：

- `pipeline`: `chat` / `openai-distill`
- `run_id`
- `input_paths`（不要记录密钥；路径可以相对化）
- `input_hash`（如对 conversations.json 做 sha256）
- `config_file`（使用的配置文件名）
- `filters`（如 allow_models、时间范围等）
- `counts`（对话数、消息数、样本数、丢弃数）
- `git_commit`（可选）

## 6. 配置文件组织（开源友好）

### 6.1 推荐的配置策略

把“可提交的模板”和“本地私有配置”拆开：

- `setting_template.jsonc`：可提交，只有示例与默认值
- `settings/`（建议新增目录）：存放多个可提交 profile
  - `settings/chat.jsonc`
  - `settings/openai.jsonc`
- `settings/local.jsonc`：本地私有（真实路径/密钥），默认忽略提交

运行时优先级建议是：

1) profile（chat/openai）
2) local 覆盖（如果存在）

> 这部分会涉及 CLI 的 `--config` 是否真正生效、以及是否支持“合并 local 配置”。实现可以放到第二阶段做，但目录与约定先定下来。

### 6.2 data_args 的分层建议

建议把 `data_args` 分为 `paths` + `sources`，避免 QQ 参数出现在 OpenAI pipeline 里：

- `data_args.paths.data_root`（默认 `./data`）
- `data_args.paths.runs_root`（默认 `./runs`）
- `data_args.sources.chat.qq.*`
- `data_args.sources.openai_export.*`

## 7. 分阶段落地计划（建议）

> 说明：分阶段的顺序取决于你的目标偏好：
>
> - 如果你优先“开源观感与目录清爽”，就先把目录与配置结构定下来；
> - 如果你优先“尽快产出可用结果”，就尽早做出一个最小可跑的 pipeline 来验证结构。
>
> 这里采用更偏“先整理，再接入 OpenAI 蒸馏”的节奏：先把 `chat/` 域（original/csv）归位，再把 `openai-export/` 接入。

### 阶段 0：规范与边界（不动实现，先把规则写死）

- 增加本说明文档（本文件）
- 约定 `data/`、`runs/` 的职责
- 更新 `.gitignore`：忽略 `data/`、`runs/`、大文件导出目录；不要忽略 `notes/`
- 现有脚本保持可跑，旧路径暂不动（先不强制迁移）

### 阶段 1：Chat 目录结构改造（先把 original/csv “整理进 chat”）

- 将 `dataset/original/` 的语义迁移为 `data/chat/**/original/`（按 qq/telegram/wechat 拆）
- 将 `dataset/csv/` 的语义迁移为 `runs/chat/<run_id>/csv/`（每次运行独立产物）
- 将最终 SFT 输出迁移为 `runs/chat/<run_id>/sft/`
- 迁移期保留兼容：旧路径仍可用，但给出明确迁移提示（避免“无声写到老目录”）
- 提供物理迁移入口：`python cli.py data migrate-layout`（默认 dry-run，追加 `--apply` 才会执行）

### 阶段 2：接入 OpenAI-Export（在新目录结构上实现蒸馏）

- 输入只从 `data/openai-export/` 读取（例如 conversations.json）
- 输出全部写入 `runs/openai-distill/<run_id>/`
- 先产出一条最小闭环：`sft/text.jsonl`（纯文本风格）
- 在闭环稳定后，再扩展：`sft/tool.jsonl`（工具轨迹，后续 P3）
- 固化统计与 manifest（与 chat 产物同一套规范）

### 阶段 2.5：OpenAI SFT 清洗（去技术/工具/搜索痕迹）

- 输入默认读取 `runs/openai-distill/<latest>/sft/text.jsonl`
- 输出写入 `runs/openai-clean/<run_id>/sft/train.jsonl`
- 目标：尽量剥离代码/工具调用/搜索痕迹，并做轻量口语去噪（不抹平语气）
- 建议在 `setting.jsonc` 的 `data_args.openai_sft_system_prompt` 中配置 OpenAI SFT 写入时的 `system` 消息注入策略

### 阶段 3：清理 legacy（对外开源的“收敛”阶段，可选）

- 将 `dataset/` 收敛为“examples + schemas”
- 将历史大文件与备份迁到 `runs/_archive/`（或直接由用户本地管理）

## 8. 与《Train for OpenAI4o》方案的衔接点

`notes/train-for-openai4o.md` 的 P1/P2/P3 可以直接对应到 `runs/openai-distill/<run_id>/` 下的中间目录：

- P1 标准化输出 → `normalized/`
- P2 清洗与剥离 → `normalized/`（或 `stats/audit/`）
- P3 工具样本 → `sft/tool.jsonl`

## 9. 验收点（建议写进后续实现的 CheckList）

- 任何真实数据默认不会进入 git（`data/`、`runs/` 被忽略）
- 新手只看 `settings/openai.jsonc` + 一条命令，就能跑出 `runs/openai-distill/<run_id>/sft/text.jsonl`
- OpenAI pipeline 不依赖 QQ/TG/WX 的任何字段（配置层隔离）
- 迁移 Chat pipeline 后，旧命令仍能用（至少给出明确报错与迁移提示）
