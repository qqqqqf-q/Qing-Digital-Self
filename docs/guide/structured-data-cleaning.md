# 结构化 数据清洗与 QA 构建流程解析

本文聚焦 结构化 `结构化-0.3.02` 中的 CSV 数据清洗、问答构建与评分策略，帮助在 Qing-Agent 中复用相同的清洗逻辑。内容依托 `结构化/data/qa_generator.py`、`结构化/data/clean/strategies.py`、`结构化/data/chat_parsers/telegram_parser.py`、`结构化/prompts/clean_data.py` 等实现整理而成。

## 1. 端到端流程概览
1. **源数据规范化**：Telegram 等平台的原始导出（示例：`dataset/telegram/<chat>/result.json`）被 `TelegramChatParser` 转成统一 CSV（`dataset/csv/<chat>/<chat>.csv`），列包含 `id / MsgSvrID / type_name / is_sender / talker / room_name / msg / src / CreateTime / is_forward`。
2. **预处理与过滤**（`load_csv`）：剔除跳过类型（系统通知、转账等）、自发的合并转发、含敏感词/PII 的文本，将找得到的他人图片标记为 `<image>` 并只保留相对路径。
3. **消息聚合**（`group_consecutive_messages`）：按 `single_combine_time_window`（默认 2 分钟）把同一人连续消息拼接成一条，遇到 `cut_type`（图片、语音、分享等）或自己发送的图片时插入 `CutMessage` 作为边界。
4. **QA 匹配**（`match_qa`）：状态机在 `qa_match_time_window`（默认 5 分钟）内匹配“对方提问/上下文”与“自己回复”，可累积多轮，超过 `messages_max_length` 或图片数大于 `max_image_num` 的样本会被丢弃。
5. **可选图像转写**：开启 `vision_api.enable` 时，通过 `ImageToTextProcessor` 并行调用多模态 LLM，把 `<image>` 占位替换成 `[图片描述: ...]` 文本后清空 `images` 列表。
6. **LLM 评分**：若 `clean_dataset.enable_clean` 为真，Offline（vLLM）或 Online 策略会把 QA 组装成特定 prompt，请求 LLM 给出 1–5 分，结果写回 `qa.score`。
7. **数据落盘**：`save_result` 输出 `dataset/res_csv/sft/sft-my.json`，每条样本包含 `{id, time, score, system, messages, images}`。
8. **训练阶段过滤**：`train_sft.py` 在正式喂给 LLaMA Factory 前调用 `LLMCleaningStrategy.clean()`，把 `score` 低于阈值 (`clean_dataset.llm.accept_score`, 默认 2) 的样本过滤到 `<dataset>-cleaned`。

## 2. CSV 来源与组织方式
### 2.1 Telegram 解析
- `TelegramChatParser` 负责将 Telegram `result.json` 里的消息转换为 `ChatMessage`，并映射 `media_type` → `type_name`（text/image/video/voice/sticker 等）。
- 解析要点：
  - `msg` 字段对复杂 text（字符串数组、带实体）做展开。
  - `src` 保存多媒体文件路径；只有在 `include_type` 包含 `STICKER` 且内容为空时才把贴纸 emoji 写入 `msg`。
  - `is_sender` 通过配置里的 `telegram_args.my_id` 判断。
  - 转写完成的 `ChatMessage` 会被写入 CSV，同时把所有 **他人发送的图片** 复制到 `dataset/media/images` 以便后续引用。

### 2.2 CSV 目录
- 解析后的 CSV 统一放在 `dataset/csv/<会话文件夹>/<会话>.csv`，文件名中 `_start_end.csv` 的 `start` 用于 `get_csv_files()` 排序，保证跨文件的时间顺序。
- 这些 CSV 即 `load_csv` 的输入，之后所有逻辑都以 `ChatMessage` 列表为基础。

## 3. 预处理与初筛 (`load_csv`)
1. **类型过滤**：`skip_type_list`（多语言定义，包含“系统通知/转账/语音通话/引用回复”等）直接删除，`cut_type_list`（图片/语音/分享类）不会删除，但会在后续用于切分。
2. **自发合并转发**：当 `is_sender==1 且 is_forward==True` 时移除，避免把复制/转入的内容算入个性语料。
3. **PII 与停用词**：
   - 根据配置语言选 `ChinesePIIDetector` 或 `PIIDetector`；批量检测文本消息，命中则整行删除。
   - `blocked_words` = 配置文件里的数组 ∪ `dataset/blocked_words.json` 中定义的词表；命中任意词也删除。
4. **非文本消息**：
   - `.gif` 统一归类为 sticker/动画表情，并清空 `src`。
   - 图片：若在 `dataset/media/images` 找到同名文件且 `is_sender==0`，则把 `msg` 设为 `<image>`、`src` 写成相对路径 `images/<file>`、`modality` 置为 `IMAGE`；否则把 `type_name` 改成 `Cut`，后续直接当边界。
   - 其他类型一律把 `msg` 清空，保证清洗阶段只围绕文本内容。
5. **时间戳标准化**：`CreateTime` 转成 `pandas.Timestamp`，为之后的窗口判断做准备。

## 4. 消息聚合 (`group_consecutive_messages`)
目的：把零碎的连续消息合并，减少 QA 匹配噪声，同时在特定情形插入 `CutMessage`。

- **合并策略**：同一 `talker`、同一 `is_sender` 且消息间隔不超过 `single_combine_time_window`（秒）时加入同一组，最后用 `_combine_text` 拼接，必要时在句尾加换行。
- **媒体拼接限制**：合并结果若超过 `combine_msg_max_length` 会被截断；结合 `<image>` 占位符的个数裁剪 `src` 列表，防止多余图片。
- **Cut 插入**：
  - 当前消息属于 `cut_type` 或“自己发的图片”则立即输出 `CutMessage`（包含 `cut_type`、时间、发送方），用于 `match_qa` 在读取到它时强制 flush。
  - 连续多个 cut 只保留一个，避免重复切段。

## 5. QA 组装 (`match_qa`)
`match_qa` 通过一个状态机把聚合后的消息转为 `QaPair` 列表，核心规则：

1. **状态设计**：
   - `WAITING_INSTRUCTION`：等待“对方”发送的消息，作为潜在的用户提问。
   - `WAITING_RESPONSE`：等待“自己”发送的消息，作为回答。
2. **时间窗口**：使用 `qa_match_strategy`（默认 `TimeWindowStrategy`）对 `last_message` 与当前消息做时间差判断，超过 `qa_match_time_window` 会强制结束上一轮对话并存档。
3. **CutMessage 处理**：一旦遇到 `CutMessage`，立即把当前对话落盘并重置状态，确保跨媒体或不可解析片段不会连在一起。
4. **多轮 QA**：在一次 `WAITING_RESPONSE` 状态下，如果多次出现“对方消息 → 我方消息”且都满足时间窗，就会把多轮 `Q/A` 依次写入 `conversation_messages`，形成一个长上下文样本。
5. **图片聚合**：只有**来自对方**的消息才会把 `src` 里的相对路径追加到 `conversation_images`，并在写入前检查不超过 `max_image_num`。
6. **长度/图片数量约束**：`_save_current_qa_pair` 会计算当前会话所有 `Message.content` 的字符总数，超过 `messages_max_length`（默认 2048）或图片超限直接放弃。
7. **System Prompt**：每个样本额外带一个 `system` 字段，内容来自配置 `default_system`；若 `add_time=True`，则会把当前对话的时间戳格式化追加在系统提示里。

## 6. 输出 QA 数据结构
`save_result` 将内存中的 `QaPair` 列表写成 JSON。单条样本示例：

```json
{
  "id": "0",
  "time": "2024-08-01T12:34:56.000",
  "score": 4,
  "system": "你是一个乐于助人的私人助理。 Current datetime: 08-01 12:34:56",
  "messages": [
    {"role": "user", "content": "下午去你那边拿机器，顺便帮我看下版本？"},
    {"role": "assistant", "content": "可以，晚上六点之前来就行，我提前打包好。"}
  ],
  "images": ["images/abcd1234.jpg"]
}
```

- `messages` 只包含 `user` 与 `assistant` 角色，顺序即对话顺序；`system` 不放在 messages 里。
- `images` 是对方发送的媒体相对路径；若启用图像转写并替换了 `<image>`，这里会被清空。
- `score` 初始为 0，LLM 评分后会更新到 1–5（含图样本在 Offline 策略中直接赋值为 6，确保清洗阶段不会被阈值过滤）。

## 7. LLM 清洗与评分
### 7.1 策略入口
- `clean_dataset.enable_clean=True` 时才会触发评分。
- 若 `online_llm_clear=True`，使用 `OlineLLMCleaningStrategy`，通过自定义 API (`base_url`, `llm_api_key`, `model_name`) 走在线模型；否则要求本地安装 vLLM (`llamafactory.extras.packages.is_vllm_available`)，使用 `LLMCleaningStrategy`。

### 7.2 提示词与输入格式
1. **Offline（vLLM）**：
   - 在 `judge` 方法中，把每条样本转成 `messages_str`，形如：
     ```
     Q: ...
     A: ...
     Q: ...
     A: ...
     ```
   - 套用 `CLEAN_PROMPT`（`结构化/prompts/clean_data.py`），并把 `{"id": "<qa_id>","messages": "<messages_str>"}` 作为 JSON 片段嵌入 prompt。
   - LLM 必须返回 `{"id": "<qa_id>","score": <1-5>}` 的 JSON，vLLM 通过 `QaPairScore` 模型做解析与校验。
2. **Online**：
   - 每个批次组装 `qa_list = [{"id": qa.id, "Q": "...", "A": "..."}]`，仅取首个 `user`/`assistant` 消息填充，随后传给 `ONLINE_LLM_CLEAN_PROMPT`。
   - 在线模型需返回一串 JSON array，`OlineLLMCleaningStrategy` 会做正则裁剪（移除 ```json 包裹）并用 `QaPairScoreWithId` 解析。

### 7.3 结果写回与统计
- 评分成功即写入 `qa.score`; 解析失败或模型异常会把该条记为 0 分，并在日志中记录。
- Offline 策略会用 pandas 打印 1–5 分布统计；在线策略同样输出分布表，便于观测清洗力度。

### 7.4 分数的二次利用
- `make-dataset` 阶段只负责打分与写入 JSON。
- 真正的样本过滤发生在 `train_sft.py`：它读取 `dataset_dir/dataset_info.json`，找到原始文件与 `<dataset>-cleaned` 文件名，调用 `LLMCleaningStrategy.clean()`：
  - 重新打开原始 JSON，丢弃 `score < accept_score` 的条目。
  - 若全部被丢弃则回退到原始数据，避免训练数据集为空。

## 8. 图像描述与媒体处理
- `ImageToTextProcessor` 只有在 `vision_api.enable=True` 且提供 `api_url/api_key/model_name` 时启用。
- 工作流程：
  1. 扫描 `QaPair` 的 `images`，拼出绝对路径（基于 `config.media_dir`）。
  2. 通过线程池批量把图片做 base64，并调用兼容 OpenAI 的 `/chat/completions`（可配置重试、并发数）。
  3. 获得描述后，按 `<image>` 占位的出现顺序逐个替换为 `[图片描述: ...]`，替换完成后清空 `images` 列表。
- 这样可以把多模态样本转成纯文本，继续沿用同一清洗策略。

## 9. 关键配置开关
| 配置项 (`结构化/utils/config_models.py`) | 作用 |
| --- | --- |
| `include_type` / `skip_type_list` / `cut_type_list` | 控制保留哪些模态、哪些消息需要切断或跳过。|
| `single_combine_time_window` / `qa_match_time_window` | 决定消息合并与 QA 匹配的时间阈值。 |
| `combine_msg_max_length` / `messages_max_length` | 限制合并后单条消息和整段 QA 的最大长度。 |
| `max_image_num` | 每个 QA 可携带的图片数量。 |
| `blocked_words` + `dataset/blocked_words.json` | 自定义敏感词过滤表。 |
| `clean_dataset.llm.accept_score` | 二次过滤的及格线。 |
| `online_llm_clear`, `model_name`, `base_url`, `llm_api_key`, `clean_batch_size` | 在线 LLM 清洗所需的连接信息与批大小。 |
| `vision_api.*` | 图像转写 API 的开关与并发控制。 |

## 10. 在 Qing-Agent 中复用的建议
1. **保持两阶段清洗**：沿用“生成数据时打分 + 训练前按阈值过滤”的分层设计，这样旧流程可继续输出未清洗数据，而新的 default parser 能无缝插入。
2. **复制状态机与窗口逻辑**：`group_consecutive_messages` 与 `match_qa` 的时间窗、Cut 机制决定了 QA 上下文的质量，移植时要保持同样的约束（尤其是图片触发 Cut、消息长度检查以及 system prompt 拼接）。
3. **沿用评分 prompt**：若要比较新旧 parser 的表现，建议直接使用 `CLEAN_PROMPT` / `ONLINE_LLM_CLEAN_PROMPT`，确保评分口径一致，再考虑扩展额外的指标。
4. **保留输出结构**：`messages` 数组 + `system` + `images` 的 schema 已与 LLaMA Factory 适配，建议在 Qing-Agent 的 `clean llm --parser default` 中继续输出相同字段，方便后续训练脚本读取 `dataset_info.json` 进行切换。

## 11. Qing-Agent 集成方式
- 通过 `python cli.py data clean llm --parser default` 调用 结构化 风格流程：先按 CSV 聚合+过滤，再复用 LLM 打分与 `accept_score` 阈值。`setting.jsonc` 的 `data_args.clean_set_args.llm_parser` 也可以设置为 `default` 以长期启用。
- 相关配置映射：`media_dir`、`max_image_num`、`add_time_to_system`、`vision_api.*`、`include_type`、`blocked_words` 等字段直接生效；当 `vision_api.enable=true` 且提供 API 信息时，会在清洗阶段自动完成图片描述并清空 `images` 列表。
- 输出仍保存到 `dataset/sft.jsonl`（或自定义 `data_path`），后续 `train_sft` 会读取 `_scored.csv` 与 `<dataset>-cleaned`，沿用原有训练链路。

通过以上梳理，可以在不破坏现有清洗逻辑的前提下，为 Qing-Agent 新增默认 parser，并将 结构化 的数据清洗策略与 LLM 评分机制完整移植。

