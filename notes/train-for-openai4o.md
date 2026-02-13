# Qwen2.5 对齐 GPT-4o/4.1（含工具）训练方案

更新时间：2026-02-12

## 1. 目标与边界

### 1.1 目标（必须满足）

- 目标模型：Qwen2.5 系列（优先 `Qwen2.5-3B-Instruct` 做技术验证，再扩到 `Qwen2.5-7B-Instruct`）。
- 目标风格：尽可能复现 `gpt-4o` 与 `gpt-4-1` 的输出风格与行为（表达结构、澄清方式、语气、拒答方式、工具触发习惯）。
- 不引入个人化：不训练任何“我本人/我的记忆/我的人设”。训练目标是“4o 风格助手”，不是“数字分身”。
- 工具必需：模型需学会在合适时机发起工具调用，并在工具返回后继续完成任务。

### 1.2 非目标（明确不做）

- 不尝试复刻/推断 OpenAI 的隐藏 system prompt、内部对齐策略、私有记忆机制。
- 不追求逐 token 复现（只能逼近可见输出分布）。
- 不把 ChatGPT 导出中的内部标记（如 `\ue200cite\ue202`、`sediment://` 等）当作稳定协议对外暴露。

### 1.3 宽限点（可接受的偏差）

这些偏差不算失败，但要可控、可解释：

- 同一问题的措辞与结构允许不同，但应保持“4o 风格区间”内（更像 4o/4.1，而不是像 Qwen 原生或像你本人）。
- 工具调用参数细节允许与原始导出不一致，但必须满足：
  - JSON 结构合法
  - 工具名与参数符合你定义的 schema
  - 失败时能自恢复（重试/降级/询问澄清）
- 安全拒答不要求与 4o 完全一致，但要保持“稳健且不胡编”的倾向。

## 2. 数据现状与关键事实（来自本仓库 `data/openai-export/conversations.json` 的只读扫描；兼容 legacy `openai_data/conversations.json`）

### 2.1 结构

- 顶层：JSON 数组，每个元素是一段 conversation，包含 `default_model_slug/create_time/update_time/mapping/current_node` 等字段。
- 消息：在 `mapping[node_id].message`，以树结构组织；`current_node` 指向当前有效分支，需要沿 `parent` 回溯得到主链。

### 2.2 与训练直接相关的统计结论

- 对话总数约 2150。
- conversation 级别存在模型标注：`default_model_slug`。
- 你要的两类（conversation 级别计数）：
  - `gpt-4o`：约 614
  - `gpt-4-1`：约 41
- 导出内存在多种 `content_type`：
  - `text`：正常文本
  - `code`：常用于“工具触发表达”，例如 `{"content_type":"code","text":"search(\"...\")"}`
  - `multimodal_text`：带图片引用（parts 内包含 `asset_pointer`）
  - `user_editable_context`：包含 `user_profile/user_instructions`（必须剥离，避免个人化）
- 工具痕迹至少包含：
  - `search("...")`（高频）
  - `open_url("...")`（低频）
  - 以及形如 JSON 的“工具指令块”（`{"search_query":[...],"open":[...],...}`），需要统一映射到你的工具协议
- 还存在大量内部标记/引用块（如 `\ue200cite\ue202`），这些不应进入训练样本或应在清洗中移除。

## 3. 总体技术路线（中等方向，先把闭环打通）

一句话：把 ChatGPT 导出转换为“可复用的 OpenAI Chat Completions 风格训练样本”，先做纯文本风格对齐，再做工具对齐，最后再考虑多模态（可选）。

核心分层（强制分离，避免互相污染）：

- 数据层：OpenAI 导出解析 + 过滤（只保留 `gpt-4o/gpt-4-1`）
- 规范层：统一成“内部标准协议”（OpenAI Chat Completions 语义，工具 schema 由你定义并固定）
- 训练层：风格 SFT（文本）+ 工具 SFT（工具调用与续写）
- 部署层：工具 adapter（把第三方实现映射到内部 schema），推理服务与回归评测

## 4. 工具协议：不要混淆“导出格式”与“API 协议”

重要结论：ChatGPT 导出里的工具过程不是 OpenAI API `tool_calls` 的原样记录，常见表现是：

- `assistant` 的 `content_type:"code"` 里写了 `search("...")`
- `tool` 角色消息的 `content` 可能为空，但 `metadata.search_result_groups` 含检索结果

因此必须做两件事：

1) 设计一个你自己的工具 schema（尽量贴近 OpenAI Chat Completions 的 `tools` 结构）
2) 写“导出格式 -> schema”转换器，生成可训练样本

### 4.1 内部工具 schema（建议最小集）

只定义你能稳定实现、且在训练集中有足够样本的工具。建议从最小闭环开始：

- `web_search(query: string, top_k?: number, recency_days?: number)`：返回搜索结果列表
- `open_url(url: string)`：抓取网页文本（可限制长度、做去噪）

可选扩展（不阻塞 P1-P4）：

- `ocr_image(image_id: string)`：当作视觉能力的“文本化接口”（优先于直接 VL 训练）
- `summarize(text: string)`：仅当你确实需要链式工具，不建议一开始就加

### 4.2 Adapter 原则

- 协议只保留一套：以 OpenAI Chat Completions 语义为中心（messages + tools + tool_calls + tool role result）。
- 第三方差异全部放在 adapter：外部 provider 的返回结构统一映射成你的标准 `tool_result`。
- 工具实现与训练数据解耦：训练只学“什么时候调用、怎么填参数、如何利用返回”，不学具体 provider 的字段细节。

## 5. 数据策略：确保“不沾你本人”

你无法避免“user prompt 来自你”，但你可以避免“模型把你当成目标人格/记住你的私人信息”：

- 必须丢弃 `user_editable_context`（里面是 user_profile/user_instructions）
- 必须做 PII/隐私过滤（邮箱、手机号、地址、账号、身份证、公司客户信息、家庭成员等）
- 对“强个人化主题”对话直接剔除，而不是脱敏硬留（避免训练出爱聊你的私事）
- 训练与评测都要做：同样的过滤规则要能在 pipeline 中重复执行（可审计）

## 6. 工具训练的推荐方式（最稳、工程可控）

不建议一次把“风格 + 工具 + 多模态”全塞进一个 LoRA。建议两阶段、可组合：

1) 风格 LoRA：只用纯文本回合（user/assistant text），让语言风格先稳定
2) 工具 LoRA：只用带工具链的样本（user -> assistant(tool_call) -> tool -> assistant），专门教工具格式与续写

推理时组合策略（按框架支持选一种）：

- 合并 LoRA（merge）得到单一权重
- 或运行时叠加（两张 LoRA 同时加载），通过路由控制工具场景权重占比

## 7. 计划拆分（P1-P5 必须按顺序推进）

说明：每个 P 都切成“薄片任务”，薄到单个 agent 可以直接写代码/写测试/写文档完成，并且每个薄片都有验收点。

---

## P1：OpenAI 导出解析与标准化（只做“读懂并吐出可控结构”）

目标：把 `data/openai-export/conversations.json` 转成“主链 messages 序列 + 元数据”，并能稳定筛选 `gpt-4o/gpt-4-1`（兼容 legacy `openai_data/conversations.json`）。

薄片任务：

- P1-1：定义标准中间表示（NormalizedConversation/NormalizedMessage）
  - 产物：一个 Python 数据结构（dataclass 或 TypedDict），字段至少包含 `source_id/model/create_time/messages/images/tool_traces`。
  - 验收：字段能覆盖 `text/code/multimodal_text/user_editable_context`，并能无损表达“主链顺序”。
- P1-2：实现 OpenAI 导出“流式/低内存”读取
  - 产物：一个迭代器 `iter_conversations(path)`，不一次性 `json.load` 146MB 数组。
  - 验收：在 Windows 上跑完整文件不爆内存，能输出总数与模型分布统计。
- P1-3：实现“主链提取”（`current_node -> parent` 回溯）
  - 产物：`extract_main_path(mapping, current_node)`。
  - 验收：不会把分叉草稿/未采用分支混入训练序列。
- P1-4：实现模型过滤与时间过滤（可配置）
  - 产物：可配置 `allow_models={"gpt-4o","gpt-4-1"}`，可选 `cutoff_ts`。
  - 验收：过滤后计数与预期一致（至少能解释差异来源：解析失败/缺字段/损坏条目）。
- P1-5：CLI 集成（新增 `data import openai` 或 `data extract --source-type openai`）
  - 产物：一条命令把导出解析为中间 JSONL（先不做清洗）。
  - 验收：命令可复跑、输出可审计（记录输入文件 hash、过滤条件、输出条数）。

风险提示：

- `mapping` 节点内容类型多，需要先完整覆盖再谈清洗。
- 解析期不要“顺便清洗”，否则排错困难。

---

## P2：隐私剥离与噪声清洗（保证“不沾你本人”，并减少无用 token）

目标：把 P1 的中间表示变成“可训练语料”，并显式保证个人化信息不过权重。

薄片任务：

- P2-1：剥离 `user_editable_context`
  - 产物：在 pipeline 中把其内容写入单独字段（仅用于审计/统计，不进入训练 messages）。
  - 验收：训练输出中不出现 `user_profile/user_instructions` 字段内容。
- P2-2：PII/隐私过滤与策略分级
  - 产物：规则过滤器（正则 + 关键词 + 长度/熵判定），输出 decision：drop / mask / keep。
  - 验收：能拦截邮箱/手机号/地址/身份证/账号等；对命中样本有日志（不打印明文）。
- P2-3：内部标记清理
  - 产物：清理 `\ue200cite\ue202`、`\ue200entity\ue202`、以及类似 “Sources” 注脚块，保留可读文本。
  - 验收：清洗后文本可读，不残留不可见控制字符。
- P2-4：代码/超长内容的处理策略（丢弃或降权）
  - 产物：`code_density` 估计（code fence/关键符号/`content_type=="code"`），并提供阈值策略。
  - 验收：工具调用样本保留必要的 `code`（用于 tool trace），其余大段代码可丢弃或截断。
- P2-5：去重与切片（训练友好）
  - 产物：对高度重复回答做去重；超长对话按窗口切片（保留最近 N 轮上下文）。
  - 验收：输出样本长度分布可控，训练不会频繁因超长报错。
- P2-6：生成 holdout 与评测集
  - 产物：固定随机种子的 train/valid/test 切分（conversation 级别，不要 message 级别随机切）。
  - 验收：任何人复跑得到同一划分；test 集永不进入训练。

---

## P3：工具轨迹重建与协议化（把导出“搜索痕迹”变成可训练 tool_calls）

目标：从 `content_type:"code"` 与 `tool` 消息 metadata 里重建工具链，转成你选定的 OpenAI Chat Completions 语义。

薄片任务：

- P3-1：工具调用识别（从 `code.text` 解析）
  - 产物：解析 `search("...")/open_url("...")`，以及 JSON 指令块（`{"search_query":[...],...}`）。
  - 验收：能输出工具名、参数、原始片段；无法识别的归类为 `unknown_tool_trace` 并可统计。
- P3-2：工具结果提取（从 `tool` role metadata 还原）
  - 产物：把 `metadata.search_result_groups` 等结构压缩成稳定的 `tool_result` JSON（字段白名单 + 长度限制）。
  - 验收：同一条工具结果可重复生成，且不会把无关大字段塞进训练（控 token）。
- P3-3：构建“标准工具样本”三段式
  - 产物：样本序列形态固定为：
    - user（问题）
    - assistant（tool_calls）
    - tool（result）
    - assistant（最终回答）
  - 验收：序列中每一段角色与字段符合你约定的 schema，JSON 可被解析。
- P3-4：工具 schema 与 adapter 定义
  - 产物：一份 `tools` 定义（JSON Schema 风格），以及运行时 adapter 接口（provider -> internal）。
  - 验收：能用同一套 schema 同时对接 vLLM / LM Studio / 其它 OpenAI-compatible server（至少在请求层一致）。
- P3-5：回归测试（最少覆盖 search/open_url）
  - 产物：合成测试样本（不含真实隐私），验证解析、结果压缩、样本序列拼装。
  - 验收：CI/本地测试能跑，避免后续改动导致工具样本 silently 变形。

关键取舍（建议写进实现注释与文档）：

- 工具结果不要追求“还原全部网页内容”，只保留“足够支撑回答”的摘要字段，避免训练 token 爆炸。
- 工具名建议使用你内部稳定名（例如 `web_search/open_url`），不要被导出的 `search/open` 名称绑死。

---

## P4：训练与评估闭环（3B 验证 -> 7B 放大，风格与工具都要能量化）

目标：用最小成本把效果跑出来，并且能用指标说清楚“像 4o”到了什么程度。

薄片任务：

- P4-1：建立基线评测集（固定 prompts）
  - 产物：一组覆盖风格与工具的 prompts（不含个人信息），并固化为评测脚本输入。
  - 验收：任何人用 base 模型与微调模型都能复跑同一评测集。
- P4-2：训练阶段 1（风格 LoRA）
  - 产物：只用纯文本样本训练 LoRA；记录超参、数据版本、commit hash。
  - 验收：风格指标提升且不明显损伤指令遵循（至少在评测集上不退化）。
- P4-3：训练阶段 2（工具 LoRA 或继续训练）
  - 产物：加入工具链样本；重点看 tool_calls 合法率、触发时机、工具后续写质量。
  - 验收：工具调用 JSON 合法率显著提升；工具结果利用率提升（不会无视工具返回）。
- P4-4：3B -> 7B 放大策略
  - 产物：在 3B 跑通后把 pipeline 与超参迁移到 7B；必要时调整 batch/seq_len/lr。
  - 验收：7B 训练可复现，效果不低于 3B（风格与工具两个维度都要过线）。
- P4-5：对齐风险控制
  - 产物：对“幻觉引用/伪造来源/伪造工具结果”建立专门回归集，并加入负样本约束。
  - 验收：模型不会凭空输出看似真实的 citations 或 tool 结果。

---

## P5：部署与运行时工具编排（让“训练出来的工具能力”在真实系统可用）

目标：保证训练时学到的 tool_calls 在生产环境能稳定跑，不因为换 provider 就崩。

薄片任务：

- P5-1：推理服务与协议一致性
  - 产物：统一用 OpenAI-compatible server（vLLM/LM Studio）暴露 `/chat/completions`，并固定 `tools` schema 版本。
  - 验收：同一请求在不同 provider 下，tool_calls 的语义一致（差异由 adapter 消化）。
- P5-2：工具执行器（Tool Runner）
  - 产物：一个独立模块负责：
    - 解析模型输出 tool_calls
    - 调用具体工具实现
    - 把 tool result 注入下一轮 messages
  - 验收：支持重试、超时、降级；失败时不会卡死对话。
- P5-3：端到端回归测试
  - 产物：最少覆盖：
    - 需要搜索才能答的问题
    - 搜索无结果的降级
    - open_url 失败/超时的降级
  - 验收：每次改动 tool schema/adapter 都能快速发现破坏性变更。
- P5-4：可选扩展：多模态两条路线（只在确实需要时启动）
  - 路线 A（低成本）：`ocr_image -> 文本工具 -> 文本模型`，不做 VL 训练
  - 路线 B（高成本）：引入 `Qwen2.5-VL-3B` 做端到端验证，再决定是否上 7B-VL 微调
  - 验收：以“图像任务评测集”量化提升，且不显著破坏文本风格与工具行为。

## 8. 最小可行里程碑（建议的验收门槛）

为了防止长期投入后才发现方向不对，建议把“可用”定义清楚：

- M1（完成 P1+P2）：能稳定输出“纯文本 SFT 数据集”，且证明不包含 `user_editable_context` 与明显 PII。
- M2（完成 P3）：能稳定输出“工具链 SFT 数据集”，并能在运行时通过 Tool Runner 跑通一次搜索问答闭环。
- M3（完成 P4）：3B 微调后在固定评测集上，风格更像 4o/4.1，工具调用合法率显著提升。
- M4（完成 P5）：换 provider/换搜索实现只改 adapter，不改训练数据、不改模型，功能不崩。

## 9. 需要提前确认的合规点（不要拖到上线）

- 你使用 OpenAI/ChatGPT 导出内容进行训练/分发是否符合你账户与使用条款（尤其是用于训练第三方模型、商用分发）。
- 训练数据中是否含有第三方版权内容（网页抓取结果、长篇引用、代码库内容等），需要在清洗阶段做“引用长度限制/来源剔除”。
