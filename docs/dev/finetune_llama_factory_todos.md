# LLaMA Factory Web UI 迁移 TODO（SFT/LoRA 全量转向）

> 目标：完全用 LLaMA Factory Web UI 承担微调（SFT/LoRA/QLoRA/MoE），暂不与 CLI 绑定，保留旧脚本为兼容层。最终形成可在 Windows/Mac/Linux 三端运行的可复现流程与调试文档。

## 阶段 0：现状盘点与冻结范围
- [ ] 标记旧微调入口为“legacy”：`run_finetune.py`、`finetune/merge.py`、`finetune/infer_lora_chat.py`（仅文档标注，暂不删除）。
- [ ] 暂不接入 CLI（仅验证与 debug 阶段），后续预留接入点以保证一致参数与目录规范。
- [ ] 数据不改动：继续由 `process_data/generate_chatml_raw.py` 与 `process_data/generate_chatml_llm.py` 产出 ChatML JSONL，字段为 `messages:[{role, content}]`。

## 阶段 1：数据格式对齐（最小改动）
- [ ] 不做数据转换，直接使用现有 ChatML JSONL。
- [ ] 在 WebUI 中选择 `formatting=sharegpt`，列映射 `messages -> messages`，标签映射 `role_tag=role, content_tag=content, user_tag=user, assistant_tag=assistant`。
- [ ] 如需工具化（可选）：新增轻量“数据清单”脚本，生成 LLaMA Factory 可识别的数据配置 JSON（不改动原数据）。

## 阶段 2：Web UI 启动封装
- [ ] 新增：`finetune/llama_factory/launcher.py`
  - [ ] 检查依赖：`llamafactory` 是否可导入，不可用则打印安装指引（不自动安装）。
  - [ ] 提供跨平台启动：
    - [ ] 方式 A：`python -m llamafactory.webui`（或官方推荐 CLI）
    - [ ] 方式 B：子进程方式附带环境变量（端口、数据目录）
  - [ ] 输出：明确的控制台提示、访问 URL、日志路径
  - [ ] 支持 Windows/Mac/Linux 的路径处理

## 阶段 3：最小工作流串联（不依赖 CLI）
- [ ] 目录规范：
  - [ ] 数据：`dataset/llamafactory/*.jsonl`
  - [ ] 训练：`finetune/outputs/llamafactory/<run_id>`
  - [ ] 日志：`finetune/logs/llamafactory/<run_id>`
- [ ] 文档 Demo：从原始数据 -> 转换 -> 启动 Web UI -> 选择基座模型 -> 启动训练 -> 产物验证。

## 阶段 4：产物验证与推理
- [ ] 复用/适配 `finetune/infer_lora_chat.py`：
  - [ ] 增加对 LLaMA Factory 产物目录结构的兼容（LoRA adapter/merged 权重）
  - [ ] 常见失败场景提示（找不到 adapter、base model 未指定）

## 阶段 5：开发者调试文档
- [ ] 新增：`docs/guide/finetune-llama-factory.md`
  - [ ] 安装：Python/驱动、`llamafactory` 包安装与常见坑（Windows/conda/torch CUDA）
  - [ ] 数据：支持格式与本项目转换器用法
  - [ ] 启动：`launcher.py` 与手动命令对照
  - [ ] 训练：关键参数解释、常见显存策略（QLoRA、gradient checkpointing、flash attn）
  - [ ] 故障排查：端口占用、显存不足、数据解析失败、LoRA 合并失败

## 阶段 6：质量与维护
- [ ] 遵循 SRP/OCP：转换器、启动器、推理解耦；可测试、可替换
- [ ] 依赖注入：入口注入路径与配置（模块自上而下注入）
- [ ] 错误处理：给出明确可操作的错误信息，避免静默失败
- [ ] 文档与代码同步：每次接口变更同步更新文档

## 验收标准
- [ ] 不依赖 CLI 的完整最小流程可用（转换->WebUI->训练->推理）
- [ ] Windows/Mac/Linux 均可启动 Web UI
- [ ] 提供至少一种可用数据格式（ShareGPT 或 Alpaca）
- [ ] 训练与推理路径约定清晰，产物结构稳定
- [ ] 调试文档覆盖常见故障场景
