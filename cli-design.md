# Qing Digital Self CLI 设计文档

## 项目概述

Qing Digital Self 是一个基于个人QQ聊天记录的数字分身项目，通过微调大语言模型来还原用户独特的表达风格。本文档设计了一个统一的CLI入口，整合项目的各个功能模块。

## 架构设计

### 核心原则

- **单一入口**：统一的 `qds` 命令作为项目入口
- **模块化**：每个功能模块独立，可单独调用
- **配置驱动**：支持配置文件和命令行参数双重配置
- **渐进式**：支持完整工作流和单步操作
- **安全性**：敏感信息保护和参数验证

### CLI架构图

```
qds (主入口)
├── config    (配置管理)
│   ├── init     - 初始化配置文件
│   ├── show     - 显示当前配置
│   ├── set      - 设置配置项
│   └── validate - 验证配置有效性
├── data      (数据处理)
│   ├── extract  - 从QQ数据库提取聊天记录
│   ├── clean    - 清洗训练数据
│   ├── merge    - 合并多个数据源
│   └── convert  - 格式转换
├── train     (模型训练)
│   ├── start    - 启动微调训练
│   ├── resume   - 恢复训练
│   ├── monitor  - 监控训练进度
│   └── merge    - 合并LoRA权重
├── test      (模型测试)
│   ├── chat     - 交互式对话测试
│   ├── batch    - 批量测试
│   └── compare  - 模型对比测试
├── workflow  (完整工作流)
│   ├── quick    - 快速启动完整流程
│   ├── data     - 仅数据处理流程
│   └── custom   - 自定义工作流
└── utils     (工具命令)
    ├── db       - 数据库工具
    ├── log      - 日志管理
    └── system   - 系统信息
```

## 命令详细设计

### 1. 主命令

```bash
qds --version              # 显示版本信息
qds --help                 # 显示帮助信息
qds status                 # 显示项目状态
```

### 2. 配置管理 (config)

```bash
# 初始化配置
qds config init [--template basic|advanced]

# 显示配置
qds config show [--format json|yaml|table]
qds config show --section database
qds config show --key model.base_model

# 设置配置
qds config set database.qq_db_path "/path/to/db"
qds config set model.base_model "qwen2.5-7b"
qds config set --file setting.jsonc

# 验证配置
qds config validate
qds config validate --section training
```

### 3. 数据处理 (data)

```bash
# 数据提取
qds data extract [--output data/raw] [--filter-user USER_ID]
qds data extract --start-date 2023-01-01 --end-date 2024-12-31

# 数据清洗
qds data clean [--method traditional|llm] [--input data/raw] [--output data/clean]
qds data clean --llm --model gpt-4 --batch-size 100

# 数据合并
qds data merge --sources "data/chat1,data/chat2" --output data/merged
qds data merge --recursive data/sources/

# 格式转换
qds data convert --from chatml --to harmony --input data.json --output data.jsonl
```

### 4. 模型训练 (train)

```bash
# 启动训练
qds train start [--config training.json] [--resume-from CHECKPOINT]
qds train start --model qwen2.5-7b --data data/clean --epochs 3

# 恢复训练
qds train resume --checkpoint checkpoints/step-1000

# 监控训练
qds train monitor [--follow] [--refresh 5s]

# 合并权重
qds train merge --base-model MODEL_PATH --lora-path LORA_PATH --output OUTPUT_PATH
```

### 5. 模型测试 (test)

```bash
# 交互式对话
qds test chat [--model MODEL_PATH] [--temperature 0.7]

# 批量测试
qds test batch --model MODEL_PATH --questions questions.txt --output results.json

# 模型对比
qds test compare --models "model1,model2" --questions questions.txt
```

### 6. 完整工作流 (workflow)

```bash
# 快速启动（全流程）
qds workflow quick [--config CONFIG_FILE]

# 仅数据处理流程
qds workflow data [--clean-method llm]

# 自定义工作流
qds workflow custom --steps "extract,clean,train" --config custom.json
```

### 7. 工具命令 (utils)

```bash
# 数据库工具
qds utils db test-connection
qds utils db stats --table C2CMessage

# 日志管理
qds utils log show [--level info] [--tail 100]
qds utils log clean --older-than 7d

# 系统信息
qds utils system info
qds utils system check-deps
```

## 参数设计

### 全局参数

```bash
--config, -c        # 指定配置文件路径
--verbose, -v       # 详细输出模式
--quiet, -q         # 静默模式
--log-level         # 日志级别 (debug|info|warn|error)
--no-color          # 禁用颜色输出
--dry-run           # 干运行模式（不执行实际操作）
```

### 常用参数组合

```bash
# 开发调试
qds --verbose --log-level debug data extract

# 生产环境
qds --quiet --log-level warn workflow quick

# 测试模式
qds --dry-run train start --epochs 1
```

## 配置文件集成

### 配置优先级

1. 命令行参数（最高优先级）
2. 环境变量
3. 指定的配置文件
4. 默认配置文件 (`seeting.jsonc`)
5. 内置默认值

### 配置文件示例

```jsonc
{
  // CLI特定配置
  "cli": {
    "default_workflow": "quick",
    "auto_backup": true,
    "confirm_destructive": true
  },
  
  // 现有配置保持不变
  "database": {
    "qq_db_path": "/path/to/qq.db"
  },
  "model": {
    "base_model": "qwen2.5-7b"
  }
}
```

## 输出格式设计

### 进度显示

```
数据提取中... [████████████████████] 100% (2500/2500 messages)
清洗数据中... [████████░░░░░░░░░░░░] 45% (1125/2500 messages)
```

### 状态信息

```
状态: 训练中
模型: qwen2.5-7b
进度: Epoch 2/3, Step 450/680
损失: 0.234
预计剩余时间: 15分钟
```

### 结果摘要

```
训练完成!

摘要:
├─ 数据集: 2,500 条对话
├─ 训练时长: 45分钟
├─ 最终损失: 0.156
├─ 模型大小: 4.2GB
└─ 输出路径: ./outputs/qwen2.5-7b-personal
```

## 错误处理

### 错误级别

1. **致命错误**：程序无法继续执行
2. **警告**：可能影响结果但不中断执行
3. **信息**：提示用户注意的信息

### 错误示例

```bash
错误: QQ数据库文件不存在
路径: /path/to/qq.db
建议: 
  1. 检查路径是否正确
  2. 使用 'qds utils db test-connection' 测试连接
  3. 运行 'qds config set database.qq_db_path PATH' 重新设置

错误代码: E001
```

## 扩展性设计

### 插件架构

```python
# plugins/custom_cleaner.py
from qds.plugin import DataCleanerPlugin

class CustomCleaner(DataCleanerPlugin):
    def clean(self, data):
        # 自定义清洗逻辑
        pass
```

### 自定义命令

```bash
# 注册自定义命令
qds plugin register custom_cleaner.py

# 使用自定义命令
qds data clean --method custom
```

## 实现计划

### 阶段一：核心框架
- CLI主入口和参数解析
- 配置管理系统
- 基础命令框架

### 阶段二：数据处理
- data 命令组实现
- 与现有数据处理模块集成

### 阶段三：训练集成
- train 命令组实现
- 训练进度监控

### 阶段四：测试和工具
- test 命令组
- utils 工具命令
- workflow 工作流

### 阶段五：优化完善
- 错误处理优化
- 性能优化
- 文档完善

## 技术选型

### 依赖库

```python
# 命令行解析
click >= 8.0.0          # 现代化CLI框架
rich >= 13.0.0          # 丰富的终端输出

# 配置管理
pydantic >= 2.0.0       # 数据验证
jsonc-parser >= 1.0.0   # JSONC支持

# 进度显示
tqdm >= 4.64.0          # 进度条

# 日志增强
structlog >= 22.0.0     # 结构化日志
```

### 文件结构

```
cli/
├── __init__.py
├── main.py              # 主入口
├── commands/            # 命令实现
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── train.py
│   ├── test.py
│   ├── workflow.py
│   └── utils.py
├── core/                # 核心组件
│   ├── __init__.py
│   ├── config_manager.py
│   ├── progress.py
│   └── output.py
└── plugins/             # 插件系统
    └── __init__.py
```

## 使用示例

### 新手快速开始

```bash
# 1. 初始化配置
qds config init

# 2. 设置QQ数据库路径
qds config set database.qq_db_path "/path/to/qq.db"

# 3. 运行完整工作流
qds workflow quick
```

### 高级用户定制

```bash
# 1. 提取特定时间段的数据
qds data extract --start-date 2024-01-01 --end-date 2024-06-30

# 2. 使用LLM清洗数据
qds data clean --method llm --model gpt-4

# 3. 自定义训练参数
qds train start --epochs 5 --learning-rate 1e-4 --batch-size 4

# 4. 监控训练进度
qds train monitor --follow
```

### 生产环境部署

```bash
# 静默模式运行完整流程
qds --quiet --log-level warn workflow quick --config production.jsonc

# 定时任务更新数据
qds workflow data --clean-method traditional
```

这个CLI设计充分考虑了项目的企业级特性，提供了灵活的配置管理、清晰的命令结构和良好的用户体验，同时保持了与现有代码架构的兼容性。