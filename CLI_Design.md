# Qing-Digital-Self CLI 入口设计文档

## 项目概述

Qing-Digital-Self 是一个数字分身项目，通过QQ聊天记录微调大语言模型，创建个人化的AI助手。项目包含完整的数据处理、模型微调和推理流程。

## 核心功能模块

### 1. 数据处理模块
- QQ数据库解密与解析 (`database/`)
- 聊天数据格式转换 (`process_data/`)
- 数据清洗与预处理 (`generate_training_data.py`)
- 多源数据合并 (`merge_data/`)

### 2. 模型训练模块
- QLoRA微调 (`finetune/`)
- 训练脚本执行 (`run_finetune.py`)
- 模型合并 (`finetune/merge.py`)

### 3. 推理模块
- 模型推理 (`finetune/infer_lora_chat.py`)
- OpenAI兼容接口 (`openai/`)

### 4. 配置管理
- 统一配置系统 (`utils/config/`)
- 日志管理 (`utils/logger/`)

## CLI架构设计

### 主入口文件：`qds_cli.py`

```
qing-digital-self/
├── qds_cli.py               # CLI主入口
├── cli/                     # CLI子模块
│   ├── __init__.py
│   ├── commands/            # 命令实现
│   │   ├── __init__.py
│   │   ├── data.py          # 数据处理命令
│   │   ├── train.py         # 训练命令
│   │   ├── infer.py         # 推理命令
│   │   ├── config.py        # 配置命令
│   │   └── utils.py         # 工具命令
│   ├── core/                # CLI核心功能
│   │   ├── __init__.py
│   │   ├── base.py          # 基础CLI类
│   │   ├── exceptions.py    # 自定义异常
│   │   └── helpers.py       # 辅助函数
│   └── interface/           # 交互界面
│       ├── __init__.py
│       ├── prompts.py       # 交互提示
│       └── validators.py    # 参数验证
└── [现有项目结构]
```

### 命令结构设计

#### 一级命令
```bash
qds --help                  # 显示帮助
qds --version              # 显示版本信息
qds config                 # 配置管理
qds data                   # 数据处理
qds train                  # 模型训练
qds infer                  # 模型推理
qds utils                  # 工具命令
```

#### 二级命令详细设计

##### 1. 配置管理 (`qds config`)
```bash
qds config init            # 初始化配置文件
qds config show            # 显示当前配置
qds config set <key> <value>  # 设置配置项
qds config validate        # 验证配置有效性
```

##### 2. 数据处理 (`qds data`)
```bash
qds data extract           # 从QQ数据库提取数据
qds data clean             # 清洗训练数据
qds data convert           # 转换数据格式
qds data merge             # 合并多源数据
qds data preview           # 预览数据样本
qds data stats             # 显示数据统计
```

##### 3. 模型训练 (`qds train`)
```bash
qds train start            # 开始训练
qds train resume           # 恢复训练
qds train stop             # 停止训练
qds train status           # 训练状态
qds train merge            # 合并LoRA权重
```

##### 4. 模型推理 (`qds infer`)
```bash
qds infer chat             # 交互式对话
qds infer serve            # 启动API服务
qds infer batch            # 批量推理
qds infer test             # 测试模型效果
```

##### 5. 工具命令 (`qds utils`)
```bash
qds utils check-deps       # 检查依赖
qds utils clean-cache      # 清理缓存
qds utils export           # 导出模型/数据
qds utils import           # 导入模型/数据
```

### 命令行参数设计

#### 全局参数
```bash
--config, -c        # 指定配置文件路径
--verbose, -v       # 详细输出模式
--quiet, -q         # 静默模式
--log-level         # 日志级别
--work-dir          # 工作目录
```

#### 具体命令参数示例

##### 数据提取
```bash
qds data extract \
  --qq-db-path ./data/qq.db \
  --qq-number 123456789 \
  --output ./data/extracted.json \
  --time-range "2023-01-01,2024-01-01"
```

##### 数据清洗
```bash
qds data clean \
  --input ./data/extracted.json \
  --output ./data/cleaned.jsonl \
  --method llm \
  --batch-size 10 \
  --workers 4
```

##### 模型训练
```bash
qds train start \
  --model-path ./models/Qwen3-8B \
  --data-path ./data/training.jsonl \
  --output-dir ./checkpoints \
  --lora-r 16 \
  --lora-alpha 32 \
  --batch-size 1 \
  --max-steps 1000
```

### 交互式功能

#### 配置向导
```bash
qds config init --interactive
# 引导用户完成配置
# 1. 选择模型路径
# 2. 配置QQ数据库
# 3. 设置训练参数
# 4. 配置API密钥
```

#### 训练监控
```bash
qds train status --follow
# 实时显示训练进度
# - 当前步数/总步数
# - 损失值变化
# - 学习率
# - 预计完成时间
```

#### 对话测试
```bash
qds infer chat
# 进入交互式对话模式
# 支持多轮对话
# 显示回复时间和置信度
```

### 错误处理与用户体验

#### 错误分类
1. **配置错误**：配置文件缺失、格式错误、参数无效
2. **依赖错误**：缺少必要的Python包或CUDA环境
3. **数据错误**：数据文件不存在、格式不正确、权限问题
4. **模型错误**：模型文件损坏、不兼容的模型格式
5. **资源错误**：内存不足、磁盘空间不够、GPU不可用

#### 错误提示设计
```bash
# 友好的错误信息
 错误：配置文件 'seeting.jsonc' 不存在
 建议：运行 'qds config init' 创建配置文件

# 带解决方案的提示
 错误：CUDA内存不足
 建议：
  1. 减少批处理大小：--batch-size 1
  2. 启用梯度检查点：--gradient-checkpointing
  3. 使用更小的模型或量化

# 进度指示器
⏳ 正在提取QQ数据... [████████████████████] 100% (1234/1234)
```

### 配置文件集成

#### 配置优先级
1. 命令行参数（最高优先级）
2. 环境变量
3. 用户配置文件 (`seeting.jsonc`)
4. 默认配置

#### 配置验证
- 启动时自动验证所有配置项
- 提供配置修复建议
- 支持配置模板和预设

### 插件系统设计

#### 扩展点
```python
# cli/plugins/
├── data_processors/     # 自定义数据处理器
├── model_adapters/      # 模型适配器
├── export_formats/      # 导出格式支持
└── custom_commands/     # 自定义命令
```

### 国际化支持

#### 多语言消息
```python
# 支持中英双语
messages = {
    'zhcn': '正在开始训练...',
    'en': 'Starting training...'
}
```

### 性能考虑

#### 异步操作
- 长时间运行的任务使用异步执行
- 提供取消和暂停功能
- 支持后台运行模式

#### 缓存机制
- 数据处理结果缓存
- 模型权重缓存
- 配置验证缓存

### 安全性

#### 敏感信息保护
- API密钥不在命令历史中显示
- 配置文件权限检查
- 数据传输加密

#### 输入验证
- 所有用户输入严格验证
- 防止路径遍历攻击
- 文件大小限制

## 实现优先级

### 第一阶段（MVP）
1. 基础CLI框架
2. 核心命令实现：`config`, `data extract`, `train start`
3. 基本错误处理
4. 简单的进度显示

### 第二阶段（完善功能）
1. 交互式功能
2. 高级数据处理命令
3. 训练监控和控制
4. 推理服务

### 第三阶段（增强体验）
1. 插件系统
2. Web界面集成
3. 性能优化
4. 高级错误恢复

这个设计提供了完整而灵活的CLI接口，既满足技术用户的需求，又为普通用户提供了友好的交互体验。