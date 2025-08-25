# Qing-Digital-Self CLI 使用说明

基于 cli_design.md 设计的企业级命令行工具，提供数字分身项目的完整生命周期管理。

## 快速开始

### 1. 基本使用

```bash
# 查看帮助
python qds_cli.py --help

# 查看版本
python qds_cli.py --version

# 详细输出模式
python qds_cli.py --verbose <command>

# 静默模式
python qds_cli.py --quiet <command>
```

### 2. 配置管理

```bash
# 初始化配置文件
python qds_cli.py config init

# 交互式配置向导
python qds_cli.py config init --interactive

# 显示当前配置
python qds_cli.py config show

# 以JSON格式显示配置
python qds_cli.py config show --format json

# 设置配置项
python qds_cli.py config set log_level DEBUG

# 验证配置
python qds_cli.py config validate
```

### 3. 数据处理

```bash

均会自动从conig获取字段,parser仅作为额外使用
# 从QQ数据库提取数据
python qds_cli.py data extract
python qds_cli.py data extract --qq-db-path ./data/qq.db --qq-number-ai 1684773595 --output ./dataset/csv

# 清洗数据（原始算法）
python qds_cli.py data clean raw

# 清洗数据（LLM方法,为实现,暂时等同于raw）
python qds_cli.py data clean llm

# 转换数据格式
python qds_cli.py data convert --input ./data/raw.json --output ./data/chatml.jsonl --format chatml

# 合并多源数据
python qds_cli.py data merge --inputs ./data/file1.jsonl ./data/file2.jsonl --output ./data/merged.jsonl --deduplicate

# 预览数据
python qds_cli.py data preview --input ./dataset/sft.jsonl --count 3

# 数据统计
python qds_cli.py data stats --input ./dataset/sft.jsonl
```

### 4. 模型训练

```bash
# 开始训练
python qds_cli.py train start --model-path ./model/Qwen3-8B --data-path ./data/training.jsonl --output-dir ./checkpoints

# 高级训练参数
python qds_cli.py train start \
  --model-path ./model/Qwen3-8B \
  --data-path ./data/training.jsonl \
  --output-dir ./checkpoints \
  --lora-r 16 \
  --lora-alpha 32 \
  --batch-size 1 \
  --max-steps 1000

# 恢复训练
python qds_cli.py train start --resume ./checkpoints/checkpoint-500

# 查看训练状态
python qds_cli.py train status

# 实时跟踪训练日志
python qds_cli.py train status --follow

# 停止训练
python qds_cli.py train stop

# 强制停止训练
python qds_cli.py train stop --force

# 合并LoRA权重
python qds_cli.py train merge --base-model ./model/Qwen3-8B --lora-path ./checkpoints/final --output ./model/merged
```

### 5. 模型推理

```bash
# 交互式对话
python qds_cli.py infer chat --model-path ./model/merged

# 自定义推理参数
python qds_cli.py infer chat \
  --model-path ./model/merged \
  --max-length 2048 \
  --temperature 0.7 \
  --top-p 0.9

# 启动API服务
python qds_cli.py infer serve --model-path ./model/merged --host 0.0.0.0 --port 8000

# 批量推理
python qds_cli.py infer batch \
  --model-path ./model/merged \
  --input ./data/test_inputs.jsonl \
  --output ./data/test_outputs.jsonl \
  --batch-size 8

# 测试模型效果
python qds_cli.py infer test --model-path ./model/merged

# 使用自定义测试数据
python qds_cli.py infer test --model-path ./model/merged --test-data ./data/test_cases.jsonl
```

### 6. 系统工具

```bash
# 检查依赖
python qds_cli.py utils check-deps

# 自动修复依赖
python qds_cli.py utils check-deps --fix

# 清理缓存
python qds_cli.py utils clean-cache

# 清理所有缓存
python qds_cli.py utils clean-cache --all

# 导出模型
python qds_cli.py utils export --type model --source ./model/merged --target ./exports/model.tar.gz

# 导出数据
python qds_cli.py utils export --type data --source ./data/training.jsonl --target ./exports/data.tar.gz

# 导出配置
python qds_cli.py utils export --type config --source ./seeting.jsonc --target ./exports/config.jsonc

# 导入资源
python qds_cli.py utils import --type model --source ./imports/model.tar.gz --target ./model/imported
```

## 全局参数

- `--config, -c`: 指定配置文件路径
- `--verbose, -v`: 详细输出模式
- `--quiet, -q`: 静默模式  
- `--log-level`: 设置日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- `--work-dir`: 设置工作目录

## 配置优先级

1. 命令行参数（最高优先级）
2. 环境变量
3. 用户配置文件 (seeting.jsonc)
4. 默认配置

## 错误处理

CLI 提供了详细的错误信息和建议：

```bash
# 示例错误信息
错误：配置文件 'seeting.jsonc' 不存在
建议：运行 'python qds_cli.py config init' 创建配置文件

# 带解决方案的提示
错误：CUDA内存不足  
建议：
  1. 减少批处理大小：--batch-size 1
  2. 启用梯度检查点：--gradient-checkpointing
  3. 使用更小的模型或量化
```

## 进度指示

长时间运行的操作会显示进度：

```bash
⏳ 正在提取QQ数据... [████████████████████] 100% (1234/1234)
🔄  正在清洗数据... [██████████▒▒▒▒▒▒▒▒▒▒] 50% (500/1000)
🚀 训练进行中... Step 750/1000, Loss: 0.23, ETA: 5分钟
```

## 架构特点

- **企业级设计**: 遵循SOLID原则和设计模式
- **模块化结构**: 清晰的目录结构和职责分离
- **异常处理**: 完整的异常分类和错误恢复
- **参数验证**: 严格的输入验证和安全检查
- **进度监控**: 实时进度显示和状态管理
- **配置管理**: 灵活的配置系统和向导式设置
- **多语言支持**: 中英双语界面（当前主要为中文）

## 开发规范

项目严格遵循 CLAUDE.md 中的开发规范：

- 使用中文输出和注释
- 遵循企业级代码质量标准
- 不使用不必要的emoji
- 保持代码整洁性和可维护性
- 严格的错误处理和异常管理

## 扩展性

CLI设计了插件系统的基础架构，支持：

- 自定义数据处理器
- 模型适配器扩展
- 导出格式支持
- 自定义命令添加