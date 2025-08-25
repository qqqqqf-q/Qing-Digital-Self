# Environment 环境管理模块

## 概述

环境管理模块提供完整的Python环境检测、虚拟环境创建、依赖安装和验证功能。严格按照`environment.md`的架构要求设计，仅使用Python标准库实现，支持Windows、Linux、macOS跨平台操作。

## 特性

- ✅ Python版本检测（要求3.10+，推荐3.12）
- ✅ 虚拟环境自动创建和管理
- ✅ 基础依赖自动安装（requirements.txt）
- ✅ CUDA检测和PyTorch版本选择
- ✅ Unsloth自动安装
- ✅ 手动安装模式
- ✅ 环境验证和报告生成
- ✅ 模型下载（ModelScope/HuggingFace）

## 快速开始

### 自动安装（推荐）

```bash
python environment/setup_env.py --auto
```

### 手动安装

```bash
python environment/setup_env.py --manual
```

### 检查环境状态

```bash
python environment/setup_env.py --check
```

## 目录结构

```
environment/
├── __init__.py              # 模块初始化
├── env_manager.py           # 主环境管理器
├── python_checker.py        # Python版本检测
├── venv_manager.py          # 虚拟环境管理
├── cuda_detector.py         # CUDA检测和PyTorch选择
├── dependency_installer.py  # 依赖安装器
├── env_validator.py         # 环境验证器
├── setup_env.py            # 主入口脚本
├── download/               # 模型下载模块
│   ├── __init__.py
│   └── model_download.py   # 模型下载器
└── README.md              # 说明文档
```

## 使用方法

### 1. 作为独立脚本

```bash
# 完整安装流程
python environment/setup_env.py --auto

# 手动选择安装
python environment/setup_env.py --manual

# 指定虚拟环境名称
python environment/setup_env.py --auto --venv my_env

# 检查环境状态
python environment/setup_env.py --check
```

### 2. 在代码中使用

```python
from environment import EnvironmentManager

# 创建环境管理器
env_manager = EnvironmentManager()

# 完整安装流程
success = env_manager.complete_environment_setup(
    venv_name="qds_env",
    ml_mode="auto"
)

# 检查环境状态
status = env_manager.get_environment_status()
```

### 3. 单独使用子模块

```python
from environment import PythonChecker, VenvManager, CudaDetector

# Python版本检查
checker = PythonChecker()
is_valid = checker.validate_environment()

# 虚拟环境管理
venv = VenvManager()
venv.create_venv("my_env")

# CUDA检测
cuda = CudaDetector()
system_info = cuda.get_system_info()
```

## 安装流程

环境安装按以下步骤进行：

1. **Python版本检测**
   - 检测当前Python版本
   - 要求Python 3.10+
   - 推荐Python 3.12
   - 显示版本兼容性警告

2. **虚拟环境创建**
   - 创建名为`qds_env`的虚拟环境（可自定义）
   - 支持Windows和Unix系统
   - 提供激活命令

3. **基础依赖安装**
   - 安装requirements.txt中的依赖
   - 包含非机器学习相关的基础包

4. **机器学习环境设置**
   
   **自动模式：**
   - CUDA检测
   - 自动选择合适的PyTorch版本
   - 自动安装Unsloth
   - 安装peft、transformers等ML包
   
   **手动模式：**
   - 显示所有可用PyTorch版本
   - 用户手动选择
   - 交互式命令行安装

5. **环境验证**
   - 验证所有关键包可正常导入
   - 生成详细验证报告
   - 检查PyTorch、CUDA、Unsloth等

## 模型下载

支持从ModelScope和HuggingFace下载模型：

```bash
# 使用配置文件中的设置
python environment/download/model_download.py

# 指定参数下载
python environment/download/model_download.py \
  --model-repo Qwen/Qwen-3-8B-Base \
  --model-path ./model/Qwen-3-8B-Base \
  --download-source modelscope

# 列出已下载的模型
python environment/download/model_download.py --list

# 查看模型信息
python environment/download/model_download.py --info ./model/Qwen-3-8B-Base
```

## 支持的PyTorch版本

- CUDA 12.6 (推荐)
- CUDA 12.8 (推荐) 
- CUDA 12.9 (推荐)
- CUDA 12.1
- CUDA 11.8
- CPU版本

## 跨平台兼容性

### Windows
- 支持PowerShell和CMD
- 自动处理路径分隔符
- 虚拟环境激活：`qds_env\Scripts\activate.bat`

### Linux/macOS
- 支持Bash、Zsh等Shell
- Unix路径处理
- 虚拟环境激活：`source qds_env/bin/activate`

## 配置集成

自动读取项目配置文件`seeting.jsonc`中的设置：

- `model_repo`: 默认模型仓库
- `model_path`: 默认模型路径
- `download_source`: 下载源（modelscope/huggingface）

## 错误处理

- 网络超时自动重试
- 备用下载地址（Unsloth）
- 详细错误日志
- 用户友好的错误提示

## 日志系统

集成项目日志系统：

- 彩色控制台输出
- 文件日志记录
- 多语言支持（中文/英文）
- 可配置日志级别

## 注意事项

1. 确保有足够的磁盘空间（机器学习环境需要几GB）
2. 网络环境良好（部分包较大）
3. 对于公司网络，可能需要配置代理
4. Windows用户建议使用PowerShell
5. 首次安装Unsloth可能需要较长时间

## 故障排除

### Python版本问题
```bash
# 检查可用Python版本
python --version
python3 --version
python3.12 --version
```

### 虚拟环境问题
```bash
# 删除并重新创建
rmdir /s qds_env  # Windows
rm -rf qds_env    # Linux/macOS
```

### CUDA问题
```bash
# 检查CUDA
nvcc --version
nvidia-smi
```

### 网络问题
```bash
# 使用代理
pip install --proxy http://proxy:port package
```

## 开发

模块采用SOLID原则设计：

- 单一职责：每个模块专注特定功能
- 开闭原则：易于扩展新的安装源
- 接口隔离：清晰的模块接口
- 依赖注入：可测试的设计

仅使用Python标准库，无外部依赖。