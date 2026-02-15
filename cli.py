#!/usr/bin/env python3
"""
Qing-Digital-Self CLI 主入口

企业级命令行工具，提供数字分身项目的完整生命周期管理功能。
支持数据处理、模型训练、推理服务等核心操作。

使用示例:
    python cli.py config init
    python cli.py data extract --help
    python cli.py train start
    python cli.py infer chat

支持的命令:
    config   - 配置管理
    data     - 数据处理
    train    - 模型训练
    infer    - 模型推理
    utils    - 工具命令
    download - 模型下载
"""

import sys
import os
import argparse
from typing import List, Optional, Dict, Any

# 设置NumExpr最大线程数，避免警告信息
os.environ.setdefault('NUMEXPR_MAX_THREADS', '12')

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import QingCLI, CLIError
from utils.logger.logger import get_logger
from utils.config.config import get_config, ConfigError


def create_parser() -> argparse.ArgumentParser:
    """创建主命令行解析器"""
    prog = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "cli.py"
    parser = argparse.ArgumentParser(
        prog=prog,
        description='Qing-Digital-Self CLI - 企业级数字分身项目管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
常用命令示例:
  {prog} config init                   初始化配置文件
  {prog} config show                   显示当前配置
  {prog} data extract                  从聊天数据中提取数据
  {prog} data clean llm --accept-score 3  使用LLM清洗数据(分数阈值3)
  {prog} train start                   开始模型训练
  {prog} infer chat                    启动交互式对话

获取更多帮助:
  {prog} <command> --help              查看特定命令的详细帮助
  
项目地址: https://github.com/qqqqqf-q/Qing-Digital-Self
文档地址: https://github.com/qqqqqf-q/Qing-Digital-Self/docs
        """
    )
    
    # 全局参数
    parser.add_argument(
        '--version', '-V',
        action='version',
        version='Qing-Digital-Self CLI v0.1.0'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='指定配置文件路径 (默认: setting.jsonc)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='设置日志级别'
    )
    
    parser.add_argument(
        '--work-dir',
        type=str,
        help='设置工作目录'
    )
    
    # 子命令
    subparsers = parser.add_subparsers(
        dest='command',
        title='可用命令',
        description='选择要执行的操作',
        help=f'使用 {prog} <command> --help 查看详细帮助'
    )
    
    # 配置管理命令
    config_parser = subparsers.add_parser(
        'config',
        help='配置管理',
        description='管理项目配置文件和设置'
    )
    config_subparsers = config_parser.add_subparsers(dest='config_action')
    
    # config init
    config_init = config_subparsers.add_parser('init', help='初始化配置文件')
    config_init.add_argument('--interactive', action='store_true', help='交互式配置')
    config_init.add_argument('--template', choices=['basic', 'advanced'], default='basic', help='配置模板')
    config_init.add_argument('--force', action='store_true', help='强制覆盖已存在的配置文件')
    
    # config show
    config_show = config_subparsers.add_parser('show', help='显示当前配置')
    config_show.add_argument('--format', choices=['json', 'yaml', 'table'], default='table', help='输出格式')
    
    # config set
    config_set = config_subparsers.add_parser('set', help='设置配置项')
    config_set.add_argument('key', help='配置键')
    config_set.add_argument('value', help='配置值')
    
    # config validate
    config_validate = config_subparsers.add_parser('validate', help='验证配置有效性')
    
    # 数据处理命令
    data_parser = subparsers.add_parser(
        'data',
        help='数据处理',
        description='QQ数据提取、清洗和格式转换'
    )
    data_subparsers = data_parser.add_subparsers(dest='data_action')
    
    # data extract
        # data extract
    data_extract = data_subparsers.add_parser('extract', help='从聊天数据中提取数据（支持QQ、Telegram和WeChat）')
    
    # 数据源选择
    data_extract.add_argument('--source-type', choices=['qq', 'tg', 'telegram', 'wx', 'wechat'], help='指定数据源类型（不指定则自动检测）')
    data_extract.add_argument('--data-dir', help='数据目录路径（默认: ./data/chat/<source>/original/，兼容 ./dataset/original/）')
    data_extract.add_argument('--output', help='输出目录路径（默认: ./runs/chat/<run_id>/csv/）')
    data_extract.add_argument('--run-id', help='指定本次运行ID（默认自动生成 YYYYMMDD_HHMMSS，可配合 --run-tag）')
    data_extract.add_argument('--run-tag', help='自动run_id的后缀标签（如 chat_qq，字符会被清洗为[a-zA-Z0-9_-]）')
    
    # QQ相关参数
    qq_group = data_extract.add_argument_group('QQ数据源参数')
    qq_group.add_argument('--qq-c2c-db-path', dest='qq_c2c_db_path', help='QQ私聊(c2c_msg_table)数据库/SQL文件路径（支持.db/.sql）')
    qq_group.add_argument('--qq-group-db-path', dest='qq_group_db_path', help='QQ群聊(group_msg_table)数据库/SQL文件路径（支持.db/.sql）')
    qq_group.add_argument('--qq-db-path', dest='qq_c2c_db_path', help='QQ数据库文件路径（兼容旧参数，等同于--qq-c2c-db-path）')
    qq_group.add_argument('--qq-number-ai', help='AI的QQ号码（用于区分发送者）')
    
    # Telegram相关参数
    tg_group = data_extract.add_argument_group('Telegram数据源参数')
    tg_group.add_argument('--telegram-chat-id', help='AI的Telegram聊天名称（用于区分发送者）')
    tg_group.add_argument('--tg-data-dir', help='Telegram数据目录（如不指定则使用--data-dir）')
    
    # data clean
    data_clean = data_subparsers.add_parser('clean', help='清洗训练数据')
    data_clean.add_argument('--run-id', help='指定 runs/chat/<run_id> 作为输入/输出上下文（不指定则自动选择最新run）')
    data_clean_subparsers = data_clean.add_subparsers(dest='clean_method', help='清洗方法')
    
    # data clean raw
    data_clean_raw = data_clean_subparsers.add_parser('raw', help='使用原始算法清洗数据')
    data_clean_raw.add_argument('--input', help='输入CSV目录路径（默认: runs/chat/<latest>/csv，兼容 dataset/csv）')
    data_clean_raw.add_argument('--output', help='输出文件路径（默认: runs/chat/<run_id>/sft/train.jsonl）')
    
    # data clean llm
    data_clean_llm = data_clean_subparsers.add_parser('llm', help='使用LLM方法清洗数据')
    data_clean_llm.add_argument('--input', help='输入CSV目录路径（默认: runs/chat/<latest>/csv，兼容 dataset/csv）')
    data_clean_llm.add_argument('--output', help='输出文件路径（默认: runs/chat/<run_id>/sft/train.jsonl）')
    data_clean_llm.add_argument('--parser', choices=['default', 'scoring', 'segment'], default='default',
                               help='LLM清洗策略: default(结构化) / scoring(打分) / segment(预留)')
    data_clean_llm.add_argument('--accept-score', type=int, default=2, choices=[1, 2, 3, 4, 5],
                               help='可接受的最低分数阈值(1-5分，仅用于scoring策略，默认2分)')
    data_clean_llm.add_argument('--batch-size', type=int, help='批处理大小（默认从配置读取）')
    data_clean_llm.add_argument('--workers', type=int, help='工作进程数（默认从配置读取）')
    
    # data clean estimate
    data_clean_estimate = data_clean_subparsers.add_parser('estimate', help='估算清洗资源消耗')
    data_clean_estimate_subparsers = data_clean_estimate.add_subparsers(dest='estimate_method', help='估算策略')
    
    data_clean_estimate_llm = data_clean_estimate_subparsers.add_parser('llm', help='估算LLM清洗字符量')
    data_clean_estimate_llm.add_argument('--input', help='输入CSV目录路径（默认: runs/chat/<latest>/csv，兼容 dataset/csv）')
    data_clean_estimate_llm.add_argument('--parser', choices=['scoring'], default='scoring', help='处理策略')
    data_clean_estimate_llm.add_argument('--accept-score', type=int, default=2, choices=[1, 2, 3, 4, 5],
                                         help='可接受的最低分数阈值(仅用于scoring策略)')
    data_clean_estimate_llm.add_argument('--batch-size', type=int, help='批处理大小（默认从配置读取）')
    data_clean_estimate_llm.add_argument('--workers', type=int, help='工作进程数（默认从配置读取）')
    
    # data clean rellm
    data_clean_rellm = data_clean_subparsers.add_parser('rellm', help='基于已有打分结果重新筛选数据')
    data_clean_rellm.add_argument('--input', help='原始数据输入路径（默认: runs/chat/<latest>/csv，兼容 dataset/csv）')
    data_clean_rellm.add_argument('--scored', help='打分结果CSV路径（默认: 输出路径对应的_scored.csv）')
    data_clean_rellm.add_argument('--output', help='输出文件路径（默认: runs/chat/<run_id>/sft/train.jsonl）')
    data_clean_rellm.add_argument('--accept-score', type=int, default=2, choices=[1, 2, 3, 4, 5],
                                  help='重新筛选可接受的最低分数阈值(1-5分，默认2分)')
    
    # data convert
    data_convert = data_subparsers.add_parser('convert', help='转换数据格式')
    data_convert.add_argument('--input', required=True, help='输入文件路径')
    data_convert.add_argument('--output', required=True, help='输出文件路径')
    data_convert.add_argument('--format', choices=['chatml', 'alpaca', 'sharegpt'], default='chatml', help='目标格式')
    
    # data merge
    data_merge = data_subparsers.add_parser('merge', help='合并多源数据')
    data_merge.add_argument('--inputs', nargs='+', required=True, help='输入文件列表')
    data_merge.add_argument('--output', required=True, help='输出文件路径')
    data_merge.add_argument('--deduplicate', action='store_true', help='去重')
    
    # data preview
    data_preview = data_subparsers.add_parser('preview', help='预览数据样本')
    data_preview.add_argument('--input', required=True, help='输入文件路径')
    data_preview.add_argument('--count', type=int, default=5, help='预览数量')
    
    # data stats
    data_stats = data_subparsers.add_parser('stats', help='显示数据统计')
    data_stats.add_argument('--input', required=True, help='输入文件路径')

    # data migrate-layout（Data Layout v2）
    data_migrate_layout = data_subparsers.add_parser(
        'migrate-layout',
        help='将 legacy 的 dataset/openai_data 迁移到 data/runs 目录结构（默认仅输出计划）'
    )
    data_migrate_layout.add_argument('--apply', action='store_true', help='实际执行迁移（不指定则仅 dry-run）')
    data_migrate_layout.add_argument('--mode', choices=['move', 'copy'], default='move',
                                    help='迁移方式: move(推荐，快速) / copy(保留旧目录)')
    data_migrate_layout.add_argument('--run-id', help='将 legacy 产物归档到 runs/chat/<run_id>/（默认自动生成）')
    data_migrate_layout.add_argument('--run-tag', default='legacy',
                                    help='自动run_id的后缀标签（默认: legacy）')
    data_migrate_layout.add_argument('--skip-openai', action='store_true',
                                    help='跳过 openai_data -> data/openai-export 的迁移')
    data_migrate_layout.add_argument('--data-root', dest='data_root',
                                    help='覆盖 data_root（默认读取配置或 ./data）')
    data_migrate_layout.add_argument('--runs-root', dest='runs_root',
                                    help='覆盖 runs_root（默认读取配置或 ./runs）')
    data_migrate_layout.add_argument('--force', action='store_true',
                                    help='目标已存在时尝试覆盖/合并（谨慎使用）')

    # data openai-distill（OpenAI-Export -> SFT）
    data_openai_distill = data_subparsers.add_parser(
        'openai-distill',
        help='从 ChatGPT 导出(OpenAI-Export)生成训练集（normalized + sft/text.jsonl）'
    )
    data_openai_distill.add_argument('--input', help='conversations.json 文件或包含多个导出的目录（默认: data/openai-export/，递归发现 conversations.json；兼容 openai_data/conversations.json）')
    data_openai_distill.add_argument('--run-id', help='指定本次运行ID（默认自动生成 YYYYMMDD_HHMMSS，可配合 --run-tag）')
    data_openai_distill.add_argument('--run-tag', default='openai4o', help='自动run_id的后缀标签（默认: openai4o）')
    data_openai_distill.add_argument('--allow-models', default='gpt-4o,gpt-4-1', help='允许的 default_model_slug，逗号分隔（默认: gpt-4o,gpt-4-1）')
    data_openai_distill.add_argument('--cutoff-ts', type=float, help='按 conversation.create_time 过滤（Unix 秒，小于该值则丢弃）')
    data_openai_distill.add_argument('--pii-policy', choices=['mask', 'drop', 'keep'], default='mask', help='PII 处理策略: mask(默认) / drop / keep')
    data_openai_distill.add_argument('--keep-system', action='store_true', help='保留 system 消息（默认丢弃，避免学习导出内的 system 提示）')
    data_openai_distill.add_argument('--keep-code', action='store_true', help='保留 content_type=code（默认丢弃，避免把工具痕迹混入文本训练）')
    data_openai_distill.add_argument('--keep-tool', action='store_true', help='保留 tool 角色消息（默认丢弃）')
    data_openai_distill.add_argument('--max-chars', type=int, default=20000, help='单样本最大字符数（默认: 20000，超出则从前裁剪）')
    data_openai_distill.add_argument('--max-messages', type=int, default=80, help='单样本最大消息数（默认: 80，超出则保留末尾）')
    data_openai_distill.add_argument('--data-root', dest='data_root', help='覆盖 data_root（默认读取配置或 ./data）')
    data_openai_distill.add_argument('--runs-root', dest='runs_root', help='覆盖 runs_root（默认读取配置或 ./runs）')

    # data openai-clean（SFT -> Cleaned SFT）
    data_openai_clean = data_subparsers.add_parser(
        'openai-clean',
        help='使用LLM清洗 OpenAI SFT（去技术/工具/搜索痕迹）'
    )
    data_openai_clean.add_argument('--input', help='输入SFT JSONL路径（默认读取 runs/openai-distill/<latest>/sft/text.jsonl）')
    data_openai_clean.add_argument('--distill-run-id', dest='distill_run_id',
                                   help='指定 runs/openai-distill/<run_id>/sft/text.jsonl 作为输入')
    data_openai_clean.add_argument('--run-id', help='指定本次清洗输出run_id（默认自动生成 YYYYMMDD_HHMMSS，可配合 --run-tag）')
    data_openai_clean.add_argument('--run-tag', default='openai-clean', help='自动run_id的后缀标签（默认: openai-clean）')
    data_openai_clean.add_argument('--model', help='覆盖清洗模型（默认读取 clean_set_args.openai_api.model_name）')
    data_openai_clean.add_argument('--temperature', type=float, default=None, help='清洗温度（默认: 0.2）')
    data_openai_clean.add_argument('--max-tokens', type=int, default=None, help='单次清洗最大输出tokens（默认: 4096）')
    data_openai_clean.add_argument('--workers', type=int, default=None, help='并发workers（默认读取 clean_set_args.openai_api.clean_workers）')
    data_openai_clean.add_argument('--max-chars', type=int, default=None, help='单样本最大字符数（默认: 20000）')
    data_openai_clean.add_argument('--max-messages', type=int, default=None, help='单样本最大消息数（默认: 80）')
    data_openai_clean.add_argument('--max-samples', type=int, help='最多处理样本数（用于小规模验证）')
    data_openai_clean.add_argument('--base-prompt', help='输出训练集用的system prompt（默认读取 data_args.openai_sft_system_prompt；传\"*\"或空串表示不添加）')
    data_openai_clean.add_argument('--base-prompt-file', help='从文件读取base prompt（优先级高于--base-prompt）')
    data_openai_clean.add_argument('--no-base-prompt', action='store_true', help='不添加system prompt（覆盖其它设置）')
    data_openai_clean.add_argument('--data-root', dest='data_root', help='覆盖 data_root（默认读取配置或 ./data）')
    data_openai_clean.add_argument('--runs-root', dest='runs_root', help='覆盖 runs_root（默认读取配置或 ./runs）')
    
    # 模型推理命令
    infer_parser = subparsers.add_parser(
        'infer',
        help='模型推理',
        description='模型对话和推理服务'
    )
    infer_subparsers = infer_parser.add_subparsers(dest='infer_action')
    
    # infer chat
    infer_chat = infer_subparsers.add_parser('chat', help='交互式对话')
    infer_chat.add_argument('--model-path', help='模型路径')
    infer_chat.add_argument('--max-length', type=int, default=2048, help='最大生成长度')
    infer_chat.add_argument('--temperature', type=float, default=0.7, help='采样温度')
    infer_chat.add_argument('--top-p', type=float, default=0.9, help='Top-p采样')
    
    # infer serve
    infer_serve = infer_subparsers.add_parser('serve', help='启动API服务')
    infer_serve.add_argument('--model-path', help='模型路径')
    infer_serve.add_argument('--host', default='0.0.0.0', help='服务地址')
    infer_serve.add_argument('--port', type=int, default=8000, help='服务端口')
    infer_serve.add_argument('--workers', type=int, default=1, help='工作进程数')
    
    # infer batch
    infer_batch = infer_subparsers.add_parser('batch', help='批量推理')
    infer_batch.add_argument('--model-path', help='模型路径')
    infer_batch.add_argument('--input', required=True, help='输入文件路径')
    infer_batch.add_argument('--output', required=True, help='输出文件路径')
    infer_batch.add_argument('--batch-size', type=int, default=8, help='批处理大小')
    
    # infer test
    infer_test = infer_subparsers.add_parser('test', help='测试模型效果')
    infer_test.add_argument('--model-path', help='模型路径')
    infer_test.add_argument('--test-data', help='测试数据路径')
    infer_test.add_argument('--metrics', nargs='+', default=['bleu', 'rouge'], help='评估指标')
    
    # 工具命令
    utils_parser = subparsers.add_parser(
        'utils',
        help='工具命令',
        description='系统工具和维护命令'
    )
    utils_subparsers = utils_parser.add_subparsers(dest='utils_action')
    
    # utils check-deps
    utils_check = utils_subparsers.add_parser('check-deps', help='检查依赖')
    utils_check.add_argument('--fix', action='store_true', help='自动修复缺失依赖')
    
    # utils clean-cache
    utils_clean = utils_subparsers.add_parser('clean-cache', help='清理缓存')
    utils_clean.add_argument('--all', action='store_true', help='清理所有缓存')
    
    # utils export
    utils_export = utils_subparsers.add_parser('export', help='导出模型/数据')
    utils_export.add_argument('--type', choices=['model', 'data', 'config'], required=True, help='导出类型')
    utils_export.add_argument('--source', required=True, help='源路径')
    utils_export.add_argument('--target', required=True, help='目标路径')
    
    # utils import
    utils_import = utils_subparsers.add_parser('import', help='导入模型/数据')
    utils_import.add_argument('--type', choices=['model', 'data', 'config'], required=True, help='导入类型')
    utils_import.add_argument('--source', required=True, help='源路径')
    utils_import.add_argument('--target', required=True, help='目标路径')
    
    # 模型管理命令
    model_parser = subparsers.add_parser(
        'model',
        help='模型管理',
        description='模型下载、列表和信息查看'
    )
    model_subparsers = model_parser.add_subparsers(dest='model_action')
    
    # model download
    model_download = model_subparsers.add_parser('download', help='下载模型')
    model_download.add_argument('--model-repo', help='模型仓库 (如: Qwen/Qwen-3-8B-Base)')
    model_download.add_argument('--model-path', help='本地保存路径 (如: ./model/Qwen-3-8B-Base)')
    model_download.add_argument('--download-source', choices=['modelscope', 'huggingface'], help='下载源')
    
    # model list
    model_list = model_subparsers.add_parser('list', help='列出已下载的模型')
    
    # model info
    model_info = model_subparsers.add_parser('info', help='查看模型信息')
    model_info.add_argument('model_path', nargs='?', help='模型路径（可选，不指定则显示所有模型信息）')

    # 训练命令
    train_parser = subparsers.add_parser(
        'train',
        help='模型训练',
        description='启动训练、查看状态、停止训练，或启动 WebUI'
    )
    train_subparsers = train_parser.add_subparsers(dest='train_action')

    # train start（传统脚本训练）
    train_start = train_subparsers.add_parser('start', help='开始训练')
    train_start.add_argument('--model-path', help='基础模型路径')
    train_start.add_argument('--data-path', help='训练数据路径')
    train_start.add_argument('--output-dir', help='输出目录')
    train_start.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    train_start.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    train_start.add_argument('--batch-size', type=int, default=1, help='批大小')
    train_start.add_argument('--max-steps', type=int, default=1000, help='最大训练步数')
    train_start.add_argument('--resume', help='恢复训练的检查点路径')

    # train status
    train_status = train_subparsers.add_parser('status', help='训练状态')
    train_status.add_argument('--follow', action='store_true', help='实时跟踪')
    train_status.add_argument('--output-dir', help='训练输出目录')

    # train stop
    train_stop = train_subparsers.add_parser('stop', help='停止训练')
    train_stop.add_argument('--force', action='store_true', help='强制停止')

    # train merge
    train_merge = train_subparsers.add_parser('merge', help='合并LoRA权重')
    train_merge.add_argument('--base-model', required=True, help='基础模型路径')
    train_merge.add_argument('--lora-path', required=True, help='LoRA权重路径')
    train_merge.add_argument('--output', required=True, help='输出路径')

    # train webui start（LLaMA Factory WebUI）
    train_webui = train_subparsers.add_parser('webui', help='LLaMA Factory WebUI')
    train_webui_sub = train_webui.add_subparsers(dest='webui_action')
    train_webui_start = train_webui_sub.add_parser('start', help='启动 WebUI')
    train_webui_start.add_argument('--host', default='0.0.0.0', help='监听地址')
    train_webui_start.add_argument('--port', type=int, default=7860, help='监听端口')
    train_webui_start.add_argument('--no-browser', action='store_true', help='不自动打开浏览器（上游限制，可能无效）')
    train_webui_start.add_argument('--share', action='store_true', help='开启公网分享 (Gradio)')
    train_webui_start.add_argument('--workdir', default=None, help='工作目录（可选）')

    return parser


def handle_global_args(args: argparse.Namespace) -> None:
    """处理全局参数"""
    logger = get_logger()
    
    # 设置工作目录
    if args.work_dir:
        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir, exist_ok=True)
        os.chdir(args.work_dir)
        logger.info(f"工作目录设置为: {os.getcwd()}")
    
    # 设置日志级别
    if args.log_level:
        logger.set_level(args.log_level)
    elif args.verbose:
        logger.set_level('DEBUG')
    elif args.quiet:
        logger.set_level('ERROR')


def main() -> int:
    """主函数"""
    args: Optional[argparse.Namespace] = None
    try:
        # 创建命令行解析器
        parser = create_parser()
        
        # 解析参数
        if len(sys.argv) == 1:
            parser.print_help()
            return 0
        
        args = parser.parse_args()
        
        # 处理全局参数
        handle_global_args(args)
        
        # 初始化CLI
        cli = QingCLI()
        
        # 执行命令
        result = cli.execute(args)
        return result
        
    except KeyboardInterrupt:
        logger = get_logger()
        logger.info("操作被用户取消")
        print("\n操作已被用户取消")
        return 130  # SIGINT exit code
    except ConfigError as e:
        logger = get_logger()
        logger.error(f"配置错误: {e}")
        print(f"配置错误: {e}")
        return 1
    except CLIError as e:
        logger = get_logger()
        logger.error(f"CLI错误: {e}")
        print(f"CLI错误: {e}")
        return 1
    except Exception as e:
        logger = get_logger()
        logger.error(f"未预期的错误: {e}")
        if args and getattr(args, 'verbose', False):
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
