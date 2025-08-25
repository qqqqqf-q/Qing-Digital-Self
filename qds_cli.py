#!/usr/bin/env python3
"""
Qing-Digital-Self CLI 主入口

企业级命令行工具，提供数字分身项目的完整生命周期管理功能。
支持数据处理、模型训练、推理服务等核心操作。

使用示例:
    qds config init                 # 初始化配置
    qds data extract --help         # 查看数据提取帮助
    qds train start                 # 开始训练
    qds infer chat                  # 启动对话模式

支持的命令:
    config  - 配置管理
    data    - 数据处理
    train   - 模型训练
    infer   - 模型推理
    utils   - 工具命令
"""

import sys
import os
import argparse
from typing import List, Optional, Dict, Any

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import QingCLI, CLIError
from utils.logger.logger import get_logger
from utils.config.config import get_config, ConfigError


def create_parser() -> argparse.ArgumentParser:
    """创建主命令行解析器"""
    parser = argparse.ArgumentParser(
        prog='qds',
        description='Qing-Digital-Self CLI - 企业级数字分身项目管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
常用命令示例:
  qds config init                   初始化配置文件
  qds config show                   显示当前配置
  qds data extract                  从QQ数据库提取数据
  qds data clean --method llm       使用LLM清洗数据
  qds train start                   开始模型训练
  qds infer chat                    启动交互式对话

获取更多帮助:
  qds <command> --help              查看特定命令的详细帮助
  
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
        help='指定配置文件路径 (默认: seeting.jsonc)'
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
        help='使用 qds <command> --help 查看详细帮助'
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
    data_extract = data_subparsers.add_parser('extract', help='从QQ数据库提取数据')
    data_extract.add_argument('--qq-db-path', help='QQ数据库文件路径')
    data_extract.add_argument('--qq-number-ai', help='AI的QQ号码(用于区分发送者)')
    data_extract.add_argument('--output', help='输出文件路径')
    data_extract.add_argument('--time-range', help='时间范围 (格式: 2023-01-01,2024-01-01)')
    
    # data clean
    data_clean = data_subparsers.add_parser('clean', help='清洗训练数据')
    data_clean_subparsers = data_clean.add_subparsers(dest='clean_method', help='清洗方法')
    
    # data clean raw
    data_clean_raw = data_clean_subparsers.add_parser('raw', help='使用原始算法清洗数据')
    data_clean_raw.add_argument('--input', help='输入CSV目录路径（默认从配置读取）')
    data_clean_raw.add_argument('--output', help='输出文件路径（默认从配置读取）')
    
    # data clean llm  
    data_clean_llm = data_clean_subparsers.add_parser('llm', help='使用LLM方法清洗数据')
    data_clean_llm.add_argument('--input', help='输入CSV目录路径（默认从配置读取）')
    data_clean_llm.add_argument('--output', help='输出文件路径（默认从配置读取）')
    data_clean_llm.add_argument('--batch-size', type=int, help='批处理大小（默认从配置读取）')
    data_clean_llm.add_argument('--workers', type=int, help='工作进程数（默认从配置读取）')
    
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
    
    # 模型训练命令
    train_parser = subparsers.add_parser(
        'train',
        help='模型训练',
        description='QLoRA微调和模型管理'
    )
    train_subparsers = train_parser.add_subparsers(dest='train_action')
    
    # train start
    train_start = train_subparsers.add_parser('start', help='开始训练')
    train_start.add_argument('--model-path', help='基础模型路径')
    train_start.add_argument('--data-path', help='训练数据路径')
    train_start.add_argument('--output-dir', help='输出目录')
    train_start.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    train_start.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    train_start.add_argument('--batch-size', type=int, default=1, help='批处理大小')
    train_start.add_argument('--max-steps', type=int, default=1000, help='最大训练步数')
    train_start.add_argument('--resume', help='恢复训练检查点路径')
    
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
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())