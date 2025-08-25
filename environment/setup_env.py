#!/usr/bin/env python3
"""
环境安装脚本

按照environment.md的要求，提供完整的环境设置流程
可以作为独立脚本运行，也可以被其他模块调用
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from environment import EnvironmentManager
from utils.logger.logger import get_logger


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Qing-Agent环境安装工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --install                  # 开始环境安装流程
  %(prog)s --check                    # 只检查当前环境状态
  %(prog)s --venv my_env --install    # 使用指定虚拟环境名称
        """
    )
    
    parser.add_argument(
        '--install', 
        action='store_true',
        help='开始环境安装流程'
    )
    
    parser.add_argument(
        '--check', 
        action='store_true',
        help='检查环境状态'
    )
    
    parser.add_argument(
        '--venv', 
        type=str,
        help='虚拟环境名称（默认: qds_env）'
    )
    
    parser.add_argument(
        '--requirements',
        type=str,
        default='requirements.txt',
        help='requirements文件路径（默认: requirements.txt）'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='跳过环境验证步骤'
    )
    
    args = parser.parse_args()
    
    # 创建环境管理器
    logger = get_logger('EnvironmentSetup')
    env_manager = EnvironmentManager(logger)
    
    try:
        if args.check:
            # 只检查环境状态
            logger.info("检查环境状态...")
            status = env_manager.get_environment_status(args.venv, detect_cuda=True)
            
            print("\n=== 环境状态报告 ===")
            
            # Python状态
            python_status = status['python']
            print(f"Python: {python_status['version_string']} ({python_status['status']})")
            
            # 虚拟环境状态
            venv_status = status['venv']
            venv_exists = "存在" if venv_status['exists'] else "不存在"
            print(f"虚拟环境: {venv_status['name']} ({venv_exists})")
            
            # CUDA状态
            cuda_status = status['cuda']
            if cuda_status['cuda']['available']:
                print(f"CUDA: {cuda_status['cuda']['version']}")
                for gpu in cuda_status['gpus']:
                    print(f"  GPU: {gpu['name']} ({gpu['memory']})")
            else:
                print("CUDA: 不可用")
            
            # 包状态
            packages_status = status['packages']
            print(f"PyTorch: {'已安装' if packages_status['pytorch'] else '未安装'}")
            print(f"Unsloth: {'已安装' if packages_status['unsloth'] else '未安装'}")
            
            for pkg, installed in packages_status['ml_packages'].items():
                print(f"{pkg}: {'已安装' if installed else '未安装'}")
            
        elif args.install:
            # 执行安装流程
            logger.info("开始环境安装...")
            
            success = env_manager.complete_environment_setup(
                venv_name=args.venv
            )
            
            if success:
                logger.info("环境设置完成!")
                
                if not args.skip_validation:
                    logger.info("开始最终验证...")
                    validation_success = env_manager.validate_environment(args.venv)
                    if validation_success:
                        logger.info("环境验证通过!")
                        sys.exit(0)
                    else:
                        logger.warning("环境验证失败，但安装可能仍然可用")
                        sys.exit(1)
                else:
                    logger.info("跳过验证步骤")
                    sys.exit(0)
            else:
                logger.error("环境设置失败!")
                sys.exit(1)
        
        else:
            # 没有指定操作，显示帮助
            parser.print_help()
            print("\n请选择一个操作:")
            print("  --install : 开始环境安装流程（交互式选择）")
            print("  --check   : 检查当前环境状态")
    
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"发生意外错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()