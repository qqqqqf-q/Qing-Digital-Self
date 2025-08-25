"""
CLI 基础类

定义了CLI框架的基础架构，包括主CLI类和基础命令类。
遵循SOLID原则和企业级设计模式。
"""

import os
import sys
import argparse
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

from utils.logger.logger import get_logger
from utils.config.config import get_config
from .exceptions import CLIError, ConfigurationError, ValidationError
from .helpers import get_system_info


class BaseCommand(ABC):
    """命令基类
    
    所有CLI命令都应该继承此类，实现execute方法。
    提供了统一的命令接口和通用功能。
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = get_logger(f"Command.{name}")
        self.config = get_config()
        
    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """
        执行命令
        
        Args:
            args: 命令行参数
            
        Returns:
            退出码 (0表示成功，非0表示失败)
        """
        pass
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """
        验证命令参数
        
        Args:
            args: 命令行参数
            
        Raises:
            ValidationError: 参数验证失败
        """
        pass
    
    def pre_execute(self, args: argparse.Namespace) -> None:
        """
        执行前的预处理
        
        Args:
            args: 命令行参数
        """
        self.logger.info(f"开始执行命令: {self.name}")
        self.validate_args(args)
    
    def post_execute(self, args: argparse.Namespace, result: int) -> None:
        """
        执行后的后处理
        
        Args:
            args: 命令行参数
            result: 执行结果
        """
        if result == 0:
            self.logger.info(f"命令执行成功: {self.name}")
        else:
            self.logger.error(f"命令执行失败: {self.name} (退出码: {result})")
    
    def run(self, args: argparse.Namespace) -> int:
        """
        运行命令的完整流程
        
        Args:
            args: 命令行参数
            
        Returns:
            退出码
        """
        try:
            self.pre_execute(args)
            result = self.execute(args)
            self.post_execute(args, result)
            return result
        except CLIError:
            raise
        except Exception as e:
            self.logger.error(f"命令执行时发生未预期错误: {e}")
            raise CLIError(f"命令 {self.name} 执行失败: {e}") from e


class QingCLI:
    """主CLI控制器
    
    负责管理所有命令，处理全局配置，以及提供统一的执行入口。
    实现了命令注册机制和插件系统的基础架构。
    """
    
    def __init__(self):
        self.logger = get_logger("QingCLI")
        self.config = get_config()
        self.commands: Dict[str, BaseCommand] = {}
        self._register_commands()
        
    def _register_commands(self) -> None:
        """注册所有可用命令"""
        try:
            # 延迟导入避免循环依赖
            from ..commands import (
                ConfigCommand,
                DataCommand,
                TrainCommand,
                InferCommand,
                UtilsCommand
            )
            
            # 注册命令
            self.register_command(ConfigCommand())
            self.register_command(DataCommand())
            self.register_command(TrainCommand())
            self.register_command(InferCommand())
            self.register_command(UtilsCommand())
            
        except ImportError as e:
            self.logger.error(f"加载命令模块失败: {e}")
            raise CLIError(f"CLI初始化失败: 无法加载命令模块") from e
    
    def register_command(self, command: BaseCommand) -> None:
        """
        注册命令
        
        Args:
            command: 命令实例
        """
        if not isinstance(command, BaseCommand):
            raise ValueError(f"命令必须继承自BaseCommand: {type(command)}")
        
        self.commands[command.name] = command
        self.logger.debug(f"注册命令: {command.name}")
    
    def get_command(self, name: str) -> Optional[BaseCommand]:
        """
        获取命令
        
        Args:
            name: 命令名称
            
        Returns:
            命令实例或None
        """
        return self.commands.get(name)
    
    def list_commands(self) -> List[str]:
        """
        获取所有已注册的命令列表
        
        Returns:
            命令名称列表
        """
        return list(self.commands.keys())
    
    def execute(self, args: argparse.Namespace) -> int:
        """
        执行CLI命令
        
        Args:
            args: 解析后的命令行参数
            
        Returns:
            退出码
        """
        try:
            # 检查是否有主命令
            if not hasattr(args, 'command') or args.command is None:
                self._print_general_help()
                return 0
            
            # 获取命令
            command = self.get_command(args.command)
            if command is None:
                self.logger.error(f"未知命令: {args.command}")
                self._print_available_commands()
                return 1
            
            # 执行命令
            return command.run(args)
            
        except KeyboardInterrupt:
            self.logger.info("操作被用户取消")
            return 130  # SIGINT exit code
        except ConfigurationError as e:
            self.logger.error(f"配置错误: {e}")
            return e.error_code
        except CLIError as e:
            self.logger.error(str(e))
            return e.error_code
        except Exception as e:
            self.logger.error(f"CLI执行时发生未预期错误: {e}")
            return 1
    
    def _print_general_help(self) -> None:
        """打印总体帮助信息"""
        print("Qing-Digital-Self CLI - 企业级数字分身项目管理工具\n")
        print("使用方法:")
        print("  qds <command> [options]\n")
        self._print_available_commands()
        print("\n使用 'qds <command> --help' 获取特定命令的详细帮助")
    
    def _print_available_commands(self) -> None:
        """打印可用命令列表"""
        print("可用命令:")
        for name, command in self.commands.items():
            print(f"  {name:<12} {command.description}")
    
    def get_version_info(self) -> Dict[str, Any]:
        """
        获取版本信息
        
        Returns:
            版本信息字典
        """
        return {
            "cli_version": "0.1.0",
            "python_version": sys.version,
            "platform": sys.platform,
            "config_version": self.config.get("version", "unknown"),
            "system_info": get_system_info(),
        }
    
    def check_environment(self) -> Dict[str, Any]:
        """
        检查运行环境
        
        Returns:
            环境检查结果
        """
        checks = {
            "python_version": {
                "status": "ok" if sys.version_info >= (3, 8) else "warning",
                "message": f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "required": "Python >= 3.8"
            },
            "config_file": {
                "status": "ok" if os.path.exists("seeting.jsonc") else "error",
                "message": "配置文件存在" if os.path.exists("seeting.jsonc") else "配置文件不存在",
                "required": "seeting.jsonc"
            },
            "dependencies": self._check_dependencies(),
        }
        
        return checks
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """检查关键依赖"""
        required_packages = [
            "torch",
            "transformers",
            "datasets",
            "peft",
            "trl",
            "openai",
            "colorama",
            "psutil",
        ]
        
        missing_packages = []
        available_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                available_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        status = "ok" if not missing_packages else "warning"
        
        return {
            "status": status,
            "message": f"可用包: {len(available_packages)}/{len(required_packages)}",
            "available": available_packages,
            "missing": missing_packages,
        }
    
    def initialize_project(self, project_dir: Optional[str] = None) -> None:
        """
        初始化项目目录结构
        
        Args:
            project_dir: 项目目录路径，默认为当前目录
        """
        if project_dir:
            os.makedirs(project_dir, exist_ok=True)
            os.chdir(project_dir)
        
        # 创建必要的目录
        directories = [
            "models",
            "checkpoints",
            "logs",
            "configs",
            "scripts",
            "outputs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"创建目录: {directory}")
        
        # 复制配置模板
        template_path = "seeting_template.jsonc"
        config_path = "seeting.jsonc"
        
        if os.path.exists(template_path) and not os.path.exists(config_path):
            import shutil
            shutil.copy2(template_path, config_path)
            self.logger.info(f"创建配置文件: {config_path}")
        
        self.logger.info("项目初始化完成")