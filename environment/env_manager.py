"""
环境管理主模块

整合所有环境管理功能，提供统一的接口
按照environment.md的要求实现完整的环境安装流程
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.config.config import get_config
from utils.logger.logger import get_logger

from .python_checker import PythonChecker
from .venv_manager import VenvManager
from .cuda_detector import CudaDetector
from .dependency_installer import DependencyInstaller
from .env_validator import EnvironmentValidator
from .download.model_download import ModelDownloader


class EnvironmentManager:
    """环境管理器主类"""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger('EnvironmentManager')
        self.config = get_config()
        
        # 初始化子模块
        self.python_checker = PythonChecker(self.logger)
        self.venv_manager = VenvManager(self.logger)
        self.cuda_detector = CudaDetector(self.logger)
        self.dependency_installer = DependencyInstaller(
            self.venv_manager, self.cuda_detector, self.logger
        )
        self.env_validator = EnvironmentValidator(self.venv_manager, self.logger)
        self.model_downloader = ModelDownloader(self.logger)
    
    def _log(self, level: str, message: str):
        """内部日志方法"""
        if self.logger:
            getattr(self.logger, level.lower())(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def check_python_requirements(self) -> bool:
        """检查Python版本要求 (步骤1)"""
        self._log("info", "=== 检查Python版本 ===")
        
        return self.python_checker.validate_environment()
    
    def setup_virtual_environment(self, venv_name: str = None) -> bool:
        """设置虚拟环境 (步骤2)"""
        if venv_name is None:
            venv_name = VenvManager.DEFAULT_VENV_NAME
        
        self._log("info", "=== 创建虚拟环境 ===")
        
        # 检查是否已存在
        if self.venv_manager.check_venv_exists(venv_name):
            self._log("info", f"虚拟环境已存在: {venv_name}")
            return True
        
        # 创建虚拟环境
        success = self.venv_manager.create_venv(venv_name)
        
        if success:
            # 显示激活命令
            activation_cmd = self.venv_manager.get_activation_command(venv_name)
            self._log("info", "虚拟环境创建成功")
            self._log("info", f"激活命令: {activation_cmd}")
            
            # 在Linux/Mac上显示source命令，Windows上显示bat命令
            if os.name == 'nt':
                self._log("info", f"Windows PowerShell用户请运行: {activation_cmd}")
            else:
                self._log("info", f"Linux/Mac用户请运行: {activation_cmd}")
        
        return success
    
    def install_basic_dependencies(self, venv_name: str = None, 
                                  requirements_file: str = "requirements.txt") -> bool:
        """安装基础依赖 (步骤3)"""
        self._log("info", "=== 安装基础依赖 ===")
        
        return self.dependency_installer.install_requirements(requirements_file, venv_name)
    
    def setup_machine_learning_environment(self, venv_name: str = None) -> bool:
        """设置机器学习环境 (步骤4) - 按照environment.md要求提供用户选择"""
        self._log("info", "=== 设置机器学习环境 ===")
        
        while True:
            self._log("info", "请选择机器学习依赖安装方式:")
            self._log("info", "1. CUDA自动安装 (torch + unsloth)")
            self._log("info", "2. 手动安装")
            self._log("info", "3. 跳过机器学习环境安装")
            
            try:
                choice = input("请输入选择 (1-3): ").strip()
                
                if choice == '1':
                    return self._cuda_auto_setup(venv_name)
                elif choice == '2':
                    return self._manual_ml_setup(venv_name)
                elif choice == '3':
                    self._log("info", "跳过机器学习环境安装")
                    self._log("info", "您可以后续手动安装 torch, unsloth 等依赖")
                    return True
                else:
                    self._log("warning", "无效选择，请重新输入")
                    continue
                    
            except (KeyboardInterrupt, EOFError):
                self._log("info", "用户中断操作")
                return False
    
    def _cuda_auto_setup(self, venv_name: str = None) -> bool:
        """CUDA自动安装 (torch + unsloth) - 严格按照environment.md要求"""
        self._log("info", "--- CUDA自动安装模式 ---")
        
        # 1. 检测CUDA版本 (避免重复检测)
        cuda_version = self.cuda_detector.detect_cuda_version()
        
        if cuda_version:
            self._log("info", f"检测到CUDA版本: {cuda_version}")
            # 显示GPU信息
            gpu_info = self.cuda_detector.get_gpu_info()
            if gpu_info:
                for i, gpu in enumerate(gpu_info):
                    self._log("info", f"GPU {i}: {gpu['name']} ({gpu['memory']})")
        else:
            self._log("warning", "未检测到CUDA，将使用CPU版本")
        
        # 2. 先安装PyTorch (按照environment.md要求)
        self._log("info", "第1步: 安装PyTorch...")
        recommendation = self.cuda_detector.recommend_pytorch_version(cuda_version)
        
        if recommendation['recommended']:
            self._log("info", f"推荐使用: {recommendation['note']}")
        else:
            self._log("warning", recommendation['note'])
        
        # 安装PyTorch，传递已检测的cuda_version避免重复检测
        pytorch_success = self.dependency_installer.install_pytorch(venv_name, cuda_version)
        if not pytorch_success:
            self._log("error", "PyTorch安装失败")
            return False
        
        # 3. 再安装Unsloth (按照environment.md要求)
        self._log("info", "第2步: 安装Unsloth...")
        unsloth_success = self.dependency_installer.install_unsloth_auto(venv_name)
        if not unsloth_success:
            self._log("warning", "Unsloth自动安装失败，但PyTorch已安装")
        
        # 4. 检查并安装其他必需包 (peft, transformers等)
        self._log("info", "第3步: 检查其他必需包...")
        required_packages = ['peft', 'transformers']
        
        for package in required_packages:
            if not self.dependency_installer.check_package_installed(package, venv_name):
                self._log("info", f"安装缺失的包: {package}")
                self.dependency_installer.install_package(package, venv_name)
        
        return True
    
    def _manual_ml_setup(self, venv_name: str = None) -> bool:
        """手动机器学习环境设置 - 按照environment.md要求"""
        self._log("info", "--- 手动安装模式 ---")
        
        while True:
            self._log("info", "\n请选择手动安装方式:")
            self._log("info", "1. 选择预设版本 (PyTorch, Unsloth, Transformers, PEFT)")
            self._log("info", "2. 自由输入命令模式")
            
            try:
                choice = input("请输入选择 (1-2): ").strip()
                
                if choice == '1':
                    return self._manual_preset_selection(venv_name)
                elif choice == '2':
                    return self._manual_command_mode(venv_name)
                else:
                    self._log("warning", "无效选择，请重新输入")
                    
            except (KeyboardInterrupt, EOFError):
                self._log("info", "用户中断操作")
                return False
    
    def _manual_preset_selection(self, venv_name: str = None) -> bool:
        """手动预设版本选择 - 按照environment.md要求"""
        packages = ['torch', 'unsloth', 'transformers', 'peft']
        
        for package in packages:
            self._log("info", f"\n--- 选择 {package} 版本 ---")
            
            if package == 'torch':
                # PyTorch特殊处理，显示所有可用版本
                if not self._install_pytorch_manually(venv_name):
                    return False
            else:
                # 其他包的手动安装
                if not self._install_package_manually(package, venv_name):
                    return False
        
        return True
    
    def _install_pytorch_manually(self, venv_name: str = None) -> bool:
        """手动安装PyTorch"""
        versions = self.cuda_detector.get_all_pytorch_versions()
        
        self._log("info", "可用的PyTorch版本:")
        for i, version in enumerate(versions, 1):
            recommended = " (推荐)" if version['recommended'] else ""
            self._log("info", f"{i}. {version['description']}{recommended}")
            self._log("info", f"   安装命令: {version['install_command']}")
        
        while True:
            try:
                choice = input(f"请选择PyTorch版本 (1-{len(versions)}) 或输入自定义版本: ").strip()
                
                # 检查是否是数字选择
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(versions):
                        selected_version = versions[index]
                        install_cmd = selected_version['install_command']
                        
                        if install_cmd.startswith("pip install "):
                            install_args = install_cmd[12:].split()
                            result = self.venv_manager.run_in_venv(
                                ["pip", "install"] + install_args,
                                venv_name,
                                capture_output=False,
                                timeout=900
                            )
                            return result.returncode == 0
                    else:
                        self._log("warning", "无效选择")
                        continue
                        
                except ValueError:
                    # 用户输入自定义版本
                    if choice.startswith("torch=="):
                        result = self.venv_manager.run_in_venv(
                            ["pip", "install", choice],
                            venv_name,
                            capture_output=False,
                            timeout=900
                        )
                        return result.returncode == 0
                    else:
                        self._log("warning", "自定义版本格式应为: torch==版本号")
                        continue
                        
            except (KeyboardInterrupt, EOFError):
                return False
    
    def _install_package_manually(self, package: str, venv_name: str = None) -> bool:
        """手动安装指定包"""
        while True:
            try:
                self._log("info", f"请输入{package}版本 (格式: {package}==版本号) 或直接按Enter使用最新版本:")
                user_input = input(">>> ").strip()
                
                if not user_input:
                    # 使用最新版本
                    package_spec = package
                elif user_input.startswith(f"{package}=="):
                    package_spec = user_input
                else:
                    self._log("warning", f"格式错误，应为: {package}==版本号")
                    continue
                
                result = self.venv_manager.run_in_venv(
                    ["pip", "install", package_spec],
                    venv_name,
                    capture_output=False,
                    timeout=600
                )
                
                if result.returncode == 0:
                    return True
                else:
                    retry = input("安装失败，是否重试? (y/N): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return False
                        
            except (KeyboardInterrupt, EOFError):
                return False
    
    def _manual_command_mode(self, venv_name: str = None) -> bool:
        """自由命令输入模式 - 按照environment.md要求"""
        self._log("info", "进入自由命令模式 (保持环境激活)")
        self._log("info", "输入 'exit' 退出")
        self._log("info", "pip install 命令将直接执行")
        self._log("info", "其他命令将提示确认")
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if user_input.lower() == 'exit':
                    self._log("info", "退出命令模式")
                    break
                
                if not user_input:
                    continue
                
                if user_input.startswith('pip install'):
                    # 直接执行pip install命令
                    cmd_parts = user_input.split()
                    try:
                        result = self.venv_manager.run_in_venv(
                            cmd_parts, venv_name, capture_output=False, timeout=600
                        )
                        if result.returncode == 0:
                            self._log("info", "命令执行成功")
                        else:
                            self._log("error", "命令执行失败")
                    except Exception as e:
                        self._log("error", f"执行命令时发生错误: {e}")
                        
                else:
                    # 其他命令需要确认
                    confirm = input(f"确认要执行命令 '{user_input}' 吗? (y/N): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        try:
                            cmd_parts = user_input.split()
                            result = self.venv_manager.run_in_venv(
                                cmd_parts, venv_name, capture_output=False, timeout=300
                            )
                            if result.returncode == 0:
                                self._log("info", "命令执行成功")
                            else:
                                self._log("error", "命令执行失败")
                        except Exception as e:
                            self._log("error", f"执行命令时发生错误: {e}")
                    else:
                        self._log("info", "取消执行")
                        
            except (KeyboardInterrupt, EOFError):
                self._log("info", "用户中断，退出命令模式")
                break
        
        return True
    
    def validate_environment(self, venv_name: str = None) -> bool:
        """验证环境设置 (步骤5)"""
        self._log("info", "=== 验证环境 ===")
        
        check_result = self.env_validator.run_comprehensive_check(venv_name)
        
        # 生成并显示报告
        report = self.env_validator.generate_report(check_result)
        self._log("info", "\n" + report)
        
        return check_result['success']
    
    def complete_environment_setup(self, venv_name: str = None) -> bool:
        """完整的环境设置流程 - 按照environment.md要求"""
        self._log("info", "开始环境设置流程...")
        
        try:
            # 步骤1: 检查Python版本
            if not self.check_python_requirements():
                self._log("error", "Python版本检查失败，请安装Python 3.10+")
                return False
            
            # 步骤2: 创建虚拟环境
            if not self.setup_virtual_environment(venv_name):
                self._log("error", "虚拟环境创建失败")
                return False
            
            # 步骤3: 安装基础依赖
            if not self.install_basic_dependencies(venv_name):
                self._log("error", "基础依赖安装失败")
                return False
            
            # 步骤4: 设置机器学习环境 (用户选择模式)
            if not self.setup_machine_learning_environment(venv_name):
                self._log("error", "机器学习环境设置失败")
                return False
            
            # 步骤5: 验证环境 (询问用户是否需要验证)
            self._log("info", "是否需要验证环境? (验证torch, unsloth等ML包)")
            verify_choice = input("验证环境? (y/N): ").strip().lower()
            
            if verify_choice in ['y', 'yes']:
                if not self.validate_environment(venv_name):
                    self._log("warning", "环境验证失败，但基础环境已可用")
            else:
                self._log("info", "跳过环境验证")
            
            self._log("info", "环境设置完成!")
            
            # 显示使用说明
            self._show_usage_instructions(venv_name)
            
            return True
            
        except KeyboardInterrupt:
            self._log("info", "用户中断环境设置")
            return False
        except Exception as e:
            self._log("error", f"环境设置过程中发生错误: {e}")
            return False
    
    def _show_usage_instructions(self, venv_name: str = None):
        """显示使用说明"""
        activation_cmd = self.venv_manager.get_activation_command(venv_name)
        
        self._log("info", "\n=== 使用说明 ===")
        self._log("info", "1. 激活虚拟环境:")
        self._log("info", f"   {activation_cmd}")
        self._log("info", "")
        self._log("info", "2. 可用的项目功能:")
        self._log("info", "   - 数据处理和清理")
        self._log("info", "   - 配置文件管理")
        self._log("info", "   - 日志记录系统")
        self._log("info", "")
        self._log("info", "3. 如果需要机器学习功能:")
        self._log("info", "   可以手动安装: pip install torch transformers unsloth")
        self._log("info", "")
        self._log("info", "4. 完成后使用 'deactivate' 退出虚拟环境")
    
    def get_environment_status(self, venv_name: str = None, detect_cuda: bool = True) -> Dict:
        """获取环境状态信息"""
        cuda_info = self.cuda_detector.get_system_info(detect_fresh=detect_cuda)
        
        return {
            'python': self.python_checker.check_current_python(),
            'venv': self.venv_manager.get_venv_info(venv_name),
            'cuda': cuda_info,
            'packages': self.dependency_installer.get_installation_status(venv_name)
        }