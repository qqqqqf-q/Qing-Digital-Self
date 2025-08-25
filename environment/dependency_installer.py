"""
依赖安装模块

处理基础依赖、机器学习依赖的安装
支持自动和手动安装模式
仅使用标准库实现
"""

import subprocess
import urllib.request
import urllib.error
import os
import tempfile
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class DependencyInstaller:
    """依赖安装器"""
    
    # Unsloth安装脚本URL
    UNSLOTH_INSTALL_URLS = [
        "https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py",
        "https://ghfast.top/https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py"
    ]
    
    # 必需的ML相关包（这些通常会被unsloth自动安装）
    REQUIRED_ML_PACKAGES = [
        "bitsandbytes",
        "numpy", 
        "peft",
        "torch",
        "transformers",
        "accelerate",
        "xformers",
        "triton"
    ]
    
    def __init__(self, venv_manager, cuda_detector, logger=None):
        self.venv_manager = venv_manager
        self.cuda_detector = cuda_detector
        self.logger = logger
        
    def _log(self, level: str, message: str):
        """内部日志方法"""
        if self.logger:
            getattr(self.logger, level.lower())(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def install_requirements(self, requirements_file: str = "requirements.txt", 
                           venv_name: str = None) -> bool:
        """安装requirements.txt中的依赖"""
        if not os.path.exists(requirements_file):
            self._log("error", f"requirements文件不存在: {requirements_file}")
            return False
        
        try:
            self._log("info", f"正在安装基础依赖: {requirements_file}")
            
            result = self.venv_manager.run_in_venv(
                ["pip", "install", "-r", requirements_file],
                venv_name,
                timeout=600  # 10分钟超时
            )
            
            if result.returncode == 0:
                self._log("info", "基础依赖安装成功")
                return True
            else:
                self._log("error", f"基础依赖安装失败: {result.stderr}")
                return False
                
        except Exception as e:
            self._log("error", f"安装基础依赖时发生错误: {e}")
            return False
    
    def install_pytorch(self, venv_name: str = None, cuda_version: str = None) -> bool:
        """安装PyTorch"""
        # 获取推荐版本，传入cuda_version避免重复检测
        recommendation = self.cuda_detector.recommend_pytorch_version(cuda_version)
        
        if not recommendation['install_command']:
            self._log("error", "无法确定PyTorch安装命令")
            return False
        
        try:
            # 提取pip install命令的参数
            install_cmd = recommendation['install_command']
            if install_cmd.startswith("pip install "):
                install_args = install_cmd[12:].split()  # 移除"pip install "
            else:
                self._log("error", f"无效的安装命令格式: {install_cmd}")
                return False
            
            self._log("info", f"正在安装PyTorch: {install_cmd}")
            self._log("info", recommendation['note'])
            
            result = self.venv_manager.run_in_venv(
                ["pip", "install", "--upgrade"] + install_args,
                venv_name,
                timeout=900  # 15分钟超时
            )
            
            if result.returncode == 0:
                self._log("info", "PyTorch安装成功")
                return True
            else:
                self._log("error", f"PyTorch安装失败: {result.stderr}")
                return False
                
        except Exception as e:
            self._log("error", f"安装PyTorch时发生错误: {e}")
            return False
    
    def download_unsloth_installer(self) -> Optional[str]:
        """下载Unsloth安装脚本"""
        for url in self.UNSLOTH_INSTALL_URLS:
            try:
                self._log("info", f"尝试从以下地址下载Unsloth安装脚本: {url}")
                
                # 创建临时文件
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as temp_file:
                    temp_filename = temp_file.name
                
                # 下载文件
                with urllib.request.urlopen(url, timeout=30) as response:
                    content = response.read().decode('utf-8')
                    with open(temp_filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                self._log("info", f"Unsloth安装脚本下载成功: {temp_filename}")
                return temp_filename
                
            except urllib.error.URLError as e:
                self._log("warning", f"下载失败 {url}: {e}")
                continue
            except Exception as e:
                self._log("warning", f"下载脚本时发生错误 {url}: {e}")
                continue
        
        self._log("error", "所有Unsloth安装脚本地址都无法访问")
        return None
    
    def install_unsloth_auto(self, venv_name: str = None) -> bool:
        """自动安装Unsloth（会自动安装所有相关依赖）"""
        # 下载安装脚本
        script_path = self.download_unsloth_installer()
        if not script_path:
            return False
        
        try:
            self._log("info", "运行Unsloth自动安装脚本获取安装命令")
            
            # 在虚拟环境中运行安装脚本
            result = self.venv_manager.run_in_venv(
                ["python", script_path],
                venv_name,
                timeout=300  # 5分钟超时
            )
            
            # 清理临时文件
            try:
                os.unlink(script_path)
            except:
                pass
            
            if result.returncode == 0:
                # 脚本会输出安装命令
                install_command = result.stdout.strip()
                
                if install_command and "pip install" in install_command:
                    self._log("info", f"获得Unsloth安装命令: {install_command}")
                    self._log("info", "正在执行Unsloth安装（包含所有深度学习依赖）...")
                    
                    # 直接执行安装命令
                    if install_command.startswith("pip install"):
                        # 提取pip参数
                        pip_args = install_command.split()[2:]  # 跳过"pip install"
                        
                        install_result = self.venv_manager.run_in_venv(
                            ["pip", "install"] + pip_args,
                            venv_name,
                            capture_output=False,  # 实时显示输出
                            timeout=1800  # 30分钟超时，unsloth安装可能很慢
                        )
                        
                        if install_result.returncode == 0:
                            self._log("info", "Unsloth及相关依赖安装成功")
                            return True
                        else:
                            self._log("error", "Unsloth安装失败")
                            return False
                    else:
                        self._log("error", f"无法解析安装命令: {install_command}")
                        return False
                else:
                    self._log("warning", "未获得有效的安装命令")
                    self._log("info", "脚本输出:")
                    self._log("info", result.stdout)
                    return False
            else:
                self._log("error", f"Unsloth安装脚本运行失败: {result.stderr}")
                return False
                
        except Exception as e:
            self._log("error", f"运行Unsloth安装脚本时发生错误: {e}")
            # 清理临时文件
            try:
                os.unlink(script_path)
            except:
                pass
            return False
    
    def install_package(self, package: str, venv_name: str = None, 
                       upgrade: bool = True) -> bool:
        """安装单个包"""
        return self.venv_manager.install_package_in_venv(
            package, venv_name, upgrade
        )
    
    def install_ml_dependencies(self, venv_name: str = None) -> bool:
        """安装机器学习相关依赖"""
        success_count = 0
        total_packages = len(self.REQUIRED_ML_PACKAGES)
        
        for package in self.REQUIRED_ML_PACKAGES:
            if self.install_package(package, venv_name):
                success_count += 1
            else:
                self._log("error", f"安装失败: {package}")
        
        if success_count == total_packages:
            self._log("info", "所有机器学习依赖安装成功")
            return True
        else:
            self._log("warning", f"部分依赖安装失败 ({success_count}/{total_packages})")
            return False
    
    def check_package_installed(self, package: str, venv_name: str = None) -> bool:
        """检查包是否已安装"""
        try:
            result = self.venv_manager.run_in_venv(
                ["python", "-c", f"import {package}"],
                venv_name,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_installation_status(self, venv_name: str = None) -> Dict:
        """获取安装状态"""
        status = {
            'pytorch': False,
            'unsloth': False,
            'ml_packages': {},
            'all_packages': []
        }
        
        # 检查PyTorch
        status['pytorch'] = self.check_package_installed('torch', venv_name)
        
        # 检查Unsloth
        status['unsloth'] = self.check_package_installed('unsloth', venv_name)
        
        # 检查机器学习包
        for package in self.REQUIRED_ML_PACKAGES:
            status['ml_packages'][package] = self.check_package_installed(package, venv_name)
        
        # 获取所有已安装的包
        status['all_packages'] = self.venv_manager.list_installed_packages(venv_name)
        
        return status
    
    def interactive_package_install(self, venv_name: str = None):
        """交互式安装包"""
        self._log("info", "进入交互安装模式")
        self._log("info", "输入pip命令进行安装，输入'exit'退出")
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if user_input.lower() == 'exit':
                    self._log("info", "退出交互安装模式")
                    break
                
                if not user_input:
                    continue
                
                if user_input.startswith('pip install'):
                    # 直接执行pip安装命令
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
                        
            except KeyboardInterrupt:
                self._log("info", "\n用户中断，退出交互模式")
                break
            except EOFError:
                self._log("info", "\n退出交互安装模式")
                break