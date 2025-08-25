"""
虚拟环境管理模块

创建、激活、管理Python虚拟环境
仅使用标准库实现，支持跨平台操作
"""

import os
import sys
import subprocess
import platform
from typing import Optional, Tuple, Dict
from pathlib import Path


class VenvManager:
    """虚拟环境管理器"""
    
    DEFAULT_VENV_NAME = "qds_env"
    
    def __init__(self, logger=None):
        self.logger = logger
        self.system = platform.system().lower()
        
    def _log(self, level: str, message: str):
        """内部日志方法"""
        if self.logger:
            getattr(self.logger, level.lower())(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def get_venv_path(self, venv_name: str = None) -> Path:
        """获取虚拟环境路径"""
        if venv_name is None:
            venv_name = self.DEFAULT_VENV_NAME
        return Path.cwd() / venv_name
    
    def get_activation_script(self, venv_path: Path) -> Path:
        """获取虚拟环境激活脚本路径"""
        if self.system == "windows":
            return venv_path / "Scripts" / "Activate.ps1"
        else:
            return venv_path / "bin" / "activate"
    
    def get_python_executable(self, venv_path: Path) -> Path:
        """获取虚拟环境中的Python可执行文件路径"""
        if self.system == "windows":
            return venv_path / "Scripts" / "python.exe"
        else:
            return venv_path / "bin" / "python"
    
    def get_pip_executable(self, venv_path: Path) -> Path:
        """获取虚拟环境中的pip可执行文件路径"""
        if self.system == "windows":
            return venv_path / "Scripts" / "pip.exe"
        else:
            return venv_path / "bin" / "pip"
    
    def check_venv_exists(self, venv_name: str = None) -> bool:
        """检查虚拟环境是否存在"""
        venv_path = self.get_venv_path(venv_name)
        python_exe = self.get_python_executable(venv_path)
        return venv_path.exists() and python_exe.exists()
    
    def create_venv(self, venv_name: str = None, python_cmd: str = None) -> bool:
        """创建虚拟环境"""
        if venv_name is None:
            venv_name = self.DEFAULT_VENV_NAME
        
        venv_path = self.get_venv_path(venv_name)
        
        # 检查是否已存在
        if self.check_venv_exists(venv_name):
            self._log("warning", f"虚拟环境已存在: {venv_path}")
            return True
        
        # 确定使用的Python命令
        if python_cmd is None:
            python_cmd = sys.executable
        
        try:
            self._log("info", f"正在创建虚拟环境: {venv_path}")
            self._log("info", f"使用Python解释器: {python_cmd}")
            
            # 创建虚拟环境
            result = subprocess.run(
                [python_cmd, "-m", "venv", str(venv_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                self._log("info", f"虚拟环境创建成功: {venv_path}")
                
                # 验证虚拟环境
                if self.check_venv_exists(venv_name):
                    self._log("info", "虚拟环境验证通过")
                    return True
                else:
                    self._log("error", "虚拟环境创建后验证失败")
                    return False
            else:
                self._log("error", f"虚拟环境创建失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self._log("error", "虚拟环境创建超时")
            return False
        except Exception as e:
            self._log("error", f"创建虚拟环境时发生错误: {e}")
            return False
    
    def get_activation_command(self, venv_name: str = None) -> str:
        """获取激活虚拟环境的命令"""
        venv_path = self.get_venv_path(venv_name)
        
        if self.system == "windows":
            # Windows PowerShell
            activate_script = venv_path / "Scripts" / "Activate.ps1"
            return f'& "{activate_script}"'
        else:
            # Unix系统
            activate_script = venv_path / "bin" / "activate"
            return f'source "{activate_script}"'
    
    def get_venv_info(self, venv_name: str = None) -> Dict:
        """获取虚拟环境信息"""
        venv_path = self.get_venv_path(venv_name)
        python_exe = self.get_python_executable(venv_path)
        pip_exe = self.get_pip_executable(venv_path)
        activation_cmd = self.get_activation_command(venv_name)
        
        info = {
            'name': venv_name or self.DEFAULT_VENV_NAME,
            'path': str(venv_path),
            'exists': self.check_venv_exists(venv_name),
            'python_executable': str(python_exe),
            'pip_executable': str(pip_exe),
            'activation_command': activation_cmd,
            'system': self.system
        }
        
        # 如果环境存在，获取Python版本信息
        if info['exists']:
            try:
                result = subprocess.run(
                    [str(python_exe), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    info['python_version'] = result.stdout.strip()
                else:
                    info['python_version'] = "未知"
            except Exception:
                info['python_version'] = "检测失败"
        
        return info
    
    def run_in_venv(self, command: list, venv_name: str = None, 
                   capture_output: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
        """在虚拟环境中执行命令"""
        if not self.check_venv_exists(venv_name):
            raise RuntimeError(f"虚拟环境不存在: {venv_name or self.DEFAULT_VENV_NAME}")
        
        venv_path = self.get_venv_path(venv_name)
        python_exe = self.get_python_executable(venv_path)
        
        # 如果命令以python开头，替换为虚拟环境中的python
        if command and command[0] in ['python', 'python3']:
            command[0] = str(python_exe)
        
        # 设置环境变量
        env = os.environ.copy()
        env['VIRTUAL_ENV'] = str(venv_path)
        env['PATH'] = f"{venv_path / ('Scripts' if self.system == 'windows' else 'bin')}{os.pathsep}{env.get('PATH', '')}"
        
        try:
            return subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                env=env
            )
        except subprocess.TimeoutExpired as e:
            self._log("error", f"命令执行超时: {' '.join(command)}")
            raise e
        except Exception as e:
            self._log("error", f"执行命令时发生错误: {e}")
            raise e
    
    def install_package_in_venv(self, package: str, venv_name: str = None, 
                               upgrade: bool = False) -> bool:
        """在虚拟环境中安装包"""
        try:
            pip_cmd = ["pip", "install"]
            if upgrade:
                pip_cmd.append("--upgrade")
            pip_cmd.append(package)
            
            self._log("info", f"在虚拟环境中安装: {package}")
            result = self.run_in_venv(pip_cmd, venv_name)
            
            if result.returncode == 0:
                self._log("info", f"成功安装: {package}")
                return True
            else:
                self._log("error", f"安装失败 {package}: {result.stderr}")
                return False
                
        except Exception as e:
            self._log("error", f"安装包时发生错误: {e}")
            return False
    
    def remove_venv(self, venv_name: str = None) -> bool:
        """删除虚拟环境"""
        venv_path = self.get_venv_path(venv_name)
        
        if not venv_path.exists():
            self._log("warning", f"虚拟环境不存在: {venv_path}")
            return True
        
        try:
            import shutil
            shutil.rmtree(venv_path)
            self._log("info", f"虚拟环境已删除: {venv_path}")
            return True
        except Exception as e:
            self._log("error", f"删除虚拟环境失败: {e}")
            return False
    
    def list_installed_packages(self, venv_name: str = None) -> list:
        """列出虚拟环境中已安装的包"""
        try:
            result = self.run_in_venv(["pip", "list", "--format=freeze"], venv_name)
            if result.returncode == 0:
                packages = []
                for line in result.stdout.strip().split('\n'):
                    if '==' in line:
                        name, version = line.split('==', 1)
                        packages.append({'name': name.strip(), 'version': version.strip()})
                return packages
            else:
                self._log("error", f"获取包列表失败: {result.stderr}")
                return []
        except Exception as e:
            self._log("error", f"获取包列表时发生错误: {e}")
            return []