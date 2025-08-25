"""
Python版本检测模块

检测系统Python版本，确保满足项目要求
仅使用标准库实现
"""

import sys
import subprocess
import os
from typing import Tuple, Optional


class PythonChecker:
    """Python版本检测器"""
    
    REQUIRED_MAJOR = 3
    REQUIRED_MINOR = 10
    RECOMMENDED_VERSION = "3.12"
    
    def __init__(self, logger=None):
        self.logger = logger
        
    def _log(self, level: str, message: str):
        """内部日志方法"""
        if self.logger:
            getattr(self.logger, level.lower())(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def get_current_python_version(self) -> Tuple[int, int, int]:
        """获取当前Python版本"""
        return sys.version_info[:3]
    
    def check_python_executable(self, python_cmd: str = "python") -> Optional[Tuple[int, int, int]]:
        """检测指定Python可执行文件的版本"""
        try:
            result = subprocess.run(
                [python_cmd, "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                version_str = result.stdout.strip()
                # 解析版本号 "Python 3.x.y"
                if version_str.startswith("Python "):
                    version_parts = version_str.split()[1].split('.')
                    if len(version_parts) >= 2:
                        major = int(version_parts[0])
                        minor = int(version_parts[1])
                        patch = int(version_parts[2]) if len(version_parts) > 2 else 0
                        return (major, minor, patch)
                        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, 
                ValueError, FileNotFoundError) as e:
            self._log("debug", f"检测{python_cmd}失败: {e}")
            
        return None
    
    def find_available_python(self) -> Optional[str]:
        """查找可用的Python解释器"""
        python_commands = ["python", "python3", "python3.12", "python3.11", "python3.10"]
        
        for cmd in python_commands:
            version = self.check_python_executable(cmd)
            if version and self.is_version_compatible(version):
                self._log("info", f"找到兼容的Python解释器: {cmd} (版本 {'.'.join(map(str, version))})")
                return cmd
        
        return None
    
    def is_version_compatible(self, version: Tuple[int, int, int]) -> bool:
        """检查版本是否兼容"""
        major, minor, patch = version
        
        if major != self.REQUIRED_MAJOR:
            return False
            
        return minor >= self.REQUIRED_MINOR
    
    def check_current_python(self) -> dict:
        """检查当前Python环境"""
        current_version = self.get_current_python_version()
        major, minor, patch = current_version
        
        result = {
            'version': current_version,
            'version_string': f"{major}.{minor}.{patch}",
            'executable': sys.executable,
            'is_compatible': self.is_version_compatible(current_version),
            'is_recommended': (major == 3 and minor >= 12),
        }
        
        # 版本检查逻辑
        if not result['is_compatible']:
            result['status'] = 'incompatible'
            result['message'] = (f"Python版本过低 ({result['version_string']})，"
                               f"需要Python {self.REQUIRED_MAJOR}.{self.REQUIRED_MINOR}+")
        elif result['is_recommended']:
            result['status'] = 'excellent'
            result['message'] = f"Python版本理想 ({result['version_string']})"
        else:
            result['status'] = 'acceptable'
            result['message'] = (f"Python版本可用 ({result['version_string']})，"
                               f"建议升级到Python {self.RECOMMENDED_VERSION}")
        
        return result
    
    def get_system_python_paths(self) -> list:
        """获取系统中所有Python路径"""
        paths = []
        
        # 检查PATH环境变量中的Python
        path_env = os.environ.get('PATH', '')
        for path_dir in path_env.split(os.pathsep):
            if os.path.isdir(path_dir):
                for exe_name in ['python.exe', 'python3.exe', 'python'] if os.name == 'nt' else ['python', 'python3']:
                    exe_path = os.path.join(path_dir, exe_name)
                    if os.path.isfile(exe_path):
                        version = self.check_python_executable(exe_path)
                        if version:
                            paths.append({
                                'path': exe_path,
                                'version': version,
                                'compatible': self.is_version_compatible(version)
                            })
        
        # 去重并排序
        unique_paths = {}
        for item in paths:
            key = item['version']
            if key not in unique_paths or len(item['path']) < len(unique_paths[key]['path']):
                unique_paths[key] = item
        
        return sorted(unique_paths.values(), key=lambda x: x['version'], reverse=True)
    
    def validate_environment(self) -> bool:
        """验证Python环境是否满足要求"""
        check_result = self.check_current_python()
        
        if check_result['status'] == 'incompatible':
            self._log("error", check_result['message'])
            self._log("error", f"建议安装Python {self.RECOMMENDED_VERSION}")
            
            # 显示可用的Python版本
            available_pythons = self.get_system_python_paths()
            if available_pythons:
                self._log("info", "系统中发现的Python版本:")
                for python_info in available_pythons:
                    status = "✓" if python_info['compatible'] else "✗"
                    version_str = '.'.join(map(str, python_info['version']))
                    self._log("info", f"  {status} {python_info['path']} (版本 {version_str})")
            
            return False
        
        elif check_result['status'] == 'acceptable':
            self._log("warning", check_result['message'])
            self._log("warning", "较低版本的Python可能存在兼容性问题")
        else:
            self._log("info", check_result['message'])
        
        return True