"""
环境验证模块

验证Python环境和依赖包是否正确安装
仅使用标准库实现
"""

import subprocess
import importlib.util
from typing import Dict, List, Tuple, Optional


class EnvironmentValidator:
    """环境验证器"""
    
    # 需要验证的包（按重要性排序）
    VALIDATION_PACKAGES = [
        'torch',
        'transformers', 
        'peft',
        'unsloth',
        'bitsandbytes',
        'accelerate',
        'numpy'
    ]
    
    # 验证脚本
    VALIDATION_SCRIPTS = {
        'torch': """
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
""",
        'transformers': """
import transformers
print(f"Transformers版本: {transformers.__version__}")
""",
        'peft': """
import peft
print(f"PEFT版本: {peft.__version__}")
""",
        'unsloth': """
import unsloth
print("Unsloth导入成功")
try:
    print(f"Unsloth版本: {unsloth.__version__}")
except AttributeError:
    print("Unsloth版本信息不可用")
"""
    }
    
    def __init__(self, venv_manager, logger=None):
        self.venv_manager = venv_manager
        self.logger = logger
        
    def _log(self, level: str, message: str):
        """内部日志方法"""
        if self.logger:
            getattr(self.logger, level.lower())(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def validate_package_import(self, package: str, venv_name: str = None) -> Tuple[bool, str]:
        """验证包是否可以正确导入"""
        try:
            script = self.VALIDATION_SCRIPTS.get(package, f"import {package}")
            
            result = self.venv_manager.run_in_venv(
                ["python", "-c", script],
                venv_name,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip()
                
        except Exception as e:
            return False, str(e)
    
    def validate_all_packages(self, venv_name: str = None) -> Dict:
        """验证所有必需包"""
        results = {}
        
        self._log("info", "开始验证环境...")
        
        for package in self.VALIDATION_PACKAGES:
            self._log("info", f"验证 {package}...")
            success, output = self.validate_package_import(package, venv_name)
            
            results[package] = {
                'success': success,
                'output': output
            }
            
            if success:
                self._log("info", f"✓ {package} 验证通过")
                if output:
                    for line in output.split('\n'):
                        if line.strip():
                            self._log("info", f"  {line}")
            else:
                self._log("error", f"✗ {package} 验证失败: {output}")
        
        return results
    
    def check_python_version(self, venv_name: str = None) -> Dict:
        """检查Python版本"""
        try:
            result = self.venv_manager.run_in_venv(
                ["python", "--version"],
                venv_name,
                timeout=10
            )
            
            if result.returncode == 0:
                version_str = result.stdout.strip()
                return {
                    'success': True,
                    'version': version_str,
                    'message': f"Python版本: {version_str}"
                }
            else:
                return {
                    'success': False,
                    'version': None,
                    'message': f"获取Python版本失败: {result.stderr}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'version': None,
                'message': f"检查Python版本时发生错误: {e}"
            }
    
    def check_pip_version(self, venv_name: str = None) -> Dict:
        """检查pip版本"""
        try:
            result = self.venv_manager.run_in_venv(
                ["pip", "--version"],
                venv_name,
                timeout=10
            )
            
            if result.returncode == 0:
                version_str = result.stdout.strip()
                return {
                    'success': True,
                    'version': version_str,
                    'message': f"pip版本: {version_str}"
                }
            else:
                return {
                    'success': False,
                    'version': None,
                    'message': f"获取pip版本失败: {result.stderr}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'version': None,
                'message': f"检查pip版本时发生错误: {e}"
            }
    
    def run_comprehensive_check(self, venv_name: str = None) -> Dict:
        """运行全面的环境检查"""
        self._log("info", "开始全面环境检查...")
        
        # 检查虚拟环境
        if not self.venv_manager.check_venv_exists(venv_name):
            return {
                'success': False,
                'message': '虚拟环境不存在',
                'details': {}
            }
        
        # 收集所有检查结果
        results = {
            'python': self.check_python_version(venv_name),
            'pip': self.check_pip_version(venv_name),
            'packages': self.validate_all_packages(venv_name)
        }
        
        # 统计成功/失败
        python_ok = results['python']['success']
        pip_ok = results['pip']['success']
        packages_ok = all(pkg['success'] for pkg in results['packages'].values())
        
        overall_success = python_ok and pip_ok and packages_ok
        
        # 生成报告
        if overall_success:
            message = "所有环境检查通过"
            self._log("info", "✓ 环境验证成功")
        else:
            failed_items = []
            if not python_ok:
                failed_items.append("Python")
            if not pip_ok:
                failed_items.append("pip")
            if not packages_ok:
                failed_packages = [pkg for pkg, result in results['packages'].items() 
                                 if not result['success']]
                failed_items.extend(failed_packages)
            
            message = f"环境检查失败: {', '.join(failed_items)}"
            self._log("error", "✗ 环境验证失败")
        
        return {
            'success': overall_success,
            'message': message,
            'details': results
        }
    
    def generate_report(self, check_results: Dict) -> str:
        """生成详细的检查报告"""
        lines = []
        lines.append("=" * 50)
        lines.append("环境验证报告")
        lines.append("=" * 50)
        
        # 总体状态
        status = "通过" if check_results['success'] else "失败"
        lines.append(f"总体状态: {status}")
        lines.append(f"检查信息: {check_results['message']}")
        lines.append("")
        
        # 详细信息
        if 'details' in check_results:
            details = check_results['details']
            
            # Python版本
            if 'python' in details:
                python_result = details['python']
                status_icon = "✓" if python_result['success'] else "✗"
                lines.append(f"{status_icon} Python: {python_result['message']}")
            
            # pip版本
            if 'pip' in details:
                pip_result = details['pip']
                status_icon = "✓" if pip_result['success'] else "✗"
                lines.append(f"{status_icon} pip: {pip_result['message']}")
            
            lines.append("")
            
            # 包验证结果
            if 'packages' in details:
                lines.append("包验证结果:")
                lines.append("-" * 30)
                
                for package, result in details['packages'].items():
                    status_icon = "✓" if result['success'] else "✗"
                    lines.append(f"{status_icon} {package}")
                    
                    if result['output']:
                        for line in result['output'].split('\n'):
                            if line.strip():
                                lines.append(f"    {line}")
                    lines.append("")
        
        lines.append("=" * 50)
        return '\n'.join(lines)
    
    def quick_check(self, packages: List[str], venv_name: str = None) -> bool:
        """快速检查指定包是否可用"""
        for package in packages:
            try:
                result = self.venv_manager.run_in_venv(
                    ["python", "-c", f"import {package}"],
                    venv_name,
                    timeout=10
                )
                if result.returncode != 0:
                    return False
            except Exception:
                return False
        return True