"""
工具命令模块

提供依赖检查、缓存清理、导入导出等实用工具功能。
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import importlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ..core.base import BaseCommand
from ..core.exceptions import ValidationError, FileOperationError, DependencyError
from ..core.helpers import format_file_size, get_system_info, ensure_directory
from ..interface.validators import validate_path, validate_choice
from ..interface.prompts import format_success_message, format_warning_message
from utils.config.config import get_config
from utils.logger.logger import get_logger


class UtilsCommand(BaseCommand):
    """工具命令"""
    
    def __init__(self):
        self.logger = get_logger('UtilsCommand')
        super().__init__("utils", "系统工具")
        
    def execute(self, args: argparse.Namespace) -> int:
        """执行工具命令"""
        action = getattr(args, 'utils_action', None)
        
        if action == 'check-deps':
            return self._check_dependencies(args)
        elif action == 'clean-cache':
            return self._clean_cache(args)
        elif action == 'export':
            return self._export_resource(args)
        elif action == 'import':
            return self._import_resource(args)
        else:
            self.logger.error("未指定工具操作")
            return 1
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """验证命令参数"""
        action = getattr(args, 'utils_action', None)
        
        if action in ['export', 'import']:
            validate_choice(args.type, 'type', ['model', 'data', 'config'])
            validate_path(args.source, must_exist=True)
            validate_path(args.target, must_exist=False, check_parent=True)
    
    def _check_dependencies(self, args: argparse.Namespace) -> int:
        """检查系统依赖"""
        try:
            self.logger.info("检查系统依赖...")
            fix_missing = getattr(args, 'fix', False)
            
            # 定义必需依赖
            required_packages = {
                'core': [
                    'torch',
                    'transformers', 
                    'datasets',
                    'tokenizers',
                ],
                'training': [
                    'peft',
                    'trl',
                    'bitsandbytes',
                    'accelerate',
                ],
                'api': [
                    'openai',
                    'fastapi',
                    'uvicorn',
                ],
                'utilities': [
                    'colorama',
                    'psutil',
                    'tqdm',
                    'requests',
                ]
            }
            
            # 检查Python版本
            python_check = self._check_python_version()
            
            # 检查包依赖
            package_results = {}
            for category, packages in required_packages.items():
                package_results[category] = self._check_package_category(packages)
            
            # 检查系统资源
            system_check = self._check_system_resources()
            
            # 检查GPU
            gpu_check = self._check_gpu_availability()
            
            # 显示检查结果
            self._display_dependency_results(python_check, package_results, system_check, gpu_check)
            
            # 自动修复
            if fix_missing:
                return self._fix_missing_dependencies(package_results)
            
            # 返回状态
            all_good = (
                python_check['status'] == 'ok' and
                all(result['status'] == 'ok' for result in package_results.values()) and
                system_check['status'] == 'ok'
            )
            
            return 0 if all_good else 1
            
        except Exception as e:
            raise DependencyError(f"依赖检查失败: {e}")
    
    def _check_python_version(self) -> Dict[str, Any]:
        """检查Python版本"""
        major, minor = sys.version_info[:2]
        current_version = f"{major}.{minor}"
        
        if major >= 3 and minor >= 8:
            status = 'ok'
            message = f"Python {current_version} (支持)"
        elif major >= 3 and minor >= 7:
            status = 'warning'
            message = f"Python {current_version} (建议升级到3.8+)"
        else:
            status = 'error'
            message = f"Python {current_version} (不支持，需要3.8+)"
        
        return {
            'status': status,
            'message': message,
            'version': current_version,
            'required': '3.8+'
        }
    
    def _check_package_category(self, packages: List[str]) -> Dict[str, Any]:
        """检查包类别"""
        available = []
        missing = []
        
        for package in packages:
            try:
                importlib.import_module(package)
                available.append(package)
            except ImportError:
                missing.append(package)
        
        if not missing:
            status = 'ok'
        elif len(missing) < len(packages) / 2:
            status = 'warning'
        else:
            status = 'error'
        
        return {
            'status': status,
            'available': available,
            'missing': missing,
            'total': len(packages)
        }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """检查系统资源"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 内存检查
            memory_gb = memory.total / (1024**3)
            if memory_gb >= 16:
                memory_status = 'ok'
            elif memory_gb >= 8:
                memory_status = 'warning'
            else:
                memory_status = 'error'
            
            # 磁盘检查
            disk_free_gb = disk.free / (1024**3)
            if disk_free_gb >= 50:
                disk_status = 'ok'
            elif disk_free_gb >= 20:
                disk_status = 'warning'
            else:
                disk_status = 'error'
            
            overall_status = 'ok' if memory_status == 'ok' and disk_status == 'ok' else 'warning'
            
            return {
                'status': overall_status,
                'memory': {
                    'status': memory_status,
                    'total_gb': round(memory_gb, 1),
                    'available_gb': round(memory.available / (1024**3), 1)
                },
                'disk': {
                    'status': disk_status,
                    'free_gb': round(disk_free_gb, 1),
                    'total_gb': round(disk.total / (1024**3), 1)
                }
            }
            
        except ImportError:
            return {
                'status': 'warning',
                'message': 'psutil未安装，无法检查系统资源'
            }
    
    def _check_gpu_availability(self) -> Dict[str, Any]:
        """检查GPU可用性"""
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = []
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    memory_gb = memory_total / (1024**3)
                    
                    gpu_info.append({
                        'id': i,
                        'name': gpu_name,
                        'memory_gb': round(memory_gb, 1)
                    })
                
                return {
                    'status': 'ok',
                    'available': True,
                    'count': gpu_count,
                    'devices': gpu_info
                }
            else:
                return {
                    'status': 'warning', 
                    'available': False,
                    'message': 'CUDA不可用，将使用CPU'
                }
                
        except ImportError:
            return {
                'status': 'warning',
                'available': False,
                'message': 'PyTorch未安装'
            }
    
    def _display_dependency_results(self, python_check: Dict[str, Any], 
                                  package_results: Dict[str, Any],
                                  system_check: Dict[str, Any],
                                  gpu_check: Dict[str, Any]) -> None:
        """显示依赖检查结果"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("依赖检查结果")
        self.logger.info("=" * 80)
        
        # Python版本
        status_symbol = self._get_status_symbol(python_check['status'])
        self.logger.info(f"\n{status_symbol} Python版本: {python_check['message']}")
        
        # 包依赖
        self.logger.info(f"\n包依赖检查:")
        for category, result in package_results.items():
            status_symbol = self._get_status_symbol(result['status'])
            available_count = len(result['available'])
            total_count = result['total']
            self.logger.info(f"  {status_symbol} {category}: {available_count}/{total_count} 可用")
            
            if result['missing']:
                self.logger.warning(f"    缺失: {', '.join(result['missing'])}")
        
        # 系统资源
        if 'memory' in system_check:
            memory_symbol = self._get_status_symbol(system_check['memory']['status'])
            disk_symbol = self._get_status_symbol(system_check['disk']['status'])
            self.logger.info(f"\n系统资源:")
            self.logger.info(f"  {memory_symbol} 内存: {system_check['memory']['available_gb']:.1f}GB / {system_check['memory']['total_gb']:.1f}GB")
            self.logger.info(f"  {disk_symbol} 磁盘: {system_check['disk']['free_gb']:.1f}GB 可用")
        
        # GPU
        if gpu_check['available']:
            self.logger.info(f"\n{self._get_status_symbol('ok')} GPU: {gpu_check['count']} 个设备可用")
            for device in gpu_check['devices']:
                self.logger.info(f"    {device['id']}: {device['name']} ({device['memory_gb']:.1f}GB)")
        else:
            self.logger.warning(f"\n{self._get_status_symbol('warning')} GPU: {gpu_check['message']}")
        
        self.logger.info("=" * 80)
    
    def _get_status_symbol(self, status: str) -> str:
        """获取状态符号"""
        symbols = {
            'ok': '✓',
            'warning': '⚠',
            'error': '✗'
        }
        return symbols.get(status, '?')
    
    def _fix_missing_dependencies(self, package_results: Dict[str, Any]) -> int:
        """修复缺失的依赖"""
        try:
            all_missing = []
            for category, result in package_results.items():
                if result['missing']:
                    all_missing.extend(result['missing'])
            
            if not all_missing:
                self.logger.info(format_success_message("所有依赖都已满足"))
                return 0
            
            self.logger.info(f"\n尝试安装缺失的包: {', '.join(all_missing)}")
            
            # 构建pip安装命令
            cmd = [sys.executable, '-m', 'pip', 'install'] + all_missing
            
            # 执行安装
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(format_success_message("依赖安装完成"))
                return 0
            else:
                self.logger.error(format_warning_message(f"依赖安装失败: {result.stderr}"))
                return 1
                
        except Exception as e:
            raise DependencyError(f"修复依赖失败: {e}")
    
    def _clean_cache(self, args: argparse.Namespace) -> int:
        """清理缓存"""
        try:
            self.logger.info("清理系统缓存...")
            clean_all = getattr(args, 'all', False)
            
            cleaned_size = 0
            cleaned_items = []
            
            # 清理Python缓存
            cleaned_size += self._clean_python_cache()
            cleaned_items.append("Python __pycache__")
            
            # 清理transformers缓存
            if clean_all or self._confirm_clean("transformers缓存"):
                size = self._clean_transformers_cache()
                if size > 0:
                    cleaned_size += size
                    cleaned_items.append("Transformers缓存")
            
            # 清理torch缓存
            if clean_all or self._confirm_clean("PyTorch缓存"):
                size = self._clean_torch_cache()
                if size > 0:
                    cleaned_size += size
                    cleaned_items.append("PyTorch缓存")
            
            # 清理日志文件
            if clean_all or self._confirm_clean("日志文件"):
                size = self._clean_logs()
                if size > 0:
                    cleaned_size += size
                    cleaned_items.append("日志文件")
            
            # 清理临时文件
            if clean_all or self._confirm_clean("临时文件"):
                size = self._clean_temp_files()
                if size > 0:
                    cleaned_size += size
                    cleaned_items.append("临时文件")
            
            # 显示清理结果
            if cleaned_items:
                self.logger.info(format_success_message(
                    f"缓存清理完成",
                    f"清理项目: {', '.join(cleaned_items)}\n释放空间: {format_file_size(cleaned_size)}"
                ))
            else:
                self.logger.info("没有需要清理的缓存")
            
            return 0
            
        except Exception as e:
            raise FileOperationError(f"缓存清理失败: {e}")
    
    def _confirm_clean(self, item: str) -> bool:
        """确认清理项目"""
        try:
            response = input(f"是否清理{item}? [y/N]: ").strip().lower()
            return response in ['y', 'yes', '是']
        except (EOFError, KeyboardInterrupt):
            return False
    
    def _clean_python_cache(self) -> int:
        """清理Python缓存"""
        cleaned_size = 0
        
        try:
            for root, dirs, files in os.walk('.'):
                # 跳过.git等目录
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                if '__pycache__' in dirs:
                    pycache_path = os.path.join(root, '__pycache__')
                    
                    # 计算大小
                    for f in os.listdir(pycache_path):
                        file_path = os.path.join(pycache_path, f)
                        if os.path.isfile(file_path):
                            cleaned_size += os.path.getsize(file_path)
                    
                    # 删除目录
                    shutil.rmtree(pycache_path, ignore_errors=True)
                    
        except Exception as e:
            self.logger.warning(f"清理Python缓存失败: {e}")
        
        return cleaned_size
    
    def _clean_transformers_cache(self) -> int:
        """清理Transformers缓存"""
        cleaned_size = 0
        
        try:
            # HuggingFace缓存目录
            cache_dir = Path.home() / '.cache' / 'huggingface'
            
            if cache_dir.exists():
                # 计算大小
                for file_path in cache_dir.rglob('*'):
                    if file_path.is_file():
                        cleaned_size += file_path.stat().st_size
                
                # 删除缓存
                shutil.rmtree(cache_dir, ignore_errors=True)
                
        except Exception as e:
            self.logger.warning(f"清理Transformers缓存失败: {e}")
        
        return cleaned_size
    
    def _clean_torch_cache(self) -> int:
        """清理PyTorch缓存"""
        cleaned_size = 0
        
        try:
            # PyTorch缓存目录
            cache_dir = Path.home() / '.cache' / 'torch'
            
            if cache_dir.exists():
                # 计算大小
                for file_path in cache_dir.rglob('*'):
                    if file_path.is_file():
                        cleaned_size += file_path.stat().st_size
                
                # 删除缓存
                shutil.rmtree(cache_dir, ignore_errors=True)
            
            # 清理CUDA缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
                
        except Exception as e:
            self.logger.warning(f"清理PyTorch缓存失败: {e}")
        
        return cleaned_size
    
    def _clean_logs(self) -> int:
        """清理日志文件"""
        cleaned_size = 0
        
        try:
            logs_dir = Path('./logs')
            
            if logs_dir.exists():
                # 保留最近7天的日志
                import time
                cutoff_time = time.time() - (7 * 24 * 3600)
                
                for log_file in logs_dir.glob('*.log'):
                    if log_file.stat().st_mtime < cutoff_time:
                        cleaned_size += log_file.stat().st_size
                        log_file.unlink()
                        
        except Exception as e:
            self.logger.warning(f"清理日志文件失败: {e}")
        
        return cleaned_size
    
    def _clean_temp_files(self) -> int:
        """清理临时文件"""
        cleaned_size = 0
        
        try:
            temp_patterns = [
                '*.tmp',
                '*.temp',
                '.DS_Store',
                'Thumbs.db',
                '*.swp',
                '*.bak'
            ]
            
            for pattern in temp_patterns:
                for file_path in Path('.').rglob(pattern):
                    if file_path.is_file():
                        cleaned_size += file_path.stat().st_size
                        file_path.unlink()
                        
        except Exception as e:
            self.logger.warning(f"清理临时文件失败: {e}")
        
        return cleaned_size
    
    def _export_resource(self, args: argparse.Namespace) -> int:
        """导出资源"""
        try:
            resource_type = args.type
            source = args.source
            target = args.target
            
            self.logger.info(f"导出{resource_type}: {source} -> {target}")
            
            # 确保目标目录存在
            ensure_directory(os.path.dirname(target))
            
            if resource_type == 'model':
                return self._export_model(source, target)
            elif resource_type == 'data':
                return self._export_data(source, target)
            elif resource_type == 'config':
                return self._export_config(source, target)
            else:
                raise ValidationError(f"不支持的资源类型: {resource_type}")
                
        except Exception as e:
            raise FileOperationError(f"导出失败: {e}")
    
    def _export_model(self, source: str, target: str) -> int:
        """导出模型"""
        try:
            if os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
            
            self.logger.info(f"模型导出完成: {target}")
            return 0
            
        except Exception as e:
            raise FileOperationError(f"模型导出失败: {e}")
    
    def _export_data(self, source: str, target: str) -> int:
        """导出数据"""
        try:
            if os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
            
            self.logger.info(f"数据导出完成: {target}")
            return 0
            
        except Exception as e:
            raise FileOperationError(f"数据导出失败: {e}")
    
    def _export_config(self, source: str, target: str) -> int:
        """导出配置"""
        try:
            # 如果是目录，导出整个配置目录
            if os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
            
            self.logger.info(f"配置导出完成: {target}")
            return 0
            
        except Exception as e:
            raise FileOperationError(f"配置导出失败: {e}")
    
    def _import_resource(self, args: argparse.Namespace) -> int:
        """导入资源"""
        try:
            resource_type = args.type
            source = args.source
            target = args.target
            
            self.logger.info(f"导入{resource_type}: {source} -> {target}")
            
            # 确保目标目录存在
            ensure_directory(os.path.dirname(target))
            
            if resource_type == 'model':
                return self._import_model(source, target)
            elif resource_type == 'data':
                return self._import_data(source, target)
            elif resource_type == 'config':
                return self._import_config(source, target)
            else:
                raise ValidationError(f"不支持的资源类型: {resource_type}")
                
        except Exception as e:
            raise FileOperationError(f"导入失败: {e}")
    
    def _import_model(self, source: str, target: str) -> int:
        """导入模型"""
        try:
            if os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
            
            self.logger.info(f"模型导入完成: {target}")
            return 0
            
        except Exception as e:
            raise FileOperationError(f"模型导入失败: {e}")
    
    def _import_data(self, source: str, target: str) -> int:
        """导入数据"""
        try:
            if os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
            
            self.logger.info(f"数据导入完成: {target}")
            return 0
            
        except Exception as e:
            raise FileOperationError(f"数据导入失败: {e}")
    
    def _import_config(self, source: str, target: str) -> int:
        """导入配置"""
        try:
            if os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
            
            self.logger.info(f"配置导入完成: {target}")
            return 0
            
        except Exception as e:
            raise FileOperationError(f"配置导入失败: {e}")