"""
模型管理命令

提供模型下载、列表查看和信息获取功能。
支持从ModelScope和HuggingFace下载模型。
"""

import os
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

from ..core.base import BaseCommand
from ..core.exceptions import ConfigurationError, ValidationError
from ..core.helpers import confirm_action, format_file_size


class ModelCommand(BaseCommand):
    """模型管理命令"""
    
    def __init__(self):
        super().__init__("model", "模型管理")
    
    def execute(self, args: argparse.Namespace) -> int:
        """执行模型管理命令"""
        # 检查子命令
        action = getattr(args, 'model_action', None)
        
        if action == 'download':
            return self._download_model(args)
        elif action == 'list':
            return self._list_models(args)
        elif action == 'info':
            return self._show_model_info(args)
        else:
            self.logger.error("未指定模型操作，请使用: download, list, info")
            return 1
    
    def _download_model(self, args: argparse.Namespace) -> int:
        """下载模型"""
        try:
            # 导入模型下载器
            from environment.download.model_download import ModelDownloader
            
            # 创建下载器实例
            downloader = ModelDownloader(self.logger)
            
            # 获取参数
            model_repo = getattr(args, 'model_repo', None)
            model_path = getattr(args, 'model_path', None)
            download_source = getattr(args, 'download_source', None)
            
            # 如果没有指定参数，使用配置文件中的设置
            if not any([model_repo, model_path, download_source]):
                self.logger.info("使用配置文件中的设置进行下载")
                success = downloader.download_model()
            else:
                self.logger.info("使用指定参数进行下载")
                success = downloader.download_model(
                    model_repo=model_repo,
                    model_path=model_path,
                    download_source=download_source
                )
            
            if success:
                self.logger.info("模型下载成功")
                return 0
            else:
                self.logger.error("模型下载失败")
                return 1
                
        except ImportError as e:
            self.logger.error(f"导入模型下载模块失败: {e}")
            return 1
        except Exception as e:
            self.logger.error(f"下载模型失败: {e}")
            return 1
    
    def _list_models(self, args: argparse.Namespace) -> int:
        """列出已下载的模型"""
        try:
            from environment.download.model_download import ModelDownloader
            from utils.config.config import get_config
            
            config = get_config()
            # 优先使用配置文件中的models_dir，如果没有则从model_path推导
            models_dir = config.get('models_dir')
            if not models_dir:
                model_path = config.get('model_path', './model/default')
                models_dir = os.path.dirname(model_path)
            
            if not os.path.exists(models_dir):
                self.logger.info(f"模型目录不存在: {models_dir}")
                return 0
            
            self.logger.info(f"已下载的模型 (位置: {models_dir}):")
            print(f"\n{'模型名称':<30} {'大小':<15} {'状态'}")
            print("-" * 60)
            
            model_count = 0
            downloader = ModelDownloader(self.logger)
            
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                # 只处理目录，且不是以点开头的隐藏文件/目录
                if os.path.isdir(item_path) and not item.startswith('.'):
                    model_info = downloader.get_model_info(item_path)
                    
                    # 计算目录大小
                    total_size = 0
                    for root, dirs, files in os.walk(item_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                total_size += os.path.getsize(file_path)
                            except:
                                pass
                    
                    size_str = format_file_size(total_size) if total_size > 0 else "未知"
                    
                    # 检查是否是完整的模型
                    config_files = ['config.json', 'configuration.json']
                    has_config = any(os.path.exists(os.path.join(item_path, cf)) for cf in config_files)
                    
                    # 检查模型文件（包括分片的safetensors文件）
                    model_files = ['pytorch_model.bin', 'model.safetensors', 'pytorch_model.safetensors']
                    has_model = any(os.path.exists(os.path.join(item_path, mf)) for mf in model_files)
                    
                    # 如果没有找到整体模型文件，检查是否有分片的safetensors文件
                    if not has_model:
                        for file in os.listdir(item_path):
                            if file.startswith('model-') and file.endswith('.safetensors'):
                                has_model = True
                                break
                    
                    status = "完整" if (has_config and has_model) else "不完整"
                    
                    print(f"{item:<30} {size_str:<15} {status}")
                    model_count += 1
            
            if model_count == 0:
                print("未找到已下载的模型")
            else:
                print(f"\n总共找到 {model_count} 个模型")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"列出模型失败: {e}")
            return 1
    
    def _show_model_info(self, args: argparse.Namespace) -> int:
        """显示模型信息"""
        try:
            from environment.download.model_download import ModelDownloader
            from utils.config.config import get_config
            
            # 获取模型路径（支持直接参数和子命令参数）
            model_path = getattr(args, 'model_path', None)
            
            # 如果没有指定模型路径，显示所有模型的信息
            if not model_path:
                return self._show_all_models_info(args)
            
            # 显示特定模型的信息
            if not os.path.exists(model_path):
                self.logger.error(f"模型路径不存在: {model_path}")
                return 1
            
            downloader = ModelDownloader(self.logger)
            model_info = downloader.get_model_info(model_path)
            
            self.logger.info(f"模型信息: {model_path}")
            print(f"\n{'属性':<20} {'值'}")
            print("-" * 50)
            print(f"{'路径':<20} {model_info['path']}")
            print(f"{'存在':<20} {'是' if model_info['exists'] else '否'}")
            print(f"{'文件数量':<20} {len(model_info.get('files', []))}")
            print(f"{'总大小':<20} {format_file_size(model_info.get('size', 0))}")
            
            # 显示主要文件
            if model_info.get('files'):
                print(f"\n主要文件:")
                important_files = []
                for file_info in model_info['files']:
                    filename = file_info['name']
                    if any(ext in filename for ext in ['.json', '.bin', '.safetensors', '.txt']):
                        important_files.append(file_info)
                
                if important_files:
                    print(f"{'文件名':<40} {'大小'}")
                    print("-" * 60)
                    for file_info in important_files[:10]:  # 最多显示10个
                        print(f"{file_info['name']:<40} {format_file_size(file_info['size'])}")
                    
                    if len(important_files) > 10:
                        print(f"... 还有 {len(important_files) - 10} 个文件")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"获取模型信息失败: {e}")
            return 1
    
    def _show_all_models_info(self, args: argparse.Namespace) -> int:
        """显示所有模型的详细信息"""
        try:
            from environment.download.model_download import ModelDownloader
            from utils.config.config import get_config
            
            config = get_config()
            # 获取模型目录
            models_dir = config.get('models_dir')
            if not models_dir:
                model_path = config.get('model_path', './model/default')
                models_dir = os.path.dirname(model_path)
            
            if not os.path.exists(models_dir):
                self.logger.info(f"模型目录不存在: {models_dir}")
                return 0
            
            downloader = ModelDownloader(self.logger)
            model_count = 0
            
            self.logger.info(f"所有模型详细信息 (位置: {models_dir}):")
            
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                # 只处理目录，且不是以点开头的隐藏文件/目录
                if os.path.isdir(item_path) and not item.startswith('.'):
                    model_count += 1
                    model_info = downloader.get_model_info(item_path)
                    
                    # 输出模型信息
                    print(f"\n{'='*80}")
                    print(f"模型 {model_count}: {item}")
                    print(f"{'='*80}")
                    print(f"{'属性':<20} {'值'}")
                    print("-" * 50)
                    print(f"{'路径':<20} {model_info['path']}")
                    print(f"{'存在':<20} {'是' if model_info['exists'] else '否'}")
                    print(f"{'文件数量':<20} {len(model_info.get('files', []))}")
                    print(f"{'总大小':<20} {format_file_size(model_info.get('size', 0))}")
                    
                    # 检查模型完整性
                    config_files = ['config.json', 'configuration.json']
                    has_config = any(os.path.exists(os.path.join(item_path, cf)) for cf in config_files)
                    
                    model_files = ['pytorch_model.bin', 'model.safetensors', 'pytorch_model.safetensors']
                    has_model = any(os.path.exists(os.path.join(item_path, mf)) for mf in model_files)
                    
                    if not has_model:
                        for file in os.listdir(item_path):
                            if file.startswith('model-') and file.endswith('.safetensors'):
                                has_model = True
                                break
                    
                    status = "完整" if (has_config and has_model) else "不完整"
                    print(f"{'模型状态':<20} {status}")
                    
                    # 显示主要文件
                    if model_info.get('files'):
                        print(f"\n主要文件:")
                        important_files = []
                        for file_info in model_info['files']:
                            filename = file_info['name']
                            if any(ext in filename for ext in ['.json', '.bin', '.safetensors', '.txt']):
                                important_files.append(file_info)
                        
                        if important_files:
                            print(f"{'文件名':<40} {'大小'}")
                            print("-" * 60)
                            for file_info in important_files[:8]:  # 最多显示8个
                                print(f"{file_info['name']:<40} {format_file_size(file_info['size'])}")
                            
                            if len(important_files) > 8:
                                print(f"... 还有 {len(important_files) - 8} 个文件")
            
            if model_count == 0:
                print("\n未找到已下载的模型")
            else:
                print(f"\n{'='*80}")
                print(f"总共找到 {model_count} 个模型")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"获取所有模型信息失败: {e}")
            return 1