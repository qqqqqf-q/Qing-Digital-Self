"""
模型下载模块

支持从ModelScope和HuggingFace下载模型
使用配置系统获取默认参数，支持命令行参数覆盖
使用官方SDK实现高效下载
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, List

try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from modelscope import snapshot_download as ms_snapshot_download
    MS_AVAILABLE = True
except ImportError:
    MS_AVAILABLE = False

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.config.config import get_config
from utils.logger.logger import get_logger


class ModelDownloader:
    """模型下载器"""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger('ModelDownloader')
        self.config = get_config()
        
    def _log(self, level: str, message: str):
        """内部日志方法"""
        if self.logger:
            getattr(self.logger, level.lower())(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def check_dependencies(self, download_source: str) -> bool:
        """检查所需依赖是否可用"""
        if download_source.lower() == 'modelscope':
            if not MS_AVAILABLE:
                self._log("error", "ModelScope库未安装")
                self._log("info", "请运行: pip install modelscope")
                return False
        elif download_source.lower() == 'huggingface':
            if not HF_AVAILABLE:
                self._log("error", "HuggingFace Hub库未安装")
                self._log("info", "请运行: pip install huggingface_hub")
                return False
        else:
            self._log("error", f"不支持的下载源: {download_source}")
            self._log("info", "支持的下载源: modelscope, huggingface")
            return False
        return True
    
    
    def download_from_modelscope(self, model_repo: str, model_path: str) -> bool:
        """从ModelScope下载模型"""
        if not MS_AVAILABLE:
            self._log("error", "ModelScope库未安装，请使用: pip install modelscope")
            return False
            
        try:
            self._log("info", f"从ModelScope下载模型: {model_repo}")
            self._log("info", f"下载位置: {model_path}")
            
            # 检查是否已存在
            if os.path.exists(model_path) and os.listdir(model_path):
                self._log("warning", f"模型目录已存在且非空: {model_path}")
                response = input("是否覆盖? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    self._log("info", "取消下载")
                    return False
            
            # 确保目录存在
            os.makedirs(model_path, exist_ok=True)
            
            # 使用ModelScope SDK下载
            self._log("info", "开始下载模型文件...")
            downloaded_path = ms_snapshot_download(
                model_id=model_repo,
                cache_dir=os.path.dirname(model_path),
                local_dir=model_path
            )
            
            if downloaded_path and os.path.exists(downloaded_path):
                self._log("info", "ModelScope模型下载成功")
                return True
            else:
                self._log("error", "ModelScope下载失败")
                return False
                
        except Exception as e:
            self._log("error", f"ModelScope下载过程中发生错误: {e}")
            return False
    
    def download_from_huggingface(self, model_repo: str, model_path: str) -> bool:
        """从HuggingFace下载模型"""
        if not HF_AVAILABLE:
            self._log("error", "HuggingFace Hub库未安装，请使用: pip install huggingface_hub")
            return False
            
        try:
            self._log("info", f"从HuggingFace下载模型: {model_repo}")
            self._log("info", f"下载位置: {model_path}")
            
            # 检查是否已存在
            if os.path.exists(model_path) and os.listdir(model_path):
                self._log("warning", f"模型目录已存在且非空: {model_path}")
                response = input("是否覆盖? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    self._log("info", "取消下载")
                    return False
            
            # 确保目录存在
            os.makedirs(model_path, exist_ok=True)
            
            # 使用HuggingFace Hub SDK下载
            self._log("info", "开始下载模型文件...")
            downloaded_path = hf_snapshot_download(
                repo_id=model_repo,
                cache_dir=os.path.dirname(model_path),
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            
            if downloaded_path and os.path.exists(downloaded_path):
                self._log("info", "HuggingFace模型下载成功")
                return True
            else:
                self._log("error", "HuggingFace下载失败")
                return False
                
        except Exception as e:
            self._log("error", f"HuggingFace下载过程中发生错误: {e}")
            return False
    
    def download_model(self, model_repo: str = None, model_path: str = None, 
                      download_source: str = None) -> bool:
        """下载模型（自动选择源）"""
        # 使用参数或配置文件的值
        if model_repo is None:
            model_repo = self.config.get('model_repo')
        if model_path is None:
            model_path = self.config.get('model_path')
        if download_source is None:
            download_source = self.config.get('download_source', 'modelscope')
        
        if not model_repo:
            self._log("error", "未指定模型仓库")
            return False
        if not model_path:
            self._log("error", "未指定模型路径")
            return False
        
        self._log("info", f"开始下载模型")
        self._log("info", f"模型仓库: {model_repo}")
        self._log("info", f"本地路径: {model_path}")
        self._log("info", f"下载源: {download_source}")
        
        # 检查依赖
        if not self.check_dependencies(download_source):
            return False
        
        # 根据下载源选择下载方法
        if download_source.lower() == 'modelscope':
            return self.download_from_modelscope(model_repo, model_path)
        elif download_source.lower() == 'huggingface':
            return self.download_from_huggingface(model_repo, model_path)
        else:
            return False
    
    def get_model_info(self, model_path: str) -> Dict:
        """获取模型信息"""
        info = {
            'path': model_path,
            'exists': False,
            'files': [],
            'size': 0
        }
        
        if not os.path.exists(model_path):
            return info
        
        info['exists'] = True
        
        try:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        relative_path = os.path.relpath(file_path, model_path)
                        info['files'].append({
                            'name': relative_path,
                            'size': file_size
                        })
                        info['size'] += file_size
                    except OSError:
                        continue
        except Exception as e:
            self._log("warning", f"获取模型信息时发生错误: {e}")
        
        return info
    
    def list_downloaded_models(self, base_path: str = None) -> List[Dict]:
        """列出已下载的模型"""
        if base_path is None:
            base_path = os.path.dirname(self.config.get('model_path', './model'))
        
        models = []
        
        if not os.path.exists(base_path):
            return models
        
        try:
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    model_info = self.get_model_info(item_path)
                    if model_info['files']:  # 有文件的才算是模型
                        model_info['name'] = item
                        models.append(model_info)
        except Exception as e:
            self._log("error", f"列出模型时发生错误: {e}")
        
        return models


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='模型下载工具')
    parser.add_argument(
        '--model-repo', 
        type=str, 
        help='模型仓库名称 (例如: Qwen/Qwen-3-8B-Base)'
    )
    parser.add_argument(
        '--model-path', 
        type=str, 
        help='本地保存路径'
    )
    parser.add_argument(
        '--download-source', 
        choices=['modelscope', 'huggingface'], 
        help='下载源'
    )
    parser.add_argument(
        '--list', 
        action='store_true', 
        help='列出已下载的模型'
    )
    parser.add_argument(
        '--info', 
        type=str,
        help='显示指定模型的信息'
    )
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = ModelDownloader()
    
    if args.list:
        # 列出已下载的模型
        models = downloader.list_downloaded_models()
        if models:
            print("已下载的模型:")
            for model in models:
                size_mb = model['size'] / (1024 * 1024)
                print(f"  {model['name']}: {len(model['files'])} 个文件, {size_mb:.1f} MB")
        else:
            print("未找到已下载的模型")
    
    elif args.info:
        # 显示模型信息
        info = downloader.get_model_info(args.info)
        if info['exists']:
            size_mb = info['size'] / (1024 * 1024)
            print(f"模型路径: {info['path']}")
            print(f"文件数量: {len(info['files'])}")
            print(f"总大小: {size_mb:.1f} MB")
            if len(info['files']) <= 20:  # 少于20个文件时显示详情
                print("文件列表:")
                for file_info in info['files']:
                    file_size_mb = file_info['size'] / (1024 * 1024)
                    print(f"  {file_info['name']}: {file_size_mb:.1f} MB")
        else:
            print(f"模型不存在: {args.info}")
    
    else:
        # 下载模型
        success = downloader.download_model(
            model_repo=args.model_repo,
            model_path=args.model_path,
            download_source=args.download_source
        )
        
        if success:
            print("模型下载成功")
            sys.exit(0)
        else:
            print("模型下载失败")
            sys.exit(1)


if __name__ == "__main__":
    main()