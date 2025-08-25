"""
配置管理命令

提供配置文件的初始化、查看、设置和验证功能。
支持交互式配置向导和配置模板系统。
"""

import os
import json
import shutil
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

from ..core.base import BaseCommand
from ..core.exceptions import ConfigurationError, ValidationError, FileOperationError
from ..core.helpers import confirm_action, format_file_size
from utils.config.config import get_config, ConfigError


class ConfigCommand(BaseCommand):
    """配置管理命令"""
    
    def __init__(self):
        super().__init__("config", "配置管理")
        self.config_file = "seeting.jsonc"
        self.template_file = "seeting_template.jsonc"
    
    def execute(self, args: argparse.Namespace) -> int:
        """执行配置命令"""
        action = getattr(args, 'config_action', None)
        
        if action == 'init':
            return self._init_config(args)
        elif action == 'show':
            return self._show_config(args)
        elif action == 'set':
            return self._set_config(args)
        elif action == 'validate':
            return self._validate_config(args)
        else:
            self.logger.error("未指定配置操作")
            return 1
    
    def _init_config(self, args: argparse.Namespace) -> int:
        """初始化配置文件"""
        try:
            # 检查配置文件是否已存在
            if os.path.exists(self.config_file):
                if not confirm_action(f"配置文件 {self.config_file} 已存在，是否覆盖？", False):
                    self.logger.info("取消配置初始化")
                    return 0
            
            # 选择配置模板
            template_type = getattr(args, 'template', 'basic')
            
            if args.interactive:
                return self._interactive_init(template_type)
            else:
                return self._template_init(template_type)
                
        except Exception as e:
            self.logger.error(f"初始化配置失败: {e}")
            return 1
    
    def _template_init(self, template_type: str) -> int:
        """使用模板初始化配置"""
        try:
            if not os.path.exists(self.template_file):
                self.logger.error(f"配置模板文件不存在: {self.template_file}")
                return 1
            
            # 复制模板文件
            shutil.copy2(self.template_file, self.config_file)
            
            # 根据模板类型进行定制
            if template_type == 'advanced':
                self._customize_advanced_config()
            
            self.logger.info(f"配置文件已创建: {self.config_file}")
            self.logger.info("请编辑配置文件以符合您的需求")
            
            return 0
            
        except Exception as e:
            raise ConfigurationError(f"创建配置文件失败: {e}")
    
    def _interactive_init(self, template_type: str) -> int:
        """交互式配置初始化"""
        try:
            self.logger.info("开始交互式配置向导...")
            
            config_data = self._get_template_config()
            
            # 基础配置
            print("\n=== 基础配置 ===")
            config_data = self._configure_basic_settings(config_data)
            
            # 模型配置
            print("\n=== 模型配置 ===")
            config_data = self._configure_model_settings(config_data)
            
            # 数据配置
            print("\n=== 数据配置 ===")
            config_data = self._configure_data_settings(config_data)
            
            if template_type == 'advanced':
                # 高级配置
                print("\n=== 高级配置 ===")
                config_data = self._configure_advanced_settings(config_data)
            
            # 保存配置
            self._save_config(config_data)
            
            self.logger.info(f"交互式配置完成，配置文件已保存: {self.config_file}")
            return 0
            
        except Exception as e:
            self.logger.error(f"交互式配置失败: {e}")
            return 1
    
    def _get_template_config(self) -> Dict[str, Any]:
        """获取模板配置"""
        if os.path.exists(self.template_file):
            try:
                with open(self.template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 简单的JSONC处理（移除注释）
                lines = content.split('\n')
                cleaned_lines = []
                for line in lines:
                    if '//' in line and not line.strip().startswith('"'):
                        line = line[:line.index('//')]
                    if line.strip():
                        cleaned_lines.append(line)
                
                cleaned_content = '\n'.join(cleaned_lines)
                return json.loads(cleaned_content)
                
            except Exception as e:
                self.logger.warning(f"无法读取模板文件: {e}")
        
        # 默认配置
        return {
            "repoid": "Qing-Agent",
            "branch": "main", 
            "version": "0.1.0",
            "model_args": {},
            "logger_args": {},
            "data_args": {}
        }
    
    def _configure_basic_settings(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """配置基础设置"""
        # 项目信息
        repo_id = input(f"项目ID [{config_data.get('repoid', 'Qing-Agent')}]: ").strip()
        if repo_id:
            config_data['repoid'] = repo_id
        
        version = input(f"版本号 [{config_data.get('version', '0.1.0')}]: ").strip()
        if version:
            config_data['version'] = version
        
        # 日志配置
        log_level = input("日志级别 (DEBUG/INFO/WARNING/ERROR) [INFO]: ").strip().upper()
        if log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            if 'logger_args' not in config_data:
                config_data['logger_args'] = {}
            config_data['logger_args']['log_level'] = log_level
        
        language = input("界面语言 (zhcn/en) [zhcn]: ").strip()
        if language in ['zhcn', 'en']:
            config_data['logger_args']['language'] = language
        
        return config_data
    
    def _configure_model_settings(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """配置模型设置"""
        if 'model_args' not in config_data:
            config_data['model_args'] = {}
        
        model_path = input("基础模型路径 [./model/Qwen3-8B-Base]: ").strip()
        if model_path:
            config_data['model_args']['model_path'] = model_path
        
        template = input("模型模板 (qwen/llama/chatglm) [qwen]: ").strip()
        if template:
            config_data['model_args']['template'] = template
        
        return config_data
    
    def _configure_data_settings(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """配置数据设置"""
        if 'data_args' not in config_data:
            config_data['data_args'] = {}
        
        # QQ数据配置
        qq_db_path = input("QQ数据库路径 [./data/qq.db]: ").strip()
        if qq_db_path:
            if 'qq_agrs' not in config_data['data_args']:
                config_data['data_args']['qq_agrs'] = {}
            config_data['data_args']['qq_agrs']['qq_db_path'] = qq_db_path
        
        qq_number = input("AI对应的QQ号码: ").strip()
        if qq_number:
            config_data['data_args']['qq_agrs']['qq_number_ai'] = qq_number
        
        # 数据清洗配置
        clean_method = input("数据清洗方法 (raw/llm) [raw]: ").strip()
        if clean_method in ['raw', 'llm']:
            if 'clean_set_args' not in config_data['data_args']:
                config_data['data_args']['clean_set_args'] = {}
            config_data['data_args']['clean_set_args']['clean_method'] = clean_method
        
        return config_data
    
    def _configure_advanced_settings(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """配置高级设置"""
        # 训练配置
        if 'train_sft_args' not in config_data['data_args']:
            config_data['data_args']['train_sft_args'] = {}
        
        lora_r = input("LoRA rank [16]: ").strip()
        if lora_r.isdigit():
            config_data['data_args']['train_sft_args']['lora_r'] = int(lora_r)
        
        lora_alpha = input("LoRA alpha [32]: ").strip()
        if lora_alpha.isdigit():
            config_data['data_args']['train_sft_args']['lora_alpha'] = int(lora_alpha)
        
        batch_size = input("训练批大小 [1]: ").strip()
        if batch_size.isdigit():
            config_data['data_args']['train_sft_args']['tper_device_train_batch_size'] = int(batch_size)
        
        return config_data
    
    def _save_config(self, config_data: Dict[str, Any]) -> None:
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            raise FileOperationError(f"保存配置文件失败: {e}", self.config_file, "write")
    
    def _customize_advanced_config(self) -> None:
        """定制高级配置"""
        # 高级配置的特殊处理逻辑
        pass
    
    def _show_config(self, args: argparse.Namespace) -> int:
        """显示当前配置"""
        try:
            format_type = getattr(args, 'format', 'table')
            
            config = get_config()
            config_data = config.all()
            
            if format_type == 'json':
                self._show_config_json(config_data)
            elif format_type == 'yaml':
                self._show_config_yaml(config_data)
            else:  # table
                self._show_config_table(config_data)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"显示配置失败: {e}")
            return 1
    
    def _show_config_json(self, config_data: Dict[str, Any]) -> None:
        """以JSON格式显示配置"""
        print(json.dumps(config_data, ensure_ascii=False, indent=2))
    
    def _show_config_yaml(self, config_data: Dict[str, Any]) -> None:
        """以YAML格式显示配置"""
        try:
            import yaml
            print(yaml.dump(config_data, allow_unicode=True, default_flow_style=False))
        except ImportError:
            self.logger.warning("YAML模块未安装，使用JSON格式显示")
            self._show_config_json(config_data)
    
    def _show_config_table(self, config_data: Dict[str, Any]) -> None:
        """以表格格式显示配置"""
        print("当前配置:")
        print("-" * 80)
        
        self._print_config_section("基础配置", {
            "项目ID": config_data.get('repoid'),
            "分支": config_data.get('branch'),
            "版本": config_data.get('version'),
        })
        
        self._print_config_section("模型配置", {
            "模型路径": config_data.get('model_path'),
            "模型仓库": config_data.get('model_repo'),
            "模板": config_data.get('template'),
            "微调类型": config_data.get('finetuning_type'),
        })
        
        self._print_config_section("日志配置", {
            "日志级别": config_data.get('log_level'),
            "语言": config_data.get('language'),
        })
        
        self._print_config_section("数据配置", {
            "QQ数据库": config_data.get('qq_db_path'),
            "QQ号码": config_data.get('qq_number_ai'),
            "清洗方法": config_data.get('clean_method'),
        })
        
        # 显示配置文件信息
        if os.path.exists(self.config_file):
            file_stat = os.stat(self.config_file)
            print(f"\n配置文件: {self.config_file}")
            print(f"文件大小: {format_file_size(file_stat.st_size)}")
            print(f"修改时间: {file_stat.st_mtime}")
    
    def _print_config_section(self, title: str, items: Dict[str, Any]) -> None:
        """打印配置部分"""
        print(f"\n{title}:")
        for key, value in items.items():
            if value is not None:
                print(f"  {key:<15}: {value}")
    
    def _set_config(self, args: argparse.Namespace) -> int:
        """设置配置项"""
        try:
            key = args.key
            value = args.value
            
            # 验证配置键
            if not self._is_valid_config_key(key):
                self.logger.error(f"无效的配置键: {key}")
                return 1
            
            # 类型转换
            converted_value = self._convert_config_value(key, value)
            
            # 设置配置
            config = get_config()
            config.set(key, converted_value)
            
            self.logger.info(f"配置项已更新: {key} = {converted_value}")
            
            # 保存到文件
            if confirm_action("是否保存配置到文件？", True):
                self._save_current_config()
                self.logger.info("配置已保存到文件")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"设置配置失败: {e}")
            return 1
    
    def _is_valid_config_key(self, key: str) -> bool:
        """验证配置键是否有效"""
        valid_keys = [
            'log_level', 'language', 'model_path', 'model_repo', 'template',
            'qq_db_path', 'qq_number_ai', 'clean_method', 'data_path',
            'lora_r', 'lora_alpha', 'batch_size', 'learning_rate'
        ]
        return key in valid_keys
    
    def _convert_config_value(self, key: str, value: str) -> Any:
        """转换配置值类型"""
        # 布尔值
        if key in ['trust_remote_code', 'use_qlora', 'fp16'] and value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # 整数值
        if key in ['lora_r', 'lora_alpha', 'batch_size', 'max_steps', 'logging_steps', 'save_steps']:
            try:
                return int(value)
            except ValueError:
                raise ValidationError(f"配置项 {key} 需要整数值")
        
        # 浮点数值
        if key in ['learning_rate', 'lora_dropout']:
            try:
                return float(value)
            except ValueError:
                raise ValidationError(f"配置项 {key} 需要浮点数值")
        
        # 字符串值
        return value
    
    def _save_current_config(self) -> None:
        """保存当前配置到文件"""
        try:
            config = get_config()
            config_data = config.all()
            
            # 重新组织配置数据结构
            organized_data = self._organize_config_data(config_data)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(organized_data, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            raise FileOperationError(f"保存配置失败: {e}", self.config_file, "write")
    
    def _organize_config_data(self, flat_data: Dict[str, Any]) -> Dict[str, Any]:
        """重新组织平坦的配置数据为嵌套结构"""
        # 这里需要根据实际的配置结构进行组织
        # 简化版本，直接返回平坦数据
        return flat_data
    
    def _validate_config(self, args: argparse.Namespace) -> int:
        """验证配置有效性"""
        try:
            # 检查配置文件是否存在
            if not os.path.exists(self.config_file):
                self.logger.error(f"配置文件不存在: {self.config_file}")
                return 1
            
            # 尝试加载配置
            try:
                config = get_config()
                self.logger.info("配置文件格式正确")
            except ConfigError as e:
                self.logger.error(f"配置文件格式错误: {e}")
                return 1
            
            # 验证关键配置项
            validation_errors = []
            
            # 验证模型路径
            model_path = config.get('model_path')
            if model_path and not os.path.exists(model_path):
                validation_errors.append(f"模型路径不存在: {model_path}")
            
            # 验证数据路径
            qq_db_path = config.get('qq_db_path')
            if qq_db_path and not os.path.exists(os.path.dirname(qq_db_path)):
                validation_errors.append(f"QQ数据库目录不存在: {os.path.dirname(qq_db_path)}")
            
            # 验证日志级别
            log_level = config.get('log_level')
            if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                validation_errors.append(f"无效的日志级别: {log_level}")
            
            # 报告验证结果
            if validation_errors:
                self.logger.error("配置验证失败:")
                for error in validation_errors:
                    self.logger.error(f"  - {error}")
                return 1
            else:
                self.logger.info("配置验证通过")
                return 0
                
        except Exception as e:
            self.logger.error(f"配置验证失败: {e}")
            return 1