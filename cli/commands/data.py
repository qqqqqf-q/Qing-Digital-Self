"""
数据处理命令

提供QQ数据提取、清洗、格式转换、合并等数据处理功能。
支持批量处理和进度监控。
"""

import os
import json
import sys
import argparse
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ..core.base import BaseCommand
from ..core.exceptions import DataProcessingError, ValidationError, FileOperationError
from ..core.helpers import (
    format_time_duration, 
    format_file_size, 
    get_file_stats, 
    ensure_directory,
    format_progress_bar
)
from ..interface.validators import validate_path, validate_positive_int
from utils.config.config import get_config


class DataCommand(BaseCommand):
    """数据处理命令"""
    
    def __init__(self):
        super().__init__("data", "数据处理")
    
    def execute(self, args: argparse.Namespace) -> int:
        """执行数据处理命令"""
        action = getattr(args, 'data_action', None)
        
        if action == 'extract':
            return self._extract_data(args)
        elif action == 'clean':
            return self._clean_data(args)
        elif action == 'convert':
            return self._convert_data(args)
        elif action == 'merge':
            return self._merge_data(args)
        elif action == 'preview':
            return self._preview_data(args)
        elif action == 'stats':
            return self._show_stats(args)
        else:
            self.logger.error("未指定数据操作")
            return 1
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """验证命令参数"""
        action = getattr(args, 'data_action', None)
        
        if action == 'extract':
            self._validate_extract_args(args)
        elif action == 'clean':
            self._validate_clean_args(args)
        elif action == 'convert':
            self._validate_convert_args(args)
        elif action == 'merge':
            self._validate_merge_args(args)
        elif action in ['preview', 'stats']:
            self._validate_input_file_args(args)
    
    def _validate_extract_args(self, args: argparse.Namespace) -> None:
        """验证数据提取参数"""
        # 获取数据源类型
        source_type = getattr(args, 'source_type', None)
        data_dir = getattr(args, 'data_dir') or self.config.get('data_dir', './dataset/original/')
        
        # 验证数据目录存在
        if not os.path.exists(data_dir):
            raise ValidationError(f"数据目录不存在: {data_dir}")
        
        # 根据数据源类型进行特定验证
        if source_type == 'qq':
            qq_db_path = getattr(args, 'qq_db_path') or self.config.get('qq_db_path')
            if qq_db_path and not os.path.exists(qq_db_path):
                raise ValidationError(f"QQ数据库文件不存在: {qq_db_path}")
        
        # 输出路径验证
        output_path = getattr(args, 'output')
        if output_path:
            validate_path(output_path, must_exist=False, check_parent=True)
    
    def _validate_clean_args(self, args: argparse.Namespace) -> None:
        """验证数据清洗参数"""
        # 获取实际使用的路径（支持从配置读取）
        input_path = getattr(args, 'input', None) or self.config.get('dataset_csv_path', 'dataset/csv')
        output_path = getattr(args, 'output', None) or self.config.get('data_path', './dataset/sft.jsonl')
        
        validate_path(input_path, must_exist=True)
        validate_path(output_path, must_exist=False, check_parent=True)
        
        if hasattr(args, 'batch_size') and getattr(args, 'batch_size', None) is not None:
            validate_positive_int(args.batch_size, "batch_size")
        
        if hasattr(args, 'workers') and getattr(args, 'workers', None) is not None:
            validate_positive_int(args.workers, "workers")
    
    def _validate_convert_args(self, args: argparse.Namespace) -> None:
        """验证数据转换参数"""
        validate_path(args.input, must_exist=True)
        validate_path(args.output, must_exist=False, check_parent=True)
        
        valid_formats = ['chatml', 'alpaca', 'sharegpt']
        if hasattr(args, 'format') and args.format not in valid_formats:
            raise ValidationError(f"无效的数据格式: {args.format}, 支持的格式: {valid_formats}")
    
    def _validate_merge_args(self, args: argparse.Namespace) -> None:
        """验证数据合并参数"""
        for input_file in args.inputs:
            validate_path(input_file, must_exist=True)
        
        validate_path(args.output, must_exist=False, check_parent=True)
    
    def _validate_input_file_args(self, args: argparse.Namespace) -> None:
        """验证输入文件参数"""
        validate_path(args.input, must_exist=True)
    
    def _extract_data(self, args: argparse.Namespace) -> int:
        """从聊天数据中提取数据（支持QQ和Telegram）"""
        try:
            self.logger.info("开始从聊天数据中提取数据...")
            
            # 准备参数
            source_type = getattr(args, 'source_type', None)
            data_dir = getattr(args, 'data_dir') or self.config.get('data_dir', './dataset/original/')
            output_path = getattr(args, 'output') or "./dataset/csv"
            
            # 确保输出目录存在
            ensure_directory(output_path)
            
            # 构建提取命令参数
            extract_args = {
                'data_dir': data_dir,
                'output_dir': output_path,
                'source_type': source_type,
                # QQ相关参数
                'qq_db_path': getattr(args, 'qq_db_path', None) or self.config.get('qq_db_path'),
                'qq_number_ai': getattr(args, 'qq_number_ai', None) or self.config.get('qq_number_ai'),
                # Telegram相关参数
                'telegram_chat_id': getattr(args, 'telegram_chat_id', None) or self.config.get('telegram_chat_id'),
                'tg_data_dir': getattr(args, 'tg_data_dir', None) or self.config.get('tg_data_dir')
            }
            
            # 执行数据提取
            result = self._execute_unified_data_extraction(extract_args)
            
            if result == 0:
                # 显示提取结果统计
                self._show_extraction_stats(output_path)
                self.logger.info(f"数据提取完成: {output_path}")
            
            return result
            
        except Exception as e:
            raise DataProcessingError(f"数据提取失败: {e}")
    
    def _execute_unified_data_extraction(self, extract_args: Dict[str, Any]) -> int:
        """执行统一数据提取过程"""
        try:
            # 导入统一解析器
            from process_data.chat_parser.generate_parser import UnifiedParser
            
            # 创建统一解析器
            unified_parser = UnifiedParser(
                data_dir=extract_args['data_dir'],
                output_dir=extract_args['output_dir']
            )
            
            # 构建解析参数
            parse_kwargs = {}
            
            # 添加QQ相关参数
            if extract_args.get('qq_db_path'):
                parse_kwargs['qq_db_path'] = extract_args['qq_db_path']
            if extract_args.get('qq_number_ai'):
                parse_kwargs['qq_number_ai'] = extract_args['qq_number_ai']
            
            # 添加Telegram相关参数
            if extract_args.get('telegram_chat_id'):
                parse_kwargs['telegram_chat_id'] = extract_args['telegram_chat_id']
            if extract_args.get('tg_data_dir'):
                parse_kwargs['tg_data_dir'] = extract_args['tg_data_dir']
            
            # 执行解析
            if extract_args.get('source_type'):
                # 指定数据源类型
                from process_data.chat_parser.generate_parser import DataSourceType
                
                if extract_args['source_type'] == 'qq':
                    source_type = DataSourceType.QQ
                elif extract_args['source_type'] in ['tg', 'telegram']:
                    source_type = DataSourceType.TELEGRAM
                elif extract_args['source_type'] in ['wx', 'wechat']:
                    source_type = DataSourceType.WECHAT
                else:
                    raise DataProcessingError(f"不支持的数据源类型: {extract_args['source_type']}")
                
                return unified_parser.parse_with_source_type(source_type, **parse_kwargs)
            else:
                # 自动检测数据源
                return unified_parser.parse_auto(**parse_kwargs)
            
        except ImportError as e:
            self.logger.error(f"无法导入统一解析器: {e}")
            # 降级到原有的QQ解析器
            return self._fallback_to_qq_parser(extract_args)
        except Exception as e:
            raise DataProcessingError(f"统一数据提取执行失败: {e}")
    
    def _fallback_to_qq_parser(self, extract_args: Dict[str, Any]) -> int:
        """降级到原有的QQ解析器方法"""
        try:
            # 导入QQ解析器
            from process_data.chat_parser.qq_parser import QQParser
            
            qq_db_path = extract_args.get('qq_db_path')
            if not qq_db_path:
                self.logger.error("降级到QQ解析器时未指定QQ数据库路径")
                return 1
            
            # 使用正确的参数初始化QQParser
            parser = QQParser(
                db_path=qq_db_path,
                output_dir=extract_args['output_dir'],
                qq_number_ai=extract_args.get('qq_number_ai')
            )
            
            # 执行提取
            parser.parse_all()
            return 0
            
        except ImportError as e:
            self.logger.error(f"无法导入QQ解析模块: {e}")
            # 继续降级到直接调用脚本
            return self._fallback_extract_data(extract_args)
        except Exception as e:
            raise DataProcessingError(f"QQ解析器降级失败: {e}")
    def _fallback_extract_data(self, extract_args: Dict[str, Any]) -> int:
        """降级的数据提取方法，直接调用脚本"""
        try:
            # 构建命令行参数
            cmd = [
                sys.executable,
                'generate_training_data.py',
                '--extract-only'
            ]
            
            if extract_args.get('qq_db_path'):
                cmd.extend(['--qq-db-path', extract_args['qq_db_path']])
            
            if extract_args.get('qq_number_ai'):
                cmd.extend(['--qq-number-ai', extract_args['qq_number_ai']])
            
            if extract_args.get('output_dir'):
                cmd.extend(['--output', extract_args['output_dir']])
            
            # 执行命令
            self.logger.info(f"执行命令: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            # 实时显示输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.logger.info(output.strip())
                    print(output.strip())
            
            return_code = process.poll()
            
            if return_code != 0:
                stderr = process.stderr.read()
                self.logger.error(f"数据提取失败: {stderr}")
            
            return return_code
            
        except Exception as e:
            raise DataProcessingError(f"降级数据提取失败: {e}")
    
    def _show_extraction_stats(self, output_path: str) -> None:
        """显示提取结果统计"""
        try:
            if not os.path.exists(output_path):
                return
            
            file_stats = get_file_stats(output_path)
            self.logger.info(f"提取结果统计: {output_path}")
            print(f"\n提取结果统计:")
            print(f"输出文件: {output_path}")
            print(f"文件大小: {file_stats['size']}")
            print(f"修改时间: {file_stats['modified']}")
            
            # 尝试统计记录数
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    if output_path.endswith('.json'):
                        data = json.load(f)
                        if isinstance(data, list):
                            self.logger.info(f"提取记录数量: {len(data)}")
                            print(f"记录数量: {len(data)}")
                        elif isinstance(data, dict) and 'messages' in data:
                            self.logger.info(f"提取记录数量: {len(data['messages'])}")
                            print(f"记录数量: {len(data['messages'])}")
                    elif output_path.endswith('.jsonl'):
                        count = sum(1 for _ in f)
                        self.logger.info(f"提取记录数量: {count}")
                        print(f"记录数量: {count}")
            except:
                pass
                
        except Exception as e:
            self.logger.warning(f"显示统计信息失败: {e}")
    
    def _clean_data(self, args: argparse.Namespace) -> int:
        """清洗训练数据"""
        try:
            method = getattr(args, 'clean_method', None)
            if not method:
                self.logger.error("未指定清洗方法，请使用 'raw' 或 'llm'")
                return 1
                
            self.logger.info(f"开始清洗训练数据，使用方法: {method}")
            
            # 获取参数，支持从配置文件读取默认值
            input_path = getattr(args, 'input', None) or self.config.get('dataset_csv_path', 'dataset/csv')
            output_path = getattr(args, 'output', None) or self.config.get('data_path', './dataset/sft.jsonl')
            batch_size = getattr(args, 'batch_size', None) or self.config.get('clean_batch_size', 10)
            workers = getattr(args, 'workers', None) or self.config.get('clean_workers', 4)
            resume = getattr(args, 'resume', False)
            
            self.logger.info(f"输入路径: {input_path}")
            self.logger.info(f"输出路径: {output_path}")
            
            # 确保输出目录存在
            ensure_directory(os.path.dirname(output_path))
            
            # 检查输入路径
            if not os.path.exists(input_path):
                raise FileOperationError("输入路径不存在", input_path)
            
            # 根据清洗方法执行
            if method == 'llm':
                # 获取LLM清洗策略参数
                parser = getattr(args, 'parser', None) or self.config.get('llm_parser', 'scoring')
                accept_score = getattr(args, 'accept_score', None) or self.config.get('accept_score', 2)
                result = self._clean_data_llm(input_path, output_path, batch_size, workers, parser, accept_score, resume)
            else:  # raw
                result = self._clean_data_raw(input_path, output_path)
            
            if result == 0:
                if os.path.exists(output_path):
                    output_stats = get_file_stats(output_path)
                    self.logger.info(f"数据清洗完成: {output_path} ({output_stats['size']})")
            
            return result
            
        except Exception as e:
            raise DataProcessingError(f"数据清洗失败: {e}")
    
    def _clean_data_llm(self, input_path: str, output_path: str, batch_size: int, workers: int, parser: str = 'scoring', accept_score: int = 2, resume: bool = False) -> int:
        """使用LLM清洗数据"""
        try:
            from process_data.generate_chatml_llm import LLMDataProcessor
            
            self.logger.info(f"开始LLM清洗，策略: {parser}")
            if parser == 'scoring':
                self.logger.info(f"分数阈值: {accept_score}")
                self.logger.info(f"批处理大小: {batch_size}")
                self.logger.info(f"工作线程数: {workers}")
            if resume:
                self.logger.info("启用断点续处理")
            
            # 创建LLM数据处理器
            processor = LLMDataProcessor(
                parser=parser,
                accept_score=accept_score,
                batch_size=batch_size,
                workers=workers
            )

            # 处理文件
            scored_csv = os.path.splitext(output_path)[0] + "_scored.csv"
            result = processor.process_file(input_path, output_path, scored_csv=scored_csv, resume=resume)

            if result == 0:
                self.logger.info(f"LLM清洗完成: {output_path}")
                self.logger.info(f"完整打分结果: {scored_csv}")
            else:
                self.logger.error("LLM清洗失败")
                # 失败时回退到raw方法
                self.logger.warning("回退到raw清洗方法")
                return self._clean_data_raw(input_path, output_path)
            
            return result
            
        except ImportError as e:
            self.logger.error(f"无法导入LLM清洗模块: {e}")
            self.logger.warning("回退到raw清洗方法")
            return self._clean_data_raw(input_path, output_path)
        except Exception as e:
            self.logger.error(f"LLM清洗失败: {e}")
            self.logger.warning("回退到raw清洗方法")
            return self._clean_data_raw(input_path, output_path)
    
    def _clean_data_raw(self, input_path: str, output_path: str) -> int:
        """使用原始算法清洗数据"""
        try:
            # 直接导入并调用ChatMLGenerator，避免子进程编码问题
            from process_data.generate_chatml_raw import ChatMLGenerator
            
            self.logger.info(f"开始原始算法清洗")
            self.logger.info(f"输入路径: {input_path}")
            self.logger.info(f"输出路径: {output_path}")
            
            generator = ChatMLGenerator(input_path=input_path, output_path=output_path)
            generator.run()
            
            return 0
            
        except Exception as e:
            raise DataProcessingError(f"原始算法清洗失败: {e}")
    
    def _convert_data(self, args: argparse.Namespace) -> int:
        """转换数据格式"""
        try:
            self.logger.info("开始转换数据格式...")
            
            input_path = args.input
            output_path = args.output
            target_format = getattr(args, 'format', 'chatml')
            
            # 确保输出目录存在
            ensure_directory(os.path.dirname(output_path))
            
            # 检查输入文件
            input_stats = get_file_stats(input_path)
            self.logger.info(f"输入文件: {input_path} ({input_stats['size']})")
            
            # 执行格式转换
            result = self._execute_format_conversion(input_path, output_path, target_format)
            
            if result == 0:
                output_stats = get_file_stats(output_path)
                self.logger.info(f"格式转换完成: {output_path} ({output_stats['size']})")
            
            return result
            
        except Exception as e:
            raise DataProcessingError(f"数据转换失败: {e}")
    
    def _execute_format_conversion(self, input_path: str, output_path: str, target_format: str) -> int:
        """执行格式转换"""
        try:
            # 目前只支持chatml格式转换，其他格式暂不支持
            if target_format == 'chatml':
                # 使用ChatMLGenerator进行chatml格式转换
                from process_data.generate_chatml_raw import ChatMLGenerator
                
                self.logger.info(f"使用ChatMLGenerator进行{target_format}格式转换")
                generator = ChatMLGenerator(input_path=input_path, output_path=output_path)
                generator.run()
                return 0
            else:
                # 其他格式暂不支持，返回错误
                self.logger.error(f"暂不支持 {target_format} 格式转换")
                self.logger.info("目前只支持 chatml 格式转换")
                raise ValidationError(f"暂不支持的格式: {target_format}，目前只支持 chatml 格式")
            
        except Exception as e:
            raise DataProcessingError(f"格式转换执行失败: {e}")
    
    def _merge_data(self, args: argparse.Namespace) -> int:
        """合并多源数据"""
        try:
            self.logger.info("开始合并多源数据...")
            
            input_files = args.inputs
            output_path = args.output
            deduplicate = getattr(args, 'deduplicate', False)
            
            # 确保输出目录存在
            ensure_directory(os.path.dirname(output_path))
            
            # 检查所有输入文件
            for input_file in input_files:
                if not os.path.exists(input_file):
                    raise FileOperationError(f"输入文件不存在", input_file)
                
                file_stats = get_file_stats(input_file)
                self.logger.info(f"输入文件: {input_file} ({file_stats['size']})")
            
            # 执行合并
            result = self._execute_data_merge(input_files, output_path, deduplicate)
            
            if result == 0:
                output_stats = get_file_stats(output_path)
                self.logger.info(f"数据合并完成: {output_path} ({output_stats['size']})")
            
            return result
            
        except Exception as e:
            raise DataProcessingError(f"数据合并失败: {e}")
    
    def _execute_data_merge(self, input_files: List[str], output_path: str, deduplicate: bool) -> int:
        """执行数据合并"""
        try:
            merged_data = []
            seen_items = set() if deduplicate else None
            
            for input_file in input_files:
                self.logger.info(f"处理文件: {input_file}")
                
                # 读取文件数据
                with open(input_file, 'r', encoding='utf-8') as f:
                    if input_file.endswith('.jsonl'):
                        data = [json.loads(line) for line in f]
                    else:
                        data = json.load(f)
                
                if not isinstance(data, list):
                    self.logger.warning(f"跳过非数组格式文件: {input_file}")
                    continue
                
                # 添加数据
                for item in data:
                    if deduplicate:
                        # 简单的去重逻辑
                        item_hash = hash(json.dumps(item, sort_keys=True))
                        if item_hash not in seen_items:
                            merged_data.append(item)
                            seen_items.add(item_hash)
                    else:
                        merged_data.append(item)
                
                self.logger.info(f"已处理 {len(data)} 条记录")
            
            # 保存合并结果
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.endswith('.jsonl'):
                    for item in merged_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                else:
                    json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
            removed_count = sum(len(data) for data in []) - len(merged_data) if deduplicate else 0
            self.logger.info(f"合并完成，总记录数: {len(merged_data)}")
            if deduplicate and removed_count > 0:
                self.logger.info(f"去重移除记录数: {removed_count}")
            
            return 0
            
        except Exception as e:
            raise DataProcessingError(f"数据合并执行失败: {e}")
    
    def _preview_data(self, args: argparse.Namespace) -> int:
        """预览数据样本"""
        try:
            input_path = args.input
            count = getattr(args, 'count', 5)
            
            if not os.path.exists(input_path):
                raise FileOperationError("输入文件不存在", input_path)
            
            self.logger.info(f"预览数据文件: {input_path}")
            
            # 读取数据
            with open(input_path, 'r', encoding='utf-8') as f:
                if input_path.endswith('.jsonl'):
                    data = []
                    for i, line in enumerate(f):
                        if i >= count:
                            break
                        data.append(json.loads(line))
                else:
                    all_data = json.load(f)
                    if isinstance(all_data, list):
                        data = all_data[:count]
                    else:
                        data = [all_data]
            
            # 显示预览
            self.logger.info(f"数据预览: 显示前{len(data)}条记录")
            print(f"\n数据预览 (前 {len(data)} 条记录):")
            print("=" * 80)
            
            for i, item in enumerate(data, 1):
                print(f"\n记录 {i}:")
                print("-" * 40)
                if isinstance(item, dict):
                    for key, value in item.items():
                        # 限制显示长度
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        print(f"  {key}: {value}")
                else:
                    print(f"  {item}")
            
            print("=" * 80)
            return 0
            
        except Exception as e:
            raise DataProcessingError(f"数据预览失败: {e}")
    
    def _show_stats(self, args: argparse.Namespace) -> int:
        """显示数据统计"""
        try:
            input_path = args.input
            
            if not os.path.exists(input_path):
                raise FileOperationError("输入文件不存在", input_path)
            
            file_stats = get_file_stats(input_path)
            
            self.logger.info(f"数据统计: {input_path}")
            print(f"\n数据统计: {input_path}")
            print("=" * 80)
            print(f"文件大小: {file_stats['size']}")
            print(f"修改时间: {file_stats['modified']}")
            
            # 读取并分析数据
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    if input_path.endswith('.jsonl'):
                        record_count = 0
                        total_length = 0
                        
                        for line in f:
                            record_count += 1
                            data = json.loads(line)
                            if isinstance(data, dict) and 'content' in data:
                                total_length += len(str(data['content']))
                            elif isinstance(data, str):
                                total_length += len(data)
                        
                        self.logger.info(f"数据记录数量: {record_count}")
                        print(f"记录数量: {record_count}")
                        if record_count > 0:
                            self.logger.info(f"数据平均长度: {total_length // record_count} 字符")
                            print(f"平均长度: {total_length // record_count} 字符")
                        
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.logger.info(f"数据记录数量: {len(data)}")
                            print(f"记录数量: {len(data)}")
                            if data:
                                # 分析数据结构
                                sample = data[0]
                                if isinstance(sample, dict):
                                    self.logger.info(f"数据字段: {list(sample.keys())}")
                                    print("字段统计:")
                                    for key in sample.keys():
                                        print(f"  - {key}")
                        elif isinstance(data, dict):
                            self.logger.info(f"数据类型: 单个对象, 字段: {list(data.keys())}")
                            print("数据类型: 单个对象")
                            print("字段统计:")
                            for key in data.keys():
                                print(f"  - {key}")
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON解析错误: {e}")
                print(f"JSON解析错误: {e}")
            except Exception as e:
                self.logger.error(f"统计分析错误: {e}")
                print(f"统计分析错误: {e}")
            
            print("=" * 80)
            return 0
            
        except Exception as e:
            raise DataProcessingError(f"数据统计失败: {e}")
    
    def _execute_subprocess(self, cmd: List[str], operation: str) -> int:
        """执行子进程命令"""
        try:
            self.logger.info(f"执行{operation}: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # 实时显示输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.logger.info(output.strip())
                    print(output.strip())
            
            return_code = process.poll()
            
            if return_code != 0:
                self.logger.error(f"{operation}失败，退出码: {return_code}")
            
            return return_code
            
        except Exception as e:
            raise DataProcessingError(f"{operation}子进程执行失败: {e}")