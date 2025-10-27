"""
数据处理命令

提供QQ数据提取、清洗、格式转换、合并等数据处理功能。
支持批量处理和进度监控。
"""

import os
import json
import sys
import csv
import re
import shutil
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass

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


@dataclass
class _EstimateMessage:
    """估算阶段使用的轻量消息结构"""
    role: str
    content: str


@dataclass
class _EstimateQaPair:
    """估算阶段使用的轻量问答结构"""
    id: str
    messages: List[_EstimateMessage]
    images: Optional[List[str]] = None


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
        method = getattr(args, 'clean_method', None)
        
        if method == 'estimate':
            self._validate_clean_estimate_args(args)
            return
        
        output_path = getattr(args, 'output', None) or self.config.get('data_path', './dataset/sft.jsonl')
        
        if method == 'rellm':
            validate_path(output_path, must_exist=False, check_parent=True)
            input_path = getattr(args, 'input', None) or self.config.get('dataset_csv_path', 'dataset/csv')
            validate_path(input_path, must_exist=True)
            scored_path = getattr(args, 'scored', None) or os.path.splitext(output_path)[0] + "_scored.csv"
            validate_path(scored_path, must_exist=True)
            accept_score = getattr(args, 'accept_score', None)
            if accept_score is not None:
                validate_positive_int(accept_score, "accept_score")
            return
        
        # 获取实际使用的路径（支持从配置读取）
        input_path = getattr(args, 'input', None) or self.config.get('dataset_csv_path', 'dataset/csv')
        
        validate_path(input_path, must_exist=True)
        validate_path(output_path, must_exist=False, check_parent=True)
        
        if hasattr(args, 'batch_size') and getattr(args, 'batch_size', None) is not None:
            validate_positive_int(args.batch_size, "batch_size")
        
        if hasattr(args, 'workers') and getattr(args, 'workers', None) is not None:
            validate_positive_int(args.workers, "workers")
    
    def _validate_clean_estimate_args(self, args: argparse.Namespace) -> None:
        """验证清洗估算参数"""
        estimate_method = getattr(args, 'estimate_method', None)
        if estimate_method != 'llm':
            raise ValidationError("估算目前仅支持 llm 策略")
        
        input_path = getattr(args, 'input', None) or self.config.get('dataset_csv_path', 'dataset/csv')
        validate_path(input_path, must_exist=True)
        
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
            
            if method == 'estimate':
                return self._clean_data_estimate(args)
            
            # 获取输出路径，支持从配置文件读取默认值
            output_path = getattr(args, 'output', None) or self.config.get('data_path', './dataset/sft.jsonl')
            
            self.logger.info(f"输出路径: {output_path}")
            
            # 确保输出目录存在
            ensure_directory(os.path.dirname(output_path))
            
            # 根据清洗方法执行
            if method == 'rellm':
                accept_score = getattr(args, 'accept_score', None) or self.config.get('accept_score', 2)
                input_path = getattr(args, 'input', None) or self.config.get('dataset_csv_path', 'dataset/csv')
                scored_path = getattr(args, 'scored', None) or os.path.splitext(output_path)[0] + "_scored.csv"
                self.logger.info(f"输入路径: {input_path}")
                self.logger.info(f"打分结果路径: {scored_path}")
                self.logger.info(f"目标分数阈值: {accept_score}")
                if not os.path.exists(scored_path):
                    raise FileOperationError("打分结果文件不存在", scored_path)
                result = self._clean_data_rellm(scored_path, input_path, output_path, accept_score)
            else:
                # 获取输入路径及并发参数
                input_path = getattr(args, 'input', None) or self.config.get('dataset_csv_path', 'dataset/csv')
                self.logger.info(f"输入路径: {input_path}")
                
                # 检查输入路径
                if not os.path.exists(input_path):
                    raise FileOperationError("输入路径不存在", input_path)
                
                if method == 'llm':
                    # 获取LLM清洗策略参数
                    batch_size = getattr(args, 'batch_size', None) or self.config.get('clean_batch_size', 10)
                    workers = getattr(args, 'workers', None) or self.config.get('clean_workers', 4)
                    self.logger.info(f"批处理大小: {batch_size}")
                    self.logger.info(f"工作线程数: {workers}")
                    parser = getattr(args, 'parser', None) or self.config.get('llm_parser', 'scoring')
                    accept_score = getattr(args, 'accept_score', None) or self.config.get('accept_score', 2)
                    result = self._clean_data_llm(input_path, output_path, batch_size, workers, parser, accept_score)
                else:  # raw
                    result = self._clean_data_raw(input_path, output_path)
            
            if result == 0:
                if os.path.exists(output_path):
                    output_stats = get_file_stats(output_path)
                    self.logger.info(f"数据清洗完成: {output_path} ({output_stats['size']})")
            
            return result
            
        except Exception as e:
            raise DataProcessingError(f"数据清洗失败: {e}")

    def _clean_data_estimate(self, args: argparse.Namespace) -> int:
        """估算LLM清洗资源开销"""
        try:
            estimate_method = getattr(args, 'estimate_method', None)
            if estimate_method != 'llm':
                self.logger.error("当前仅支持 llm 估算策略")
                return 1
            
            input_path = getattr(args, 'input', None) or self.config.get('dataset_csv_path', 'dataset/csv')
            parser = getattr(args, 'parser', 'scoring')
            batch_size = getattr(args, 'batch_size', None) or self.config.get('clean_batch_size', 10)
            workers = getattr(args, 'workers', None) or self.config.get('clean_workers', 4)
            accept_score = getattr(args, 'accept_score', None) or self.config.get('accept_score', 2)
            
            self.logger.info(f"估算输入路径: {input_path}")
            self.logger.info(f"估算处理策略: {parser}")
            self.logger.info(f"估算批处理大小: {batch_size}")
            self.logger.info(f"估算工作线程数(对齐配置使用): {workers}")
            self.logger.info(f"估算分数阈值: {accept_score}")
            
            return self._estimate_llm_clean(
                input_path=input_path,
                batch_size=batch_size,
                parser=parser,
                accept_score=accept_score
            )
        except Exception as e:
            raise DataProcessingError(f"清洗资源估算失败: {e}")

    def _clean_data_llm(self, input_path: str, output_path: str, batch_size: int, workers: int, parser: str = 'scoring', accept_score: int = 2) -> int:
        """使用LLM清洗数据"""
        try:
            from process_data.generate_chatml_llm import LLMDataProcessor
            
            self.logger.info(f"开始LLM清洗，策略: {parser}")
            if parser == 'scoring':
                self.logger.info(f"分数阈值: {accept_score}")
                self.logger.info(f"批处理大小: {batch_size}")
                self.logger.info(f"工作线程数: {workers}")
            
            # 创建LLM数据处理器
            processor = LLMDataProcessor(
                parser=parser,
                accept_score=accept_score,
                batch_size=batch_size,
                workers=workers
            )

            # 处理文件
            scored_csv = os.path.splitext(output_path)[0] + "_scored.csv"
            result = processor.process_file(input_path, output_path, scored_csv=scored_csv)

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
    
    def _estimate_llm_clean(self, input_path: str, batch_size: int, parser: str, accept_score: int) -> int:
        """估算LLM清洗时的字符传输量"""
        try:
            if parser != 'scoring':
                raise ValidationError("估算暂时仅支持 scoring 策略")
            
            clean_set_args = self.config.get('clean_set_args', {})
            if not isinstance(clean_set_args, dict):
                clean_set_args = {}
            openai_api = clean_set_args.get('openai_api', {})
            if not isinstance(openai_api, dict):
                openai_api = {}
            configured_batch = openai_api.get('clean_batch_size')
            fallback_batch = self.config.get('clean_batch_size', 10)
            candidate_batch = batch_size or configured_batch or fallback_batch
            try:
                actual_batch_size = int(candidate_batch)
            except (TypeError, ValueError):
                actual_batch_size = fallback_batch
            actual_batch_size = max(1, actual_batch_size)
            
            scoring_prompt = None
            qa_pairs: List[Any] = []
            
            try:
                from process_data.generate_chatml_llm import LLMDataProcessor  # type: ignore
                processor = LLMDataProcessor(
                    parser=parser,
                    accept_score=accept_score,
                    batch_size=actual_batch_size
                )
                qa_pairs = processor._load_qa_pairs(input_path)
                scoring_prompt = processor.strategy._build_scoring_prompt()
            except ImportError as import_error:
                if 'pandas' not in str(import_error):
                    raise DataProcessingError(f"无法加载LLM估算模块: {import_error}") from import_error
                self.logger.warning("检测到缺少pandas，使用轻量估算逻辑")
                scoring_prompt = self._fallback_scoring_prompt()
                qa_pairs = self._load_qa_pairs_for_estimate(input_path)
            except Exception as module_error:
                raise DataProcessingError(f"初始化LLM估算模块失败: {module_error}") from module_error
            
            if not qa_pairs:
                self.logger.warning("没有找到可估算的问答对")
                print("未找到可估算的问答对")
                return 0
            
            prompt_prefix = "请评估以下问答对：\n"
            
            total_input_english = 0
            total_input_chinese = 0
            total_output_english = 0
            total_output_chinese = 0
            skipped_with_images = 0
            effective_pairs = 0
            request_batches = 0
            
            for batch in self._iter_batches(qa_pairs, actual_batch_size):
                qa_list = []
                for qa in batch:
                    if getattr(qa, 'images', None):
                        skipped_with_images += 1
                        continue
                    
                    user_msg = next((msg.content for msg in qa.messages if msg.role == 'user'), '')
                    assistant_msg = next((msg.content for msg in qa.messages if msg.role == 'assistant'), '')
                    qa_list.append({
                        "id": qa.id,
                        "Q": user_msg,
                        "A": assistant_msg
                    })
                
                if not qa_list:
                    continue
                
                qa_list_json = json.dumps(qa_list, ensure_ascii=False)
                eng_in, chi_in = self._count_char_types(scoring_prompt + prompt_prefix + qa_list_json)
                total_input_english += eng_in
                total_input_chinese += chi_in
                
                estimated_output = json.dumps(
                    [{"id": item["id"], "score": 0} for item in qa_list],
                    ensure_ascii=False
                )
                eng_out, chi_out = self._count_char_types(estimated_output)
                total_output_english += eng_out
                total_output_chinese += chi_out
                
                effective_pairs += len(qa_list)
                request_batches += 1
            
            if request_batches == 0:
                self.logger.warning("所有问答均包含图片或无有效内容，未产生估算请求")
                print("所有问答均被跳过，未产生估算请求")
                print("模型输入字符总数: 0")
                print("模型输出字符总数: 0")
                print("模型输入字符预估Token数(英0.3/中0.6): 0.00")
                print("模型输出字符预估Token数(英0.3/中0.6): 0.00")
                return 0
            
            total_input_chars = total_input_english + total_input_chinese
            total_output_chars = total_output_english + total_output_chinese
            input_tokens = self._estimate_tokens(total_input_english, total_input_chinese)
            output_tokens = self._estimate_tokens(total_output_english, total_output_chinese)
            
            print(f"估算批次数: {request_batches}")
            print(f"参与估算的问答数量: {effective_pairs}")
            if skipped_with_images:
                print(f"包含图片而跳过的问答数量: {skipped_with_images}")
            print(f"模型输入字符总数: {total_input_chars}")
            print(f"模型输出字符总数: {total_output_chars}")
            print(f"模型输入字符预估Token数(英0.3/中0.6): {input_tokens:.2f}")
            print(f"模型输出字符预估Token数(英0.3/中0.6): {output_tokens:.2f}")
            
            self.logger.info(
                f"估算完成，批次数: {request_batches}, 输入字符: {total_input_chars}, 输出字符: {total_output_chars}\n"
                f"输入token: {input_tokens:.2f}\n输出token: {output_tokens:.2f}"
            )
            return 0
        
        except ValidationError:
            raise
        except Exception as e:
            raise DataProcessingError(f"LLM清洗字符估算失败: {e}")
    
    def _fallback_scoring_prompt(self) -> str:
        """当无法导入LLM模块时使用的默认打分提示词"""
        return """# 角色
你是一个数据质量评估员。

# 任务
你的任务是评估下面提供的聊天记录的**逻辑性**、**相关性**以及**风格代表性**。目标是识别并过滤掉那些回答与问题**明显不匹配**、**逻辑严重混乱**的样本，筛选出具有人类聊天风格独特性与辨识度的样本。请根据以下核心评估点给出一个1到5的整数分数，并将该分数与原始 `id` 一起输出。

**重要考量:**
1.  **简短回答的有效性:** 请注意，诸如“好的”、“是的”、“收到”、“嗯”、“知道了”等简短的肯定、确认或应答，在合适的语境下是完全**有逻辑且相关的**。**不要仅仅因为回答简短就将其评为低分。** 只有当这类简短回答与【问题/上下文 Q】**明显不符**时，才应考虑低分。
2.  **处理错别字和自我纠正:** 聊天记录中可能包含常见的打字错误（错别字）或用户先打错字随后又自行纠正的情况（例如，发送“我想去1楼”紧接着又发送“*2楼”进行更正）。在评估时，请**聚焦于用户想要表达的最终意图和信息的核心内容**，而**不应仅仅因为存在错别字或纠正过程就判定为低质量**。

# 核心评估点 (请在心中衡量)
1.  **相关性 (Relevance):** 【回答 A】是否直接回应或恰当地衔接了【问题/上下文 Q】？只有当两者**明显矛盾**或**完全不相关**时，才应给出低分。
2.  **逻辑性 (Coherence):** 【回答 A】是否在语义与结构上自洽？当回答**逻辑混乱**或**与上下文冲突**时，才应给出低分。
3. **风格代表性 (Style Representativeness):** 关注回答是否展现出自然的人类聊天风格，例如特定语气、情绪表达、俚语或口头禅等。风格代表性是获得5分的必要条件，但不是低分的主要依据。

# 评分标准 (1-5分)
*   **1分 (极差):** 聊天记录完全不相关，或逻辑严重混乱。
*   **2分 (差):** 大部分问答相关性低，存在明显逻辑问题。
*   **3分 (中等):** 相关性尚可但不突出，逻辑基本成立。
*   **4分 (良好):** 问答高度相关，逻辑清晰。
*   **5分 (优秀):** 满足4分标准，并展现明显的人类聊天风格特征。

# 输出要求
请严格按照以下 JSON 格式输出，每条记录仅包含原始 `id` 与评估的整数分数 `score`，不要包含任何额外说明：
[
  {
    "id": "<这里填入第1条输入数据的id值>",
    "score": <1-5的整数评分>
  },
  {
    "id": "<这里填入第2条输入数据的id值>",
    "score": <1-5的整数评分>
  }
  …
]"""
    
    def _load_qa_pairs_for_estimate(self, input_path: str) -> List[_EstimateQaPair]:
        """在缺少依赖时加载问答数据用于估算"""
        qa_pairs: List[_EstimateQaPair] = []
        
        if os.path.isfile(input_path):
            lower_name = input_path.lower()
            if lower_name.endswith('.jsonl'):
                qa_pairs.extend(self._load_qa_pairs_from_jsonl_for_estimate(input_path))
            elif lower_name.endswith('.csv'):
                qa_pairs.extend(self._load_qa_pairs_from_csv_for_estimate(input_path))
            else:
                raise DataProcessingError(f"不支持的估算输入格式: {input_path}")
        elif os.path.isdir(input_path):
            csv_files = self._collect_csv_files_for_estimate(input_path)
            for csv_file in csv_files:
                qa_pairs.extend(self._load_qa_pairs_from_csv_for_estimate(csv_file))
            if not csv_files:
                raise DataProcessingError("估算目录中未找到CSV文件")
        else:
            raise DataProcessingError(f"估算输入路径无效: {input_path}")
        
        return qa_pairs
    
    def _load_qa_pairs_from_jsonl_for_estimate(self, file_path: str) -> List[_EstimateQaPair]:
        """备用JSONL加载逻辑"""
        qa_pairs: List[_EstimateQaPair] = []
        try:
            with open(file_path, 'r', encoding='utf-8') as fp:
                for idx, line in enumerate(fp, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    raw_messages = []
                    if isinstance(data, dict):
                        if 'messages' in data and isinstance(data['messages'], list):
                            raw_messages = data['messages']
                        elif 'conversations' in data and isinstance(data['conversations'], list):
                            raw_messages = data['conversations']
                    
                    messages: List[_EstimateMessage] = []
                    for msg in raw_messages:
                        role = msg.get('role') or msg.get('from') or 'user'
                        content = msg.get('content') or msg.get('value') or ''
                        content = str(content).strip()
                        if not content:
                            continue
                        messages.append(_EstimateMessage(role=role, content=content))
                    
                    if messages:
                        qa_pairs.append(_EstimateQaPair(
                            id=f"{os.path.basename(file_path)}_{idx}",
                            messages=messages,
                            images=data.get('images')
                        ))
        except OSError as exc:
            raise DataProcessingError(f"读取JSONL失败: {exc}") from exc
        
        return qa_pairs
    
    def _load_qa_pairs_from_csv_for_estimate(self, csv_path: str) -> List[_EstimateQaPair]:
        """备用CSV加载逻辑"""
        qa_pairs: List[_EstimateQaPair] = []
        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as fp:
                reader = csv.DictReader(fp)
                if not reader.fieldnames:
                    return qa_pairs
                
                conversations: List[List[_EstimateMessage]] = []
                current: List[_EstimateMessage] = []
                
                for row in reader:
                    if row is None:
                        continue
                    message_raw = row.get('msg') or row.get('message') or ''
                    message = str(message_raw).strip()
                    if not message:
                        continue
                    
                    sender_flag = row.get('is_sender', 0)
                    try:
                        is_sender = int(float(sender_flag))
                    except (TypeError, ValueError):
                        is_sender = 0
                    
                    role = 'assistant' if is_sender == 1 else 'user'
                    current.append(_EstimateMessage(role=role, content=message))
                    
                    if role == 'user' and len(current) > 1 and current[-2].role == 'assistant':
                        if len(current) > 1:
                            conversations.append(current[:-1])
                        current = [current[-1]]
                
                if current:
                    conversations.append(current)
                
                for index, conv in enumerate(conversations):
                    if len(conv) < 2:
                        continue
                    roles = {item.role for item in conv}
                    if not {'user', 'assistant'}.issubset(roles):
                        continue
                    qa_pairs.append(_EstimateQaPair(
                        id=f"{os.path.basename(csv_path)}_{index}",
                        messages=conv
                    ))
        
        except OSError as exc:
            raise DataProcessingError(f"读取CSV失败: {exc}") from exc
        
        return qa_pairs
    
    @staticmethod
    def _collect_csv_files_for_estimate(root_path: str) -> List[str]:
        """收集目录下所有CSV文件"""
        csv_files: List[str] = []
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.lower().endswith('.csv'):
                    csv_files.append(os.path.join(dirpath, filename))
        return csv_files
    
    @staticmethod
    def _count_char_types(text: str) -> Tuple[int, int]:
        """统计文本中英文字符与中文字符数量"""
        english = 0
        chinese = 0
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                chinese += 1
            else:
                english += 1
        return english, chinese
    
    @staticmethod
    def _iter_batches(items: List[Any], batch_size: int):
        """生成固定大小的批次"""
        size = max(1, batch_size or 1)
        for start in range(0, len(items), size):
            yield items[start:start + size]
    
    @staticmethod
    def _estimate_tokens(english_chars: int, chinese_chars: int) -> float:
        """根据经验系数估算token数量"""
        english = max(0, english_chars or 0)
        chinese = max(0, chinese_chars or 0)
        return english * 0.3 + chinese * 0.6

    def _clean_data_rellm(self, scored_path: str, input_path: str, output_path: str, accept_score: int) -> int:
        """基于已有打分结果重新筛选数据"""
        try:
            if os.path.exists(output_path):
                backup_path = f"{output_path}.{datetime.now().strftime('%Y%m%d%H%M%S')}.bak"
                shutil.copy(output_path, backup_path)
                self.logger.info(f"已备份现有输出文件: {backup_path}")
            
            accepted_scores: Dict[str, float] = {}
            total_records = 0
            
            with open(scored_path, 'r', encoding='utf-8', newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                fieldnames = {name.strip() for name in (reader.fieldnames or []) if name}
                required_fields = {'id', 'score'}
                missing_fields = required_fields - fieldnames
                if missing_fields:
                    raise DataProcessingError(f"打分结果缺少必要字段: {', '.join(sorted(missing_fields))}")
                
                for row in reader:
                    total_records += 1
                    score_raw = row.get('score')
                    uid = str(row.get('id') or '').strip()
                    if not uid:
                        continue
                    try:
                        score_value = float(score_raw)
                    except (TypeError, ValueError):
                        self.logger.debug(f"跳过无法解析分数的记录: {row}")
                        continue
                    if score_value >= accept_score:
                        accepted_scores[uid] = score_value
            
            if not accepted_scores:
                self.logger.warning("没有记录满足当前分数阈值，未生成新数据")
                return 0
            
            self.logger.info(f"打分文件共 {total_records} 条，满足阈值的记录 {len(accepted_scores)} 条")
            
            target_indices: Dict[str, Set[int]] = defaultdict(set)
            for uid in accepted_scores.keys():
                if '_' not in uid:
                    continue
                file_name, idx_str = uid.rsplit('_', 1)
                try:
                    target_indices[file_name].add(int(idx_str))
                except ValueError:
                    self.logger.debug(f"跳过无法解析编号的记录ID: {uid}")
                    continue
            
            system_prompt = (self.config.get('system_prompt', '') or '').strip()
            include_system = system_prompt and system_prompt != "*"
            
            def replace_spaces_with_newlines(content: str) -> str:
                if not content:
                    return ''
                content = re.sub(r'[。！？；，、：] +', lambda m: m.group(0)[0] + '\n', content)
                content = re.sub(r' +', '\n', content)
                return content.strip()
            
            def process_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
                processed: List[Dict[str, str]] = []
                current_role = None
                current_chunks: List[str] = []
                
                for msg in messages:
                    content = replace_spaces_with_newlines(msg['content'])
                    role = msg['role']
                    if not content:
                        continue
                    if role == current_role:
                        current_chunks.append(content)
                    else:
                        if current_chunks and current_role:
                            processed.append({"role": current_role, "content": "\n".join(current_chunks)})
                        current_role = role
                        current_chunks = [content]
                
                if current_chunks and current_role:
                    processed.append({"role": current_role, "content": "\n".join(current_chunks)})
                
                return processed
            
            selected = 0
            found_ids: Set[str] = set()
            
            def handle_conversation(file_name: str, conv_index: int, messages: List[Dict[str, str]], writer) -> None:
                nonlocal selected
                if conv_index not in target_indices.get(file_name, set()):
                    return
                conv_id = f"{file_name}_{conv_index}"
                score = accepted_scores.get(conv_id)
                if score is None:
                    return
                found_ids.add(conv_id)
                
                processed_msgs = process_messages(messages)
                if len(processed_msgs) < 2:
                    return
                
                output_messages: List[Dict[str, str]] = []
                if include_system:
                    output_messages.append({"role": "system", "content": system_prompt})
                output_messages.extend(processed_msgs)
                
                writer.write(json.dumps({"messages": output_messages}, ensure_ascii=False) + '\n')
                selected += 1
            
            def iter_csv_files(path: str) -> List[str]:
                if os.path.isfile(path) and path.lower().endswith('.csv'):
                    return [path]
                csv_files: List[str] = []
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith('.csv'):
                            csv_files.append(os.path.join(root, file))
                return csv_files
            
            def load_conversations_from_csv(csv_path: str, writer) -> None:
                file_name = os.path.basename(csv_path)
                indices_for_file = target_indices.get(file_name)
                if not indices_for_file:
                    return
                conv_index = 0
                conversation: List[Dict[str, str]] = []
                
                with open(csv_path, 'r', encoding='utf-8', newline='') as fp:
                    reader = csv.DictReader(fp)
                    if not reader.fieldnames or 'msg' not in reader.fieldnames:
                        return
                    
                    for row in reader:
                        msg = str(row.get('msg') or '').strip()
                        if not msg:
                            continue
                        is_sender_raw = row.get('is_sender', 0)
                        try:
                            is_sender = int(float(is_sender_raw))
                        except (TypeError, ValueError):
                            is_sender = 0
                        role = 'assistant' if is_sender == 1 else 'user'
                        
                        conversation.append({"role": role, "content": msg})
                        
                        if role == 'user' and len(conversation) > 1 and conversation[-2]['role'] == 'assistant':
                            finished = conversation[:-1]
                            if any(m['role'] == 'user' for m in finished) and any(m['role'] == 'assistant' for m in finished):
                                handle_conversation(file_name, conv_index, finished, writer)
                            conv_index += 1
                            conversation = [conversation[-1]]
                    
                    if conversation and any(m['role'] == 'user' for m in conversation) and any(m['role'] == 'assistant' for m in conversation):
                        handle_conversation(file_name, conv_index, conversation, writer)
            
            with open(output_path, 'w', encoding='utf-8') as out_fp:
                csv_files = iter_csv_files(input_path)
                if not csv_files:
                    raise DataProcessingError("未在输入路径中找到任何CSV文件")
                
                for csv_file in csv_files:
                    load_conversations_from_csv(csv_file, out_fp)
            
            if selected == 0:
                self.logger.warning("没有问答对满足筛选条件或未找到匹配的原始记录")
            
            missing = sorted(set(accepted_scores.keys()) - found_ids)
            if missing:
                self.logger.warning(f"有 {len(missing)} 条满足分数阈值的记录未在原始数据中匹配到，示例: {missing[:5]}")
            
            self.logger.info(f"重新筛选完成: {selected}/{len(accepted_scores)} 条记录写入 (阈值 {accept_score})")
            return 0
        
        except DataProcessingError:
            raise
        except Exception as e:
            raise DataProcessingError(f"重新筛选打分数据失败: {e}") from e
    
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
