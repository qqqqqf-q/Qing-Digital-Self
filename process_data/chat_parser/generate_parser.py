#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一聊天数据解析器

支持自动检测数据源类型（QQ/Telegram/WeChat）并调用对应的解析器，
提供统一的数据解析入口，符合工厂模式设计。
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any, List
from pathlib import Path
from enum import Enum
import json

# 设置NumExpr最大线程数，避免警告信息
os.environ.setdefault('NUMEXPR_MAX_THREADS', '12')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.config.config import get_config
from utils.logger.logger import get_logger


class DataSourceType(Enum):
    """数据源类型枚举"""
    QQ = "qq"
    TELEGRAM = "telegram"
    WECHAT = "wechat"
    UNKNOWN = "unknown"


class DataSourceDetector:
    """数据源类型检测器"""
    
    def __init__(self, data_dir: str = "./dataset/original/"):
        self.data_dir = Path(data_dir)
        self.logger = get_logger('DataSourceDetector')
    
    def detect_qq_data(self) -> bool:
        """检测是否存在QQ数据"""
        # 检查QQ数据库文件
        qq_db_files = list(self.data_dir.glob("*.db"))
        if qq_db_files:
            self.logger.info(f"检测到QQ数据库文件: {[str(f) for f in qq_db_files]}")
            return True
        
        # 检查其他可能的QQ数据格式
        qq_data_patterns = ["qq*.db", "QQ*.db", "Msg*.db", "msg*.db"]
        for pattern in qq_data_patterns:
            if list(self.data_dir.glob(pattern)):
                self.logger.info(f"检测到QQ数据文件: {pattern}")
                return True
        
        return False

    def detect_wechat_data(self) -> bool:
        """检测是否存在WeChat数据"""
        wechat_dir = self.data_dir / "wechat"
        if wechat_dir.is_dir():
            wechat_db_files = list(wechat_dir.glob("MSG*.db"))
            if wechat_db_files:
                self.logger.info(f"检测到WeChat数据库文件: {[str(f) for f in wechat_db_files]}")
                return True
        return False
    
    def detect_telegram_data(self) -> bool:
        """检测是否存在Telegram数据"""
        if not self.data_dir.exists():
            return False
        
        # 检查Telegram导出的目录结构
        # Telegram导出通常为 ChatExport_YYYY-MM-DD_HH-MM-SS 格式的目录
        telegram_dirs = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # 检查是否包含 result.json 文件
                result_file = item / "result.json"
                if result_file.exists():
                    # 进一步验证是否为Telegram格式
                    if self._validate_telegram_json(result_file):
                        telegram_dirs.append(item)
        
        if telegram_dirs:
            self.logger.info(f"检测到Telegram数据目录: {[str(d) for d in telegram_dirs]}")
            return True
        
        return False
    
    def _validate_telegram_json(self, json_file: Path) -> bool:
        """验证JSON文件是否为Telegram格式"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查Telegram特有字段
            required_fields = ['name', 'type', 'id', 'messages']
            return all(field in data for field in required_fields)
        
        except (json.JSONDecodeError, Exception):
            return False
    
    def detect_data_sources(self) -> List[DataSourceType]:
        """检测所有可用的数据源类型"""
        sources = []
        
        if self.detect_qq_data():
            sources.append(DataSourceType.QQ)
        
        if self.detect_telegram_data():
            sources.append(DataSourceType.TELEGRAM)

        if self.detect_wechat_data():
            sources.append(DataSourceType.WECHAT)
        
        if not sources:
            sources.append(DataSourceType.UNKNOWN)
        
        return sources


class UnifiedParser:
    """统一解析器管理器"""
    
    def __init__(self, data_dir: str = "./dataset/original/", output_dir: str = "./dataset/csv/"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.config = get_config()
        self.logger = get_logger('UnifiedParser')
        self.detector = DataSourceDetector(data_dir)
    
    def parse_with_source_type(self, source_type: DataSourceType, **kwargs) -> int:
        """根据指定的数据源类型进行解析"""
        if source_type == DataSourceType.QQ:
            return self._parse_qq_data(**kwargs)
        elif source_type == DataSourceType.TELEGRAM:
            return self._parse_telegram_data(**kwargs)
        elif source_type == DataSourceType.WECHAT:
            return self._parse_wechat_data(**kwargs)
        else:
            self.logger.error(f"不支持的数据源类型: {source_type}")
            return 1
    
    def parse_auto(self, prefer_source: Optional[str] = None, **kwargs) -> int:
        """自动检测数据源并解析。
        如果指定了 prefer_source，则只解析该来源。
        否则，将解析所有检测到的数据源。
        """
        available_sources = self.detector.detect_data_sources()
        
        if DataSourceType.UNKNOWN in available_sources and len(available_sources) == 1:
            self.logger.error(f"未检测到有效数据源，请检查数据目录: {self.data_dir}")
            return 1
        
        # 移除UNKNOWN类型
        available_sources = [s for s in available_sources if s != DataSourceType.UNKNOWN]
        
        if not available_sources:
            self.logger.error("未检测到有效数据源")
            return 1
        
        # 如果指定了首选数据源，优先使用
        if prefer_source:
            prefer_type = None
            if prefer_source.lower() in ['qq']:
                prefer_type = DataSourceType.QQ
            elif prefer_source.lower() in ['tg', 'telegram']:
                prefer_type = DataSourceType.TELEGRAM
            elif prefer_source.lower() in ['wx', 'wechat']:
                prefer_type = DataSourceType.WECHAT
            
            if prefer_type and prefer_type in available_sources:
                self.logger.info(f"使用指定的数据源: {prefer_type.value}")
                return self.parse_with_source_type(prefer_type, **kwargs)
            else:
                self.logger.warning(f"指定的数据源 '{prefer_source}' 不可用或未检测到，将解析所有可用源。")

        # 自动解析所有检测到的数据源
        self.logger.info(f"自动模式启动，将解析所有检测到的数据源: {[s.value for s in available_sources]}")
        
        overall_success = True
        for source in available_sources:
            self.logger.info(f"--- 开始解析: {source.value} ---")
            result = self.parse_with_source_type(source, **kwargs)
            if result != 0:
                self.logger.error(f"解析数据源 {source.value} 失败。")
                overall_success = False
            else:
                self.logger.info(f"--- 完成解析: {source.value} ---")

        if overall_success:
            self.logger.info("所有可用数据源均已成功解析。")
        else:
            self.logger.warning("部分数据源在解析过程中出现错误。")
            
        return 0 if overall_success else 1
    
    def _parse_qq_data(self, **kwargs) -> int:
        """解析QQ数据"""
        try:
            from .qq_parser import QQParser
            
            # 获取QQ解析参数
            qq_db_path = kwargs.get('qq_db_path') or self.config.get('qq_db_path')
            qq_number_ai = kwargs.get('qq_number_ai') or self.config.get('qq_number_ai')
            
            if not qq_db_path:
                # 尝试自动查找QQ数据库文件
                qq_db_files = list(Path(self.data_dir).glob("*.db"))
                if qq_db_files:
                    qq_db_path = str(qq_db_files[0])
                    self.logger.info(f"自动选择QQ数据库文件: {qq_db_path}")
                else:
                    self.logger.error("未找到QQ数据库文件")
                    return 1
            
            self.logger.info(f"开始解析QQ数据: {qq_db_path}")
            
            # 创建QQ解析器并执行解析
            parser = QQParser(
                db_path=qq_db_path,
                output_dir=self.output_dir,
                qq_number_ai=qq_number_ai
            )
            
            parser.parse_all()
            self.logger.info("QQ数据解析完成")
            return 0
            
        except ImportError as e:
            self.logger.error(f"无法导入QQ解析器: {e}")
            return 1
        except Exception as e:
            self.logger.error(f"QQ数据解析失败: {e}")
            return 1

    def _parse_wechat_data(self, **kwargs) -> int:
        """解析WeChat数据"""
        try:
            from .wx_parser import WXParser
            
            wechat_data_dir = os.path.join(self.data_dir, 'wechat')
            
            if not os.path.isdir(wechat_data_dir):
                self.logger.error(f"WeChat数据目录不存在: {wechat_data_dir}")
                return 1

            self.logger.info(f"开始解析WeChat数据: {wechat_data_dir}")
            
            parser = WXParser(
                input_dir=wechat_data_dir,
                output_dir=self.output_dir
            )
            
            parser.run()
            self.logger.info("WeChat数据解析完成")
            return 0
            
        except ImportError as e:
            self.logger.error(f"无法导入WeChat解析器: {e}")
            return 1
        except Exception as e:
            self.logger.error(f"WeChat数据解析失败: {e}")
            return 1
    
    def _parse_telegram_data(self, **kwargs) -> int:
        """解析Telegram数据"""
        try:
            from .tg_parser import TGParser
            
            # 获取Telegram解析参数
            telegram_chat_id = kwargs.get('telegram_chat_id') or self.config.get('telegram_chat_id')
            tg_data_dir = kwargs.get('tg_data_dir') or self.config.get('tg_data_dir') or self.data_dir
            
            self.logger.info(f"开始解析Telegram数据: {tg_data_dir}")
            
            # 创建Telegram解析器并执行解析
            parser = TGParser(
                data_dir=tg_data_dir,
                output_dir=self.output_dir,
                telegram_chat_id=telegram_chat_id
            )
            
            parser.parse_all()
            self.logger.info("Telegram数据解析完成")
            return 0
            
        except ImportError as e:
            self.logger.error(f"无法导入Telegram解析器: {e}")
            return 1
        except Exception as e:
            self.logger.error(f"Telegram数据解析失败: {e}")
            return 1
    
    def list_available_sources(self) -> Dict[str, Any]:
        """列出所有可用的数据源"""
        sources = self.detector.detect_data_sources()
        
        result = {
            'available_sources': [s.value for s in sources if s != DataSourceType.UNKNOWN],
            'data_directory': str(self.data_dir),
            'output_directory': str(self.output_dir)
        }
        
        # 添加详细信息
        if DataSourceType.QQ in sources:
            qq_files = list(Path(self.data_dir).glob("*.db"))
            result['qq_files'] = [str(f) for f in qq_files]
        
        if DataSourceType.TELEGRAM in sources:
            tg_dirs = []
            for item in Path(self.data_dir).iterdir():
                if item.is_dir() and (item / "result.json").exists():
                    tg_dirs.append(str(item))
            result['telegram_directories'] = tg_dirs

        if DataSourceType.WECHAT in sources:
            wechat_dir = Path(self.data_dir) / 'wechat'
            if wechat_dir.is_dir():
                wx_files = list(wechat_dir.glob("MSG*.db"))
                result['wechat_files'] = [str(f) for f in wx_files]
        
        return result


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='统一聊天数据解析器 - 支持自动检测和解析QQ/Telegram/WeChat数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 自动检测数据源并解析
  python generate_parser.py

  # 指定数据源类型
  python generate_parser.py --source qq
  python generate_parser.py --source tg
  python generate_parser.py --source wx

  # 列出可用数据源
  python generate_parser.py --list

  # 指定输入输出目录
  python generate_parser.py --data-dir "./dataset/original/" --output-dir "./dataset/csv/"

  # QQ特定参数
  python generate_parser.py --source qq --qq-db-path "./dataset/original/qq.db" --qq-number-ai "123456789"

  # Telegram特定参数
  python generate_parser.py --source tg --telegram-chat-id "Your Chat Name"
        """
    )
    
    parser.add_argument(
        '--source',
        choices=['qq', 'tg', 'telegram', 'wx', 'wechat'],
        help='指定数据源类型 (qq: QQ, tg/telegram: Telegram, wx/wechat: WeChat)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./dataset/original/',
        help='数据目录路径 (默认: "./dataset/original/")'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./dataset/csv/',
        help='输出目录路径 (默认: "./dataset/csv/")'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用的数据源'
    )
    
    # QQ特定参数
    qq_group = parser.add_argument_group('QQ数据源参数')
    qq_group.add_argument(
        '--qq-db-path',
        type=str,
        help='QQ数据库文件路径'
    )
    qq_group.add_argument(
        '--qq-number-ai',
        type=str,
        help='AI的QQ号码(用于区分发送者)'
    )
    
    # Telegram特定参数
    tg_group = parser.add_argument_group('Telegram数据源参数')
    tg_group.add_argument(
        '--telegram-chat-id',
        type=str,
        help='AI的Telegram聊天名称(用于区分发送者)'
    )
    tg_group.add_argument(
        '--tg-data-dir',
        type=str,
        help='Telegram数据目录(如不指定则使用--data-dir)'
    )
    
    return parser


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 创建统一解析器
    unified_parser = UnifiedParser(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # 列出可用数据源
    if args.list:
        sources_info = unified_parser.list_available_sources()
        print("\n可用数据源信息:")
        print("=" * 50)
        print(f"数据目录: {sources_info['data_directory']}")
        print(f"输出目录: {sources_info['output_directory']}")
        print(f"可用数据源: {', '.join(sources_info['available_sources'])}")
        
        if 'qq_files' in sources_info:
            print(f"QQ数据文件: {', '.join(sources_info['qq_files'])}")
        
        if 'telegram_directories' in sources_info:
            print(f"Telegram数据目录: {', '.join(sources_info['telegram_directories'])}")

        if 'wechat_files' in sources_info:
            print(f"WeChat数据文件: {', '.join(sources_info['wechat_files'])}")
        
        return 0
    
    # 构建解析参数
    parse_kwargs = {}
    
    if args.qq_db_path:
        parse_kwargs['qq_db_path'] = args.qq_db_path
    if args.qq_number_ai:
        parse_kwargs['qq_number_ai'] = args.qq_number_ai
    if args.telegram_chat_id:
        parse_kwargs['telegram_chat_id'] = args.telegram_chat_id
    if args.tg_data_dir:
        parse_kwargs['tg_data_dir'] = args.tg_data_dir
    
    # 执行解析
    if args.source:
        # 指定数据源类型
        source_type = None
        if args.source == 'qq':
            source_type = DataSourceType.QQ
        elif args.source in ['tg', 'telegram']:
            source_type = DataSourceType.TELEGRAM
        elif args.source in ['wx', 'wechat']:
            source_type = DataSourceType.WECHAT
        
        return unified_parser.parse_with_source_type(source_type, **parse_kwargs)
    else:
        # 自动检测数据源
        return unified_parser.parse_auto(**parse_kwargs)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
