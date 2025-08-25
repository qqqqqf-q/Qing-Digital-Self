import logging
import os
import sys
from datetime import datetime
from typing import Optional
from utils.config.config import get_config
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and COLORAMA_AVAILABLE and self._supports_color()
        
        # 颜色映射
        if self.use_colors:
            self.colors = {
                'DEBUG': Fore.CYAN,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.RED + Style.BRIGHT,
            }
            self.reset = Style.RESET_ALL
        else:
            self.colors = {}
            self.reset = ''
    
    def _supports_color(self) -> bool:
        """检测终端是否支持颜色输出"""
        # Windows cmd和PowerShell支持
        if sys.platform == "win32":
            return True
        # Unix系统检查TERM环境变量
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    
    def format(self, record):
        if self.use_colors:
            # 获取颜色
            color = self.colors.get(record.levelname, '')
            
            # 格式化时间戳
            formatted_time = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
            
            # 构建彩色格式
            log_message = (
                f"{Fore.BLUE}[{formatted_time}]{self.reset} "
                f"{Fore.MAGENTA}{record.name}{self.reset} "
                f"{color}[{record.levelname}]{self.reset} "
                f"{record.getMessage()}"
            )
            
            return log_message
        else:
            # 无颜色的格式
            return super().format(record)

class Logger:
    def __init__(self, name: str = 'QingAgent'):
        # 获取配置
        self.config = get_config()
        self.log_level = self.config.get('log_level', 'INFO')
        self.language = self.config.get('language', 'zhcn')  # 默认中文
        self.enable_colors = self.config.get('enable_colors', True)  # 默认启用颜色
        self.log_dir = 'logs'
        
        # 创建日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_log_level())
        
        # 防止向父级logger传播，避免重复输出
        self.logger.propagate = False
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 移除已有的处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 添加控制台处理器
        self._add_console_handler()
        
        # 添加文件处理器
        self._add_file_handler()

    def _get_log_level(self):
        """将字符串日志级别转换为logging模块对应的级别"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return level_map.get(self.log_level.upper(), logging.INFO)

    def _add_console_handler(self):
        """添加控制台处理器"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._get_log_level())
        
        # 设置彩色格式化器
        if self.enable_colors:
            formatter = ColoredFormatter(use_colors=True)
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)

    def _add_file_handler(self):
        """添加文件处理器"""
        # 日志文件名格式：年-月-日.log
        log_file = os.path.join(self.log_dir, f'{datetime.now().strftime("%Y-%m-%d")}.log')
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件日志始终记录所有级别的日志
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)

    def _get_message(self, message: Optional[str] = None, zhcn: Optional[str] = None, en: Optional[str] = None) -> str:
        """根据配置的语言返回相应的消息"""
        if message is not None:
            return message
        
        if self.language == 'en' and en is not None:
            return en
        elif self.language == 'zhcn' and zhcn is not None:
            return zhcn
        
        # 如果没有对应语言的消息，返回可用的消息
        if zhcn is not None:
            return zhcn
        elif en is not None:
            return en
        else:
            return "No message provided"

    def debug(self, message: Optional[str] = None, zhcn: Optional[str] = None, en: Optional[str] = None):
        """记录DEBUG级别的日志
        
        Args:
            message: 直接传入的消息（优先级最高）
            zhcn: 中文消息
            en: 英文消息
        """
        msg = self._get_message(message, zhcn, en)
        self.logger.debug(msg)

    def info(self, message: Optional[str] = None, zhcn: Optional[str] = None, en: Optional[str] = None):
        """记录INFO级别的日志
        
        Args:
            message: 直接传入的消息（优先级最高）
            zhcn: 中文消息
            en: 英文消息
        """
        msg = self._get_message(message, zhcn, en)
        self.logger.info(msg)

    def warning(self, message: Optional[str] = None, zhcn: Optional[str] = None, en: Optional[str] = None):
        """记录WARNING级别的日志
        
        Args:
            message: 直接传入的消息（优先级最高）
            zhcn: 中文消息
            en: 英文消息
        """
        msg = self._get_message(message, zhcn, en)
        self.logger.warning(msg)

    def error(self, message: Optional[str] = None, zhcn: Optional[str] = None, en: Optional[str] = None):
        """记录ERROR级别的日志
        
        Args:
            message: 直接传入的消息（优先级最高）
            zhcn: 中文消息
            en: 英文消息
        """
        msg = self._get_message(message, zhcn, en)
        self.logger.error(msg)

    def critical(self, message: Optional[str] = None, zhcn: Optional[str] = None, en: Optional[str] = None):
        """记录CRITICAL级别的日志
        
        Args:
            message: 直接传入的消息（优先级最高）
            zhcn: 中文消息
            en: 英文消息
        """
        msg = self._get_message(message, zhcn, en)
        self.logger.critical(msg)

    def set_level(self, level: str):
        """动态设置日志级别"""
        self.log_level = level
        self.logger.setLevel(self._get_log_level())
        
        # 更新控制台处理器的级别
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(self._get_log_level())
    
    def set_language(self, language: str):
        """动态设置日志语言
        
        Args:
            language: 语言代码，支持 'zhcn' 或 'en'
        """
        if language in ['zhcn', 'en']:
            self.language = language
            self.config.set('language', language)
        else:
            raise ValueError(f"Unsupported language: {language}. Supported languages: 'zhcn', 'en'")
    
    def set_colors(self, enable: bool):
        """动态设置彩色输出
        
        Args:
            enable: 是否启用彩色输出
        """
        self.enable_colors = enable
        self.config.set('enable_colors', enable)
        
        # 重新配置控制台处理器
        console_handlers = [h for h in self.logger.handlers if isinstance(h, logging.StreamHandler) 
                           and not isinstance(h, logging.FileHandler)]
        
        for handler in console_handlers:
            self.logger.removeHandler(handler)
        
        self._add_console_handler()

# 创建全局日志实例
logger = Logger()

def get_logger(name: str = 'QingAgent') -> Logger:
    """获取日志实例"""
    return Logger(name)