import logging
import os
from datetime import datetime
from typing import Optional
from config.config import get_config

class Logger:
    def __init__(self, name: str = 'QingAgent'):
        # 获取配置
        self.config = get_config()
        self.log_level = self.config.get('log_level', 'INFO')
        self.language = self.config.get('language', 'zhcn')  # 默认中文
        self.log_dir = 'logs'
        
        # 创建日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_log_level())
        
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
        
        # 设置日志格式
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

# 创建全局日志实例
logger = Logger()

def get_logger(name: str = 'QingAgent') -> Logger:
    """获取日志实例"""
    return Logger(name)