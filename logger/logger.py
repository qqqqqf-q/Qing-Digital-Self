import logging
import os
from datetime import datetime
from config.config import get_config

class Logger:
    def __init__(self, name: str = 'QingAgent'):
        # 获取配置
        self.config = get_config()
        self.log_level = self.config.get('log_level', 'INFO')
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

    def debug(self, message: str):
        """记录DEBUG级别的日志"""
        self.logger.debug(message)

    def info(self, message: str):
        """记录INFO级别的日志"""
        self.logger.info(message)

    def warning(self, message: str):
        """记录WARNING级别的日志"""
        self.logger.warning(message)

    def error(self, message: str):
        """记录ERROR级别的日志"""
        self.logger.error(message)

    def critical(self, message: str):
        """记录CRITICAL级别的日志"""
        self.logger.critical(message)

    def set_level(self, level: str):
        """动态设置日志级别"""
        self.log_level = level
        self.logger.setLevel(self._get_log_level())
        
        # 更新控制台处理器的级别
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(self._get_log_level())

# 创建全局日志实例
logger = Logger()

def get_logger(name: str = 'QingAgent') -> Logger:
    """获取日志实例"""
    return Logger(name)