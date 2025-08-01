import os
from typing import Dict, Any, Optional

class Config:
    def __init__(self):
        # 初始化默认配置
        self._config: Dict[str, Any] = {
            'db_path':os.getenv('db_path','./c2c.db'),
            'qq_number_ai':os.getenv('qq_number_ai'),
            # 其他配置
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'use_unsloth': os.getenv('USE_UNSLOTH', 'false').lower() == 'true',

            'system_prompt':os.getenv('system_prompt',''),
        }

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置配置项"""
        self._config[key] = value

    def update(self, config_dict: Dict[str, Any]) -> None:
        """批量更新配置"""
        self._config.update(config_dict)

    def all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()

# 创建全局配置实例
config = Config()

# 加载.env文件（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
    # 重新初始化配置以应用.env文件中的环境变量
    config = Config()
except ImportError:
    # 如果没有安装python-dotenv，忽略
    pass


def get_config() -> Config:
    """获取配置实例"""
    return config