import os
from typing import Dict, Any, Optional


class Config:
    def __init__(self):
        # 初始化默认配置
        self._config: Dict[str, Any] = {
            "db_path": os.getenv("db_path", "./c2c.db"),
            "qq_number_ai": os.getenv(
                "qq_number_ai",
            ),
            # 其他配置
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "language": os.getenv("LANGUAGE", "zhcn"),  # 日志语言配置，支持zhcn和en
            "use_unsloth": os.getenv("USE_UNSLOTH", "false").lower() == "true",
            "system_prompt": os.getenv("system_prompt", ""),
            # LM Studio配置
            "OpenAI_URL": os.getenv("OpenAI_URL", "http://localhost:1234"),
            "OpenAI_Model": os.getenv("OpenAI_MODEL", "/qwen3-8b-fp8"),
            "OpenAI_timeout": int(os.getenv("OpenAI_TIMEOUT", "20")),
            "OpenAI_max_retries": int(os.getenv("OpenAI_MAX_RETRIES", "3")),
            "OpenAI_retry_delay": float(os.getenv("OpenAI_RETRY_DELAY", "1.0")),
            "use_llm_clean": os.getenv("USE_LLM_CLEAN", "true").lower() == "true",
            # 并发配置
            "max_workers": int(os.getenv("MAX_WORKERS", "6")),  # 并发处理线程数
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
