import os
import logging
from typing import Dict, Any, Optional, Union
from urllib.parse import urlparse


class ConfigError(Exception):
    """配置相关异常"""
    pass


class Config:
    def __init__(self):
        # 设置基础日志
        self._setup_basic_logging()
        self.logger = logging.getLogger('Config')
        
        # 加载.env文件
        self._load_env_file()
        
        # 初始化配置
        self._config: Dict[str, Any] = {}
        self._load_default_config()
        
        # 验证关键配置
        self._validate_config()
        
        self.logger.info("配置系统初始化完成")

    def _setup_basic_logging(self):
        """设置基础日志配置"""
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    def _load_env_file(self):
        """安全加载.env文件"""
        try:
            from dotenv import load_dotenv
            
            env_path = '.env'
            if os.path.exists(env_path):
                result = load_dotenv(env_path)
                if result:
                    self.logger.info(f"成功加载.env文件: {env_path}")
                else:
                    self.logger.warning(f".env文件存在但加载失败: {env_path}")
            else:
                self.logger.info(".env文件不存在，使用系统环境变量")
                
        except ImportError:
            self.logger.warning("python-dotenv未安装，跳过.env文件加载")
        except Exception as e:
            self.logger.error(f"加载.env文件时发生错误: {e}")
            raise ConfigError(f"配置加载失败: {e}")

    def _safe_int(self, value: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
        """安全的整数转换"""
        try:
            result = int(value)
            if min_val is not None and result < min_val:
                self.logger.warning(f"值 {result} 小于最小值 {min_val}，使用默认值 {default}")
                return default
            if max_val is not None and result > max_val:
                self.logger.warning(f"值 {result} 大于最大值 {max_val}，使用默认值 {default}")
                return default
            return result
        except (ValueError, TypeError) as e:
            self.logger.warning(f"无法转换为整数: {value}，使用默认值 {default}，错误: {e}")
            return default

    def _safe_float(self, value: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """安全的浮点数转换"""
        try:
            result = float(value)
            if min_val is not None and result < min_val:
                self.logger.warning(f"值 {result} 小于最小值 {min_val}，使用默认值 {default}")
                return default
            if max_val is not None and result > max_val:
                self.logger.warning(f"值 {result} 大于最大值 {max_val}，使用默认值 {default}")
                return default
            return result
        except (ValueError, TypeError) as e:
            self.logger.warning(f"无法转换为浮点数: {value}，使用默认值 {default}，错误: {e}")
            return default

    def _safe_bool(self, value: str, default: bool) -> bool:
        """安全的布尔值转换"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return default

    def _load_default_config(self):
        """加载默认配置"""
        self._config = {
            # 数据库配置
            "db_path": os.getenv("db_path", "./c2c.db"),
            "qq_number_ai": os.getenv("qq_number_ai"),
            
            # 日志配置
            "log_level": os.getenv("LOG_LEVEL", "INFO").upper(),
            "language": os.getenv("LANGUAGE", "zhcn"),
            
            # 模型配置
            "use_unsloth": self._safe_bool(os.getenv("USE_UNSLOTH", "false"), False),
            "system_prompt": os.getenv("system_prompt", ""),
            
            # OpenAI/LM Studio配置
            "OpenAI_URL": os.getenv("OpenAI_URL", "http://localhost:1234"),
            "OpenAI_Model": os.getenv("OpenAI_MODEL", "/qwen3-8b-fp8"),
            "OpenAI_timeout": self._safe_int(os.getenv("OpenAI_TIMEOUT", "20"), 20, 1, 300),
            "OpenAI_max_retries": self._safe_int(os.getenv("OpenAI_MAX_RETRIES", "3"), 3, 0, 10),
            "OpenAI_retry_delay": self._safe_float(os.getenv("OpenAI_RETRY_DELAY", "1.0"), 1.0, 0.1, 10.0),
            "OpenAI_temperature": self._safe_float(os.getenv("OpenAI_temperature", "0.1"), 0.1, 0.0, 2.0),
            "OpenAI_max_tokens": self._safe_int(os.getenv("OpenAI_max_tokens", "10000"), 10000, 1, 100000),
            
            # 数据处理配置
            "use_llm_clean": self._safe_bool(os.getenv("USE_LLM_CLEAN", "true"), True),
            
            # 并发配置
            "max_workers": self._safe_int(os.getenv("MAX_WORKERS", "6"), 6, 1, 32),
        }

    def _validate_config(self):
        """验证配置项"""
        errors = []
        
        # 验证日志级别
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self._config['log_level'] not in valid_log_levels:
            errors.append(f"无效的日志级别: {self._config['log_level']}，有效值: {valid_log_levels}")
        
        # 验证语言设置
        valid_languages = ['zhcn', 'en']
        if self._config['language'] not in valid_languages:
            errors.append(f"无效的语言设置: {self._config['language']}，有效值: {valid_languages}")
        
        # 验证URL格式
        try:
            parsed_url = urlparse(self._config['OpenAI_URL'])
            if not parsed_url.scheme or not parsed_url.netloc:
                errors.append(f"无效的OpenAI URL格式: {self._config['OpenAI_URL']}")
        except Exception as e:
            errors.append(f"OpenAI URL解析错误: {e}")
        
        # 验证数据库路径的目录是否存在
        db_dir = os.path.dirname(os.path.abspath(self._config['db_path']))
        if not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
                self.logger.info(f"创建数据库目录: {db_dir}")
            except Exception as e:
                errors.append(f"无法创建数据库目录 {db_dir}: {e}")
        
        # 如果有错误，记录并抛出异常
        if errors:
            error_msg = "配置验证失败:\n" + "\n".join(f"- {error}" for error in errors)
            self.logger.error(error_msg)
            raise ConfigError(error_msg)
        
        self.logger.info("配置验证通过")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置配置项"""
        old_value = self._config.get(key)
        self._config[key] = value
        self.logger.debug(f"配置项 {key} 从 {old_value} 更新为 {value}")

    def update(self, config_dict: Dict[str, Any]) -> None:
        """批量更新配置"""
        for key, value in config_dict.items():
            self.set(key, value)
        self.logger.info(f"批量更新了 {len(config_dict)} 个配置项")

    def all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()

    def get_status(self) -> Dict[str, Any]:
        """获取配置加载状态"""
        return {
            "config_loaded": True,
            "env_file_exists": os.path.exists('.env'),
            "total_configs": len(self._config),
            "critical_configs_set": {
                "db_path": bool(self._config.get('db_path')),
                "OpenAI_URL": bool(self._config.get('OpenAI_URL')),
                "system_prompt": bool(self._config.get('system_prompt')),
            }
        }


# 全局配置实例 - 只创建一次
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """获取配置实例（单例模式）"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


# 为了向后兼容，保留config变量
config = get_config()
