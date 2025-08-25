import os
import json
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
        
        # 初始化配置
        self._config: Dict[str, Any] = {}
        self._jsonc_config: Dict[str, Any] = {}
        
        # 加载配置文件
        self._load_jsonc_config()
        
        # 加载.env文件（作为补充）
        self._load_env_file()
        
        # 加载默认配置
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

    def _load_jsonc_config(self):
        """加载JSONC配置文件"""
        config_path = "seeting.jsonc"
        if not os.path.exists(config_path):
            self.logger.warning(f"配置文件不存在: {config_path}")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 改进的JSONC支持 - 移除注释
            lines = content.split('\n')
            cleaned_lines = []
            in_string = False
            
            for line in lines:
                cleaned_line = ""
                i = 0
                while i < len(line):
                    char = line[i]
                    
                    # 处理字符串状态
                    if char == '"' and (i == 0 or line[i-1] != '\\'):
                        in_string = not in_string
                    
                    # 如果不在字符串内且遇到//，截断行
                    if not in_string and i < len(line) - 1 and line[i:i+2] == '//':
                        break
                    
                    cleaned_line += char
                    i += 1
                
                # 去除尾随空格并跳过空行
                cleaned_line = cleaned_line.rstrip()
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
            
            cleaned_content = '\n'.join(cleaned_lines)
            self._jsonc_config = json.loads(cleaned_content)
            self.logger.info(f"成功加载配置文件: {config_path}")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"配置文件JSON解析错误: {e}")
            # 输出更多调试信息
            self.logger.error(f"清理后的内容:\n{cleaned_content}")
            raise ConfigError(f"配置文件格式错误: {e}")
        except Exception as e:
            self.logger.error(f"加载配置文件时发生错误: {e}")
            raise ConfigError(f"配置文件加载失败: {e}")

    def _get_nested_value(self, path: str, default: Any = None) -> Any:
        """从嵌套字典中获取值，使用点分隔路径"""
        keys = path.split('.')
        value = self._jsonc_config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def _load_default_config(self):
        """加载默认配置"""
        # 从JSONC配置文件中加载，如果不存在则使用环境变量或默认值
        self._config = {
            # 基础信息
            "repoid": self._get_nested_value("repoid", "Qing-Agent"),
            "branch": self._get_nested_value("branch", "main"),
            "version": self._get_nested_value("version", "0.1.0"),
            
            # 模型配置
            "model_path": self._get_nested_value("model_args.model_path", os.getenv("MODEL_PATH", "./model/Qwen3-8B-Base")),
            "model_repo": self._get_nested_value("model_args.model_repo", os.getenv("MODEL_REPO", "Qwen/Qwen-3-8B-Base")),
            "model_output_path": self._get_nested_value("model_args.model_output_path", os.getenv("MODEL_OUTPUT_PATH", "./model_output")),
            "template": self._get_nested_value("model_args.template", os.getenv("TEMPLATE", "qwen")),
            "finetuning_type": self._get_nested_value("model_args.finetuning_type", os.getenv("FINETUNING_TYPE", "qlora")),
            "trust_remote_code": self._get_nested_value("model_args.trust_remote_code", self._safe_bool(os.getenv("TRUST_REMOTE_CODE", "true"), True)),
            "download_source": self._get_nested_value("model_args.download_source", os.getenv("DOWNLOAD_SOURCE", "modelscope")),
            
            # 日志配置
            "log_level": self._get_nested_value("logger_args.log_level", os.getenv("LOG_LEVEL", "INFO")).upper(),
            "language": self._get_nested_value("logger_args.language", os.getenv("LANGUAGE", "zhcn")),
            
            # QQ数据配置（重命名db_path为qq_db_path）
            "qq_db_path": self._get_nested_value("data_args.qq_agrs.qq_db_path", os.getenv("DB_PATH", "./data/qq.db")),
            "qq_number_ai": self._get_nested_value("data_args.qq_agrs.qq_number_ai", os.getenv("QQ_NUMBER_AI")),
            
            # Telegram配置
            "telegram_chat_id": self._get_nested_value("data_args.telegram_args.telegram_chat_id", os.getenv("TELEGRAM_CHAT_ID")),
            
            # 数据处理配置
            "include_type": self._get_nested_value("data_args.include_type", ["text"]),
            "blocked_words": self._get_nested_value("data_args.blocked_words", ["密码", "账号", "手机号"]),
            "single_combine_time_window": self._get_nested_value("data_args.single_combine_time_window", 2),
            "qa_match_time_window": self._get_nested_value("data_args.qa_match_time_window", 5),
            "combine_msg_max_length": self._get_nested_value("data_args.combine_msg_max_length", 2048),
            "messages_max_length": self._get_nested_value("data_args.messages_max_length", 2048),
            
            # 清理配置
            "clean_method": self._get_nested_value("data_args.clean_set_args.clean_method", os.getenv("CLEAN_METHOD", "llm")),
            "use_llm_clean": self._get_nested_value("data_args.clean_set_args.clean_method", os.getenv("CLEAN_METHOD", "llm")) == "llm",
            
            # OpenAI API配置
            "OpenAI_URL": self._get_nested_value("data_args.clean_set_args.openai_api.api_base", os.getenv("OpenAI_URL", "https://api.openai.com/v1")),
            "OpenAI_api_key": self._get_nested_value("data_args.clean_set_args.openai_api.api_key", os.getenv("OpenAI_API_KEY", "sk-1234567890abcdef1234567890abcdef")),
            "OpenAI_Model": self._get_nested_value("data_args.clean_set_args.openai_api.model_name", os.getenv("OpenAI_MODEL", "xxx")),
            "clean_batch_size": self._get_nested_value("data_args.clean_set_args.openai_api.clean_batch_size", self._safe_int(os.getenv("CLEAN_BATCH_SIZE", "10"), 10)),
            "clean_workers": self._get_nested_value("data_args.clean_set_args.openai_api.clean_workers", self._safe_int(os.getenv("CLEAN_WORKERS", "4"), 4)),
            "OpenAI_timeout": self._get_nested_value("data_args.clean_set_args.openai_api.timeout", self._safe_int(os.getenv("OpenAI_TIMEOUT", "120"), 120, 1, 300)),
            "OpenAI_max_retries": self._safe_int(os.getenv("OpenAI_MAX_RETRIES", "3"), 3, 0, 10),
            "OpenAI_retry_delay": self._safe_float(os.getenv("OpenAI_RETRY_DELAY", "1.0"), 1.0, 0.1, 10.0),
            "OpenAI_temperature": self._safe_float(os.getenv("OpenAI_temperature", "0.1"), 0.1, 0.0, 2.0),
            "OpenAI_max_tokens": self._safe_int(os.getenv("OpenAI_max_tokens", "10000"), 10000, 1, 100000),
            
            # 训练配置
            "use_qlora": self._get_nested_value("data_args.train_sft_args.use_qlora", self._safe_bool(os.getenv("USE_QLORA", "true"), True)),
            "data_path": self._get_nested_value("data_args.train_sft_args.data_path", os.getenv("DATA_PATH", "./dataset/sft.jsonl")),
            "lora_r": self._get_nested_value("data_args.train_sft_args.lora_r", self._safe_int(os.getenv("LORA_R", "16"), 16)),
            "lora_alpha": self._get_nested_value("data_args.train_sft_args.lora_alpha", self._safe_int(os.getenv("LORA_ALPHA", "32"), 32)),
            "lora_dropout": self._get_nested_value("data_args.train_sft_args.lora_dropout", self._safe_float(os.getenv("LORA_DROPOUT", "0.1"), 0.1)),
            "lora_target_modules": self._get_nested_value("data_args.train_sft_args.lora_target_modules", ["q_proj", "v_proj"]),
            "per_device_train_batch_size": self._get_nested_value("data_args.train_sft_args.tper_device_train_batch_size", self._safe_int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1"), 1)),
            "per_device_eval_batch_size": self._get_nested_value("data_args.train_sft_args.per_device_eval_batch_size", self._safe_int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE", "1"), 1)),
            "gradient_accumulation_steps": self._get_nested_value("data_args.train_sft_args.gradient_accumulation_steps", self._safe_int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "16"), 16)),
            "max_steps": self._get_nested_value("data_args.train_sft_args.max_steps", self._safe_int(os.getenv("MAX_STEPS", "-1"), -1)),
            "learning_rate": self._get_nested_value("data_args.train_sft_args.learning_rate", self._safe_float(os.getenv("LEARNING_RATE", "2e-4"), 2e-4)),
            "fp16": self._get_nested_value("data_args.train_sft_args.fp16", self._safe_bool(os.getenv("FP16", "true"), True)),
            "logging_steps": self._get_nested_value("data_args.train_sft_args.logging_steps", self._safe_int(os.getenv("LOGGING_STEPS", "10"), 10)),
            "save_steps": self._get_nested_value("data_args.train_sft_args.save_steps", self._safe_int(os.getenv("SAVE_STEPS", "100"), 100)),
            "load_precision": self._get_nested_value("data_args.train_sft_args.load_precision", os.getenv("LOAD_PRECISION", "int8")),
            
            # 兼容性配置
            "max_workers": self._safe_int(os.getenv("MAX_WORKERS", "6"), 6, 1, 32),
            "system_prompt": os.getenv("system_prompt", ""),
            "use_unsloth": self._safe_bool(os.getenv("USE_UNSLOTH", "false"), False),
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
        qq_db_dir = os.path.dirname(os.path.abspath(self._config['qq_db_path']))
        if not os.path.exists(qq_db_dir):
            try:
                os.makedirs(qq_db_dir, exist_ok=True)
                self.logger.info(f"创建数据库目录: {qq_db_dir}")
            except Exception as e:
                errors.append(f"无法创建数据库目录 {qq_db_dir}: {e}")
        
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
                "qq_db_path": bool(self._config.get('qq_db_path')),
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
