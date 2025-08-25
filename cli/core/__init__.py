"""
CLI 核心模块

包含CLI框架的基础组件和核心功能。
"""

from .base import QingCLI, BaseCommand
from .exceptions import (
    CLIError,
    ConfigurationError,
    DataProcessingError,
    TrainingError,
    InferenceError
)
from .helpers import (
    format_time_duration,
    format_file_size,
    safe_path_join,
    confirm_action,
    get_system_info
)

__all__ = [
    'QingCLI',
    'BaseCommand',
    'CLIError',
    'ConfigurationError',
    'DataProcessingError',
    'TrainingError',
    'InferenceError',
    'format_time_duration',
    'format_file_size',
    'safe_path_join',
    'confirm_action',
    'get_system_info'
]