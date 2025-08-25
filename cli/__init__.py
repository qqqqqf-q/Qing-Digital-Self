"""
Qing-Digital-Self CLI 包

企业级命令行界面，用于数字分身项目的完整生命周期管理。
"""

__version__ = "0.1.0"
__author__ = "Qing-Agent"
__email__ = ""

from .core.base import QingCLI
from .core.exceptions import (
    CLIError, 
    ConfigurationError, 
    DataProcessingError,
    TrainingError,
    InferenceError
)

__all__ = [
    'QingCLI',
    'CLIError',
    'ConfigurationError', 
    'DataProcessingError',
    'TrainingError',
    'InferenceError'
]