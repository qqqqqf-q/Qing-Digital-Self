"""
参数验证器

提供各种参数验证功能，确保用户输入的正确性和安全性。
"""

import os
import re
from pathlib import Path
from typing import Union, Optional, List, Any
from urllib.parse import urlparse

from ..core.exceptions import ValidationError


def validate_path(path: Union[str, Path], must_exist: bool = True, check_parent: bool = False) -> Path:
    """
    验证路径参数
    
    Args:
        path: 路径字符串或Path对象
        must_exist: 路径是否必须存在
        check_parent: 是否检查父目录存在
        
    Returns:
        验证后的Path对象
        
    Raises:
        ValidationError: 路径验证失败
    """
    try:
        path_obj = Path(path)
        
        # 检查路径是否为空
        if not str(path_obj).strip():
            raise ValidationError("路径不能为空")
        
        # 检查路径是否包含危险字符
        if '..' in str(path_obj) or str(path_obj).startswith('/'):
            raise ValidationError(f"路径包含危险字符: {path_obj}")
        
        # 检查路径是否存在
        if must_exist and not path_obj.exists():
            raise ValidationError(f"路径不存在: {path_obj}")
        
        # 检查父目录是否存在
        if check_parent:
            parent_dir = path_obj.parent
            if not parent_dir.exists():
                raise ValidationError(f"父目录不存在: {parent_dir}")
        
        return path_obj
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"路径验证失败: {e}")


def validate_positive_int(value: Union[int, str], param_name: str) -> int:
    """
    验证正整数参数
    
    Args:
        value: 要验证的值
        param_name: 参数名称
        
    Returns:
        验证后的整数值
        
    Raises:
        ValidationError: 参数验证失败
    """
    try:
        int_value = int(value)
        
        if int_value <= 0:
            raise ValidationError(f"参数 {param_name} 必须为正整数: {value}")
        
        return int_value
        
    except ValueError:
        raise ValidationError(f"参数 {param_name} 必须为整数: {value}")


def validate_float_range(value: Union[float, str], param_name: str, 
                        min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """
    验证浮点数范围
    
    Args:
        value: 要验证的值
        param_name: 参数名称
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        验证后的浮点数值
        
    Raises:
        ValidationError: 参数验证失败
    """
    try:
        float_value = float(value)
        
        if min_val is not None and float_value < min_val:
            raise ValidationError(f"参数 {param_name} 不能小于 {min_val}: {value}")
        
        if max_val is not None and float_value > max_val:
            raise ValidationError(f"参数 {param_name} 不能大于 {max_val}: {value}")
        
        return float_value
        
    except ValueError:
        raise ValidationError(f"参数 {param_name} 必须为数字: {value}")


def validate_choice(value: str, param_name: str, choices: List[str], case_sensitive: bool = True) -> str:
    """
    验证选择参数
    
    Args:
        value: 要验证的值
        param_name: 参数名称
        choices: 可选值列表
        case_sensitive: 是否区分大小写
        
    Returns:
        验证后的值
        
    Raises:
        ValidationError: 参数验证失败
    """
    if not case_sensitive:
        value = value.lower()
        choices = [choice.lower() for choice in choices]
    
    if value not in choices:
        raise ValidationError(f"参数 {param_name} 无效值: {value}, 可选值: {choices}")
    
    return value


def validate_url(url: str, param_name: str = "url") -> str:
    """
    验证URL格式
    
    Args:
        url: 要验证的URL
        param_name: 参数名称
        
    Returns:
        验证后的URL
        
    Raises:
        ValidationError: URL验证失败
    """
    try:
        parsed = urlparse(url)
        
        if not parsed.scheme:
            raise ValidationError(f"URL缺少协议: {url}")
        
        if not parsed.netloc:
            raise ValidationError(f"URL缺少主机名: {url}")
        
        if parsed.scheme not in ['http', 'https']:
            raise ValidationError(f"URL协议不支持: {parsed.scheme}")
        
        return url
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"URL格式错误: {e}")


def validate_model_path(path: Union[str, Path], param_name: str = "model_path") -> Path:
    """
    验证模型路径
    
    Args:
        path: 模型路径
        param_name: 参数名称
        
    Returns:
        验证后的Path对象
        
    Raises:
        ValidationError: 模型路径验证失败
    """
    path_obj = validate_path(path, must_exist=True)
    
    # 检查是否为目录
    if not path_obj.is_dir():
        raise ValidationError(f"模型路径必须是目录: {path_obj}")
    
    # 检查是否包含模型文件
    model_files = [
        'config.json',
        'pytorch_model.bin',
        'model.safetensors',
        'tokenizer.json',
        'tokenizer_config.json'
    ]
    
    found_files = []
    for model_file in model_files:
        if (path_obj / model_file).exists():
            found_files.append(model_file)
    
    if not found_files:
        raise ValidationError(f"模型目录不包含必要文件: {path_obj}")
    
    return path_obj


def validate_data_path(path: Union[str, Path], param_name: str = "data_path") -> Path:
    """
    验证数据文件路径
    
    Args:
        path: 数据文件路径
        param_name: 参数名称
        
    Returns:
        验证后的Path对象
        
    Raises:
        ValidationError: 数据路径验证失败
    """
    path_obj = validate_path(path, must_exist=True)
    
    # 检查是否为文件
    if not path_obj.is_file():
        raise ValidationError(f"数据路径必须是文件: {path_obj}")
    
    # 检查文件扩展名
    valid_extensions = ['.json', '.jsonl', '.csv', '.txt']
    if path_obj.suffix.lower() not in valid_extensions:
        raise ValidationError(f"数据文件格式不支持: {path_obj.suffix}, 支持格式: {valid_extensions}")
    
    return path_obj


def validate_qq_number(qq_number: Union[str, int], param_name: str = "qq_number") -> str:
    """
    验证QQ号码格式
    
    Args:
        qq_number: QQ号码
        param_name: 参数名称
        
    Returns:
        验证后的QQ号码字符串
        
    Raises:
        ValidationError: QQ号码验证失败
    """
    qq_str = str(qq_number).strip()
    
    # 检查是否为数字
    if not qq_str.isdigit():
        raise ValidationError(f"QQ号码必须为数字: {qq_number}")
    
    # 检查长度（QQ号码通常5-11位）
    if len(qq_str) < 5 or len(qq_str) > 11:
        raise ValidationError(f"QQ号码长度无效: {qq_number}")
    
    return qq_str


from typing import Tuple

def validate_time_range(time_range: str, param_name: str = "time_range") -> Tuple[str, str]:
    """
    验证时间范围格式
    
    Args:
        time_range: 时间范围字符串，格式: "2023-01-01,2024-01-01"
        param_name: 参数名称
        
    Returns:
        开始时间和结束时间的元组
        
    Raises:
        ValidationError: 时间范围验证失败
    """
    try:
        if ',' not in time_range:
            raise ValidationError(f"时间范围格式错误，应包含逗号分隔: {time_range}")
        
        start_str, end_str = time_range.split(',', 1)
        start_str = start_str.strip()
        end_str = end_str.strip()
        
        # 验证日期格式 (YYYY-MM-DD)
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        
        if not re.match(date_pattern, start_str):
            raise ValidationError(f"开始时间格式错误，应为YYYY-MM-DD: {start_str}")
        
        if not re.match(date_pattern, end_str):
            raise ValidationError(f"结束时间格式错误，应为YYYY-MM-DD: {end_str}")
        
        # 验证日期有效性
        from datetime import datetime
        try:
            start_date = datetime.strptime(start_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_str, '%Y-%m-%d')
        except ValueError as e:
            raise ValidationError(f"日期格式无效: {e}")
        
        # 验证时间顺序
        if start_date >= end_date:
            raise ValidationError("开始时间必须早于结束时间")
        
        return start_str, end_str
        
    except ValueError as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"时间范围解析失败: {e}")


def validate_json_file(path: Union[str, Path], param_name: str = "json_file") -> Path:
    """
    验证JSON文件格式
    
    Args:
        path: JSON文件路径
        param_name: 参数名称
        
    Returns:
        验证后的Path对象
        
    Raises:
        ValidationError: JSON文件验证失败
    """
    import json
    
    path_obj = validate_path(path, must_exist=True)
    
    # 检查文件扩展名
    if path_obj.suffix.lower() not in ['.json', '.jsonc']:
        raise ValidationError(f"文件不是JSON格式: {path_obj}")
    
    # 验证JSON格式
    try:
        with open(path_obj, 'r', encoding='utf-8') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"JSON格式错误: {e}")
    except Exception as e:
        raise ValidationError(f"JSON文件读取失败: {e}")
    
    return path_obj
