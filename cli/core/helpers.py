"""
CLI 辅助函数

提供各种实用工具函数，用于格式化输出、文件操作、系统信息获取等。
"""

import os
import sys
import time
import platform
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta

from .exceptions import ValidationError, FileOperationError


def format_time_duration(seconds: float) -> str:
    """
    格式化时间持续时间为人类可读格式
    
    Args:
        seconds: 时间秒数
        
    Returns:
        格式化的时间字符串
        
    Examples:
        >>> format_time_duration(3661.5)
        '1小时1分1秒'
        >>> format_time_duration(125.3)
        '2分5秒'
    """
    if seconds < 0:
        return "0秒"
    
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}小时")
    if minutes > 0:
        parts.append(f"{minutes}分")
    if secs > 0 or not parts:
        parts.append(f"{secs}秒")
    
    return "".join(parts)


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小为人类可读格式
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        格式化的文件大小字符串
        
    Examples:
        >>> format_file_size(1024)
        '1.0 KB'
        >>> format_file_size(1536)
        '1.5 KB'
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def safe_path_join(*args: Union[str, Path]) -> Path:
    """
    安全的路径拼接，防止路径遍历攻击
    
    Args:
        *args: 路径组件
        
    Returns:
        安全的路径对象
        
    Raises:
        ValidationError: 如果路径包含危险字符
    """
    path_parts = []
    
    for part in args:
        part_str = str(part)
        
        # 检查危险字符
        if '..' in part_str or part_str.startswith('/') or ':' in part_str:
            raise ValidationError(f"路径包含危险字符: {part_str}")
        
        path_parts.append(part_str)
    
    return Path(*path_parts)


def confirm_action(message: str, default: bool = False) -> bool:
    """
    获取用户确认
    
    Args:
        message: 确认消息
        default: 默认选择
        
    Returns:
        用户选择结果
    """
    default_hint = "[Y/n]" if default else "[y/N]"
    
    try:
        # 检查标准输入是否可用和打开状态
        if not sys.stdin.isatty() or sys.stdin.closed:
            print(f"{message} {default_hint}: 使用默认值 ({'是' if default else '否'})")
            return default
        
        # 尝试读取输入前先检查stdin状态
        if hasattr(sys.stdin, 'readable') and not sys.stdin.readable():
            print(f"{message} {default_hint}: 标准输入不可读，使用默认值 ({'是' if default else '否'})")
            return default
        
        response = input(f"{message} {default_hint}: ").strip().lower()
        
        if not response:
            return default
        
        return response in ['y', 'yes', '是', '确认']
    
    except (EOFError, KeyboardInterrupt):
        print(f"\n用户取消输入，使用默认值: {'是' if default else '否'}")
        return False
    except Exception as e:
        # 处理所有其他异常，包括 "I/O operation on closed file"
        print(f"输入错误 ({type(e).__name__}: {e})")
        print(f"使用默认值: {'是' if default else '否'}")
        return default


def get_system_info() -> Dict[str, Any]:
    """
    获取系统信息
    
    Returns:
        系统信息字典
    """
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
            },
            "memory": {
                "total": format_file_size(memory.total),
                "available": format_file_size(memory.available),
                "used": format_file_size(memory.used),
                "percent": memory.percent,
            },
            "disk": {
                "total": format_file_size(disk.total),
                "free": format_file_size(disk.free),
                "used": format_file_size(disk.used),
                "percent": (disk.used / disk.total) * 100,
            },
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
        }
        
        # 添加GPU信息（如果可用）
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                system_info["gpu"] = []
                for gpu in gpus:
                    system_info["gpu"].append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total": f"{gpu.memoryTotal} MB",
                        "memory_used": f"{gpu.memoryUsed} MB",
                        "memory_free": f"{gpu.memoryFree} MB",
                        "temperature": f"{gpu.temperature}°C",
                        "load": f"{gpu.load * 100:.1f}%",
                    })
        except ImportError:
            pass
        
        return system_info
        
    except Exception as e:
        return {"error": f"获取系统信息失败: {e}"}


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        目录路径对象
        
    Raises:
        FileOperationError: 如果无法创建目录
    """
    path_obj = Path(path)
    
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    except Exception as e:
        raise FileOperationError(f"无法创建目录", str(path_obj), "mkdir") from e


def check_file_permissions(path: Union[str, Path], operation: str = "read") -> bool:
    """
    检查文件权限
    
    Args:
        path: 文件路径
        operation: 操作类型 ("read", "write", "execute")
        
    Returns:
        是否有权限
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        return False
    
    if operation == "read":
        return os.access(path_obj, os.R_OK)
    elif operation == "write":
        return os.access(path_obj, os.W_OK)
    elif operation == "execute":
        return os.access(path_obj, os.X_OK)
    else:
        return False


def get_file_stats(path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取文件或目录的统计信息。
    如果路径是目录，则递归计算其中所有文件的总大小。
    
    Args:
        path: 文件或目录路径
        
    Returns:
        统计信息字典
        
    Raises:
        FileOperationError: 如果路径不存在或无法访问
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileOperationError(f"路径不存在", str(path_obj), "stat")
    
    try:
        # 获取基本信息
        stat_info = path_obj.stat()
        is_dir = path_obj.is_dir()
        
        total_size = 0
        if is_dir:
            # 如果是目录，递归计算总大小
            for dirpath, dirnames, filenames in os.walk(path_obj):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    # 检查是否是符号链接，避免重复计算和错误
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
        else:
            # 如果是文件，直接获取大小
            total_size = stat_info.st_size

        return {
            "path": str(path_obj.absolute()),
            "size": format_file_size(total_size),
            "size_bytes": total_size,
            "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            "permissions": oct(stat_info.st_mode)[-3:],
            "is_file": path_obj.is_file(),
            "is_directory": is_dir,
            "is_symlink": path_obj.is_symlink(),
        }
    
    except Exception as e:
        raise FileOperationError(f"无法获取路径信息: {e}", str(path_obj), "stat") from e


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    截断字符串到指定长度
    
    Args:
        text: 原始字符串
        max_length: 最大长度
        suffix: 后缀字符串
        
    Returns:
        截断后的字符串
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def parse_time_range(time_range_str: str) -> tuple[datetime, datetime]:
    """
    解析时间范围字符串
    
    Args:
        time_range_str: 时间范围字符串，格式: "2023-01-01,2024-01-01"
        
    Returns:
        开始时间和结束时间的元组
        
    Raises:
        ValidationError: 如果时间格式不正确
    """
    try:
        start_str, end_str = time_range_str.split(',')
        
        start_time = datetime.strptime(start_str.strip(), '%Y-%m-%d')
        end_time = datetime.strptime(end_str.strip(), '%Y-%m-%d')
        
        if start_time >= end_time:
            raise ValidationError("开始时间必须早于结束时间")
        
        return start_time, end_time
    
    except ValueError as e:
        raise ValidationError(f"时间格式不正确，应为 YYYY-MM-DD,YYYY-MM-DD: {e}") from e


def format_progress_bar(current: int, total: int, width: int = 50, fill: str = '█', empty: str = ' ') -> str:
    """
    生成进度条字符串
    
    Args:
        current: 当前进度
        total: 总数
        width: 进度条宽度
        fill: 填充字符
        empty: 空白字符
        
    Returns:
        进度条字符串
    """
    if total <= 0:
        return f"[{empty * width}] 0%"
    
    percent = min(current / total, 1.0)
    filled_length = int(width * percent)
    
    bar = fill * filled_length + empty * (width - filled_length)
    percentage = percent * 100
    
    return f"[{bar}] {percentage:.1f}%"


def validate_json_file(file_path: Union[str, Path]) -> bool:
    """
    验证JSON文件格式
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        是否为有效的JSON文件
    """
    import json
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, FileNotFoundError, IOError):
        return False


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """
    寻找可用端口
    
    Args:
        start_port: 起始端口
        max_attempts: 最大尝试次数
        
    Returns:
        可用端口号
        
    Raises:
        ValidationError: 如果找不到可用端口
    """
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    raise ValidationError(f"无法找到可用端口 (尝试范围: {start_port}-{start_port + max_attempts})")