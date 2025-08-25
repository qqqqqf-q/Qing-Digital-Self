"""
CLI 交互提示和界面模块

提供进度条、用户交互、消息格式化等界面功能。
"""

import sys
import time
import threading
from typing import Optional, Any, Dict, List
from datetime import datetime


class ProgressBar:
    """进度条类"""
    
    def __init__(self, total: int, width: int = 50, desc: str = "进度"):
        self.total = total
        self.width = width
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def update(self, n: int = 1) -> None:
        """更新进度"""
        with self._lock:
            self.current = min(self.current + n, self.total)
            self._display()
    
    def set(self, value: int) -> None:
        """设置进度值"""
        with self._lock:
            self.current = min(max(value, 0), self.total)
            self._display()
    
    def _display(self) -> None:
        """显示进度条"""
        if self.total <= 0:
            return
        
        percent = self.current / self.total
        filled_length = int(self.width * percent)
        
        bar = '█' * filled_length + '▒' * (self.width - filled_length)
        percentage = percent * 100
        
        # 计算预估时间
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f"ETA: {self._format_time(eta)}"
        else:
            eta_str = "ETA: --:--"
        
        # 显示进度条
        sys.stdout.write(f'\r{self.desc}: [{bar}] {percentage:.1f}% ({self.current}/{self.total}) {eta_str}')
        sys.stdout.flush()
        
        if self.current >= self.total:
            print()  # 换行
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m{int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}m"
    
    def finish(self) -> None:
        """完成进度条"""
        self.set(self.total)


class InteractivePrompter:
    """交互式提示器"""
    
    def __init__(self):
        pass
    
    def ask_yes_no(self, question: str, default: Optional[bool] = None) -> bool:
        """询问是/否问题"""
        if default is True:
            prompt = f"{question} [Y/n]: "
        elif default is False:
            prompt = f"{question} [y/N]: "
        else:
            prompt = f"{question} [y/n]: "
        
        while True:
            try:
                response = input(prompt).strip().lower()
                
                if not response and default is not None:
                    return default
                
                if response in ['y', 'yes', '是', '确认']:
                    return True
                elif response in ['n', 'no', '否', '取消']:
                    return False
                else:
                    print("请输入 y/yes 或 n/no")
                    
            except (EOFError, KeyboardInterrupt):
                print()
                return False
    
    def ask_choice(self, question: str, choices: List[str], default: Optional[str] = None) -> str:
        """询问选择问题"""
        print(question)
        for i, choice in enumerate(choices, 1):
            marker = " (默认)" if choice == default else ""
            print(f"  {i}. {choice}{marker}")
        
        while True:
            try:
                if default:
                    prompt = f"请选择 (1-{len(choices)}) [默认: {default}]: "
                else:
                    prompt = f"请选择 (1-{len(choices)}): "
                
                response = input(prompt).strip()
                
                if not response and default:
                    return default
                
                try:
                    index = int(response) - 1
                    if 0 <= index < len(choices):
                        return choices[index]
                    else:
                        print(f"请输入 1 到 {len(choices)} 之间的数字")
                except ValueError:
                    print("请输入有效的数字")
                    
            except (EOFError, KeyboardInterrupt):
                print()
                return default or choices[0]
    
    def ask_string(self, question: str, default: Optional[str] = None, 
                   validator=None) -> str:
        """询问字符串输入"""
        if default:
            prompt = f"{question} [{default}]: "
        else:
            prompt = f"{question}: "
        
        while True:
            try:
                response = input(prompt).strip()
                
                if not response and default:
                    response = default
                
                if not response:
                    print("输入不能为空")
                    continue
                
                if validator:
                    try:
                        response = validator(response)
                        break
                    except Exception as e:
                        print(f"输入无效: {e}")
                        continue
                else:
                    break
                    
            except (EOFError, KeyboardInterrupt):
                print()
                return default or ""
        
        return response
    
    def ask_number(self, question: str, default: Optional[float] = None,
                   min_val: Optional[float] = None, max_val: Optional[float] = None,
                   is_int: bool = False) -> float:
        """询问数字输入"""
        suffix = ""
        if min_val is not None and max_val is not None:
            suffix = f" ({min_val}-{max_val})"
        elif min_val is not None:
            suffix = f" (>={min_val})"
        elif max_val is not None:
            suffix = f" (<={max_val})"
        
        if default is not None:
            prompt = f"{question}{suffix} [{default}]: "
        else:
            prompt = f"{question}{suffix}: "
        
        while True:
            try:
                response = input(prompt).strip()
                
                if not response and default is not None:
                    return default
                
                try:
                    if is_int:
                        value = int(response)
                    else:
                        value = float(response)
                    
                    if min_val is not None and value < min_val:
                        print(f"值必须大于等于 {min_val}")
                        continue
                    
                    if max_val is not None and value > max_val:
                        print(f"值必须小于等于 {max_val}")
                        continue
                    
                    return float(value) if not is_int else int(value)
                    
                except ValueError:
                    print("请输入有效的数字")
                    
            except (EOFError, KeyboardInterrupt):
                print()
                return default or 0


def format_error_message(message: str, details: Optional[str] = None) -> str:
    """格式化错误消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    try:
        # 尝试使用彩色输出
        from colorama import Fore, Style, init
        init(autoreset=True)
        
        formatted = f"{Fore.RED}[{timestamp}] 错误: {message}{Style.RESET_ALL}"
        if details:
            formatted += f"\n{Fore.RED}详细信息: {details}{Style.RESET_ALL}"
        
        return formatted
        
    except ImportError:
        # 降级到纯文本
        formatted = f"[{timestamp}] 错误: {message}"
        if details:
            formatted += f"\n详细信息: {details}"
        
        return formatted


def format_success_message(message: str, details: Optional[str] = None) -> str:
    """格式化成功消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    try:
        # 尝试使用彩色输出
        from colorama import Fore, Style, init
        init(autoreset=True)
        
        formatted = f"{Fore.GREEN}[{timestamp}] 成功: {message}{Style.RESET_ALL}"
        if details:
            formatted += f"\n{Fore.GREEN}详细信息: {details}{Style.RESET_ALL}"
        
        return formatted
        
    except ImportError:
        # 降级到纯文本
        formatted = f"[{timestamp}] 成功: {message}"
        if details:
            formatted += f"\n详细信息: {details}"
        
        return formatted


def format_warning_message(message: str, details: Optional[str] = None) -> str:
    """格式化警告消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    try:
        # 尝试使用彩色输出
        from colorama import Fore, Style, init
        init(autoreset=True)
        
        formatted = f"{Fore.YELLOW}[{timestamp}] 警告: {message}{Style.RESET_ALL}"
        if details:
            formatted += f"\n{Fore.YELLOW}详细信息: {details}{Style.RESET_ALL}"
        
        return formatted
        
    except ImportError:
        # 降级到纯文本
        formatted = f"[{timestamp}] 警告: {message}"
        if details:
            formatted += f"\n详细信息: {details}"
        
        return formatted


def format_info_message(message: str, details: Optional[str] = None) -> str:
    """格式化信息消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    try:
        # 尝试使用彩色输出
        from colorama import Fore, Style, init
        init(autoreset=True)
        
        formatted = f"{Fore.BLUE}[{timestamp}] 信息: {message}{Style.RESET_ALL}"
        if details:
            formatted += f"\n{Fore.BLUE}详细信息: {details}{Style.RESET_ALL}"
        
        return formatted
        
    except ImportError:
        # 降级到纯文本
        formatted = f"[{timestamp}] 信息: {message}"
        if details:
            formatted += f"\n详细信息: {details}"
        
        return formatted


def print_table(data: List[Dict[str, Any]], headers: Optional[List[str]] = None) -> None:
    """打印表格"""
    if not data:
        print("(无数据)")
        return
    
    # 确定列
    if headers is None:
        headers = list(data[0].keys()) if data else []
    
    # 计算列宽
    col_widths = {}
    for header in headers:
        col_widths[header] = len(str(header))
        for row in data:
            value = str(row.get(header, ''))
            col_widths[header] = max(col_widths[header], len(value))
    
    # 打印表头
    header_row = " | ".join(str(header).ljust(col_widths[header]) for header in headers)
    print(header_row)
    print("-" * len(header_row))
    
    # 打印数据行
    for row in data:
        data_row = " | ".join(str(row.get(header, '')).ljust(col_widths[header]) for header in headers)
        print(data_row)


def print_section(title: str, content: str, width: int = 80) -> None:
    """打印节区内容"""
    print()
    print("=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width)
    print()
    print(content)
    print()


def clear_line() -> None:
    """清除当前行"""
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()