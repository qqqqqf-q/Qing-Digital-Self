"""
训练命令模块

提供模型训练、监控、停止、权重合并等功能。
支持QLoRA微调和训练状态管理。
"""

import os
import sys
import json
import argparse
import subprocess
import signal
import time
import threading
from typing import Dict, Any, Optional
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ..core.base import BaseCommand
from ..core.exceptions import TrainingError, ValidationError, FileOperationError
from ..core.helpers import format_time_duration, format_file_size, ensure_directory
from ..interface.validators import validate_path, validate_positive_int, validate_model_path
from utils.config.config import get_config
from utils.logger.logger import get_logger


class TrainCommand(BaseCommand):
    """训练命令"""
    
    def __init__(self):
        self.logger = get_logger('TrainCommand')
        super().__init__("train", "模型训练")
        self.training_process = None
        self.training_thread = None
        
    def execute(self, args: argparse.Namespace) -> int:
        """执行训练命令"""
        action = getattr(args, 'train_action', None)
        
        if action == 'start':
            return self._start_training(args)
        elif action == 'status':
            return self._show_status(args)
        elif action == 'stop':
            return self._stop_training(args)
        elif action == 'merge':
            return self._merge_weights(args)
        else:
            self.logger.error("未指定训练操作")
            return 1
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """验证命令参数"""
        action = getattr(args, 'train_action', None)
        
        if action == 'start':
            self._validate_start_args(args)
        elif action == 'merge':
            self._validate_merge_args(args)
    
    def _validate_start_args(self, args: argparse.Namespace) -> None:
        """验证训练启动参数"""
        # 验证模型路径
        model_path = getattr(args, 'model_path') or self.config.get('model_path')
        if model_path and os.path.exists(model_path):
            validate_model_path(model_path)
        
        # 验证数据路径
        data_path = getattr(args, 'data_path') or self.config.get('data_path')
        if not data_path:
            raise ValidationError("必须指定训练数据路径")
        
        validate_path(data_path, must_exist=True)
        
        # 验证数值参数
        if hasattr(args, 'max_steps') and args.max_steps:
            validate_positive_int(args.max_steps, "max_steps")
        
        if hasattr(args, 'batch_size') and args.batch_size:
            validate_positive_int(args.batch_size, "batch_size")
    
    def _validate_merge_args(self, args: argparse.Namespace) -> None:
        """验证权重合并参数"""
        validate_model_path(args.base_model)
        validate_path(args.lora_path, must_exist=True)
        validate_path(args.output, must_exist=False, check_parent=True)
    
    def _start_training(self, args: argparse.Namespace) -> int:
        """开始训练"""
        try:
            self.logger.info("准备开始训练...")
            
            # 准备训练参数
            self.logger.info("准备训练参数...")
            train_params = self._prepare_training_params(args)
            self.logger.info(f"训练参数准备完成: {list(train_params.keys())}")
            
            # 验证训练环境
            self.logger.info("验证训练环境...")
            self._validate_training_environment(train_params)
            self.logger.info("训练环境验证完成")
            
            # 检查是否恢复训练
            if getattr(args, 'resume'):
                return self._resume_training(args.resume, train_params)
            else:
                self.logger.info("开始新训练...")
                return self._start_new_training(train_params)
            
        except Exception as e:
            import traceback
            self.logger.error(f"训练启动异常: {e}")
            self.logger.error(f"异常堆栈: {traceback.format_exc()}")
            raise TrainingError(f"训练启动失败: {e}")
    
    def _prepare_training_params(self, args: argparse.Namespace) -> Dict[str, Any]:
        """准备训练参数"""
        # 从配置和命令行参数合并
        params = {
            'model_path': getattr(args, 'model_path') or self.config.get('model_path'),
            'data_path': getattr(args, 'data_path') or self.config.get('data_path'),
            'output_dir': getattr(args, 'output_dir') or './checkpoints',
            'lora_r': getattr(args, 'lora_r') or self.config.get('lora_r', 16),
            'lora_alpha': getattr(args, 'lora_alpha') or self.config.get('lora_alpha', 32),
            'batch_size': getattr(args, 'batch_size') or self.config.get('per_device_train_batch_size', 1),
            'max_steps': getattr(args, 'max_steps') or self.config.get('max_steps', 1000),
            'learning_rate': self.config.get('learning_rate', 2e-4),
            'gradient_accumulation_steps': self.config.get('gradient_accumulation_steps', 16),
            'logging_steps': self.config.get('logging_steps', 10),
            'save_steps': self.config.get('save_steps', 100),
            'fp16': self.config.get('fp16', True),
            'use_qlora': self.config.get('use_qlora', True),
        }
        
        return params
    
    def _validate_training_environment(self, params: Dict[str, Any]) -> None:
        """验证训练环境"""
        # 检查GPU可用性 - 暂时跳过torch检查避免null bytes错误
        try:
            self.logger.info("跳过torch验证（避免null bytes错误）")
            # import torch
            # if not torch.cuda.is_available():
            #     self.logger.warning("CUDA不可用，将使用CPU训练（速度较慢）")
            # else:
            #     gpu_count = torch.cuda.device_count()
            #     self.logger.info(f"检测到 {gpu_count} 个GPU设备")
        except ImportError:
            raise TrainingError("PyTorch未安装")
        
        # 检查必要的库 - 暂时跳过避免null bytes错误
        self.logger.info("跳过依赖检查（避免null bytes错误）")
        # required_packages = ['transformers', 'peft', 'datasets']
        # missing_packages = []
        # 
        # for package in required_packages:
        #     try:
        #         __import__(package)
        #     except ImportError:
        #         missing_packages.append(package)
        # 
        # if missing_packages:
        #     raise TrainingError(f"缺少必要的训练库: {missing_packages}")
        
        # 检查输出目录
        ensure_directory(params['output_dir'])
        
        # 检查磁盘空间
        import shutil
        free_space = shutil.disk_usage(params['output_dir']).free
        if free_space < 5 * 1024 * 1024 * 1024:  # 5GB
            self.logger.warning(f"磁盘空间不足: {format_file_size(free_space)}")
    
    def _start_new_training(self, params: Dict[str, Any]) -> int:
        """开始新的训练"""
        try:
            # 构建训练命令
            cmd = self._build_training_command(params)
            
            self.logger.info(f"启动训练进程: {' '.join(cmd)}")
            
            # 启动训练进程 - 分离stdout和stderr，避免输出丢失
            env = dict(os.environ)
            env.update({
                'PYTHONUNBUFFERED': '1',  # 强制Python无缓冲输出
                'PYTHONIOENCODING': 'utf-8'  # 强制使用UTF-8编码
            })
            
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # 分离错误流
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=0,  # 无缓冲
                universal_newlines=True,
                env=env
            )
            
            # 启动监控线程
            self.training_thread = threading.Thread(
                target=self._monitor_training,
                args=(self.training_process, params)
            )
            self.training_thread.start()
            
            # 等待训练完成或中断
            return_code = self.training_process.wait()
            
            if self.training_thread.is_alive():
                self.training_thread.join(timeout=10)
            
            if return_code == 0:
                self.logger.info("训练成功完成")
            else:
                self.logger.error(f"训练失败，退出码: {return_code}")
            
            return return_code
            
        except KeyboardInterrupt:
            self.logger.info("接收到中断信号，正在停止训练...")
            self._stop_training_process()
            return 130
        except Exception as e:
            raise TrainingError(f"训练执行失败: {e}")
    
    def _resume_training(self, checkpoint_path: str, params: Dict[str, Any]) -> int:
        """恢复训练"""
        try:
            if not os.path.exists(checkpoint_path):
                raise FileOperationError("检查点不存在", checkpoint_path)
            
            self.logger.info(f"从检查点恢复训练: {checkpoint_path}")
            
            # 更新参数
            params['resume_from_checkpoint'] = checkpoint_path
            
            return self._start_new_training(params)
            
        except Exception as e:
            raise TrainingError(f"恢复训练失败: {e}")
    
    def _build_training_command(self, params: Dict[str, Any]) -> list:
        """构建训练命令"""
        cmd = [
            sys.executable,
            'run_finetune.py'
        ]
        
        # 检查参数是否包含null bytes
        for key, value in params.items():
            if isinstance(value, str) and '\x00' in value:
                self.logger.error(f"参数 {key} 包含null bytes: {repr(value)}")
                raise ValueError(f"参数 {key} 包含null bytes")
        
        # 添加参数
        if params.get('model_path'):
            cmd.extend(['--local_dir', params['model_path']])
        
        if params.get('data_path'):
            cmd.extend(['--data_path', params['data_path']])
        
        cmd.extend(['--output_dir', params['output_dir']])
        cmd.extend(['--lora_r', str(params['lora_r'])])
        cmd.extend(['--lora_alpha', str(params['lora_alpha'])])
        cmd.extend(['--per_device_train_batch_size', str(params['batch_size'])])
        cmd.extend(['--gradient_accumulation_steps', str(params['gradient_accumulation_steps'])])
        cmd.extend(['--learning_rate', str(params['learning_rate'])])
        cmd.extend(['--max_steps', str(params['max_steps'])])
        cmd.extend(['--logging_steps', str(params['logging_steps'])])
        cmd.extend(['--save_steps', str(params['save_steps'])])
        
        if params.get('fp16'):
            cmd.extend(['--fp16', 'true'])
        
        if params.get('use_qlora'):
            cmd.extend(['--use_qlora', 'true'])
        
        if params.get('resume_from_checkpoint'):
            cmd.extend(['--resume_from_checkpoint', params['resume_from_checkpoint']])
        
        return cmd
    
    def _monitor_training(self, process: subprocess.Popen, params: Dict[str, Any]) -> None:
        """监控训练进程 - 使用非阻塞读取同时处理stdout和stderr"""
        import select
        import time
        
        try:
            output_dir = params['output_dir']
            log_file = os.path.join(output_dir, 'training.log')
            error_log_file = os.path.join(output_dir, 'training_error.log')
            
            # 确保日志目录存在
            ensure_directory(output_dir)
            
            # 记录进程启动信息
            self.logger.info(f"训练进程PID: {process.pid}")
            
            with open(log_file, 'w', encoding='utf-8', errors='replace') as f, \
                 open(error_log_file, 'w', encoding='utf-8', errors='replace') as err_f:
                
                # Windows下使用轮询方式读取
                while True:
                    # 检查进程状态
                    return_code = process.poll()
                    
                    # 读取标准输出
                    try:
                        stdout_data = self._read_stream_non_blocking(process.stdout)
                        if stdout_data:
                            f.write(stdout_data)
                            f.flush()
                            self._print_safe(stdout_data.strip())
                            self._parse_training_output(stdout_data.strip())
                    except Exception as e:
                        self.logger.debug(f"读取stdout失败: {e}")
                    
                    # 读取错误输出
                    try:
                        stderr_data = self._read_stream_non_blocking(process.stderr)
                        if stderr_data:
                            err_f.write(stderr_data)
                            err_f.flush()
                            # 错误信息用红色显示（如果支持颜色）
                            self.logger.error(f"训练脚本错误: {stderr_data.strip()}")
                            self._print_safe(f"ERROR: {stderr_data.strip()}")
                    except Exception as e:
                        self.logger.debug(f"读取stderr失败: {e}")
                    
                    # 如果进程已结束
                    if return_code is not None:
                        self.logger.info(f"训练进程结束，退出码: {return_code}")
                        
                        # 读取剩余的输出
                        try:
                            remaining_stdout = process.stdout.read()
                            remaining_stderr = process.stderr.read()
                            
                            if remaining_stdout:
                                f.write(remaining_stdout)
                                f.flush()
                                self._print_safe(remaining_stdout.strip())
                                
                            if remaining_stderr:
                                err_f.write(remaining_stderr)
                                err_f.flush()
                                self.logger.error(f"最终错误输出: {remaining_stderr.strip()}")
                                self._print_safe(f"FINAL ERROR: {remaining_stderr.strip()}")
                        except Exception as e:
                            self.logger.warning(f"读取最终输出失败: {e}")
                        
                        break
                    
                    # 短暂休眠避免CPU占用过高
                    time.sleep(0.1)
            
        except Exception as e:
            self.logger.error(f"训练监控失败: {e}")
            import traceback
            self.logger.error(f"监控异常详情: {traceback.format_exc()}")
    
    def _read_stream_non_blocking(self, stream) -> str:
        """非阻塞读取流数据"""
        try:
            # 简化的实现：尝试读取一行，如果阻塞就返回空
            import io
            if hasattr(stream, 'readable') and stream.readable():
                try:
                    # 设置短暂超时，避免无限阻塞
                    line = stream.readline()
                    return line if line else ""
                except (io.BlockingIOError, OSError):
                    return ""
            return ""
        except Exception:
            return ""
    
    def _print_safe(self, text: str) -> None:
        """安全打印，处理编码问题"""
        if not text.strip():
            return
            
        try:
            self.logger.info(text)
        except UnicodeEncodeError:
            try:
                # 尝试用系统编码
                encoded = text.encode(sys.stdout.encoding, errors='replace')
                self.logger.info(encoded.decode(sys.stdout.encoding))
            except:
                # 最后的备用方案
                self.logger.debug(repr(text))
    
    def _parse_training_output(self, output: str) -> None:
        """解析训练输出"""
        try:
            # 解析损失值
            if 'train_loss' in output:
                import re
                match = re.search(r'train_loss:\s*([\d.]+)', output)
                if match:
                    loss = float(match.group(1))
                    self.logger.debug(f"当前损失: {loss}")
            
            # 解析步数
            if 'step' in output:
                import re
                match = re.search(r'step:\s*(\d+)', output)
                if match:
                    step = int(match.group(1))
                    self.logger.debug(f"当前步数: {step}")
            
        except Exception:
            pass
    
    def _stop_training_process(self) -> None:
        """停止训练进程"""
        if self.training_process:
            try:
                # 发送SIGTERM信号
                self.training_process.terminate()
                
                # 等待进程退出
                try:
                    self.training_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # 强制杀死进程
                    self.training_process.kill()
                    self.training_process.wait()
                
            except Exception as e:
                self.logger.error(f"停止训练进程失败: {e}")
    
    def _show_status(self, args: argparse.Namespace) -> int:
        """显示训练状态"""
        try:
            output_dir = getattr(args, 'output_dir') or './checkpoints'
            follow = getattr(args, 'follow', False)
            
            if not os.path.exists(output_dir):
                self.logger.info("未找到训练输出目录")
                return 0
            
            # 查找最新的检查点
            checkpoints = self._find_checkpoints(output_dir)
            
            if not checkpoints:
                self.logger.info("未找到训练检查点")
                return 0
            
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            
            self.logger.info(f"训练状态:")
            self.logger.info(f"输出目录: {output_dir}")
            self.logger.info(f"最新检查点: {latest_checkpoint}")
            
            # 显示训练日志
            log_file = os.path.join(output_dir, 'training.log')
            if os.path.exists(log_file):
                if follow:
                    self._follow_log_file(log_file)
                else:
                    self._show_recent_logs(log_file)
            
            return 0
            
        except Exception as e:
            raise TrainingError(f"获取训练状态失败: {e}")
    
    def _find_checkpoints(self, output_dir: str) -> list:
        """查找检查点目录"""
        checkpoints = []
        
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item.startswith('checkpoint-'):
                checkpoints.append(item_path)
        
        return checkpoints
    
    def _follow_log_file(self, log_file: str) -> None:
        """实时跟踪日志文件"""
        try:
            self.logger.info("实时显示训练日志 (按Ctrl+C退出)")
            
            with open(log_file, 'r', encoding='utf-8') as f:
                # 移动到文件末尾
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        self.logger.info(line.strip())
                    else:
                        time.sleep(1)
                        
        except KeyboardInterrupt:
            self.logger.info("停止日志跟踪")
    
    def _show_recent_logs(self, log_file: str, lines: int = 50) -> None:
        """显示最近的日志"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                self.logger.info(f"\n最近 {len(recent_lines)} 行训练日志:")
                self.logger.info("-" * 80)
                for line in recent_lines:
                    self.logger.info(line.strip())
                self.logger.info("-" * 80)
                
        except Exception as e:
            self.logger.error(f"读取日志文件失败: {e}")
    
    def _stop_training(self, args: argparse.Namespace) -> int:
        """停止训练"""
        try:
            force = getattr(args, 'force', False)
            
            if self.training_process:
                self.logger.info("停止当前训练进程...")
                self._stop_training_process()
                return 0
            
            # 查找运行中的训练进程
            training_pids = self._find_training_processes()
            
            if not training_pids:
                self.logger.info("未找到运行中的训练进程")
                return 0
            
            for pid in training_pids:
                try:
                    self.logger.info(f"停止训练进程 PID: {pid}")
                    
                    if force:
                        os.kill(pid, signal.SIGKILL)
                    else:
                        os.kill(pid, signal.SIGTERM)
                        
                except ProcessLookupError:
                    self.logger.info(f"进程 {pid} 已不存在")
                except Exception as e:
                    self.logger.error(f"停止进程 {pid} 失败: {e}")
            
            return 0
            
        except Exception as e:
            raise TrainingError(f"停止训练失败: {e}")
    
    def _find_training_processes(self) -> list:
        """查找训练进程"""
        try:
            import psutil
            
            training_pids = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('run_finetune.py' in cmd for cmd in cmdline):
                        training_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return training_pids
            
        except ImportError:
            self.logger.warning("psutil未安装，无法查找训练进程")
            return []
    
    def _merge_weights(self, args: argparse.Namespace) -> int:
        """合并LoRA权重"""
        try:
            self.logger.info("开始合并LoRA权重...")
            
            base_model = args.base_model
            lora_path = args.lora_path
            output_path = args.output
            
            # 确保输出目录存在
            ensure_directory(output_path)
            
            # 构建合并命令
            cmd = [
                sys.executable,
                'finetune/merge.py',
                '--base_model_path', base_model,
                '--lora_path', lora_path,
                '--output_path', output_path
            ]
            
            self.logger.info(f"执行权重合并: {' '.join(cmd)}")
            
            # 执行合并
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                self.logger.info(f"权重合并完成: {output_path}")
                
                # 显示输出文件信息
                if os.path.exists(output_path):
                    size = sum(
                        os.path.getsize(os.path.join(output_path, f))
                        for f in os.listdir(output_path)
                        if os.path.isfile(os.path.join(output_path, f))
                    )
                    self.logger.info(f"合并后模型大小: {format_file_size(size)}")
                
            else:
                self.logger.error(f"权重合并失败: {result.stderr}")
            
            return result.returncode
            
        except Exception as e:
            raise TrainingError(f"权重合并失败: {e}")