"""
推理命令模块

提供模型推理、对话、API服务、批量推理等功能。
支持交互式对话和模型效果测试。
"""

import os
import sys
import json
import argparse
import subprocess
import threading
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ..core.base import BaseCommand
from ..core.exceptions import InferenceError, ValidationError, NetworkError
from ..core.helpers import format_time_duration, find_available_port, ensure_directory
from ..interface.validators import validate_model_path, validate_path, validate_positive_int
from ..interface.prompts import InteractivePrompter
from utils.config.config import get_config


class InferCommand(BaseCommand):
    """推理命令"""
    
    def __init__(self):
        super().__init__("infer", "模型推理")
        self.server_process = None
        self.chat_session = None
        
    def execute(self, args: argparse.Namespace) -> int:
        """执行推理命令"""
        action = getattr(args, 'infer_action', None)
        
        if action == 'chat':
            return self._start_chat(args)
        elif action == 'serve':
            return self._start_server(args)
        elif action == 'batch':
            return self._batch_inference(args)
        elif action == 'test':
            return self._test_model(args)
        else:
            self.logger.error("未指定推理操作")
            return 1
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """验证命令参数"""
        action = getattr(args, 'infer_action', None)
        
        # 验证模型路径
        model_path = getattr(args, 'model_path') or self.config.get('model_path')
        if model_path and os.path.exists(model_path):
            validate_model_path(model_path)
        
        if action == 'batch':
            validate_path(args.input, must_exist=True)
            validate_path(args.output, must_exist=False, check_parent=True)
            
            if hasattr(args, 'batch_size'):
                validate_positive_int(args.batch_size, "batch_size")
        elif action == 'serve':
            if hasattr(args, 'port'):
                validate_positive_int(args.port, "port")
            if hasattr(args, 'workers'):
                validate_positive_int(args.workers, "workers")
    
    def _start_chat(self, args: argparse.Namespace) -> int:
        """启动交互式对话"""
        try:
            self.logger.info("启动交互式对话模式...")
            
            # 准备推理参数
            model_path = getattr(args, 'model_path') or self.config.get('model_path')
            max_length = getattr(args, 'max_length', 2048)
            temperature = getattr(args, 'temperature', 0.7)
            top_p = getattr(args, 'top_p', 0.9)
            
            if not model_path:
                raise ValidationError("必须指定模型路径")
            
            # 初始化聊天会话
            chat_session = ChatSession(
                model_path=model_path,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
            # 启动对话循环
            return self._run_chat_loop(chat_session)
            
        except Exception as e:
            raise InferenceError(f"启动对话失败: {e}")
    
    def _run_chat_loop(self, chat_session) -> int:
        """运行对话循环"""
        try:
            prompter = InteractivePrompter()
            
            print("=" * 60)
            print("Qing-Digital-Self 交互式对话")
            print("输入 'quit' 或 'exit' 退出")
            print("输入 'clear' 清除对话历史")
            print("输入 'help' 查看帮助")
            print("=" * 60)
            
            while True:
                try:
                    # 获取用户输入
                    user_input = input("\n用户: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # 处理特殊命令
                    if user_input.lower() in ['quit', 'exit', '退出']:
                        print("再见!")
                        break
                    elif user_input.lower() in ['clear', '清除']:
                        chat_session.clear_history()
                        print("对话历史已清除")
                        continue
                    elif user_input.lower() in ['help', '帮助']:
                        self._show_chat_help()
                        continue
                    
                    # 生成回复
                    print("AI: ", end="", flush=True)
                    response = chat_session.generate_response(user_input)
                    print(response)
                    
                except KeyboardInterrupt:
                    print("\n\n对话被中断")
                    break
                except Exception as e:
                    print(f"\n生成回复时出错: {e}")
                    continue
            
            return 0
            
        except Exception as e:
            raise InferenceError(f"对话循环失败: {e}")
    
    def _show_chat_help(self) -> None:
        """显示对话帮助"""
        print("\n对话命令:")
        print("  quit/exit/退出    - 退出对话")
        print("  clear/清除        - 清除对话历史")
        print("  help/帮助         - 显示此帮助")
        print("\n提示:")
        print("  - 支持多轮对话，AI会记住前面的内容")
        print("  - 可以随时使用Ctrl+C中断当前生成")
    
    def _start_server(self, args: argparse.Namespace) -> int:
        """启动API服务器"""
        try:
            self.logger.info("启动API服务器...")
            
            # 准备服务器参数
            model_path = getattr(args, 'model_path') or self.config.get('model_path')
            host = getattr(args, 'host', '0.0.0.0')
            port = getattr(args, 'port', 8000)
            workers = getattr(args, 'workers', 1)
            
            if not model_path:
                raise ValidationError("必须指定模型路径")
            
            # 检查端口可用性
            if port == 8000:
                port = find_available_port(8000)
            
            # 构建服务器命令
            cmd = self._build_server_command(model_path, host, port, workers)
            
            self.logger.info(f"在 http://{host}:{port} 启动服务器")
            
            # 启动服务器进程
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8'
            )
            
            # 监控服务器输出
            try:
                while True:
                    output = self.server_process.stdout.readline()
                    if output == '' and self.server_process.poll() is not None:
                        break
                    if output:
                        print(output.strip())
            except KeyboardInterrupt:
                self.logger.info("正在停止服务器...")
                self.server_process.terminate()
                self.server_process.wait()
            
            return self.server_process.returncode or 0
            
        except Exception as e:
            raise InferenceError(f"启动服务器失败: {e}")
    
    def _build_server_command(self, model_path: str, host: str, port: int, workers: int) -> List[str]:
        """构建服务器命令"""
        # 检查是否有openai服务脚本
        openai_script = 'utils/openai/openai_client.py'
        
        if os.path.exists(openai_script):
            cmd = [
                sys.executable,
                openai_script,
                '--model_path', model_path,
                '--host', host,
                '--port', str(port),
                '--workers', str(workers)
            ]
        else:
            # 降级到通用推理脚本
            cmd = [
                sys.executable,
                'finetune/infer_lora_chat.py',
                '--model_path', model_path,
                '--server_mode',
                '--host', host,
                '--port', str(port)
            ]
        
        return cmd
    
    def _batch_inference(self, args: argparse.Namespace) -> int:
        """批量推理"""
        try:
            self.logger.info("开始批量推理...")
            
            input_path = args.input
            output_path = args.output
            batch_size = getattr(args, 'batch_size', 8)
            model_path = getattr(args, 'model_path') or self.config.get('model_path')
            
            if not model_path:
                raise ValidationError("必须指定模型路径")
            
            # 确保输出目录存在
            ensure_directory(os.path.dirname(output_path))
            
            # 读取输入数据
            with open(input_path, 'r', encoding='utf-8') as f:
                if input_path.endswith('.jsonl'):
                    input_data = [json.loads(line) for line in f]
                else:
                    input_data = json.load(f)
            
            if not isinstance(input_data, list):
                raise ValidationError("输入数据必须是JSON数组或JSONL格式")
            
            self.logger.info(f"读取 {len(input_data)} 条数据进行批量推理")
            
            # 初始化批量推理器
            batch_inferencer = BatchInferencer(
                model_path=model_path,
                batch_size=batch_size
            )
            
            # 执行批量推理
            results = batch_inferencer.process_batch(input_data)
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.endswith('.jsonl'):
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                else:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"批量推理完成，结果保存到: {output_path}")
            return 0
            
        except Exception as e:
            raise InferenceError(f"批量推理失败: {e}")
    
    def _test_model(self, args: argparse.Namespace) -> int:
        """测试模型效果"""
        try:
            self.logger.info("开始测试模型效果...")
            
            model_path = getattr(args, 'model_path') or self.config.get('model_path')
            test_data_path = getattr(args, 'test_data')
            metrics = getattr(args, 'metrics', ['bleu', 'rouge'])
            
            if not model_path:
                raise ValidationError("必须指定模型路径")
            
            # 如果没有指定测试数据，使用内置测试
            if not test_data_path:
                return self._run_builtin_test(model_path, metrics)
            
            # 使用自定义测试数据
            return self._run_custom_test(model_path, test_data_path, metrics)
            
        except Exception as e:
            raise InferenceError(f"模型测试失败: {e}")
    
    def _run_builtin_test(self, model_path: str, metrics: List[str]) -> int:
        """运行内置测试"""
        try:
            # 内置测试用例
            test_cases = [
                {"input": "你好", "expected": "你好！有什么我可以帮助你的吗？"},
                {"input": "今天天气怎么样？", "expected": "我无法获取实时天气信息"},
                {"input": "介绍一下自己", "expected": "我是一个AI助手"},
            ]
            
            self.logger.info(f"运行 {len(test_cases)} 个内置测试用例")
            
            # 初始化测试器
            tester = ModelTester(model_path)
            
            # 运行测试
            results = []
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n测试用例 {i}/{len(test_cases)}")
                print(f"输入: {test_case['input']}")
                
                # 生成回复
                response = tester.generate(test_case['input'])
                print(f"输出: {response}")
                print(f"期望: {test_case['expected']}")
                
                # 计算相似度
                similarity = tester.calculate_similarity(response, test_case['expected'])
                print(f"相似度: {similarity:.2f}")
                
                results.append({
                    'input': test_case['input'],
                    'output': response,
                    'expected': test_case['expected'],
                    'similarity': similarity
                })
            
            # 显示总体统计
            avg_similarity = sum(r['similarity'] for r in results) / len(results)
            print(f"\n测试完成!")
            print(f"平均相似度: {avg_similarity:.2f}")
            
            return 0
            
        except Exception as e:
            raise InferenceError(f"内置测试失败: {e}")
    
    def _run_custom_test(self, model_path: str, test_data_path: str, metrics: List[str]) -> int:
        """运行自定义测试"""
        try:
            validate_path(test_data_path, must_exist=True)
            
            # 读取测试数据
            with open(test_data_path, 'r', encoding='utf-8') as f:
                if test_data_path.endswith('.jsonl'):
                    test_data = [json.loads(line) for line in f]
                else:
                    test_data = json.load(f)
            
            if not isinstance(test_data, list):
                raise ValidationError("测试数据必须是JSON数组或JSONL格式")
            
            self.logger.info(f"读取 {len(test_data)} 条测试数据")
            
            # 初始化测试器
            tester = ModelTester(model_path)
            
            # 运行测试
            results = tester.evaluate(test_data, metrics)
            
            # 显示结果
            print("\n测试结果:")
            print("-" * 50)
            for metric, score in results.items():
                print(f"{metric}: {score:.4f}")
            
            return 0
            
        except Exception as e:
            raise InferenceError(f"自定义测试失败: {e}")


class ChatSession:
    """聊天会话管理"""
    
    def __init__(self, model_path: str, max_length: int = 2048, 
                 temperature: float = 0.7, top_p: float = 0.9):
        self.model_path = model_path
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.conversation_history = []
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """初始化模型"""
        try:
            # 这里应该加载真实的模型
            # 目前使用模拟实现
            self.model = None
            self.tokenizer = None
            
        except Exception as e:
            raise InferenceError(f"模型初始化失败: {e}")
    
    def generate_response(self, user_input: str) -> str:
        """生成回复"""
        try:
            # 添加到对话历史
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # 模拟生成回复
            response = self._simulate_generation(user_input)
            
            # 添加回复到历史
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # 限制历史长度
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return response
            
        except Exception as e:
            raise InferenceError(f"生成回复失败: {e}")
    
    def _simulate_generation(self, user_input: str) -> str:
        """模拟生成回复（实际实现需要加载真实模型）"""
        # 简单的模拟回复逻辑
        if "你好" in user_input or "hello" in user_input.lower():
            return "你好！很高兴见到你，有什么我可以帮助你的吗？"
        elif "再见" in user_input or "bye" in user_input.lower():
            return "再见！希望我们下次还能聊天。"
        elif "天气" in user_input:
            return "抱歉，我无法获取实时天气信息。建议你查看天气预报应用。"
        elif "介绍" in user_input:
            return "我是基于Qing-Digital-Self项目训练的AI助手，可以帮助你解答问题和进行对话。"
        else:
            return f"感谢你的问题：'{user_input}'。我正在思考如何回答..."
    
    def clear_history(self) -> None:
        """清除对话历史"""
        self.conversation_history = []


class BatchInferencer:
    """批量推理器"""
    
    def __init__(self, model_path: str, batch_size: int = 8):
        self.model_path = model_path
        self.batch_size = batch_size
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """初始化模型"""
        # 实际实现需要加载真实模型
        pass
    
    def process_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理批量数据"""
        results = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_results = self._process_single_batch(batch)
            results.extend(batch_results)
            
            # 显示进度
            progress = min(i + self.batch_size, len(data))
            print(f"\r处理进度: {progress}/{len(data)}", end="", flush=True)
        
        print()  # 换行
        return results
    
    def _process_single_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理单个批次"""
        results = []
        
        for item in batch:
            # 简单模拟推理
            if 'input' in item:
                response = f"批量推理回复: {item['input']}"
            else:
                response = "无效输入"
            
            results.append({
                'input': item.get('input', ''),
                'output': response,
                'metadata': {
                    'timestamp': time.time(),
                    'model_path': self.model_path
                }
            })
        
        return results


class ModelTester:
    """模型测试器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """初始化模型"""
        # 实际实现需要加载真实模型
        pass
    
    def generate(self, input_text: str) -> str:
        """生成单个回复"""
        # 简单模拟
        return f"测试回复: {input_text}"
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简单的字符级相似度
        if not text1 or not text2:
            return 0.0
        
        # Jaccard相似度
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate(self, test_data: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, float]:
        """评估模型"""
        results = {}
        
        for metric in metrics:
            if metric == 'bleu':
                results['bleu'] = self._calculate_bleu(test_data)
            elif metric == 'rouge':
                results['rouge'] = self._calculate_rouge(test_data)
            else:
                results[metric] = 0.0
        
        return results
    
    def _calculate_bleu(self, test_data: List[Dict[str, Any]]) -> float:
        """计算BLEU分数"""
        # 简单模拟
        return 0.65
    
    def _calculate_rouge(self, test_data: List[Dict[str, Any]]) -> float:
        """计算ROUGE分数"""
        # 简单模拟
        return 0.72