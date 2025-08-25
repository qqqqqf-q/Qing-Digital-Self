import os
import csv
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.config.config import get_config
from utils.logger.logger import get_logger


@dataclass
class Message:
    """消息数据结构"""
    id: str
    msg_svr_id: str
    type_name: str
    is_sender: int
    talker: str
    msg: str
    src: str
    create_time: datetime
    room_name: str
    is_forward: int


@dataclass
class ChatMLMessage:
    """ChatML消息数据结构"""
    role: str
    content: str


class ChatMLGenerator:
    """ChatML格式生成器"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('ChatMLGenerator')
        
        # 从配置文件读取参数
        self.system_prompt = self.config.get('system_prompt', '')
        self.blocked_words = self.config.get('blocked_words', [])
        self.include_type = self.config.get('include_type', ['text'])
        self.single_combine_time_window = self.config.get('single_combine_time_window', 2)
        self.qa_match_time_window = self.config.get('qa_match_time_window', 5)
        self.combine_msg_max_length = self.config.get('combine_msg_max_length', 2048)
        self.messages_max_length = self.config.get('messages_max_length', 2048)
        
        # AI用户标识
        self.qq_number_ai = self.config.get('qq_number_ai')
        
        self.logger.info("ChatML生成器初始化完成")
        self.logger.info(f"合并时间窗口: {self.single_combine_time_window}分钟")
        self.logger.info(f"QA匹配时间窗口: {self.qa_match_time_window}分钟")
        self.logger.info(f"消息最大长度: {self.combine_msg_max_length}")
        self.logger.info(f"对话最大长度: {self.messages_max_length}")

    def _parse_time(self, time_str: str) -> datetime:
        """解析时间字符串"""
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            self.logger.error(f"时间解析错误: {time_str}, 错误: {e}")
            return datetime.now()

    def _is_blocked_message(self, message: str) -> bool:
        """检查消息是否包含禁用词"""
        if not message:
            return True
            
        for blocked_word in self.blocked_words:
            if blocked_word.lower() in message.lower():
                return True
        return False

    def _clean_message(self, message: str) -> str:
        """清理消息内容"""
        if not message:
            return ""
            
        # 移除特殊字符和格式化
        message = message.strip()
        # 移除多余的空白字符
        message = re.sub(r'\s+', ' ', message)
        return message

    def load_csv_files(self, dataset_path: str) -> List[Message]:
        """加载所有CSV文件"""
        messages = []
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            self.logger.error(f"数据集目录不存在: {dataset_path}")
            return messages
            
        csv_files = list(dataset_dir.glob("**/*.csv"))
        self.logger.info(f"找到 {len(csv_files)} 个CSV文件")
        
        for csv_file in csv_files:
            try:
                self.logger.debug(f"正在处理文件: {csv_file}")
                file_messages = self._load_single_csv(csv_file)
                messages.extend(file_messages)
                self.logger.debug(f"从文件 {csv_file.name} 加载了 {len(file_messages)} 条消息")
            except Exception as e:
                self.logger.error(f"加载文件 {csv_file} 时发生错误: {e}")
                continue
                
        self.logger.info(f"总共加载了 {len(messages)} 条消息")
        return messages

    def _load_single_csv(self, csv_file: Path) -> List[Message]:
        """加载单个CSV文件"""
        messages = []
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 检查消息类型是否在允许范围内
                    if row['type_name'] not in self.include_type:
                        continue
                        
                    # 检查是否为转发消息
                    if int(row.get('is_forward', 0)) == 1:
                        continue
                    
                    # 清理和验证消息内容
                    cleaned_msg = self._clean_message(row['msg'])
                    if not cleaned_msg or self._is_blocked_message(cleaned_msg):
                        continue
                    
                    message = Message(
                        id=row['id'],
                        msg_svr_id=row['MsgSvrID'],
                        type_name=row['type_name'],
                        is_sender=int(row['is_sender']),
                        talker=row['talker'],
                        msg=cleaned_msg,
                        src=row['src'],
                        create_time=self._parse_time(row['CreateTime']),
                        room_name=row['room_name'],
                        is_forward=int(row['is_forward'])
                    )
                    messages.append(message)
                    
        except Exception as e:
            self.logger.error(f"读取CSV文件 {csv_file} 时发生错误: {e}")
            
        return messages

    def combine_messages_by_time_window(self, messages: List[Message]) -> List[Message]:
        """按时间窗口合并消息"""
        if not messages:
            return messages
            
        # 按房间和时间排序
        messages.sort(key=lambda x: (x.room_name, x.create_time))
        
        combined_messages = []
        current_group = [messages[0]]
        
        for i in range(1, len(messages)):
            current_msg = messages[i]
            last_msg = current_group[-1]
            
            # 检查是否应该合并
            time_diff = (current_msg.create_time - last_msg.create_time).total_seconds() / 60
            same_room = current_msg.room_name == last_msg.room_name
            same_sender = current_msg.is_sender == last_msg.is_sender  # 使用is_sender而非talker
            
            if same_room and same_sender and time_diff <= self.single_combine_time_window:
                # 检查合并后长度是否超限
                combined_content = last_msg.msg + " " + current_msg.msg
                if len(combined_content) <= self.combine_msg_max_length:
                    current_group.append(current_msg)
                    continue
            
            # 合并当前组并开始新组
            if len(current_group) > 1:
                combined_msg = self._merge_message_group(current_group)
                combined_messages.append(combined_msg)
            else:
                combined_messages.append(current_group[0])
                
            current_group = [current_msg]
        
        # 处理最后一组
        if current_group:
            if len(current_group) > 1:
                combined_msg = self._merge_message_group(current_group)
                combined_messages.append(combined_msg)
            else:
                combined_messages.append(current_group[0])
        
        self.logger.info(f"消息合并完成: {len(messages)} -> {len(combined_messages)}")
        return combined_messages

    def _merge_message_group(self, group: List[Message]) -> Message:
        """合并消息组"""
        if not group:
            return None
            
        first_msg = group[0]
        combined_content = " ".join([msg.msg for msg in group])
        
        # 创建合并后的消息
        merged_msg = Message(
            id=first_msg.id,
            msg_svr_id=first_msg.msg_svr_id,
            type_name=first_msg.type_name,
            is_sender=first_msg.is_sender,
            talker=first_msg.talker,
            msg=combined_content[:self.combine_msg_max_length],
            src=first_msg.src,
            create_time=first_msg.create_time,
            room_name=first_msg.room_name,
            is_forward=first_msg.is_forward
        )
        
        return merged_msg

    def generate_qa_pairs(self, messages: List[Message]) -> List[List[ChatMLMessage]]:
        """生成QA对话对"""
        if not messages or not self.qq_number_ai:
            self.logger.warning("无法生成QA对：消息为空或未设置AI用户ID")
            return []
            
        # 按房间分组
        room_messages = {}
        for msg in messages:
            if msg.room_name not in room_messages:
                room_messages[msg.room_name] = []
            room_messages[msg.room_name].append(msg)
        
        all_qa_pairs = []
        
        for room_name, room_msgs in room_messages.items():
            self.logger.debug(f"处理房间: {room_name}, 消息数量: {len(room_msgs)}")
            room_msgs.sort(key=lambda x: x.create_time)
            qa_pairs = self._generate_room_qa_pairs(room_msgs)
            all_qa_pairs.extend(qa_pairs)
        
        self.logger.info(f"生成了 {len(all_qa_pairs)} 个QA对话对")
        return all_qa_pairs

    def _generate_room_qa_pairs(self, messages: List[Message]) -> List[List[ChatMLMessage]]:
        """为单个房间生成QA对"""
        qa_pairs = []
        
        for i, msg in enumerate(messages):
            # 只处理用户发送的消息（is_sender=0，非AI发送）
            if msg.is_sender == 1:  # AI发送的消息，跳过
                continue
                
            # 寻找AI的回复
            ai_reply = self._find_ai_reply(messages, i)
            if ai_reply:
                # 构建上下文
                context_messages = self._build_context(messages, i)
                
                # 创建ChatML格式的对话
                chatml_messages = []
                
                # 添加系统提示（如果有）
                if self.system_prompt and self.system_prompt != "*":
                    chatml_messages.append(ChatMLMessage("system", self.system_prompt))
                
                # 添加上下文消息
                chatml_messages.extend(context_messages)
                
                # 添加当前用户消息和AI回复
                chatml_messages.append(ChatMLMessage("user", msg.msg))
                chatml_messages.append(ChatMLMessage("assistant", ai_reply.msg))
                
                # 验证对话质量
                if self._validate_dialogue_quality(chatml_messages):
                    # 检查总长度
                    total_length = sum(len(m.content) for m in chatml_messages)
                    if total_length <= self.messages_max_length:
                        qa_pairs.append(chatml_messages)
                    else:
                        # 尝试缩减上下文
                        reduced_context = self._reduce_context(chatml_messages)
                        if reduced_context and self._validate_dialogue_quality(reduced_context):
                            qa_pairs.append(reduced_context)
        
        return qa_pairs

    def _find_ai_reply(self, messages: List[Message], user_msg_index: int) -> Optional[Message]:
        """寻找AI对用户消息的回复"""
        user_msg = messages[user_msg_index]
        time_window = timedelta(minutes=self.qa_match_time_window)
        
        # 在时间窗口内寻找AI的回复
        for i in range(user_msg_index + 1, len(messages)):
            reply_msg = messages[i]
            
            # 检查时间窗口
            if reply_msg.create_time - user_msg.create_time > time_window:
                break
                
            # 检查是否为AI回复（is_sender=1表示AI发送）
            if reply_msg.is_sender == 1:
                return reply_msg
                
        return None

    def _build_context(self, messages: List[Message], current_index: int) -> List[ChatMLMessage]:
        """构建上下文消息"""
        context_messages = []
        context_length = 0
        
        # 向前查找上下文，但要控制长度
        for i in range(max(0, current_index - 10), current_index):
            msg = messages[i]
            
            # 确定角色：is_sender=1表示AI发送（assistant），is_sender=0表示用户发送（user）
            role = "assistant" if msg.is_sender == 1 else "user"
            
            # 检查添加后是否超长
            if context_length + len(msg.msg) > self.messages_max_length // 2:
                break
                
            context_messages.append(ChatMLMessage(role, msg.msg))
            context_length += len(msg.msg)
        
        return context_messages

    def _reduce_context(self, messages: List[ChatMLMessage]) -> Optional[List[ChatMLMessage]]:
        """缩减上下文以满足长度限制"""
        # 保留系统提示和最后的用户消息、助手回复
        essential_messages = []
        
        # 添加系统提示（如果有）
        if messages and messages[0].role == "system":
            essential_messages.append(messages[0])
            start_index = 1
        else:
            start_index = 0
        
        # 添加最后的用户消息和助手回复
        if len(messages) >= 2:
            essential_messages.extend(messages[-2:])
        
        # 检查基本长度
        essential_length = sum(len(m.content) for m in essential_messages)
        if essential_length > self.messages_max_length:
            return None
            
        # 尝试添加部分上下文
        remaining_length = self.messages_max_length - essential_length
        context_messages = messages[start_index:-2] if len(messages) > 2 else []
        
        # 从最近的上下文开始添加
        selected_context = []
        for msg in reversed(context_messages):
            if len(msg.content) <= remaining_length:
                selected_context.insert(0, msg)
                remaining_length -= len(msg.content)
            else:
                break
        
        # 组合最终消息
        final_messages = []
        if messages and messages[0].role == "system":
            final_messages.append(messages[0])
        final_messages.extend(selected_context)
        if len(messages) >= 2:
            final_messages.extend(messages[-2:])
            
        return final_messages

    def _validate_dialogue_quality(self, messages: List[ChatMLMessage]) -> bool:
        """验证对话质量"""
        if len(messages) < 2:
            return False
            
        # 检查最后两条消息是否为有效的QA对
        last_two = messages[-2:]
        if len(last_two) != 2 or last_two[0].role != "user" or last_two[1].role != "assistant":
            return False
            
        # 检查用户消息和助手回复是否有意义
        user_msg = last_two[0].content.strip()
        assistant_msg = last_two[1].content.strip()
        
        if not user_msg or not assistant_msg:
            return False
            
        # 检查长度是否合理
        if len(user_msg) < 2 or len(assistant_msg) < 2:
            return False
            
        # 检查是否包含过多的无意义字符
        meaningless_patterns = [
            r'^[a-zA-Z]{1,3}$',  # 单个字母或很短的字母组合
            r'^[\d\s]+$',        # 纯数字
            r'^[^\u4e00-\u9fa5a-zA-Z\d]{3,}$',  # 大量特殊字符
            r'^([哈嘿呵]{3,}|[233333]{3,}|[wwww]{3,})$',  # 重复的无意义字符
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, user_msg) or re.match(pattern, assistant_msg):
                return False
                
        # 检查角色连续性（不应该有连续的相同角色）
        roles = [msg.role for msg in messages if msg.role in ["user", "assistant"]]
        for i in range(1, len(roles)):
            if roles[i] == roles[i-1]:
                return False
                
        return True

    def export_to_chatml_format(self, qa_pairs: List[List[ChatMLMessage]], output_file: str):
        """导出为ChatML格式文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for qa_pair in qa_pairs:
                    # 转换为ChatML标准格式
                    chatml_data = {
                        "messages": [
                            {"role": msg.role, "content": msg.content}
                            for msg in qa_pair
                        ]
                    }
                    f.write(json.dumps(chatml_data, ensure_ascii=False) + '\n')
            
            self.logger.info(f"ChatML数据已导出到: {output_file}")
            self.logger.info(f"总共导出了 {len(qa_pairs)} 个对话")
            
        except Exception as e:
            self.logger.error(f"导出ChatML数据时发生错误: {e}")

    def run(self):
        """主运行函数"""
        try:
            self.logger.info("开始生成ChatML格式训练数据")
            
            # 加载CSV数据
            dataset_path = "dataset/csv"
            messages = self.load_csv_files(dataset_path)
            
            if not messages:
                self.logger.error("未加载到任何消息数据")
                return
            
            # 合并消息
            combined_messages = self.combine_messages_by_time_window(messages)
            
            # 生成QA对
            qa_pairs = self.generate_qa_pairs(combined_messages)
            
            if not qa_pairs:
                self.logger.error("未生成任何QA对话对")
                return
            
            # 导出数据
            output_file = "./dataset/chatml_training_data.jsonl"
            self.export_to_chatml_format(qa_pairs, output_file)
            
            self.logger.info("ChatML数据生成完成")
            
        except Exception as e:
            self.logger.error(f"生成ChatML数据时发生错误: {e}")


def main():
    """主函数"""
    generator = ChatMLGenerator()
    generator.run()


if __name__ == "__main__":
    main()