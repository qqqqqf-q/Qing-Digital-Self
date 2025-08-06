#!/usr/bin/env python3
"""
Video to ChatML Converter
将视频中的双音轨对话转换为ChatML格式
"""

import argparse
import os
import tempfile
import json
import subprocess
import whisper
from pathlib import Path
import torch

class VideoToChatML:
    def __init__(self, model_name="base"):
        self.model_name = model_name
        self.model = None
        
    def load_whisper_model(self):
        """加载Whisper模型"""
        if self.model is None:
            print(f"正在加载Whisper模型: {self.model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = whisper.load_model(self.model_name, device=device)
            print(f"模型已加载到设备: {device}")
    
    def extract_audio_track(self, video_path, track_index, output_path):
        """从视频中提取指定音轨"""
        cmd = [
            'ffmpeg', '-i', video_path,
            '-map', f'0:a:{track_index}',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"成功提取音轨 {track_index} 到 {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"提取音轨失败: {e}")
            print(f"错误输出: {e.stderr}")
            return False
    
    def transcribe_audio(self, audio_path, language=None):
        """转录音频文件"""
        if self.model is None:
            self.load_whisper_model()
        
        print(f"正在转录音频: {audio_path}")
        if language:
            print(f"使用语言: {language}")
            result = self.model.transcribe(audio_path, language=language)
        else:
            print("自动检测语言")
            result = self.model.transcribe(audio_path)
        return result
    
    def merge_transcripts(self, user_transcript, assistant_transcript):
        """合并两个转录结果，按时间戳排序"""
        all_segments = []
        
        # 添加用户段落
        for segment in user_transcript['segments']:
            all_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'role': 'user'
            })
        
        # 添加助手段落
        for segment in assistant_transcript['segments']:
            all_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'role': 'assistant'
            })
        
        # 按时间戳排序
        all_segments.sort(key=lambda x: x['start'])
        
        return all_segments
    
    def segments_to_chatml(self, segments):
        """将段落转换为ChatML格式"""
        chatml_data = []
        
        for segment in segments:
            if segment['text']:  # 只处理非空文本
                chatml_data.append({
                    "role": segment['role'],
                    "content": segment['text'],
                    "timestamp": {
                        "start": segment['start'],
                        "end": segment['end']
                    }
                })
        
        return chatml_data
    
    def process_video(self, video_path, user_track, assistant_track, output_path, language=None):
        """处理视频文件，生成ChatML格式输出"""
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            user_audio_path = temp_dir / "user_audio.wav"
            assistant_audio_path = temp_dir / "assistant_audio.wav"
            
            print("正在提取音轨...")
            
            # 提取用户音轨
            if not self.extract_audio_track(str(video_path), user_track, str(user_audio_path)):
                raise RuntimeError("用户音轨提取失败")
            
            # 提取助手音轨
            if not self.extract_audio_track(str(video_path), assistant_track, str(assistant_audio_path)):
                raise RuntimeError("助手音轨提取失败")
            
            print("正在转录音频...")
            
            # 转录音频
            user_transcript = self.transcribe_audio(str(user_audio_path), language)
            assistant_transcript = self.transcribe_audio(str(assistant_audio_path), language)
            
            print("正在合并转录结果...")
            
            # 合并转录结果
            merged_segments = self.merge_transcripts(user_transcript, assistant_transcript)
            
            # 转换为ChatML格式
            chatml_data = self.segments_to_chatml(merged_segments)
            
            # 保存结果
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chatml_data, f, ensure_ascii=False, indent=2)
            
            print(f"转换完成！输出文件: {output_path}")
            print(f"共生成 {len(chatml_data)} 条对话")

def main():
    parser = argparse.ArgumentParser(description="将视频中的双音轨对话转换为ChatML格式")
    parser.add_argument("video", help="输入视频文件路径 (mkv, mp4等)")
    parser.add_argument("-u", "--user-track", type=int, default=0, 
                       help="用户音轨索引 (默认: 0)")
    parser.add_argument("-a", "--assistant-track", type=int, default=1,
                       help="助手音轨索引 (默认: 1)")
    parser.add_argument("-o", "--output", required=True,
                       help="输出的ChatML文件路径")
    parser.add_argument("-m", "--model", default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper模型大小 (默认: base)")
    parser.add_argument("-l", "--language", 
                       help="音频语言代码 (如: zh=中文, en=英文, ja=日文等，不指定则自动检测)")
    
    args = parser.parse_args()
    
    try:
        converter = VideoToChatML(model_name=args.model)
        converter.process_video(args.video, args.user_track, args.assistant_track, args.output, args.language)
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())