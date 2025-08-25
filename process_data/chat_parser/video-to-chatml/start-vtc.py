#!/usr/bin/env python3
"""
Video to ChatML 启动脚本
自动配置环境并启动转换程序
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

class VTCStarter:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.main_script = self.script_dir / "video-to-chatml.py"
        
    def check_dependencies(self):
        """检查依赖项"""
        print("检查依赖项...")
        
        # 检查Python包
        required_packages = ["whisper", "torch", "ffmpeg-python"]
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == "ffmpeg-python":
                    import ffmpeg
                elif package == "whisper":
                    import whisper
                elif package == "torch":
                    import torch
            except ImportError:
                missing_packages.append(package)
        
        # 检查ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            print("[OK] ffmpeg 已安装")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[ERROR] ffmpeg 未找到，请安装 ffmpeg")
            self.install_ffmpeg_guide()
            return False
        
        if missing_packages:
            print(f"缺少以下Python包: {', '.join(missing_packages)}")
            self.install_packages(missing_packages)
            return False
        
        print("[OK] 所有依赖项已满足")
        return True
    
    def install_packages(self, packages):
        """安装缺少的Python包"""
        print("正在安装缺少的Python包...")
        
        for package in packages:
            if package == "whisper":
                cmd = [sys.executable, "-m", "pip", "install", "openai-whisper"]
            elif package == "ffmpeg-python":
                cmd = [sys.executable, "-m", "pip", "install", "ffmpeg-python"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", package]
            
            try:
                print(f"安装 {package}...")
                subprocess.run(cmd, check=True)
                print(f"[OK] {package} 安装成功")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] {package} 安装失败: {e}")
    
    def install_ffmpeg_guide(self):
        """显示ffmpeg安装指南"""
        print("\n=== ffmpeg 安装指南 ===")
        print("Windows:")
        print("1. 访问 https://ffmpeg.org/download.html")
        print("2. 下载Windows构建版本")
        print("3. 解压到合适位置（如 C:\\ffmpeg）")
        print("4. 将bin目录添加到系统PATH环境变量")
        print("\n或者使用包管理器:")
        print("chocolatey: choco install ffmpeg")
        print("winget: winget install ffmpeg")
        print("\nLinux/macOS:")
        print("Ubuntu/Debian: sudo apt install ffmpeg")
        print("CentOS/RHEL: sudo yum install ffmpeg")
        print("macOS: brew install ffmpeg")
    
    def auto_install_dependencies(self):
        """自动安装依赖"""
        print("正在自动安装依赖...")
        
        # 安装Python包
        packages_to_install = [
            "openai-whisper",
            "torch",
            "ffmpeg-python"
        ]
        
        for package in packages_to_install:
            try:
                print(f"安装 {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                print(f"[OK] {package} 安装成功")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] {package} 安装失败: {e}")
        
        # 检查CUDA支持
        try:
            # 重新导入torch以确保获取最新版本
            import importlib
            import sys
            if 'torch' in sys.modules:
                importlib.reload(sys.modules['torch'])
            import torch
            if torch.cuda.is_available():
                print("[OK] 检测到CUDA支持")
                print(f"CUDA设备数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"设备 {i}: {torch.cuda.get_device_name(i)}")
            else:
                print("[INFO] 未检测到CUDA支持，将使用CPU进行推理")
        except ImportError:
            print("[WARNING] 无法检查CUDA支持")
    
    def get_video_info(self, video_path):
        """获取视频信息"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            data = json.loads(result.stdout)
            
            audio_streams = []
            for i, stream in enumerate(data['streams']):
                if stream['codec_type'] == 'audio':
                    audio_streams.append({
                        'index': i,
                        'codec': stream.get('codec_name', 'unknown'),
                        'channels': stream.get('channels', 'unknown'),
                        'sample_rate': stream.get('sample_rate', 'unknown')
                    })
            
            return audio_streams
        except Exception as e:
            print(f"获取视频信息失败: {e}")
            return []
    
    def interactive_setup(self):
        """交互式配置"""
        print("\n=== 交互式配置 ===")
        
        # 输入视频文件
        while True:
            video_path = input("请输入视频文件路径: ").strip().strip('"')
            if os.path.exists(video_path):
                break
            print("文件不存在，请重新输入")
        
        # 获取音轨信息
        print("\n分析视频音轨...")
        audio_streams = self.get_video_info(video_path)
        
        if audio_streams:
            print("发现以下音轨:")
            for stream in audio_streams:
                print(f"音轨 {stream['index']}: {stream['codec']}, {stream['channels']}声道, {stream['sample_rate']}Hz")
        else:
            print("无法获取音轨信息，将使用默认设置")
        
        # 选择音轨
        user_track = input(f"请输入用户音轨索引 (默认: 0): ").strip()
        user_track = int(user_track) if user_track.isdigit() else 0
        
        assistant_track = input(f"请输入助手音轨索引 (默认: 1): ").strip()
        assistant_track = int(assistant_track) if assistant_track.isdigit() else 1
        
        # 输出文件
        default_output = os.path.splitext(video_path)[0] + "_chatml.json"
        output_path = input(f"请输入输出文件路径 (默认: {default_output}): ").strip()
        output_path = output_path if output_path else default_output
        
        # 选择模型
        models = ["tiny", "base", "small", "medium", "large"]
        print(f"\n可用的Whisper模型: {', '.join(models)}")
        print("模型越大精度越高，但速度越慢")
        model = input("请选择模型 (默认: base): ").strip()
        model = model if model in models else "base"
        
        # 选择语言
        languages = {
            "zh": "中文",
            "en": "英文", 
            "ja": "日文",
            "ko": "韩文",
            "fr": "法文",
            "de": "德文",
            "es": "西班牙文",
            "ru": "俄文"
        }
        print(f"\n支持的语言:")
        for code, name in languages.items():
            print(f"  {code}: {name}")
        print("  auto: 自动检测")
        language = input("请选择语言代码 (默认: auto): ").strip()
        language = language if language and language != "auto" else None
        
        return {
            'video': video_path,
            'user_track': user_track,
            'assistant_track': assistant_track,
            'output': output_path,
            'model': model,
            'language': language
        }
    
    def run_conversion(self, config):
        """运行转换程序"""
        cmd = [
            sys.executable, str(self.main_script),
            config['video'],
            '-u', str(config['user_track']),
            '-a', str(config['assistant_track']),
            '-o', config['output'],
            '-m', config['model']
        ]
        
        if config.get('language'):
            cmd.extend(['-l', config['language']])
        
        print(f"\n执行命令: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"转换失败: {e}")
            return False
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Video to ChatML 启动脚本")
    parser.add_argument("--install", action="store_true", help="自动安装依赖")
    parser.add_argument("--check", action="store_true", help="检查依赖项")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式配置")
    
    # 直接转换参数
    parser.add_argument("video", nargs="?", help="输入视频文件路径")
    parser.add_argument("-u", "--user-track", type=int, default=0, help="用户音轨索引")
    parser.add_argument("-a", "--assistant-track", type=int, default=1, help="助手音轨索引")
    parser.add_argument("-o", "--output", help="输出ChatML文件路径")
    parser.add_argument("-m", "--model", default="base", help="Whisper模型")
    parser.add_argument("-l", "--language", help="音频语言代码")
    
    args = parser.parse_args()
    
    starter = VTCStarter()
    
    # 安装依赖
    if args.install:
        starter.auto_install_dependencies()
        return 0
    
    # 检查依赖
    if args.check:
        if starter.check_dependencies():
            print("所有依赖项已满足")
        return 0
    
    # 检查依赖项
    if not starter.check_dependencies():
        print("\n请先安装缺少的依赖项:")
        print(f"python {__file__} --install")
        return 1
    
    # 交互式模式
    if args.interactive or not args.video:
        config = starter.interactive_setup()
    else:
        # 直接模式
        if not args.output:
            args.output = os.path.splitext(args.video)[0] + "_chatml.json"
        
        config = {
            'video': args.video,
            'user_track': args.user_track,
            'assistant_track': args.assistant_track,
            'output': args.output,
            'model': args.model
        }
    
    # 运行转换
    if starter.run_conversion(config):
        print(f"\n转换成功！输出文件: {config['output']}")
        return 0
    else:
        print("\n转换失败")
        return 1

if __name__ == "__main__":
    exit(main())