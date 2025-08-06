## 从媒体文件中获取聊天数据

# 1.从双音轨的视频/音频中提取(需要有音轨分离的文件)

# Video to ChatML 转换器

将包含双音轨对话的视频文件转换为ChatML格式的JSON文件。

## 功能特性

- 支持多种视频格式（MKV、MP4等）
- 使用OpenAI Whisper进行高质量语音识别
- 支持CUDA加速（如果可用）
- 自动按时间戳排序对话
- 生成标准ChatML格式输出

## 快速开始

### 1. 安装依赖

```bash
# 自动安装所有依赖
python start-vtc.py --install

# 或手动安装
pip install -r requirements.txt
```

**注意**: 还需要安装ffmpeg:
- Windows: 从 https://ffmpeg.org 下载或使用 `choco install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

### 2. 使用方法

#### 交互式模式（推荐新用户）
```bash
python start-vtc.py -i
```

#### 直接命令行模式
```bash
python start-vtc.py video.mp4 -u 0 -a 1 -o output.json -m base
```

#### 使用主程序
```bash
python video-to-chatml.py video.mp4 -u 0 -a 1 -o output.json -m base
```

## 参数说明

- `video`: 输入视频文件路径
- `-u, --user-track`: 用户音轨索引（默认: 0）
- `-a, --assistant-track`: 助手音轨索引（默认: 1）
- `-o, --output`: 输出的ChatML文件路径
- `-m, --model`: Whisper模型大小（tiny/base/small/medium/large，默认: base）

## Whisper模型选择

| 模型 | 参数量 | 内存占用 | 速度 | 精度 |
|------|--------|----------|------|------|
| tiny | 39M | ~1GB | 最快 | 最低 |
| base | 74M | ~1GB | 快 | 低 |
| small | 244M | ~2GB | 中等 | 中等 |
| medium | 769M | ~5GB | 慢 | 高 |
| large | 1550M | ~10GB | 最慢 | 最高 |

## 输出格式

生成的ChatML文件格式如下：

```json
[
  {
    "role": "user",
    "content": "用户说的话",
    "timestamp": {
      "start": 0.0,
      "end": 2.5
    }
  },
  {
    "role": "assistant", 
    "content": "助手回复的话",
    "timestamp": {
      "start": 2.5,
      "end": 5.0
    }
  }
]
```

## 常见问题

### 1. CUDA支持
如果有NVIDIA GPU，程序会自动使用CUDA加速。检查CUDA支持：
```bash
python start-vtc.py --check
```

### 2. 音轨识别
使用交互式模式可以查看视频的所有音轨信息，帮助选择正确的音轨索引。

### 3. 内存不足
如果遇到内存不足，尝试使用更小的Whisper模型（如tiny或base）。

## 依赖项

- Python 3.7+
- openai-whisper
- torch
- ffmpeg-python
- ffmpeg (系统依赖)

## 支持的格式

**视频格式**: MP4, MKV, AVI, MOV, WMV等ffmpeg支持的格式
**音频编码**: 大部分常见音频编码（AAC, MP3, WAV等）

# 2.[TODO]从单音轨的视频/音频中自动识别并提取(暂时没写)
