## Video-to-ChatML 视频对话转换

> 将包含双音轨对话的视频文件转换为ChatML格式的训练数据
> 适用于从录制的对话视频中提取聊天数据进行模型训练

## 功能简介

Video-to-ChatML 是一个强大的工具，可以将包含双音轨对话的视频文件转换为ChatML格式的训练数据。这个功能特别适用于：

- 从录制的对话视频中提取聊天数据
- 将语音对话转换为文本格式
- 生成可用于模型微调的ChatML格式数据
- 支持多种语言的语音识别

## 环境要求

### 系统要求
- Python 3.7+
- FFmpeg (用于音频处理)
- CUDA (可选，用于GPU加速)

### 依赖安装

进入 `video-to-chatml` 目录：
```bash
cd video-to-chatml
```

安装Python依赖：
```bash
pip install -r requirements.txt
```

### FFmpeg 安装

**Windows:**
1. 下载 FFmpeg: https://ffmpeg.org/download.html
2. 解压到任意目录
3. 将 FFmpeg 的 bin 目录添加到系统 PATH

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS (使用 Homebrew)
brew install ffmpeg
```

## 使用方法

### 方法一：使用启动脚本（推荐）

运行启动脚本，它会自动检查依赖并引导你完成配置：

```bash
python start-vtc.py
```

启动脚本会：
1. 自动检查所有依赖项
2. 提供交互式配置界面
3. 自动检测视频的音轨信息
4. 引导你选择正确的音轨配置

### 方法二：直接使用转换脚本

```bash
python video-to-chatml.py [选项] 视频文件路径
```

#### 基本参数

- `video_path`: 输入视频文件路径
- `--user-track`: 用户音轨索引 (默认: 0)
- `--assistant-track`: 助手音轨索引 (默认: 1)
- `--output`: 输出ChatML文件路径 (默认: output.jsonl)
- `--model`: Whisper模型名称 (默认: base)
- `--language`: 指定语言 (可选，如: zh, en)

#### 使用示例

```bash
# 基本使用
python video-to-chatml.py conversation.mp4

# 指定音轨和输出文件
python video-to-chatml.py conversation.mp4 --user-track 0 --assistant-track 1 --output chat_data.jsonl

# 使用更大的Whisper模型提高准确性
python video-to-chatml.py conversation.mp4 --model large --language zh

# 完整参数示例
python video-to-chatml.py conversation.mp4 \
  --user-track 0 \
  --assistant-track 1 \
  --output training_data.jsonl \
  --model medium \
  --language zh
```

## Whisper 模型选择

| 模型名称 | 参数量 | 英语准确性 | 多语言准确性 | 速度 | 显存需求 |
|---------|--------|-----------|-------------|------|----------|
| tiny    | 39M    | 较低      | 较低        | 最快 | ~1GB     |
| base    | 74M    | 中等      | 中等        | 快   | ~1GB     |
| small   | 244M   | 良好      | 良好        | 中等 | ~2GB     |
| medium  | 769M   | 很好      | 很好        | 较慢 | ~5GB     |
| large   | 1550M  | 最好      | 最好        | 最慢 | ~10GB    |

**推荐选择：**
- 快速测试：`base`
- 平衡性能：`small` 或 `medium`
- 最高质量：`large`

## 视频格式要求

### 音轨配置
视频文件需要包含**两个独立的音轨**：
- 音轨 0：用户语音
- 音轨 1：助手语音

### 支持的视频格式
- MP4 (推荐)
- AVI
- MOV
- MKV
- 其他FFmpeg支持的格式

### 音频要求
- 采样率：建议 16kHz 或更高
- 声道：单声道或立体声
- 编码：任何FFmpeg支持的音频编码

## 输出格式

转换后的数据将保存为ChatML格式的JSONL文件，每行包含一个对话：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "你好，今天天气怎么样？"
    },
    {
      "role": "assistant",
      "content": "今天天气很不错，阳光明媚，温度适宜。"
    }
  ]
}
```

## 高级配置

### 语言设置

支持的语言代码（部分）：
- `zh`: 中文
- `en`: 英语
- `ja`: 日语
- `ko`: 韩语
- `fr`: 法语
- `de`: 德语
- `es`: 西班牙语

### 性能优化

1. **GPU加速**：确保安装了CUDA版本的PyTorch
2. **模型选择**：根据硬件配置选择合适的Whisper模型
3. **批处理**：对于大量视频文件，可以编写脚本批量处理

## 故障排除

### 常见问题

**1. FFmpeg 未找到**
```
错误: ffmpeg 未找到，请安装 ffmpeg
```
解决方案：确保FFmpeg已正确安装并添加到系统PATH

**2. 音轨提取失败**
```
错误: 提取音轨失败
```
解决方案：
- 检查视频文件是否包含指定的音轨
- 使用 `ffprobe` 命令查看视频的音轨信息：
  ```bash
  ffprobe -v quiet -print_format json -show_streams input.mp4
  ```

**3. 显存不足**
```
错误: CUDA out of memory
```
解决方案：
- 使用更小的Whisper模型（如 `base` 或 `small`）
- 在CPU上运行（会较慢但不需要显存）

**4. 识别准确性低**
解决方案：
- 使用更大的Whisper模型
- 确保音频质量良好
- 指定正确的语言参数

### 调试模式

如果遇到问题，可以查看详细的错误信息：

```bash
python video-to-chatml.py conversation.mp4 --verbose
```

## 最佳实践

1. **音频质量**：确保录制的音频清晰，背景噪音较少
2. **音轨分离**：录制时确保用户和助手的声音在不同的音轨上
3. **语言一致性**：指定正确的语言参数以提高识别准确性
4. **模型选择**：根据硬件配置和质量要求选择合适的Whisper模型
5. **数据验证**：转换完成后检查输出数据的质量和格式

## 与其他功能的集成

转换得到的ChatML数据可以直接用于：

1. **数据混合**：与其他数据源混合 → [混合数据](mix-data.md)
2. **模型微调**：用于微调大语言模型 → [微调模型](fine-tune-model.md)
3. **数据清洗**：进一步清洗和优化 → [清洗数据](clean-data.md)

---

**提示**：Video-to-ChatML 功能为数字分身项目提供了从多媒体内容中获取训练数据的能力，让你可以利用更多样化的数据源来训练个性化的AI模型。