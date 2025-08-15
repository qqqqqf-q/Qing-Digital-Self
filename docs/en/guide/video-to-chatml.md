## Video-to-ChatML Conversion

> Convert video files with dual audio tracks into ChatML format training data
> Perfect for extracting chat data from recorded conversation videos

## Overview

Video-to-ChatML is a powerful tool that converts video files containing dual-track conversations into ChatML format training data. This feature is particularly useful for:

- Extracting chat data from recorded conversation videos
- Converting voice conversations to text format
- Generating ChatML format data for model fine-tuning
- Supporting multi-language speech recognition

## Requirements

### System Requirements
- Python 3.7+
- FFmpeg (for audio processing)
- CUDA (optional, for GPU acceleration)

### Dependency Installation

Navigate to the `video-to-chatml` directory:
```bash
cd video-to-chatml
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### FFmpeg Installation

**Windows:**
1. Download FFmpeg: https://ffmpeg.org/download.html
2. Extract to any directory
3. Add FFmpeg's bin directory to system PATH

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS (using Homebrew)
brew install ffmpeg
```

## Usage

### Method 1: Using Startup Script (Recommended)

Run the startup script, which will automatically check dependencies and guide you through configuration:

```bash
python start-vtc.py
```

The startup script will:
1. Automatically check all dependencies
2. Provide an interactive configuration interface
3. Auto-detect video track information
4. Guide you to select correct track configuration

### Method 2: Direct Script Usage

```bash
python video-to-chatml.py [options] video_file_path
```

#### Basic Parameters

- `video_path`: Input video file path
- `--user-track`: User audio track index (default: 0)
- `--assistant-track`: Assistant audio track index (default: 1)
- `--output`: Output ChatML file path (default: output.jsonl)
- `--model`: Whisper model name (default: base)
- `--language`: Specify language (optional, e.g.: zh, en)

#### Usage Examples

```bash
# Basic usage
python video-to-chatml.py conversation.mp4

# Specify tracks and output file
python video-to-chatml.py conversation.mp4 --user-track 0 --assistant-track 1 --output chat_data.jsonl

# Use larger Whisper model for better accuracy
python video-to-chatml.py conversation.mp4 --model large --language en

# Complete parameter example
python video-to-chatml.py conversation.mp4 \
  --user-track 0 \
  --assistant-track 1 \
  --output training_data.jsonl \
  --model medium \
  --language en
```

## Whisper Model Selection

| Model   | Parameters | English Accuracy | Multilingual Accuracy | Speed | VRAM Required |
|---------|------------|------------------|----------------------|-------|---------------|
| tiny    | 39M        | Lower            | Lower                | Fastest | ~1GB        |
| base    | 74M        | Medium           | Medium               | Fast    | ~1GB        |
| small   | 244M       | Good             | Good                 | Medium  | ~2GB        |
| medium  | 769M       | Very Good        | Very Good            | Slower  | ~5GB        |
| large   | 1550M      | Best             | Best                 | Slowest | ~10GB       |

**Recommendations:**
- Quick testing: `base`
- Balanced performance: `small` or `medium`
- Highest quality: `large`

## Video Format Requirements

### Audio Track Configuration
Video files need to contain **two separate audio tracks**:
- Track 0: User voice
- Track 1: Assistant voice

### Supported Video Formats
- MP4 (recommended)
- AVI
- MOV
- MKV
- Other FFmpeg-supported formats

### Audio Requirements
- Sample rate: 16kHz or higher recommended
- Channels: Mono or stereo
- Encoding: Any FFmpeg-supported audio codec

## Output Format

Converted data will be saved as ChatML format JSONL file, with each line containing one conversation:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, how's the weather today?"
    },
    {
      "role": "assistant",
      "content": "The weather is great today, sunny and pleasant temperature."
    }
  ]
}
```

## Advanced Configuration

### Language Settings

Supported language codes (partial list):
- `zh`: Chinese
- `en`: English
- `ja`: Japanese
- `ko`: Korean
- `fr`: French
- `de`: German
- `es`: Spanish

### Performance Optimization

1. **GPU Acceleration**: Ensure CUDA version of PyTorch is installed
2. **Model Selection**: Choose appropriate Whisper model based on hardware
3. **Batch Processing**: Write scripts for processing multiple video files

## Troubleshooting

### Common Issues

**1. FFmpeg Not Found**
```
Error: ffmpeg not found, please install ffmpeg
```
Solution: Ensure FFmpeg is properly installed and added to system PATH

**2. Audio Track Extraction Failed**
```
Error: Audio track extraction failed
```
Solution:
- Check if video file contains specified audio tracks
- Use `ffprobe` command to view video track information:
  ```bash
  ffprobe -v quiet -print_format json -show_streams input.mp4
  ```

**3. Out of Memory**
```
Error: CUDA out of memory
```
Solution:
- Use smaller Whisper model (like `base` or `small`)
- Run on CPU (slower but doesn't require VRAM)

**4. Low Recognition Accuracy**
Solution:
- Use larger Whisper model
- Ensure good audio quality
- Specify correct language parameter

### Debug Mode

For detailed error information:

```bash
python video-to-chatml.py conversation.mp4 --verbose
```

## Best Practices

1. **Audio Quality**: Ensure clear recorded audio with minimal background noise
2. **Track Separation**: Ensure user and assistant voices are on different tracks during recording
3. **Language Consistency**: Specify correct language parameter for better recognition
4. **Model Selection**: Choose appropriate Whisper model based on hardware and quality requirements
5. **Data Validation**: Check output data quality and format after conversion

## Integration with Other Features

The converted ChatML data can be directly used for:

1. **Data Mixing**: Mix with other data sources → [Mix Data](mix-data.md)
2. **Model Fine-tuning**: Use for fine-tuning large language models → [Fine-tune Model](fine-tune-model.md)
3. **Data Cleaning**: Further clean and optimize → [Clean Data](clean-data.md)

---

**Tip**: The Video-to-ChatML feature provides the digital avatar project with the ability to obtain training data from multimedia content, allowing you to use more diverse data sources to train personalized AI models.