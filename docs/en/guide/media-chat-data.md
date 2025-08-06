## Extracting Chat Data from Media Files

# 1. Extracting from Dual-Audio Track Video/Audio Files (Requires Separated Audio Tracks)

# Video to ChatML Converter

Converts video files containing dual-track conversations into ChatML-format JSON files.

## Features

* Supports various video formats (MKV, MP4, etc.)
* Uses OpenAI Whisper for high-quality speech recognition
* CUDA acceleration support (if available)
* Automatically sorts dialogue by timestamp
* Generates standard ChatML output

## Quick Start

### 1. Install Dependencies

```bash
# Automatically install all dependencies
python start-vtc.py --install

# Or install manually
pip install -r requirements.txt
```

**Note**: You also need to install ffmpeg:

* Windows: Download from [https://ffmpeg.org](https://ffmpeg.org) or use `choco install ffmpeg`
* Ubuntu: `sudo apt install ffmpeg`
* macOS: `brew install ffmpeg`

### 2. Usage

#### Interactive Mode (Recommended for New Users)

```bash
python start-vtc.py -i
```

#### Direct Command-Line Mode

```bash
python start-vtc.py video.mp4 -u 0 -a 1 -o output.json -m base
```

#### Using the Main Program Directly

```bash
python video-to-chatml.py video.mp4 -u 0 -a 1 -o output.json -m base
```

## Parameter Description

* `video`: Path to the input video file
* `-u, --user-track`: User audio track index (default: 0)
* `-a, --assistant-track`: Assistant audio track index (default: 1)
* `-o, --output`: Output ChatML file path
* `-m, --model`: Whisper model size (tiny/base/small/medium/large, default: base)

## Whisper Model Options

| Model  | Parameters | RAM Usage | Speed   | Accuracy |
| ------ | ---------- | --------- | ------- | -------- |
| tiny   | 39M        | \~1GB     | Fastest | Lowest   |
| base   | 74M        | \~1GB     | Fast    | Low      |
| small  | 244M       | \~2GB     | Medium  | Medium   |
| medium | 769M       | \~5GB     | Slow    | High     |
| large  | 1550M      | \~10GB    | Slowest | Highest  |

## Output Format

The generated ChatML file format looks like this:

```json
[
  {
    "role": "user",
    "content": "User's spoken content",
    "timestamp": {
      "start": 0.0,
      "end": 2.5
    }
  },
  {
    "role": "assistant", 
    "content": "Assistant's response",
    "timestamp": {
      "start": 2.5,
      "end": 5.0
    }
  }
]
```

## Frequently Asked Questions

### 1. CUDA Support

If you have an NVIDIA GPU, the program will automatically use CUDA acceleration. Check CUDA support:

```bash
python start-vtc.py --check
```

### 2. Identifying Audio Tracks

Use interactive mode to view all audio tracks in the video, which helps select the correct track index.

### 3. Out of Memory

If you run into memory issues, try using a smaller Whisper model (e.g., `tiny` or `base`).

## Dependencies

* Python 3.7+
* openai-whisper
* torch
* ffmpeg-python
* ffmpeg (system dependency)

## Supported Formats

**Video formats**: MP4, MKV, AVI, MOV, WMV, and other ffmpeg-supported formats
**Audio codecs**: Most common codecs (AAC, MP3, WAV, etc.)

# 2. \[TODO] Automatically Detect and Extract from Single-Track Video/Audio Files (Not Implemented Yet)