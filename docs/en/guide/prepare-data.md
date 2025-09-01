## Getting QQ Chat Data

* Tutorial reference: [NTQQ Windows Data Decryption](https://qq.sbcnm.top/decrypt/NTQQ%20%28Windows%29.html)
* Supplementary material: [Database Decoding Reference](https://qq.sbcnm.top/decrypt/decode_db.html)
* The above two are different chapters of the same tutorial, read them patiently, it's not complicated (if you don't know how, scroll to the bottom to find me)
* Use DB Browser for SQLite, enter the 16-digit key you obtained as the password
* HMAC algorithm is generally SHA1, some people use SHA512 and 256, test yourself, wrong algorithm will fail to open the database (so you need to test until it opens, you can also use AI to help you adapt)
* In DB Browser **export the SQL of `c2c_msg_table`**
* Create a new database, **import the SQL file you just exported**
* Get a database like this
* Structure as shown below, it's a plaintext database (you can open it and see the data, which means it's normal)
* Rename the database to `qq.db` and place it in the `dataset/original` folder
> Or modify the `qq_db_path` in `setting.jsonc`



* <img src="https://cdn.nodeimage.com/i/oBfbWfVLhJI0CeZHTwwxq6G7XGO40Vy4.webp" alt="Database Image">

## Getting Telegram (TG) Chat Data

* Please use [Telegram Desktop](https://desktop.telegram.org/) to export chat data
* Click the `Export chat history` button
<img src="https://cdn.nodeimage.com/i/8PmL1yOyWbk1tTUkoLSk14sqrXN1HhYS.png" alt="8PmL1yOyWbk1tTUkoLSk14sqrXN1HhYS.png">
* Select the `JSON(Machine-readable JSON)` button
* No need to check other buttons, as this project does not support multimodal yet
<img src="https://cdn.nodeimage.com/i/ZOx12BovPbYXo89k4xIF9yRlEamneq4g.png" alt="ZOx12BovPbYXo89k4xIF9yRlEamneq4g.png">

* Move all **ChatExport_** folders from the export folder to the `dataset/original/` folder, as shown below
<img src="https://cdn.nodeimage.com/i/zbc3iDHiqJrIOtWwrHkzX7TMONYatB8G.png" alt="zbc3iDHiqJrIOtWwrHkzX7TMONYatB8G">

* **Important**
* Modify the `setting.jsonc` file, change `telegram_chat_id` to your telegram chat id
> **Including spaces!!!**  
* For example, if the following ID needs to be filled in as `qqqqq f`

---


## (Optional) Getting Chat Data from Video/Audio Files

*  Extract from dual-track video/audio (requires files with separated audio tracks)

### 1. Install Dependencies

```bash
# Automatically install all dependencies
python process_data/chat_parser/video-to-chatml/start-vtc.py --install

# Or install manually
pip install -r requirements.txt
```

**Note**: You also need to install ffmpeg:
- Windows: Download from https://ffmpeg.org or use `choco install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

### 2. Usage

#### Interactive Mode (Recommended for new users)
```bash
python process_data/chat_parser/video-to-chatml/start-vtc.py -i
```

#### Direct Command Line Mode
```bash
python process_data/chat_parser/video-to-chatml/start-vtc.py video.mp4 -u 0 -a 1 -o output.json -m base
```

#### Using Main Program
```bash
python process_data/chat_parser/video-to-chatml/video-to-chatml.py video.mp4 -u 0 -a 1 -o output.json -m base
```

### Parameter Description

- `video`: Input video file path
- `-u, --user-track`: User audio track index (default: 0)
- `-a, --assistant-track`: Assistant audio track index (default: 1)
- `-o, --output`: Output ChatML file path
- `-m, --model`: Whisper model size (tiny/base/small/medium/large, default: base)

### Whisper Model Selection

| Model | Parameters | Memory Usage | Speed | Accuracy |
|-------|------------|--------------|-------|----------|
| tiny | 39M | ~1GB | Fastest | Lowest |
| base | 74M | ~1GB | Fast | Low |
| small | 244M | ~2GB | Medium | Medium |
| medium | 769M | ~5GB | Slow | High |
| large | 1550M | ~10GB | Slowest | Highest |

### Output Format

The generated ChatML file format is as follows:

```json
[
  {
    "role": "user",
    "content": "What the user said",
    "timestamp": {
      "start": 0.0,
      "end": 2.5
    }
  },
  {
    "role": "assistant", 
    "content": "Assistant's reply",
    "timestamp": {
      "start": 2.5,
      "end": 5.0
    }
  }
]
```

### Common Issues

### 1. CUDA Support
If you have an NVIDIA GPU, the program will automatically use CUDA acceleration. Check CUDA support:
```bash
python process_data/chat_parser/video-to-chatml/start-vtc.py --check
```

### 2. Audio Track Recognition
Using interactive mode allows you to view all audio track information in the video, helping you select the correct track index.

### 3. Out of Memory
If you encounter memory issues, try using smaller Whisper models (like tiny or base).

### Dependencies

- Python 3.7+
- openai-whisper
- torch
- ffmpeg-python
- ffmpeg (system dependency)

### Supported Formats

**Video Formats**: MP4, MKV, AVI, MOV, WMV and other formats supported by ffmpeg
**Audio Codecs**: Most common audio codecs (AAC, MP3, WAV, etc.)

### 2. [TODO] Automatic recognition and extraction from single-track video/audio (not implemented yet)