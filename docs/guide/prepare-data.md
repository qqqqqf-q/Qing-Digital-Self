##  获取 QQ 聊天数据

* 教程请参考：[NTQQ Windows 数据解密](https://qq.sbcnm.top/decrypt/NTQQ%20%28Windows%29.html)
* 补充资料：[数据库解码参考](https://qq.sbcnm.top/decrypt/decode_db.html)
* 上面这两个是同一个教程的不同章节,耐心看完就好,不复杂(如果不会可以翻到最底下找我哦)
* 使用 DB Browser for SQLite，密码填写你获取到的 16 位密钥
* HMAC 算法一般为SHA1，也有人是SHA512和256,自行测试,算法错误了会打不开数据库（所以需要测试到打开为之,也可以用 AI 帮你适配）
* 在 DB Browser 里**导出 `c2c_msg_table` 的 SQL**
* 新建数据库，**导入刚才导出的 SQL 文件**
* 获得一个这样的数据库
* 结构如下图,是明文数据库(你能打开并且能看到数据就是正常的)
* 将数据库重命名为 `qq.db`并放在`dataset/original`文件夹下
> 或修改`setting.jsonc`中的`qq_db_path`



* <img src="https://cdn.nodeimage.com/i/oBfbWfVLhJI0CeZHTwwxq6G7XGO40Vy4.webp" alt="数据库图片">

---

## 获取Telegram(TG)聊天数据

* 请使用[Telegram Desktop](https://desktop.telegram.org/)导出聊天数据
* 点击`Export chat history`按钮
<img src="https://cdn.nodeimage.com/i/8PmL1yOyWbk1tTUkoLSk14sqrXN1HhYS.png" alt="8PmL1yOyWbk1tTUkoLSk14sqrXN1HhYS.png">
* 选择`JSON(Machine-readable JSON)`按钮
* 不必勾选其他按钮,因为此项目暂不支持多模态
<img src="https://cdn.nodeimage.com/i/ZOx12BovPbYXo89k4xIF9yRlEamneq4g.png" alt="ZOx12BovPbYXo89k4xIF9yRlEamneq4g.png">

* 将`导出文件夹下的**ChatExport_**文件夹全部移至`dataset/original/`文件夹内,如下图所示
<img src="https://cdn.nodeimage.com/i/zbc3iDHiqJrIOtWwrHkzX7TMONYatB8G.png" alt="zbc3iDHiqJrIOtWwrHkzX7TMONYatB8G">

* **重要**
* 修改`setting.jsonc`文件,将`telegram_chat_id`改为你的telegram聊天id
> **包含空格!!!**  
* 比如以下ID需要填写的是`qqqqq f`


## 获取微信(WX)聊天数据
* 前往[WeChatBakTool的Github项目](https://github.com/SuxueCode/WechatBakTool)并且去release下载最新版
* 或点此下载[WechatBakTool](https://github.com/SuxueCode/WechatBakTool/releases/download/v0.9.7.6/WechatBakTool.zip)
* 前往[此项目](https://github.com/tom-snow/wechat-windows-versions/releases)下载老版本微信(v3.9.12.15)
* 或点此链接快速下载[微信](https://github.com/tom-snow/wechat-windows-versions/releases/download/v3.9.12.15/WeChatSetup-3.9.12.15.exe)

* 安装微信并登录
* 手机打开`设置-聊天-聊天记录迁移与备份-迁移-迁移至电脑微信`并继续
* 解压baktool

* 0.安装.NET Desktop Runtime(注意是6.0版本的Desktop Runtime，如已经安装忽略)
* 1.打开微信，并登录。
* 2.在软件左侧下方点击新建工作区，
* 3.在新建工作区界面，选择要创建工作区的微信进程，并确认下方微信号是否正确
* 4.解密方式推荐选择用户名推断查找！该方式理论支持所有64位版本微信。但该模式需要确保微信账号正确
* 5.新手请忽略其他选项，直接点击创建工作区，程序会自动进行工作区创建、解密。

* 右键`工作区`-`管理`,导出好友聊天,全部
* 将前往`baktool`的文件夹,进入`workspace-[随机名文件夹]-DecDB`
* 找到所有`MSG*.db`的文件,例如`MSG1.db`,全部移动至`dataset/original/wechat`文件夹内
<img src="https://cdn.nodeimage.com/i/TRbknJP4C4KkBsfTKBUN3CXJPPVagMaP.png" alt="TRbknJP4C4KkBsfTKBUN3CXJPPVagMaP">
## (可选) 从视频/音频文件中获取聊天数据

*  从双音轨的视频/音频中提取(需要有音轨分离的文件)

### 1. 安装依赖

```bash
# 自动安装所有依赖
python process_data/chat_parser/video-to-chatml/start-vtc.py --install

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
python process_data/chat_parser/video-to-chatml/start-vtc.py -i
```

#### 直接命令行模式
```bash
python process_data/chat_parser/video-to-chatml/start-vtc.py video.mp4 -u 0 -a 1 -o output.json -m base
```

#### 使用主程序
```bash
python process_data/chat_parser/video-to-chatml/video-to-chatml.py video.mp4 -u 0 -a 1 -o output.json -m base
```

### 参数说明

- `video`: 输入视频文件路径
- `-u, --user-track`: 用户音轨索引（默认: 0）
- `-a, --assistant-track`: 助手音轨索引（默认: 1）
- `-o, --output`: 输出的ChatML文件路径
- `-m, --model`: Whisper模型大小（tiny/base/small/medium/large，默认: base）

### Whisper模型选择

| 模型 | 参数量 | 内存占用 | 速度 | 精度 |
|------|--------|----------|------|------|
| tiny | 39M | ~1GB | 最快 | 最低 |
| base | 74M | ~1GB | 快 | 低 |
| small | 244M | ~2GB | 中等 | 中等 |
| medium | 769M | ~5GB | 慢 | 高 |
| large | 1550M | ~10GB | 最慢 | 最高 |

### 输出格式

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

### 常见问题

### 1. CUDA支持
如果有NVIDIA GPU，程序会自动使用CUDA加速。检查CUDA支持：
```bash
python process_data/chat_parser/video-to-chatml/start-vtc.py --check
```

### 2. 音轨识别
使用交互式模式可以查看视频的所有音轨信息，帮助选择正确的音轨索引。

### 3. 内存不足
如果遇到内存不足，尝试使用更小的Whisper模型（如tiny或base）。

### 依赖项

- Python 3.7+
- openai-whisper
- torch
- ffmpeg-python
- ffmpeg (系统依赖)

### 支持的格式

**视频格式**: MP4, MKV, AVI, MOV, WMV等ffmpeg支持的格式
**音频编码**: 大部分常见音频编码（AAC, MP3, WAV等）

### 2.[TODO]从单音轨的视频/音频中自动识别并提取(暂时没写)
