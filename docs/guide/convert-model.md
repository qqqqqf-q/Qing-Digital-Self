## 编译 llama.cpp

> 下面三步都依赖于编译好的 llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp --depth 1
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_NATIVE=ON
cmake --build . --config Release
```

---

## HuggingFace 权重转 GGUF

### 命令格式：

```bash
python3 ./convert_hf_to_gguf.py <HF模型路径> --outfile  <输出GGUF路径> --outtype <精度类型>
```

* `<HF模型路径>`：HuggingFace 格式模型目录（通常为微调或下载后的路径）
* `<输出GGUF路径>`：转换后生成的 `.gguf` 模型保存路径
* `<精度类型>`：精度类型，'f32', 'f16', 'bf16', 'q8_0', 'tq1_0', 'tq2_0', 'auto'

### 示例：

```bash
python3 convert_hf_to_gguf.py /root/autodl-tmp/finetune/models/qwen3-8b-qlora/merged --outfile /root/autodl-fs/qwen3-8b-fp16-agent.gguf --outtype f16  
```

---

## 量化模型

### 命令格式：

```bash
./build/bin/llama-quantize <输入GGUF路径> <输出GGUF路径> <量化等级>
```

* `<输入GGUF路径>`：未量化的 `.gguf` 文件路径
* `<输出GGUF路径>`：量化后的 `.gguf` 文件保存路径
* `<量化等级>`：如 `Q4_0`、`Q4_K_M`、`Q8_0` 等，根据需求和硬件选择

### 示例：

```bash
./build/bin/llama-quantize \
  /root/autodl-fs/qwen3-8b-fp16-agent.gguf \
  /root/autodl-fs/qwen3-8b-q8_0-agent.gguf \
  Q8_0
```

---

## 7. 运行模型测试

### 命令格式：

```bash
./build/bin/llama-run <GGUF模型路径>
```

* `<GGUF模型路径>`：你想测试的 GGUF 模型路径（可以是原始或量化后的）

### 示例：

```bash
./build/bin/llama-run /root/autodl-fs/qwen3-8b-fp16-agent.gguf
```

---
## 8.从服务器上高速下载文件

## 可以直接从服务器提供商的数据存储中下载,就不用开机付费了

### 或者
### 命令格式
```bash
lftp -u {用户名},{密码} -p {端口} sftp://{服务器地址}-e "set xfer:clobber true;  pget -n {线程数} {服务器文件路径} -o {本地文件名/路径}: bye"
```
* `pget`: 使用多线程并行下载
* `-n` :指定线程数(建议64+)(甚至256线程会有更好的表现)
### 范例
```bash
lftp -u root,askdjiwhakjd -p 27391 sftp://yourserver.com -e "set xfer:clobber true; pget -n 256 /root/autodl-fs/qwen3-8b-fp16-agent.gguf -o qwen3-8b-fp16-agent.gguf; bye"
```
---
