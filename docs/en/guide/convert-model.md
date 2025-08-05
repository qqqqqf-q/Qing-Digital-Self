## 4. Compile llama.cpp

> The following three steps all depend on compiled llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp --depth 1
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_NATIVE=ON
cmake --build . --config Release
```

---

## 5. Convert HuggingFace Weights to GGUF

### Command Format:

```bash
python3 .convert_hf_to_gguf.py <HF_model_path> --outfile <output_GGUF_path> --outtype <precision_type>
```

* `<HF_model_path>`: HuggingFace format model directory (usually the path after fine-tuning or downloading)
* `<output_GGUF_path>`: Save path for the converted `.gguf` model
* `<precision_type>`: Precision type, such as `f16`, `f32`, or other supported formats

### Example:

```bash
python3 convert_hf_to_gguf.py /root/autodl-tmp/finetune/models/qwen3-8b-qlora/merged --outfile /root/autodl-fs/qwen3-8b-fp16-agent.gguf --outtype f16
```

---

## 6. Quantize Model

### Command Format:

```bash
./build/bin/llama-quantize <input_GGUF_path> <output_GGUF_path> <quantization_level>
```

* `<input_GGUF_path>`: Unquantized `.gguf` file path
* `<output_GGUF_path>`: Save path for quantized `.gguf` file
* `<quantization_level>`: Such as `Q4_0`, `Q4_K_M`, `Q8_0`, etc., choose based on needs and hardware

### Example:

```bash
./build/bin/llama-quantize \
  /root/autodl-fs/qwen3-8b-fp16-agent.gguf \
  /root/autodl-fs/qwen3-8b-q8_0-agent.gguf \
  Q8_0
```

---

## 7. Run Model Test

### Command Format:

```bash
./build/bin/llama-run <GGUF_model_path>
```

* `<GGUF_model_path>`: The GGUF model path you want to test (can be original or quantized)

### Example:

```bash
./build/bin/llama-run /root/autodl-fs/qwen3-8b-fp16-agent.gguf
```

---
## 8. High-speed File Download from Server

### Command Format
```bash
lftp -u {username},{password} -p {port} sftp://{server_address} -e "set xfer:clobber true; pget -n {thread_count} {server_file_path} -o {local_file_name/path}; bye"
```
* `pget`: Use multi-threaded parallel download
* `-n`: Specify thread count (recommend 64+) (even 256 threads will have better performance)
### Example
```bash
lftp -u root,askdjiwhakjd -p 27391 sftp://yourserver.com -e "set xfer:clobber true; pget -n 256 /root/autodl-fs/qwen3-8b-fp16-agent.gguf -o qwen3-8b-fp16-agent.gguf; bye"
```
---