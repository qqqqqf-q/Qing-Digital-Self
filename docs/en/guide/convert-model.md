## Compile llama.cpp

> The following three steps depend on having `llama.cpp` compiled.

```bash
git clone https://github.com/ggerganov/llama.cpp --depth 1
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_NATIVE=ON
cmake --build . --config Release -j

cd ..
python3 -m venv venv
source venv/bin/activate
pip install -r "./requirements/requirements-convert_hf_to_gguf.txt"
```

---

## Convert HuggingFace Weights to GGUF

### Command format:

```bash
python3 ./convert_hf_to_gguf.py <HF_model_path> --outfile <output_GGUF_path> --outtype <precision_type>
```

* `<HF_model_path>`: Path to the HuggingFace format model directory (usually after fine-tuning or downloading).
* `<output_GGUF_path>`: Path where the converted `.gguf` model will be saved.
* `<precision_type>`: Precision type — `'f32'`, `'f16'`, `'bf16'`, `'q8_0'`, `'tq1_0'`, `'tq2_0'`, `'auto'`.

### Example:

```bash
python3 convert_hf_to_gguf.py /root/autodl-tmp/finetune/models/qwen3-8b-qlora/merged --outfile /root/autodl-fs/qwen3-8b-fp16-agent.gguf --outtype f16
```

---

## Quantize the Model

### Command format:

```bash
./build/bin/llama-quantize <input_GGUF_path> <output_GGUF_path> <quantization_level>
```

* `<input_GGUF_path>`: Path to the unquantized `.gguf` file.
* `<output_GGUF_path>`: Path where the quantized `.gguf` file will be saved.
* `<quantization_level>`: For example `Q4_0`, `Q4_K_M`, `Q8_0`, etc., depending on your needs and hardware.

### Example:

```bash
./build/bin/llama-quantize \
  /root/autodl-fs/qwen3-8b-fp16-agent.gguf \
  /root/autodl-fs/qwen3-8b-q8_0-agent.gguf \
  Q8_0
```

---

## 7. Run Model Test

### Command format:

```bash
./build/bin/llama-run <GGUF_model_path>
```

* `<GGUF_model_path>`: Path to the GGUF model you want to test (can be original or quantized)

### Example:

```bash
./build/bin/llama-run /root/autodl-fs/qwen3-8b-fp16-agent.gguf
```

---
## 8. High-speed File Download from Server

## You can download directly from the service provider's data storage without keeping the machine running and paying fees

### Or
### Command format
```bash
lftp -u {username},{password} -p {port} sftp://{server_address} -e "set xfer:clobber true; pget -n {thread_count} {server_file_path} -o {local_file_name/path}; bye"
```
* `pget`: Use multi-threaded parallel download
* `-n`: Specify thread count (recommended 64+) (even 256 threads may perform better)
### Example
```bash
lftp -u root,askdjiwhakjd -p 27391 sftp://yourserver.com -e "set xfer:clobber true; pget -n 256 /root/autodl-fs/qwen3-8b-fp16-agent.gguf -o qwen3-8b-fp16-agent.gguf; bye"
```
---
