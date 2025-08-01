import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    ap = argparse.ArgumentParser(description="Infer with LoRA adapter or merged full model")
    ap.add_argument("--base_dir", type=str, default="qwen3-8b-base", help="Base model dir (for LoRA mode)")
    ap.add_argument("--adapter_dir", type=str, default="finetune/models/qwen3-8b-qlora", help="LoRA adapter or merged dir")
    ap.add_argument("--merged", type=lambda x: str(x).lower() == "true", default=True, help="If true, load merged full weights from adapter_dir/merged")
    ap.add_argument("--prompt", type=str, default="你好，给我一个自我介绍。")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--do_sample", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--trust_remote_code", type=lambda x: str(x).lower() == "true", default=True)
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.merged:
        model_dir = os.path.join(args.adapter_dir, "merged")
        print(f"Loading merged full model from: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=args.trust_remote_code)
    else:
        # 加载 base + LoRA 适配器
        from peft import PeftModel
        print(f"Loading base model from: {args.base_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True, trust_remote_code=args.trust_remote_code)
        base = AutoModelForCausalLM.from_pretrained(args.base_dir, device_map="auto", trust_remote_code=args.trust_remote_code)
        print(f"Loading LoRA adapter from: {args.adapter_dir}")
        model = PeftModel.from_pretrained(base, args.adapter_dir, device_map="auto")
        model = model.merge_and_unload()  # 推理期可合并到内存中，提高速度（可选）

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    print("==== Generation ====")
    print(out)

if __name__ == "__main__":
    main()