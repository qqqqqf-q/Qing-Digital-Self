## Project Introduction

## This is a digital avatar project, the core idea is to **use QQ's C2C chat records as a dataset to fine-tune large models**, making the model restore your unique expression style and chat patterns as much as possible.
<p align="center">
  <img src="https://img.shields.io/badge/Downloads-1-00bfff?style=for-the-badge" style="display:inline-block;margin-right:8px;">
  <img src="https://img.shields.io/github/stars/qqqqqf-q/Qing-Digital-Self?style=for-the-badge&color=ff69b4" style="display:inline-block;margin-right:8px;">
  <img src="https://img.shields.io/badge/Status-MVP-ff69b4?style=for-the-badge" style="display:inline-block;margin-right:8px;">
  <img src="https://img.shields.io/badge/Version-v0.1.1-9370DB?style=for-the-badge" style="display:inline-block;margin-right:8px;">
  <img src="https://img.shields.io/github/license/qqqqqf-q/Qing-Digital-Self?style=for-the-badge&color=8A2BE2" style="display:inline-block;">
</p>


## The project includes **complete tutorials**, including:

* QQ database decryption and processing
* Chat data cleaning and conversion
* QLora fine-tuning process
* Testing and using fine-tuned models
* Using unsloth to accelerate training!

I know there are already quite a few similar projects, but maybe my tutorials, processes, and code implementations can give you some different help or inspiration. If it's useful to you, feel free to give it a star, I'll be very happy!

Currently this project still has many shortcomings:

* Don't know what shortcomings there are yet
* (If you have issues, welcome to open Issues)
* But it can already fine-tune Qwen3-8B with fp8 precision on 4090 24G graphics card (tested and working)

**If you also want to create your own digital avatar, come and try it!**

——
X: [@qqqqqf5](https://twitter.com/qqqqqf5)

---

<a href="https://qqqqqf-q.github.io/Qing-Digital-Self/">
  <img src="https://cdn.nodeimage.com/i/MfTsvmkJD2dQj9c9XZg9XXXS6CYwZBvx.png" alt="Quick Start Button" width="140" style="margin-top: 1em;">
</a>

---

## Project Version
# V 0.1.1(MVP)
## ~~Warning~~ Good News
* The Qlora_qwen3.py of this version has been tested on 4090 (generate_training_data_llm.py+run_finetune.py)
* Data cleaning has also been tested on real machines (current version)
## TODO
* [Completed] Add unsloth support (important! can speed up fine-tuning)
> ↑ ↑ ↑ Updated on 2025/8/4, now supported (actually it worked before, might be my graphics card issue?)
* [Completed but not tested] Add support for MoE models