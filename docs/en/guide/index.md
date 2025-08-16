## Project Introduction

## This is a digital avatar project, the core idea is to **use C2C chat records as a dataset to fine-tune large models**, making the model restore your unique expression style and chat patterns as much as possible.
<p align="center">
  <img src="https://img.shields.io/badge/Downloads-1-00bfff?style=for-the-badge" style="display:inline-block;margin-right:8px;">
  <img src="https://img.shields.io/github/stars/qqqqqf-q/Qing-Digital-Self?style=for-the-badge&color=ff69b4" style="display:inline-block;margin-right:8px;">
  <img src="https://img.shields.io/badge/Status-MVP-ff69b4?style=for-the-badge" style="display:inline-block;margin-right:8px;">
  <img src="https://img.shields.io/badge/Version-v0.1.1-9370DB?style=for-the-badge" style="display:inline-block;margin-right:8px;">
  <img src="https://img.shields.io/github/license/qqqqqf-q/Qing-Digital-Self?style=for-the-badge&color=8A2BE2" style="display:inline-block;">
</p>


## This project includes a **complete tutorial**, covering:

* Decrypting and processing QQ databases
* Cleaning and converting chat data
* QLoRA fine-tuning workflow
* Testing and using fine-tuned models
* Accelerating training with Unsloth!

I know there are already quite a few similar projects out there, but maybe my tutorial, workflow, and code implementation can offer you something different or spark new ideas.
If you find it useful, feel free to give it a star — it’ll make me happy!

The project still has its shortcomings:

* I’m not sure what they are for now
* (If you run into issues, feel free to open an Issue)
* But it’s already capable of fine-tuning Qwen3-8B with FP8 precision on a 4090 24G GPU (tested and working)

**If you also want to create your own digital persona, give it a try!**

——
X: [@qqqqqf5](https://twitter.com/qqqqqf5)
Email: qingf622@outlook.com
Github:[@qqqqqf-q](https://github.com/qqqqqf-q)
---


## Project Version
# V 0.1.2
## ~~Warning~~ Good News
* This version's Qlora_qwen3.py has been tested on 4090 (generate_training_data_llm.py+run_finetune.py)
* Data cleaning has also been tested on the current version
## TODO
* [Completed but untested] Add support for OSS models (and MXFP4? This is a 50-series computation, seems I still can't test it)
> Challenges: 1.MOE models 2.Non-original Qwen series models 3.My 3080 seems unable to test locally (neither fine-tuning nor MXFP4)
> Well, it's actually not difficult at all, just been working on other projects these days
* [In planning] Add WebUI support
>
> 1. MOE models
> 2. Non-Qwen series models
> 3. My 3080 seems unable to run local tests (for either fine-tuning or MXFP4)
>    Well, it’s actually not that hard — I’ve just been busy with other projects these past few days.

## Changelog

> Written in the commits — don’t feel like writing it here.
