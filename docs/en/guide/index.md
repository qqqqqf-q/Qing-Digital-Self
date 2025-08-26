## Project Introduction

## This is a digital avatar project, the core idea is to **use C2C chat records as a dataset to fine-tune large models**, making the model restore your unique expression style and chat patterns as much as possible.

## This project is a personal digital twin built by fine-tuning a large language model on your own chat history. The goal is to recreate your unique style of expression and conversational behavior with high fidelity.
<p align="center">
  <img src="https://img.shields.io/badge/Downloads-1-00bfff?style=for-the-badge">
<img src="https://img.shields.io/github/stars/qqqqqf-q/Qing-Digital-Self?style=for-the-badge&color=ff69b4">
<img src="https://img.shields.io/badge/Status-MVP-ff69b4?style=for-the-badge">
<img src="https://img.shields.io/badge/Version-v0.1.4Dev-9370DB?style=for-the-badge">
<img src="https://img.shields.io/github/license/qqqqqf-q/Qing-Digital-Self?style=for-the-badge&color=8A2BE2">

</p>

# The project includes bilingual support.
## Project includes bilingual support

## Chinese Documentation
<a href="https://qqqqqf-q.github.io/Qing-Digital-Self/">
  <img src="https://cdn.nodeimage.com/i/MfTsvmkJD2dQj9c9XZg9XXXS6CYwZBvx.png" alt="Quick Start Button" width="140" style="margin-top: 1em;">
</a>

## English Documents
<a href="https://qqqqqf-q.github.io/Qing-Digital-Self/en/">
  <img src="https://cdn.nodeimage.com/i/wy2ngEGv8sYYpcJ0zdIlMnHgm8lLmbIA.png" alt="Quick Start Button" width="140" style="margin-top: 1em;">
</a>

## This project includes a **complete tutorial**, covering:

* Decrypting and processing QQ databases
* Cleaning and converting chat data
* QLoRA fine-tuning workflow
* Testing and using fine-tuned models
* Accelerating training with Unsloth!

I know there are already quite a few similar projects out there, but maybe my tutorial, workflow, and code implementation can offer you something different or spark new ideas.
If you find it useful, feel free to give it a star — it'll make me happy!

The project still has its shortcomings:

* I'm not sure what they are for now
* (If you run into issues, feel free to open an Issue)
* But it's already capable of fine-tuning Qwen3-8B with FP8 precision on a 4090 24G GPU (tested and working)

**"Some code referenced from Weclone"**
**If you also want to create your own digital persona, give it a try!**

——
X: [@qqqqqf5](https://twitter.com/qqqqqf5)
Email: qingf622@outlook.com
Github:[@qqqqqf-q](https://github.com/qqqqqf-q)
---


## Project Version
# V 0.1.4 Develop

# Project Status
* Due to many code refactorings in version 0.1.4
* There might be more bugs
* Welcome all developers to submit Issues and PRs
* Contribute to this small project

# Development Issues
* The cli train and data convert functions have issues, for now you can only use the old version calls
* New LLM cleaning functionality is under development (needs to include LLM scoring, LLM output usable segments, etc.)
* Support for more Parsers and tutorials (including Telegram, Wechat, etc.)
* Fine-tuning script needs refactoring (thinking about whether to continue with Qlora+Unsloth or switch to Llama Factory)
* Some documentation hasn't been updated due to project refactoring
* Refactored parts don't have bilingual support yet
* todo1. Add server API in preparation for WebUI
