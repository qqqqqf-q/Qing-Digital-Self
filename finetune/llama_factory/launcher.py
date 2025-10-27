#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA Factory WebUI 启动器（调试/验证阶段用）

说明：
- 不改动 CLI；此模块提供最小可用的 WebUI 启动入口。
- 不改动数据清洗/导出流程；直接使用本项目生成的 ChatML JSONL。
- 仅依赖标准库，避免额外耦合。
"""

from __future__ import annotations

import os
import sys
import subprocess
import shutil
from dataclasses import dataclass
from typing import Optional


@dataclass
class WebUILaunchConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    open_browser: bool = True
    working_dir: Optional[str] = None  # 指定运行目录（可选）


def _has_llamafactory() -> bool:
    """检查 llamafactory 是否可用（优先检测 CLI，其次模块）。"""
    if shutil.which("llamafactory-cli"):
        return True
    try:
        __import__("llamafactory")
        return True
    except Exception:
        return False


def _python_exec() -> str:
    """返回当前 Python 可执行文件路径。"""
    return sys.executable or "python"


def _build_env() -> dict:
    """构建子进程环境变量。保留现有变量，按需注入。"""
    env = os.environ.copy()
    # 可在此注入自定义路径或缓存目录
    # env.setdefault("HF_HOME", os.path.abspath(".cache/hf"))
    return env


def launch_webui(cfg: WebUILaunchConfig) -> int:
    """启动 LLaMA Factory WebUI。

    返回：子进程退出码。0 表示成功退出。
    """
    if not _has_llamafactory():
        _print_install_help()
        return 1

    # 启动前生成一次 finetune 配置，供调试/追踪
    try:
        cwd = cfg.working_dir or os.getcwd()
        # 确保项目根目录在 sys.path，便于导入 utils 包
        project_root_candidates = [
            cwd,
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        ]
        for cand in project_root_candidates:
            if os.path.isdir(os.path.join(cand, "utils")) and cand not in sys.path:
                sys.path.insert(0, cand)
        from utils.config.finetune_yaml import write_finetune_yaml
        outp = write_finetune_yaml(os.path.join(cwd, "config", "finetune-config.yaml"))
        print(f"已生成配置: {outp}")
    except Exception as e:
        print(f"生成 finetune 配置失败: {e}")

    # 优先使用官方 CLI：llamafactory-cli webui
    # 回退方案：python -m llamafactory.cli webui（避免依赖 entry_points 暴露）
    env = _build_env()
    # Gradio 环境变量
    env["GRADIO_SERVER_NAME"] = cfg.host
    env["GRADIO_SERVER_PORT"] = str(cfg.port)
    # 是否公开分享链接由 GRADIO_SHARE 控制
    if cfg.share:
        env["GRADIO_SHARE"] = "1"
    else:
        env.setdefault("GRADIO_SHARE", "0")

    cli = shutil.which("llamafactory-cli")
    if cli:
        args = [cli, "webui"]
        launcher_desc = cli + " webui"
        py = _python_exec()
    else:
        py = _python_exec()
        args = [py, "-m", "llamafactory.cli", "webui"]
        launcher_desc = f"{py} -m llamafactory.cli webui"

    cwd = cfg.working_dir or os.getcwd()
    print("==== LLaMA Factory WebUI ====")
    print(f"Launcher : {launcher_desc}")
    print(f"Python   : {py}")
    print(f"Workdir  : {cwd}")
    print(f"URL      : http://{cfg.host}:{cfg.port}")
    if cfg.share:
        print("Share    : enabled (waiting for public URL)")
    print("================================")

    try:
        proc = subprocess.Popen(args, cwd=cwd, env=env)
        proc.wait()
        return int(proc.returncode or 0)
    except KeyboardInterrupt:
        try:
            proc.terminate()  # 尽量优雅退出
        except Exception:
            pass
        return 130
    except FileNotFoundError:
        print("未找到可执行文件，请检查 Python/llamafactory-cli 是否在 PATH 中。")
        return 2
    except Exception as e:
        print(f"启动失败: {e}")
        return 3


def _print_install_help() -> None:
    """打印 llamafactory 安装指引（跨平台）。"""
    print("未检测到 llamafactory 包，请先安装：\n")
    print("1) 确保已安装匹配 CUDA 的 torch：")
    print("   - pip:    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio")
    print("   - conda:  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    print("2) 安装 LLaMA Factory：")
    print("   - pip install -U llamafactory")
    print("3) 启动 WebUI：")
    print("   - python -m llamafactory.webui")


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="LLaMA Factory WebUI 启动器")
    p.add_argument("--host", default="0.0.0.0", help="监听地址")
    p.add_argument("--port", default=7860, type=int, help="监听端口")
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器（受上游实现限制，可能无效）")
    p.add_argument("--share", action="store_true", help="开启公网分享链接（Gradio 隧道）")
    p.add_argument("--workdir", default=None, help="工作目录(可选)")
    args = p.parse_args(argv)

    cfg = WebUILaunchConfig(
        host=args.host,
        port=args.port,
        share=args.share,
        open_browser=not args.no_browser,
        working_dir=args.workdir,
    )
    return launch_webui(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
