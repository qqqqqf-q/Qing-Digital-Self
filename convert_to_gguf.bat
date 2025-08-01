@echo off
echo Qing-Agent 模型转换工具 - GGUF格式转换
echo =====================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未检测到Python，请先安装Python
    pause
    exit /b 1
)

REM 检查是否提供了参数
if "%1"=="" (
    echo 使用方法:
    echo   convert_to_gguf.bat [模型路径] [量化类型] [输出目录]
    echo.
    echo 示例:
    echo   convert_to_gguf.bat ./finetune/models/qwen3-8b-qlora q4_k_m ./gguf_models
    echo.
    echo 支持的量化类型:
    echo   q4_0, q4_1, q5_0, q5_1, q8_0
    echo   q2_k, q3_k, q4_k, q5_k, q6_k, q8_k, q4_k_m
    pause
    exit /b 1
)

set MODEL_PATH=%1
set QUANT_TYPE=%2
set OUTPUT_DIR=%3

if "%QUANT_TYPE%"=="" set QUANT_TYPE=q4_k_m
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=./gguf_models

echo 正在转换模型...
echo 模型路径: %MODEL_PATH%
echo 量化类型: %QUANT_TYPE%
echo 输出目录: %OUTPUT_DIR%
echo.

python convert_to_gguf.py --model_path "%MODEL_PATH%" --quantization "%QUANT_TYPE%" --output_dir "%OUTPUT_DIR%" --install_deps

if errorlevel 1 (
    echo.
    echo 转换失败！请检查错误信息
) else (
    echo.
    echo 转换完成！
)

pause