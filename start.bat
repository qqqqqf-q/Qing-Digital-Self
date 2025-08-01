@echo off

REM NVIDIA 4090 优化启动脚本
REM 为 Qwen3-8B QLoRA 微调进行完整优化

echo === NVIDIA 4090 优化启动脚本 ===
echo 平台: Windows
echo GPU: NVIDIA GeForce RTX 4090
echo 优化: 启用所有可用优化选项
echo.

REM 检查CUDA是否可用
echo 检查CUDA可用性...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到NVIDIA GPU或CUDA驱动
    pause
    exit /b 1
)

REM 显示GPU信息
echo GPU信息:
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name,memory.total --format^=csv^,noheader^,nounits ^| head -n 1') do set gpu_info=%%i
echo %gpu_info%
echo.

REM 设置环境变量
set CUDA_VISIBLE_DEVICES=0
set CUDA_LAUNCH_BLOCKING=0
set TORCH_LOGS=
set TORCHDYNAMO_DISABLE=1
set TORCH_COMPILE_DISABLE=1
set TRITON_DISABLE=1
set TORCH_USE_TRITON=0
set PYTORCH_DISABLE_TRITON=1
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2

REM 检查Unsloth是否已安装
echo 检查Unsloth安装状态...
python -c "import unsloth" >nul 2>&1
if %errorlevel% equ 0 (
    echo Unsloth: 已安装
    set USE_UNSLOTH=true
) else (
    echo Unsloth: 未安装，将使用普通4bit量化
    set USE_UNSLOTH=false
)
echo.

REM 构建训练命令
set CMD=python finetune/qlora_qwen3.py ^
  --use_unsloth %USE_UNSLOTH% ^
  --use_qlora true ^
  --data_path training_data.jsonl ^
  --output_dir finetune/models/qwen3-8b-qlora ^
  --per_device_train_batch_size 1 ^
  --gradient_accumulation_steps 16 ^
  --learning_rate 2e-4 ^
  --num_train_epochs 3 ^
  --lora_r 16 ^
  --lora_alpha 32 ^
  --lora_dropout 0.05 ^
  --logging_steps 20 ^
  --save_steps 200 ^
  --warmup_ratio 0.05 ^
  --lr_scheduler_type cosine ^
  --gradient_checkpointing ^
  --merge_and_save ^
  --fp16 true ^
  --optim adamw_torch_fused ^
  --dataloader_pin_memory false ^
  --dataloader_num_workers 0

REM 显示将要执行的命令
echo 将要执行的命令:
echo %CMD%
echo.

echo 环境变量设置:
echo   CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo   CUDA_LAUNCH_BLOCKING=%CUDA_LAUNCH_BLOCKING%
echo   TORCHDYNAMO_DISABLE=%TORCHDYNAMO_DISABLE%
echo   TRITON_DISABLE=%TRITON_DISABLE%
echo   USE_UNSLOTH=%USE_UNSLOTH%
echo.

echo 开始训练...
echo.

REM 执行训练
%CMD%

REM 检查训练结果
if %errorlevel% equ 0 (
    echo.
    echo === 训练成功完成 ===
    echo 模型已保存到: finetune/models/qwen3-8b-qlora
    if "%USE_UNSLOTH%" == "true" (
        echo 提示: 您已使用 Unsloth 加速训练，推理时也建议使用 Unsloth 加载模型以获得最佳性能
    )
) else (
    echo.
    echo === 训练失败 ===
    pause
    exit /b 1
)

echo.
pause