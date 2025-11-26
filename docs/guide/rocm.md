## 安装普通ROCm环境
请访问此处[AMD Wiki](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)

总命令(基于Ubuntu24.04)
```bash
wget https://repo.radeon.com/amdgpu-install/7.1/ubuntu/noble/amdgpu-install_7.1.70100-1_all.deb
sudo apt install ./amdgpu-install_7.1.70100-1_all.deb
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm

wget https://repo.radeon.com/amdgpu-install/7.1/ubuntu/noble/amdgpu-install_7.1.70100-1_all.deb
sudo apt install ./amdgpu-install_7.1.70100-1_all.deb
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms

docker pull rocm/pytorch:latest

docker run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 8G \
    rocm/pytorch:latest
```

## 安装ROCm-vLLM
### 方法1.Docker拉取AMD Infinity Hub官方镜像

[此处为AMD Infinity Hub的文档](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm.html?model=pyt_vllm_llama-2-70b)
* 命令
```bash
docker pull rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103
```
### 方法2.从`rocm/pytorch`镜像自己编译vLLM
* 使用此命令复制`rocm/pytorch:latest`镜像
```bash
docker run --name rocm-vllm \
  --hostname rocm-vllm \
  --network bridge \
  --device /dev/kfd:/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -it \
  rocm/pytorch:latest
```
### vLLM搭建
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

export VLLM_TARGET_DEVICE=rocm
pip install -r requirements/rocm.txt
pip install -e . --no-build-isolation
```
> 也许需要手动设置芯片组型号,例如`export PYTORCH_ROCM_ARCH="gfx1030`