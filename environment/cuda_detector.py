"""
CUDA检测和PyTorch版本选择模块

检测系统CUDA版本，推荐合适的PyTorch版本
仅使用标准库实现
"""

import subprocess
import re
import platform
import time
from typing import Optional, Dict, List, Tuple


class CudaDetector:
    """CUDA检测器"""
    
    # PyTorch版本映射 (CUDA版本 -> PyTorch安装命令)
    PYTORCH_CUDA_MAP = {
        "12.6": "torch torchvision --index-url https://download.pytorch.org/whl/cu126",
        "12.8": "torch torchvision --index-url https://download.pytorch.org/whl/cu128", 
        "12.9": "torch torchvision --index-url https://download.pytorch.org/whl/cu129",
        "12.1": "torch==2.7.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121",
        "11.8": "torch==2.7.1+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118",
        "cpu": "torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    }
    
    # 推荐的PyTorch版本
    RECOMMENDED_PYTORCH_VERSION = "2.8"
    PREFERRED_CUDA_VERSIONS = ["12.6", "12.8", "12.9"]
    
    def __init__(self, logger=None):
        self.logger = logger
        self.system = platform.system().lower()
        # 缓存CUDA检测结果，避免重复检测
        self._cuda_version_cache = None
        self._cache_timestamp = 0
        
    def _log(self, level: str, message: str):
        """内部日志方法"""
        if self.logger:
            getattr(self.logger, level.lower())(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def detect_cuda_version(self) -> Optional[str]:
        """检测系统CUDA版本"""
        # 缓存机制：如果在30秒内已检测过，直接返回缓存结果
        current_time = time.time()
        if (self._cache_timestamp > 0 and 
            current_time - self._cache_timestamp < 30):
            return self._cuda_version_cache
            
        self._log("info", "正在检测CUDA版本...")
        
        try:
            # 尝试使用nvcc命令检测CUDA版本
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # 解析nvcc输出
                version_match = re.search(r'release (\d+\.\d+)', result.stdout)
                if version_match:
                    cuda_version = version_match.group(1)
                    self._log("info", f"通过nvcc检测到CUDA版本: {cuda_version}")
                    # 更新缓存
                    self._cuda_version_cache = cuda_version
                    self._cache_timestamp = current_time
                    return cuda_version
                    
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self._log("debug", "nvcc命令不可用，尝试nvidia-smi")
        
        try:
            # 尝试使用nvidia-smi检测
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                # 解析nvidia-smi输出中的CUDA版本
                cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
                if cuda_match:
                    cuda_version = cuda_match.group(1)
                    self._log("info", f"通过nvidia-smi检测到CUDA版本: {cuda_version}")
                    # 更新缓存
                    self._cuda_version_cache = cuda_version
                    self._cache_timestamp = current_time
                    return cuda_version
                    
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self._log("debug", "nvidia-smi命令不可用")
        
        self._log("warning", "未检测到CUDA，将使用CPU版本")
        # 缓存None结果
        self._cuda_version_cache = None
        self._cache_timestamp = current_time
        return None
    
    def get_gpu_info(self) -> List[Dict]:
        """获取GPU信息"""
        gpu_info = []
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) >= 3:
                            gpu_info.append({
                                'name': parts[0],
                                'memory': f"{parts[1]} MB",
                                'driver_version': parts[2]
                            })
                            
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self._log("debug", "无法获取GPU详细信息")
        
        return gpu_info
    
    def recommend_pytorch_version(self, cuda_version: Optional[str] = None) -> Dict:
        """推荐PyTorch版本和安装命令"""
        # 只有在没有提供cuda_version时才检测
        if cuda_version is None:
            cuda_version = self.detect_cuda_version()
        
        recommendation = {
            'cuda_version': cuda_version,
            'has_cuda': cuda_version is not None,
            'install_command': None,
            'note': ""
        }
        
        if cuda_version is None:
            # CPU版本
            recommendation['install_command'] = f"pip install {self.PYTORCH_CUDA_MAP['cpu']}"
            recommendation['note'] = "未检测到CUDA，使用CPU版本PyTorch"
            recommendation['recommended'] = True
            
        elif cuda_version in self.PYTORCH_CUDA_MAP:
            # 直接匹配的CUDA版本
            recommendation['install_command'] = f"pip install {self.PYTORCH_CUDA_MAP[cuda_version]}"
            recommendation['note'] = f"为CUDA {cuda_version}优化的PyTorch版本"
            recommendation['recommended'] = cuda_version in self.PREFERRED_CUDA_VERSIONS
            
        else:
            # 寻找最接近的CUDA版本
            cuda_float = float(cuda_version)
            best_match = None
            best_distance = float('inf')
            
            for supported_version in self.PYTORCH_CUDA_MAP.keys():
                if supported_version == 'cpu':
                    continue
                    
                try:
                    supported_float = float(supported_version)
                    distance = abs(cuda_float - supported_float)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = supported_version
                        
                except ValueError:
                    continue
            
            if best_match:
                recommendation['install_command'] = f"pip install {self.PYTORCH_CUDA_MAP[best_match]}"
                recommendation['note'] = (f"CUDA {cuda_version}不直接支持，"
                                        f"使用最接近的版本CUDA {best_match}")
                recommendation['recommended'] = False
            else:
                # 回退到CPU版本
                recommendation['install_command'] = f"pip install {self.PYTORCH_CUDA_MAP['cpu']}"
                recommendation['note'] = f"CUDA {cuda_version}不支持，回退到CPU版本"
                recommendation['recommended'] = False
        
        return recommendation
    
    def get_all_pytorch_versions(self) -> List[Dict]:
        """获取所有可用的PyTorch版本选项"""
        versions = []
        
        for cuda_ver, install_cmd in self.PYTORCH_CUDA_MAP.items():
            if cuda_ver == 'cpu':
                versions.append({
                    'cuda_version': 'CPU',
                    'description': 'CPU版本 (无GPU加速)',
                    'install_command': f"pip install {install_cmd}",
                    'recommended': False
                })
            else:
                is_preferred = cuda_ver in self.PREFERRED_CUDA_VERSIONS
                versions.append({
                    'cuda_version': cuda_ver,
                    'description': f'CUDA {cuda_ver} 版本',
                    'install_command': f"pip install {install_cmd}",
                    'recommended': is_preferred
                })
        
        return versions
    
    def detect_pytorch_installation(self) -> Dict:
        """检测现有PyTorch安装"""
        try:
            result = subprocess.run(
                ["python", "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda if torch.cuda.is_available() else 'CPU')"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 3:
                    torch_version = lines[0]
                    cuda_available = lines[1] == 'True'
                    cuda_version = lines[2] if cuda_available else None
                    
                    return {
                        'installed': True,
                        'version': torch_version,
                        'cuda_available': cuda_available,
                        'cuda_version': cuda_version
                    }
            
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        return {'installed': False}
    
    def get_system_info(self, detect_fresh: bool = True) -> Dict:
        """获取完整的系统信息"""
        if detect_fresh:
            cuda_version = self.detect_cuda_version()
        else:
            # 简单检测，不输出详细日志
            cuda_version = None
            try:
                result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version_match = re.search(r'release (\d+\.\d+)', result.stdout)
                    if version_match:
                        cuda_version = version_match.group(1)
            except:
                try:
                    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
                        if cuda_match:
                            cuda_version = cuda_match.group(1)
                except:
                    pass
        
        gpu_info = self.get_gpu_info()
        pytorch_info = self.detect_pytorch_installation()
        # 避免重复检测，直接传入已检测的cuda_version
        recommendation = self.recommend_pytorch_version(cuda_version)
        
        return {
            'system': platform.system(),
            'cuda': {
                'version': cuda_version,
                'available': cuda_version is not None
            },
            'gpus': gpu_info,
            'pytorch': pytorch_info,
            'recommendation': recommendation
        }