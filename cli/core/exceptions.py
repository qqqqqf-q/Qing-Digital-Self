"""
CLI 自定义异常类

定义了所有CLI操作中可能抛出的异常类型，提供清晰的错误分类和处理机制。
"""


class CLIError(Exception):
    """CLI基础异常类"""
    
    def __init__(self, message: str, error_code: int = 1):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        
    def __str__(self) -> str:
        return f"CLI错误 [{self.error_code}]: {self.message}"


class ConfigurationError(CLIError):
    """配置相关异常"""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, error_code=10)
        self.config_key = config_key
        
    def __str__(self) -> str:
        if self.config_key:
            return f"配置错误 [{self.error_code}]: 配置项 '{self.config_key}' - {self.message}"
        return f"配置错误 [{self.error_code}]: {self.message}"


class DataProcessingError(CLIError):
    """数据处理相关异常"""
    
    def __init__(self, message: str, file_path: str = None, line_number: int = None):
        super().__init__(message, error_code=20)
        self.file_path = file_path
        self.line_number = line_number
        
    def __str__(self) -> str:
        location_info = ""
        if self.file_path:
            location_info = f" 文件: {self.file_path}"
            if self.line_number:
                location_info += f" 行: {self.line_number}"
        
        return f"数据处理错误 [{self.error_code}]: {self.message}{location_info}"


class TrainingError(CLIError):
    """训练相关异常"""
    
    def __init__(self, message: str, checkpoint_path: str = None, step: int = None):
        super().__init__(message, error_code=30)
        self.checkpoint_path = checkpoint_path
        self.step = step
        
    def __str__(self) -> str:
        training_info = ""
        if self.step:
            training_info = f" 训练步数: {self.step}"
        if self.checkpoint_path:
            training_info += f" 检查点: {self.checkpoint_path}"
            
        return f"训练错误 [{self.error_code}]: {self.message}{training_info}"


class InferenceError(CLIError):
    """推理相关异常"""
    
    def __init__(self, message: str, model_path: str = None):
        super().__init__(message, error_code=40)
        self.model_path = model_path
        
    def __str__(self) -> str:
        model_info = ""
        if self.model_path:
            model_info = f" 模型路径: {self.model_path}"
            
        return f"推理错误 [{self.error_code}]: {self.message}{model_info}"


class ValidationError(CLIError):
    """参数验证异常"""
    
    def __init__(self, message: str, parameter_name: str = None, parameter_value: str = None):
        super().__init__(message, error_code=50)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        
    def __str__(self) -> str:
        param_info = ""
        if self.parameter_name:
            param_info = f" 参数: {self.parameter_name}"
            if self.parameter_value:
                param_info += f" 值: {self.parameter_value}"
                
        return f"参数验证错误 [{self.error_code}]: {self.message}{param_info}"


class ResourceError(CLIError):
    """资源相关异常 (内存、磁盘、GPU等)"""
    
    def __init__(self, message: str, resource_type: str = None, required: str = None, available: str = None):
        super().__init__(message, error_code=60)
        self.resource_type = resource_type
        self.required = required
        self.available = available
        
    def __str__(self) -> str:
        resource_info = ""
        if self.resource_type:
            resource_info = f" 资源类型: {self.resource_type}"
            if self.required and self.available:
                resource_info += f" 需要: {self.required}, 可用: {self.available}"
                
        return f"资源错误 [{self.error_code}]: {self.message}{resource_info}"


class DependencyError(CLIError):
    """依赖相关异常"""
    
    def __init__(self, message: str, dependency_name: str = None, required_version: str = None, current_version: str = None):
        super().__init__(message, error_code=70)
        self.dependency_name = dependency_name
        self.required_version = required_version
        self.current_version = current_version
        
    def __str__(self) -> str:
        dep_info = ""
        if self.dependency_name:
            dep_info = f" 依赖: {self.dependency_name}"
            if self.required_version:
                dep_info += f" 要求版本: {self.required_version}"
            if self.current_version:
                dep_info += f" 当前版本: {self.current_version}"
                
        return f"依赖错误 [{self.error_code}]: {self.message}{dep_info}"


class NetworkError(CLIError):
    """网络相关异常"""
    
    def __init__(self, message: str, url: str = None, status_code: int = None):
        super().__init__(message, error_code=80)
        self.url = url
        self.status_code = status_code
        
    def __str__(self) -> str:
        network_info = ""
        if self.url:
            network_info = f" URL: {self.url}"
        if self.status_code:
            network_info += f" 状态码: {self.status_code}"
            
        return f"网络错误 [{self.error_code}]: {self.message}{network_info}"


class FileOperationError(CLIError):
    """文件操作相关异常"""
    
    def __init__(self, message: str, file_path: str = None, operation: str = None):
        super().__init__(message, error_code=90)
        self.file_path = file_path
        self.operation = operation
        
    def __str__(self) -> str:
        file_info = ""
        if self.file_path:
            file_info = f" 文件: {self.file_path}"
        if self.operation:
            file_info += f" 操作: {self.operation}"
            
        return f"文件操作错误 [{self.error_code}]: {self.message}{file_info}"