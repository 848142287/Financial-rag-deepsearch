"""
统一的基础配置类

避免在多个文件中重复定义Config类
"""

@dataclass
class BaseConfig:
    """
    基础配置类

    所有配置类的基类，提供统一的配置管理接口
    """

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """从字典创建配置"""
        return cls(**data)

    def update(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def validate(self) -> bool:
        """验证配置"""
        return True

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}({self.to_dict()})"

@dataclass
class ServiceConfig(BaseConfig):
    """服务配置基类"""

    enabled: bool = True
    debug: bool = False
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

    def validate(self) -> bool:
        """验证服务配置"""
        if self.timeout < 0:
            raise ValueError("timeout must be >= 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        return True

@dataclass
class CacheConfig(BaseConfig):
    """缓存配置"""

    enabled: bool = True
    ttl: int = 3600  # 默认1小时
    max_size: int = 1000
    cleanup_interval: int = 300  # 5分钟

    def validate(self) -> bool:
        """验证缓存配置"""
        if self.ttl < 0:
            raise ValueError("ttl must be >= 0")
        if self.max_size < 0:
            raise ValueError("max_size must be >= 0")
        return True

__all__ = [
    'BaseConfig',
    'ServiceConfig',
    'CacheConfig',
]
