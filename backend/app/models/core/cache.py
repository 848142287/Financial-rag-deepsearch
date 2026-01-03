"""
统一的缓存相关数据模型

包括：CacheEntry（缓存条目）、CacheLevel（缓存级别）
"""

from dataclasses import dataclass, field
from enum import Enum

class CacheLevel(Enum):
    """缓存级别"""
    MEMORY = "memory"  # 内存缓存（最快）
    REDIS = "redis"  # Redis缓存（中等）
    DATABASE = "database"  # 数据库缓存（最慢）

@dataclass
class CacheEntry:
    """
    统一的缓存条目类

    替代在7个文件中重复定义的CacheEntry类
    """
    key: str
    value: Any
    ttl: int = 3600  # 生存时间（秒）
    level: CacheLevel = CacheLevel.MEMORY
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    metadata: dict = field(default_factory=dict)

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl <= 0:
            return False  # 永不过期
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl

    def touch(self):
        """更新访问时间和计数"""
        self.accessed_at = datetime.now()
        self.access_count += 1

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "key": self.key,
            "value": self.value,
            "ttl": self.ttl,
            "level": self.level.value,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "metadata": self.metadata
        }

@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "hit_rate": f"{self.hit_rate:.2%}"
        }

__all__ = [
    'CacheLevel',
    'CacheEntry',
    'CacheStats',
]
