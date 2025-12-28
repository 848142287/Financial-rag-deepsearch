"""
Base Caching Abstractions

Defines the foundational abstractions for the unified caching system including
cache levels, policies, and base cache implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import hashlib
import json
import logging
import pickle
import time
import threading

logger = logging.getLogger(__name__)


class CacheLevel(str, Enum):
    """Cache levels in the hierarchy"""
    L1_MEMORY = "l1_memory"      # Fastest, smallest, local
    L2_REDIS = "l2_redis"        # Fast, distributed
    L3_DISK = "l3_disk"          # Slower, persistent
    L4_DATABASE = "l4_database"  # Slowest, fallback


class CachePolicy(str, Enum):
    """Cache eviction policies"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    FIFO = "fifo"                  # First In, First Out
    TTL = "ttl"                    # Time To Live
    SIZE_BASED = "size_based"      # Based on memory size


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    ttl: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate size after initialization"""
        self.size_bytes = self._calculate_size()
        self.last_accessed = datetime.utcnow()

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl

    def touch(self):
        """Update last access time and count"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def _calculate_size(self) -> int:
        """Calculate approximate size in bytes"""
        try:
            serialized = pickle.dumps(self.value)
            return len(serialized) + len(self.key.encode())
        except Exception:
            return len(str(self.value).encode()) + len(self.key.encode())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'key': self.key,
            'value': self.value,
            'ttl': self.ttl,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'size_bytes': self.size_bytes,
            'tags': self.tags,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary"""
        entry = cls(
            key=data['key'],
            value=data['value'],
            ttl=data.get('ttl'),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )
        entry.created_at = datetime.fromisoformat(data['created_at'])
        entry.last_accessed = datetime.fromisoformat(data['last_accessed'])
        entry.access_count = data.get('access_count', 0)
        entry.size_bytes = data.get('size_bytes', 0)
        return entry


@dataclass
class CacheConfig:
    """Configuration for cache instances"""
    max_size: Optional[int] = None  # Maximum number of entries
    max_memory_mb: Optional[int] = None  # Maximum memory in MB
    ttl: Optional[float] = None  # Default TTL in seconds
    policy: CachePolicy = CachePolicy.LRU
    cleanup_interval: float = 60.0  # Cleanup interval in seconds
    enable_stats: bool = True
    enable_compression: bool = False
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    serializer: str = "pickle"  # pickle, json, msgpack

    def __post_init__(self):
        """Validate configuration"""
        if self.max_size is not None and self.max_size <= 0:
            raise ValueError("max_size must be positive")
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    total_entries: int = 0
    memory_usage_mb: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate"""
        total_requests = self.hits + self.misses
        return self.hits / total_requests if total_requests > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Calculate miss rate"""
        total_requests = self.hits + self.misses
        return self.misses / total_requests if total_requests > 0 else 0.0

    def reset(self):
        """Reset all metrics"""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0


class BaseCache(ABC):
    """Base cache implementation"""

    def __init__(self, config: CacheConfig, level: CacheLevel):
        self.config = config
        self.level = level
        self.metrics = CacheMetrics()
        self._lock = threading.RLock() if not hasattr(self, '_is_async') else None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value with optional TTL"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete entry by key"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all entries"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get keys matching pattern"""
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get number of entries"""
        pass

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values"""
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def set_many(self, items: Dict[str, Any], ttl: Optional[float] = None) -> Dict[str, bool]:
        """Set multiple values"""
        result = {}
        for key, value in items.items():
            result[key] = await self.set(key, value, ttl)
        return result

    async def delete_many(self, keys: List[str]) -> Dict[str, bool]:
        """Delete multiple entries"""
        result = {}
        for key in keys:
            result[key] = await self.delete(key)
        return result

    async def get_or_set(self, key: str, factory: Callable[[], Awaitable[Any]],
                        ttl: Optional[float] = None) -> Any:
        """Get value or set if not exists"""
        value = await self.get(key)
        if value is not None:
            return value

        # Generate value
        new_value = await factory()
        await self.set(key, new_value, ttl)
        return new_value

    async def get_ttl(self, key: str) -> Optional[float]:
        """Get remaining TTL for key"""
        # Default implementation - override in subclasses
        return None

    async def touch(self, key: str) -> bool:
        """Update access time for key"""
        # Default implementation - override in subclasses
        return False

    async def increment(self, key: str, delta: int = 1) -> Optional[int]:
        """Increment numeric value"""
        # Default implementation - override in subclasses
        value = await self.get(key)
        if value is None:
            return None
        if isinstance(value, (int, float)):
            new_value = value + delta
            await self.set(key, new_value)
            return new_value
        return None

    async def get_stats(self) -> CacheMetrics:
        """Get cache statistics"""
        return self.metrics

    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        self._is_running = True
        while self._is_running:
            try:
                await self.cleanup_expired()
                await asyncio.sleep(self.config.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(self.config.cleanup_interval)

    async def cleanup_expired(self) -> int:
        """Clean up expired entries - override in subclasses"""
        return 0

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value based on configuration"""
        try:
            if self.config.serializer == "json":
                return json.dumps(value, default=str).encode()
            elif self.config.serializer == "pickle":
                return pickle.dumps(value)
            else:
                raise ValueError(f"Unsupported serializer: {self.config.serializer}")
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value based on configuration"""
        try:
            if self.config.serializer == "json":
                return json.loads(data.decode())
            elif self.config.serializer == "pickle":
                return pickle.loads(data)
            else:
                raise ValueError(f"Unsupported serializer: {self.config.serializer}")
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_cleanup_task()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop_cleanup_task()


class CacheException(Exception):
    """Base cache exception"""
    pass


class CacheKeyException(CacheException):
    """Key-related exceptions"""
    pass


class CacheSerializationException(CacheException):
    """Serialization-related exceptions"""
    pass


class CacheCapacityException(CacheException):
    """Capacity-related exceptions"""
    pass


class CacheConnectionException(CacheException):
    """Connection-related exceptions"""
    pass