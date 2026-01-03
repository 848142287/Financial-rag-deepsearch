"""
统一缓存管理器
提供Redis-based缓存功能，支持多级缓存策略
"""

import json
import pickle
from enum import Enum
from typing import Any, Optional, Dict

from app.core.structured_logging import get_structured_logger
from app.core.redis_client import get_redis_client

logger = get_structured_logger(__name__)

class CacheLevel(Enum):
    """缓存级别"""
    MEMORY = "memory"      # 内存缓存（最快，容量小）
    REDIS = "redis"        # Redis缓存（快速，容量中等，支持持久化）
    DISTRIBUTED = "distributed"  # 分布式缓存（较慢，容量大，支持集群）

class UnifiedCacheManager:
    """统一缓存管理器 - 基于Redis实现"""

    def __init__(self, default_ttl: int = 3600, cache_level: CacheLevel = CacheLevel.REDIS):
        """
        初始化缓存管理器

        Args:
            default_ttl: 默认TTL（秒），默认1小时
            cache_level: 缓存级别
        """
        self.default_ttl = default_ttl
        self.cache_level = cache_level
        self._redis = None

    async def _get_redis(self):
        """延迟获取Redis连接"""
        if self._redis is None:
            self._redis = await get_redis_client()
        return self._redis

    def _serialize_value(self, value: Any) -> str:
        """序列化值"""
        try:
            # 尝试JSON序列化（更快、可读）
            return json.dumps(value)
        except (TypeError, ValueError):
            # 对于无法JSON序列化的对象，使用pickle
            return pickle.dumps(value)

    def _deserialize_value(self, data: str) -> Any:
        """反序列化值"""
        try:
            # 尝试JSON反序列化
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            # 尝试pickle反序列化
            try:
                return pickle.loads(data)
            except Exception:
                # 如果都失败，返回原始字符串
                return data

    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存的值，如果不存在或已过期返回None
        """
        try:
            redis = await self._get_redis()

            # 根据缓存级别处理
            if self.cache_level == CacheLevel.REDIS:
                # Redis缓存
                data = await redis.get(key)
                if data:
                    return self._deserialize_value(data)
                return None

            elif self.cache_level == CacheLevel.MEMORY:
                # 内存缓存（使用Redis的内存缓存）
                data = await redis.get(f"memory:{key}")
                if data:
                    return self._deserialize_value(data)
                return None

            else:  # CacheLevel.DISTRIBUTED
                # 分布式缓存
                data = await redis.get(f"dist:{key}")
                if data:
                    return self._deserialize_value(data)
                return None

        except Exception as e:
            logger.error(f"获取缓存失败 [key={key}]: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），如果为None则使用default_ttl

        Returns:
            是否设置成功
        """
        try:
            redis = await self._get_redis()

            if ttl is None:
                ttl = self.default_ttl

            serialized_value = self._serialize_value(value)

            # 根据缓存级别处理
            if self.cache_level == CacheLevel.REDIS:
                # Redis缓存
                await redis.setex(key, ttl, serialized_value)

            elif self.cache_level == CacheLevel.MEMORY:
                # 内存缓存（较短的TTL）
                memory_ttl = min(ttl, 300)  # 最多5分钟
                await redis.setex(f"memory:{key}", memory_ttl, serialized_value)

            else:  # CacheLevel.DISTRIBUTED
                # 分布式缓存（较长的TTL）
                dist_ttl = max(ttl, 3600)  # 至少1小时
                await redis.setex(f"dist:{key}", dist_ttl, serialized_value)

            return True

        except Exception as e:
            logger.error(f"设置缓存失败 [key={key}]: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        删除缓存

        Args:
            key: 缓存键

        Returns:
            是否删除成功
        """
        try:
            redis = await self._get_redis()

            if self.cache_level == CacheLevel.REDIS:
                await redis.delete(key)
            elif self.cache_level == CacheLevel.MEMORY:
                await redis.delete(f"memory:{key}")
            else:  # CacheLevel.DISTRIBUTED
                await redis.delete(f"dist:{key}")

            return True

        except Exception as e:
            logger.error(f"删除缓存失败 [key={key}]: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            缓存是否存在
        """
        try:
            redis = await self._get_redis()

            if self.cache_level == CacheLevel.REDIS:
                return await redis.exists(key) > 0
            elif self.cache_level == CacheLevel.MEMORY:
                return await redis.exists(f"memory:{key}") > 0
            else:  # CacheLevel.DISTRIBUTED
                return await redis.exists(f"dist:{key}") > 0

        except Exception as e:
            logger.error(f"检查缓存存在性失败 [key={key}]: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """
        设置缓存的过期时间

        Args:
            key: 缓存键
            ttl: 过期时间（秒）

        Returns:
            是否设置成功
        """
        try:
            redis = await self._get_redis()

            if self.cache_level == CacheLevel.REDIS:
                await redis.expire(key, ttl)
            elif self.cache_level == CacheLevel.MEMORY:
                await redis.expire(f"memory:{key}", min(ttl, 300))
            else:  # CacheLevel.DISTRIBUTED
                await redis.expire(f"dist:{key}", max(ttl, 3600))

            return True

        except Exception as e:
            logger.error(f"设置缓存过期时间失败 [key={key}]: {e}")
            return False

    async def clear(self) -> bool:
        """
        清空所有缓存

        Returns:
            是否清空成功
        """
        try:
            redis = await self._get_redis()

            if self.cache_level == CacheLevel.REDIS:
                # 谨慎操作，只清除带特定前缀的键
                keys = await redis.keys("cache:*")
                if keys:
                    await redis.delete(*keys)

            elif self.cache_level == CacheLevel.MEMORY:
                keys = await redis.keys("memory:cache:*")
                if keys:
                    await redis.delete(*keys)

            else:  # CacheLevel.DISTRIBUTED
                keys = await redis.keys("dist:cache:*")
                if keys:
                    await redis.delete(*keys)

            return True

        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            缓存统计数据
        """
        return {
            'cache_level': self.cache_level.value,
            'default_ttl': self.default_ttl,
            'backend': 'redis'
        }

# 全局缓存管理器实例（单例模式）
_cache_manager_instance: Optional[UnifiedCacheManager] = None

def get_cache_manager(
    cache_level: CacheLevel = CacheLevel.REDIS,
    default_ttl: int = 3600
) -> UnifiedCacheManager:
    """
    获取全局缓存管理器实例（单例模式）

    Args:
        cache_level: 缓存级别
        default_ttl: 默认TTL（秒）

    Returns:
        缓存管理器实例
    """
    global _cache_manager_instance

    if _cache_manager_instance is None:
        _cache_manager_instance = UnifiedCacheManager(
            default_ttl=default_ttl,
            cache_level=cache_level
        )
        logger.info(f"初始化全局缓存管理器: level={cache_level.value}, ttl={default_ttl}s")

    return _cache_manager_instance

# 为了向后兼容，创建一个默认实例
cache_manager = get_cache_manager()

# 导出
__all__ = [
    'UnifiedCacheManager',
    'CacheLevel',
    'get_cache_manager',
    'cache_manager'
]
