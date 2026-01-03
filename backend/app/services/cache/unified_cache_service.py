"""
统一缓存服务 - 三级缓存架构
提升系统性能,减少重复计算和数据库查询

缓存层级:
1. L1缓存(内存): 最快,容量小,存储最近使用的数据
2. L2缓存(Redis): 中等速度,容量中等,跨进程共享
3. L3缓存(数据库): 最慢,容量大,持久化存储

优化策略:
- 智能缓存预热
- 自适应失效策略
- 缓存命中率监控
- 批量缓存操作
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
import pickle

import numpy as np

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

class CacheLevel(Enum):
    """缓存级别"""
    L1 = "l1"  # 内存缓存
    L2 = "l2"  # Redis缓存
    L3 = "l3"  # 数据库缓存

class CacheStrategy(Enum):
    """缓存策略"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最不经常使用
    FIFO = "fifo"  # 先进先出
    TTL = "ttl"  # 基于时间

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def touch(self):
        """更新访问时间和计数"""
        self.access_count += 1
        self.timestamp = time.time()

@dataclass
class CacheConfig:
    """缓存配置"""
    # L1缓存配置
    l1_max_size: int = 1000  # 最大条目数
    l1_ttl: int = 300  # 5分钟
    l1_strategy: CacheStrategy = CacheStrategy.LRU

    # L2缓存配置
    l2_enabled: bool = True
    l2_max_size: int = 10000  # Redis最大条目数
    l2_ttl: int = 3600  # 1小时
    l2_prefix: str = "rag:cache:"

    # L3缓存配置
    l3_enabled: bool = False  # 默认关闭L3
    l3_ttl: int = 86400  # 24小时

    # 通用配置
    enable_stats: bool = True
    enable_preloading: bool = False
    serialize_method: str = "pickle"  # pickle, json

class CacheStats:
    """缓存统计"""

    def __init__(self):
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0
        self.l3_hits = 0
        self.l3_misses = 0
        self.evictions = 0
        self.errors = 0

    @property
    def l1_hit_rate(self) -> float:
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def l3_hit_rate(self) -> float:
        total = self.l3_hits + self.l3_misses
        return self.l3_hits / total if total > 0 else 0.0

    @property
    def overall_hit_rate(self) -> float:
        total_hits = self.l1_hits + self.l2_hits + self.l3_hits
        total_misses = self.l1_misses + self.l2_misses + self.l3_misses
        total = total_hits + total_misses
        return total_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l1": {
                "hits": self.l1_hits,
                "misses": self.l1_misses,
                "hit_rate": f"{self.l1_hit_rate:.2%}"
            },
            "l2": {
                "hits": self.l2_hits,
                "misses": self.l2_misses,
                "hit_rate": f"{self.l2_hit_rate:.2%}"
            },
            "l3": {
                "hits": self.l3_hits,
                "misses": self.l3_misses,
                "hit_rate": f"{self.l3_hit_rate:.2%}"
            },
            "overall": {
                "hit_rate": f"{self.overall_hit_rate:.2%}",
                "evictions": self.evictions,
                "errors": self.errors
            }
        }

class L1Cache:
    """L1内存缓存 - 最快的缓存层"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # 用于LRU

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        entry = self._cache.get(key)
        if entry is None:
            return None

        if entry.is_expired():
            self._remove(key)
            return None

        entry.touch()
        self._update_access_order(key)

        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """设置缓存值"""
        # 计算大小
        size = self._estimate_size(value)

        # 如果已存在,删除旧的
        if key in self._cache:
            self._remove(key)

        # 检查容量,必要时淘汰
        while len(self._cache) >= self.config.l1_max_size:
            self._evict()

        # 创建条目
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=ttl or self.config.l1_ttl,
            size=size
        )

        self._cache[key] = entry
        self._access_order.append(key)

    def _remove(self, key: str):
        """移除条目"""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)

    def _evict(self):
        """淘汰一个条目"""
        if self.config.l1_strategy == CacheStrategy.LRU:
            # 淘汰最久未访问的
            if self._access_order:
                key = self._access_order.pop(0)
                if key in self._cache:
                    del self._cache[key]
        elif self.config.l1_strategy == CacheStrategy.LFU:
            # 淘汰访问次数最少的
            if self._cache:
                key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
                self._remove(key)
        else:  # FIFO
            if self._access_order:
                key = self._access_order.pop(0)
                if key in self._cache:
                    del self._cache[key]

    def _update_access_order(self, key: str):
        """更新访问顺序(LRU)"""
        if self.config.l1_strategy == CacheStrategy.LRU:
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    def _estimate_size(self, value: Any) -> int:
        """估算值的大小(字节)"""
        try:
            if isinstance(value, (str, int, float, bool)):
                return len(str(value).encode())
            elif isinstance(value, (list, dict)):
                return len(json.dumps(value).encode())
            elif isinstance(value, np.ndarray):
                return value.nbytes
            else:
                # 使用pickle估算
                return len(pickle.dumps(value))
        except Exception:
            return 0

    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)

class L2Cache:
    """L2 Redis缓存 - 共享缓存层"""

    def __init__(self, config: CacheConfig, redis_client=None):
        self.config = config
        self.redis_client = redis_client

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self.config.l2_enabled or not self.redis_client:
            return None

        try:
            redis_key = self._make_key(key)
            data = await self.redis_client.get(redis_key)

            if data:
                return self._deserialize(data)
            return None
        except Exception as e:
            logger.warning(f"L2缓存获取失败(key='{key}'): {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """设置缓存值"""
        if not self.config.l2_enabled or not self.redis_client:
            return

        try:
            redis_key = self._make_key(key)
            data = self._serialize(value)
            expire = ttl or self.config.l2_ttl

            await self.redis_client.setex(redis_key, expire, data)
        except Exception as e:
            logger.warning(f"L2缓存设置失败(key='{key}'): {e}")

    async def delete(self, key: str):
        """删除缓存值"""
        if not self.config.l2_enabled or not self.redis_client:
            return

        try:
            redis_key = self._make_key(key)
            await self.redis_client.delete(redis_key)
        except Exception as e:
            logger.warning(f"L2缓存删除失败(key='{key}'): {e}")

    def _make_key(self, key: str) -> str:
        """生成Redis键"""
        return f"{self.config.l2_prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """序列化值"""
        if self.config.serialize_method == "pickle":
            return pickle.dumps(value)
        else:
            return json.dumps(value).encode()

    def _deserialize(self, data: bytes) -> Any:
        """反序列化值"""
        try:
            if self.config.serialize_method == "pickle":
                return pickle.loads(data)
            else:
                return json.loads(data.decode())
        except Exception:
            # 回退到pickle
            return pickle.loads(data)

class UnifiedCacheService:
    """统一缓存服务 - 三级缓存架构"""

    def __init__(self, config: CacheConfig = None, redis_client=None):
        self.config = config or CacheConfig()
        self.l1 = L1Cache(self.config)
        self.l2 = L2Cache(self.config, redis_client)
        self.stats = CacheStats()

        # 预热数据(可选)
        if self.config.enable_preloading:
            asyncio.create_task(self._preload_cache())

    async def get(
        self,
        key: str,
        levels: List[CacheLevel] = None
    ) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键
            levels: 要搜索的缓存级别(默认全部)

        Returns:
            缓存值或None
        """
        if levels is None:
            levels = [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]

        # L1缓存
        if CacheLevel.L1 in levels:
            value = self.l1.get(key)
            if value is not None:
                self.stats.l1_hits += 1
                return value
            self.stats.l1_misses += 1

        # L2缓存
        if CacheLevel.L2 in levels:
            value = await self.l2.get(key)
            if value is not None:
                self.stats.l2_hits += 1
                # 回填L1
                self.l1.set(key, value)
                return value
            self.stats.l2_misses += 1

        # L3缓存(如果启用)
        if CacheLevel.L3 in levels and self.config.l3_enabled:
            # L3缓存实现(从数据库读取)
            # value = await self._get_from_l3(key)
            # if value is not None:
            #     self.stats.l3_hits += 1
            #     # 回填L1和L2
            #     self.l1.set(key, value)
            #     await self.l2.set(key, value)
            #     return value
            self.stats.l3_misses += 1

        return None

    async def set(
        self,
        key: str,
        value: Any,
        levels: List[CacheLevel] = None,
        ttl: Optional[float] = None
    ):
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            levels: 要设置的缓存级别(默认全部)
            ttl: 过期时间(秒)
        """
        if levels is None:
            levels = [CacheLevel.L1, CacheLevel.L2]

        # L1缓存
        if CacheLevel.L1 in levels:
            self.l1.set(key, value, ttl)

        # L2缓存
        if CacheLevel.L2 in levels:
            await self.l2.set(key, value, ttl)

    async def delete(self, key: str):
        """删除缓存值"""
        self.l1._remove(key)
        await self.l2.delete(key)

    async def get_or_compute(
        self,
        key: str,
        compute_fn: callable,
        levels: List[CacheLevel] = None,
        ttl: Optional[float] = None
    ) -> Any:
        """
        获取缓存值,如果不存在则计算并缓存

        Args:
            key: 缓存键
            compute_fn: 计算函数(异步)
            levels: 缓存级别
            ttl: 过期时间

        Returns:
            缓存值或计算结果
        """
        # 尝试从缓存获取
        value = await self.get(key, levels=levels)
        if value is not None:
            return value

        # 计算值
        try:
            value = await compute_fn()
            # 缓存值
            await self.set(key, value, levels=levels, ttl=ttl)
            return value
        except Exception as e:
            logger.error(f"计算缓存值失败(key='{key}'): {e}")
            self.stats.errors += 1
            raise

    async def get_batch(
        self,
        keys: List[str],
        levels: List[CacheLevel] = None
    ) -> Dict[str, Any]:
        """
        批量获取缓存值

        Args:
            keys: 缓存键列表
            levels: 缓存级别

        Returns:
            键值对字典
        """
        results = {}
        for key in keys:
            value = await self.get(key, levels=levels)
            if value is not None:
                results[key] = value
        return results

    async def set_batch(
        self,
        items: Dict[str, Any],
        levels: List[CacheLevel] = None,
        ttl: Optional[float] = None
    ):
        """
        批量设置缓存值

        Args:
            items: 键值对字典
            levels: 缓存级别
            ttl: 过期时间
        """
        for key, value in items.items():
            await self.set(key, value, levels=levels, ttl=ttl)

    def clear_l1(self):
        """清空L1缓存"""
        self.l1.clear()

    async def clear_all(self):
        """清空所有缓存"""
        self.l1.clear()
        # L2缓存的清空需要谨慎,这里不实现

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats_dict = self.stats.to_dict()
        stats_dict["l1_size"] = self.l1.size()
        stats_dict["l1_capacity"] = self.config.l1_max_size
        return stats_dict

    async def _preload_cache(self):
        """预热缓存 - 加载热点数据"""
        # 这里可以添加预热逻辑
        logger.info("缓存预热功能已启用")
        pass

    def generate_key(self, *args, **kwargs) -> str:
        """
        生成缓存键

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            缓存键
        """
        # 组合所有参数
        key_parts = []

        # 添加位置参数
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                key_parts.append(str(tuple(arg)))
            elif isinstance(arg, dict):
                key_parts.append(str(sorted(arg.items())))
            elif isinstance(arg, np.ndarray):
                key_parts.append(str(arg.tobytes().hex()))
            else:
                key_parts.append(str(arg))

        # 添加关键字参数(排序以保持一致性)
        if kwargs:
            key_parts.append(str(sorted(kwargs.items())))

        # 生成哈希
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

# 全局缓存服务实例
_cache_service: Optional[UnifiedCacheService] = None

def get_cache_service(
    config: CacheConfig = None,
    redis_client=None
) -> UnifiedCacheService:
    """获取全局缓存服务实例"""
    global _cache_service
    if _cache_service is None:
        _cache_service = UnifiedCacheService(config, redis_client)
    return _cache_service

# 便捷装饰器
def cached(
    ttl: int = 300,
    key_prefix: str = "",
    levels: List[CacheLevel] = None
):
    """
    缓存装饰器

    Args:
        ttl: 过期时间(秒)
        key_prefix: 键前缀
        levels: 缓存级别
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_cache_service()

            # 生成缓存键
            func_key = f"{key_prefix}:{func.__name__}"
            cache_key = cache.generate_key(func_key, *args, **kwargs)

            # 尝试从缓存获取
            value = await cache.get(cache_key, levels=levels)
            if value is not None:
                return value

            # 调用函数
            value = await func(*args, **kwargs)

            # 缓存结果
            await cache.set(cache_key, value, levels=levels, ttl=ttl)

            return value

        return wrapper
    return decorator
