"""
缓存管理器
实现L1-L4多级缓存策略，提供统一缓存接口
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import threading
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """缓存级别"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"
    L4_DATABASE = "l4_database"


class CachePolicy(Enum):
    """缓存策略"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[int] = None
    size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl

    def update_access(self) -> None:
        """更新访问信息"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """缓存统计"""
    level: CacheLevel
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    avg_response_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        """命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


class CacheBackend(ABC):
    """缓存后端抽象类"""

    def __init__(self, level: CacheLevel):
        self.level = level
        self.stats = CacheStats(level=level)
        self._lock = threading.RLock()

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """清空缓存"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass

    async def update_stats(self, hit: bool, response_time: float) -> None:
        """更新统计信息"""
        with self._lock:
            self.stats.total_requests += 1
            if hit:
                self.stats.cache_hits += 1
            else:
                self.stats.cache_misses += 1
            # 更新平均响应时间
            self.stats.avg_response_time = (
                (self.stats.avg_response_time * (self.stats.total_requests - 1) + response_time)
                / self.stats.total_requests
            )


class MemoryCache(CacheBackend):
    """内存缓存后端"""

    def __init__(self, max_size: int = 1024 * 1024 * 100):  # 100MB
        super().__init__(CacheLevel.L1_MEMORY)
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                await self.update_stats(False, time.time() - start_time)
                return None

            if entry.is_expired():
                await self.delete(key)
                await self.update_stats(False, time.time() - start_time)
                return None

            entry.update_access()
            # 更新LRU顺序
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            await self.update_stats(True, time.time() - start_time)
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            # 计算值的大小
            serialized = pickle.dumps(value)
            size = len(serialized)

            # 检查是否需要清理空间
            while self._get_current_size() + size > self.max_size and self._cache:
                await self._evict_lru()

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl=ttl,
                size=size
            )

            with self._lock:
                self._cache[key] = entry
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                self.stats.entry_count = len(self._cache)
                self.stats.size_bytes = self._get_current_size()

            return True
        except Exception as e:
            logger.error(f"内存缓存设置失败: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry:
                if key in self._access_order:
                    self._access_order.remove(key)
                self.stats.entry_count = len(self._cache)
                self.stats.size_bytes = self._get_current_size()
                self.stats.evictions += 1
                return True
            return False

    async def clear(self) -> bool:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self.stats.entry_count = 0
            self.stats.size_bytes = 0
            return True

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            entry = self._cache.get(key)
            return entry is not None and not entry.is_expired()

    def _get_current_size(self) -> int:
        """获取当前缓存大小"""
        return sum(entry.size for entry in self._cache.values())

    async def _evict_lru(self) -> None:
        """LRU淘汰"""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            self._cache.pop(lru_key, None)
            self.stats.evictions += 1


class RedisCache(CacheBackend):
    """Redis缓存后端"""

    def __init__(self, redis_client, prefix: str = "cache"):
        super().__init__(CacheLevel.L2_REDIS)
        self.redis = redis_client
        self.prefix = prefix

    def _make_key(self, key: str) -> str:
        """生成Redis键"""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()

        try:
            redis_key = self._make_key(key)
            value = await self.redis.get(redis_key)

            if value is None:
                await self.update_stats(False, time.time() - start_time)
                return None

            # 反序列化
            try:
                deserialized = pickle.loads(value)
                await self.update_stats(True, time.time() - start_time)
                return deserialized
            except pickle.PickleError:
                await self.update_stats(False, time.time() - start_time)
                return None

        except Exception as e:
            logger.error(f"Redis缓存获取失败: {e}")
            await self.update_stats(False, time.time() - start_time)
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            redis_key = self._make_key(key)
            serialized = pickle.dumps(value)

            if ttl:
                await self.redis.setex(redis_key, ttl, serialized)
            else:
                await self.redis.set(redis_key, serialized)

            with self._lock:
                self.stats.entry_count += 1
                self.stats.size_bytes += len(serialized)

            return True
        except Exception as e:
            logger.error(f"Redis缓存设置失败: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            redis_key = self._make_key(key)
            result = await self.redis.delete(redis_key)

            with self._lock:
                if result:
                    self.stats.entry_count = max(0, self.stats.entry_count - 1)
                    self.stats.evictions += 1

            return result > 0
        except Exception as e:
            logger.error(f"Redis缓存删除失败: {e}")
            return False

    async def clear(self) -> bool:
        """清空缓存"""
        try:
            pattern = f"{self.prefix}:*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)

            with self._lock:
                self.stats.entry_count = 0
                self.stats.size_bytes = 0

            return True
        except Exception as e:
            logger.error(f"Redis缓存清空失败: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            redis_key = self._make_key(key)
            return bool(await self.redis.exists(redis_key))
        except Exception as e:
            logger.error(f"Redis缓存检查存在失败: {e}")
            return False


class DiskCache(CacheBackend):
    """磁盘缓存后端"""

    def __init__(self, cache_dir: str = "cache/disk"):
        super().__init__(CacheLevel.L3_DISK)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._file_index: Dict[str, Path] = {}

    def _get_file_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        # 使用MD5哈希作为文件名
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()

        try:
            file_path = self._get_file_path(key)

            if not file_path.exists():
                await self.update_stats(False, time.time() - start_time)
                return None

            # 读取文件
            with open(file_path, 'rb') as f:
                entry_data = pickle.load(f)

            entry = CacheEntry(**entry_data)

            if entry.is_expired():
                file_path.unlink()
                await self.update_stats(False, time.time() - start_time)
                return None

            entry.update_access()

            # 更新访问时间
            entry_data = entry.__dict__
            with open(file_path, 'wb') as f:
                pickle.dump(entry_data, f)

            await self.update_stats(True, time.time() - start_time)
            return entry.value

        except Exception as e:
            logger.error(f"磁盘缓存获取失败: {e}")
            await self.update_stats(False, time.time() - start_time)
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            file_path = self._get_file_path(key)

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl=ttl,
                size=0  # 将在序列化后计算
            )

            # 序列化并计算大小
            entry_data = entry.__dict__
            serialized = pickle.dumps(value)
            entry.size = len(serialized)
            entry_data['size'] = entry.size

            # 写入文件
            with open(file_path, 'wb') as f:
                pickle.dump(entry_data, f)

            with self._lock:
                self._file_index[key] = file_path
                self.stats.entry_count += 1
                self.stats.size_bytes += entry.size

            return True
        except Exception as e:
            logger.error(f"磁盘缓存设置失败: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            file_path = self._get_file_path(key)

            if file_path.exists():
                # 读取大小信息
                try:
                    with open(file_path, 'rb') as f:
                        entry_data = pickle.load(f)
                    size = entry_data.get('size', 0)
                except:
                    size = 0

                file_path.unlink()

                with self._lock:
                    self._file_index.pop(key, None)
                    self.stats.entry_count = max(0, self.stats.entry_count - 1)
                    self.stats.size_bytes = max(0, self.stats.size_bytes - size)
                    self.stats.evictions += 1

                return True

            return False
        except Exception as e:
            logger.error(f"磁盘缓存删除失败: {e}")
            return False

    async def clear(self) -> bool:
        """清空缓存"""
        try:
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink()

            with self._lock:
                self._file_index.clear()
                self.stats.entry_count = 0
                self.stats.size_bytes = 0

            return True
        except Exception as e:
            logger.error(f"磁盘缓存清空失败: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            file_path = self._get_file_path(key)
            return file_path.exists()
        except Exception as e:
            logger.error(f"磁盘缓存检查存在失败: {e}")
            return False


class DatabaseCache(CacheBackend):
    """数据库缓存后端"""

    def __init__(self, db_session):
        super().__init__(CacheLevel.L4_DATABASE)
        self.db = db_session

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()

        try:
            # 这里应该查询数据库缓存表
            # 暂时返回None，需要根据具体数据库实现
            await self.update_stats(False, time.time() - start_time)
            return None
        except Exception as e:
            logger.error(f"数据库缓存获取失败: {e}")
            await self.update_stats(False, time.time() - start_time)
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            # 这里应该插入或更新数据库缓存表
            # 暂时返回True，需要根据具体数据库实现
            with self._lock:
                self.stats.entry_count += 1

            return True
        except Exception as e:
            logger.error(f"数据库缓存设置失败: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            # 这里应该删除数据库缓存记录
            # 暂时返回True，需要根据具体数据库实现
            with self._lock:
                self.stats.entry_count = max(0, self.stats.entry_count - 1)
                self.stats.evictions += 1

            return True
        except Exception as e:
            logger.error(f"数据库缓存删除失败: {e}")
            return False

    async def clear(self) -> bool:
        """清空缓存"""
        try:
            # 这里应该清空数据库缓存表
            # 暂时返回True，需要根据具体数据库实现
            with self._lock:
                self.stats.entry_count = 0
                self.stats.size_bytes = 0

            return True
        except Exception as e:
            logger.error(f"数据库缓存清空失败: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            # 这里应该查询数据库缓存表
            # 暂时返回False，需要根据具体数据库实现
            return False
        except Exception as e:
            logger.error(f"数据库缓存检查存在失败: {e}")
            return False


class CacheManager:
    """缓存管理器

    提供多级缓存的统一接口，自动管理缓存层级和策略
    """

    def __init__(self):
        self._levels: List[CacheBackend] = []
        self._default_ttl = settings.CACHE_DEFAULT_TTL if hasattr(settings, 'CACHE_DEFAULT_TTL') else 3600
        self._enabled = True
        self._stats = {
            'total_requests': 0,
            'total_hits': 0,
            'total_misses': 0
        }

    async def initialize(self, redis_client=None, db_session=None) -> None:
        """初始化缓存层级"""
        # L1: 内存缓存
        memory_cache = MemoryCache(max_size=100 * 1024 * 1024)  # 100MB
        self._levels.append(memory_cache)

        # L2: Redis缓存
        if redis_client:
            redis_cache = RedisCache(redis_client)
            self._levels.append(redis_cache)

        # L3: 磁盘缓存
        disk_cache = DiskCache(cache_dir="cache/disk")
        self._levels.append(disk_cache)

        # L4: 数据库缓存
        if db_session:
            db_cache = DatabaseCache(db_session)
            self._levels.append(db_cache)

        logger.info(f"缓存管理器初始化完成，共 {len(self._levels)} 层缓存")

    def _generate_key(self, key: str, namespace: str = None) -> str:
        """生成缓存键"""
        if namespace:
            return f"{namespace}:{key}"
        return key

    async def get(self, key: str, namespace: str = None) -> Optional[Any]:
        """获取缓存值，按层级查询"""
        if not self._enabled or not self._levels:
            return None

        cache_key = self._generate_key(key, namespace)

        self._stats['total_requests'] += 1

        # 按层级顺序查询
        for level in self._levels:
            try:
                value = await level.get(cache_key)
                if value is not None:
                    # 缓存命中，更新统计
                    self._stats['total_hits'] += 1

                    # 回写机制：将数据写回更高层级的缓存
                    await self._write_back(cache_key, value, level.level)

                    logger.debug(f"缓存命中: {cache_key} 在 {level.level.value}")
                    return value
            except Exception as e:
                logger.warning(f"查询 {level.level.value} 缓存失败: {e}")
                continue

        # 所有层级都未命中
        self._stats['total_misses'] += 1
        logger.debug(f"缓存未命中: {cache_key}")
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                 namespace: str = None, levels: Optional[List[CacheLevel]] = None) -> bool:
        """设置缓存值"""
        if not self._enabled or not self._levels:
            return False

        cache_key = self._generate_key(key, namespace)
        if ttl is None:
            ttl = self._default_ttl

        # 确定要写入的层级
        target_levels = levels
        if target_levels is None:
            # 默认写入所有启用的层级
            target_levels = [level.level for level in self._levels]

        success_count = 0
        for level in self._levels:
            if level.level in target_levels:
                try:
                    success = await level.set(cache_key, value, ttl)
                    if success:
                        success_count += 1
                        logger.debug(f"缓存设置成功: {cache_key} 到 {level.level.value}")
                except Exception as e:
                    logger.warning(f"设置 {level.level.value} 缓存失败: {e}")

        return success_count > 0

    async def delete(self, key: str, namespace: str = None) -> bool:
        """删除缓存值"""
        if not self._enabled or not self._levels:
            return False

        cache_key = self._generate_key(key, namespace)
        success_count = 0

        for level in self._levels:
            try:
                success = await level.delete(cache_key)
                if success:
                    success_count += 1
                    logger.debug(f"缓存删除成功: {cache_key} 从 {level.level.value}")
            except Exception as e:
                logger.warning(f"删除 {level.level.value} 缓存失败: {e}")

        return success_count > 0

    async def clear(self, level: Optional[CacheLevel] = None) -> bool:
        """清空缓存"""
        if not self._enabled or not self._levels:
            return False

        if level:
            # 清空指定层级
            for cache_level in self._levels:
                if cache_level.level == level:
                    try:
                        return await cache_level.clear()
                    except Exception as e:
                        logger.warning(f"清空 {level.value} 缓存失败: {e}")
                        return False
            return False
        else:
            # 清空所有层级
            success_count = 0
            for cache_level in self._levels:
                try:
                    success = await cache_level.clear()
                    if success:
                        success_count += 1
                        logger.debug(f"清空缓存成功: {cache_level.level.value}")
                except Exception as e:
                    logger.warning(f"清空 {cache_level.level.value} 缓存失败: {e}")

            return success_count > 0

    async def exists(self, key: str, namespace: str = None) -> bool:
        """检查键是否存在"""
        if not self._enabled or not self._levels:
            return False

        cache_key = self._generate_key(key, namespace)

        for level in self._levels:
            try:
                if await level.exists(cache_key):
                    return True
            except Exception as e:
                logger.warning(f"检查 {level.level.value} 缓存存在失败: {e}")
                continue

        return False

    async def _write_back(self, key: str, value: Any, source_level: CacheLevel) -> None:
        """回写机制：将数据写回更高层级的缓存"""
        try:
            # 找到源层级的索引
            source_index = None
            for i, level in enumerate(self._levels):
                if level.level == source_level:
                    source_index = i
                    break

            if source_index is None or source_index == 0:
                return  # 已经是最顶层或未找到源层级

            # 写入更高层级的缓存
            target_levels = [level.level for level in self._levels[:source_index]]
            if target_levels:
                await self.set(key, value, levels=target_levels)
                logger.debug(f"缓存回写: {key} 到更高层级")

        except Exception as e:
            logger.warning(f"缓存回写失败: {e}")

    def enable(self) -> None:
        """启用缓存"""
        self._enabled = True
        logger.info("缓存已启用")

    def disable(self) -> None:
        """禁用缓存"""
        self._enabled = False
        logger.info("缓存已禁用")

    def is_enabled(self) -> bool:
        """检查缓存是否启用"""
        return self._enabled

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        level_stats = {}
        for level in self._levels:
            level_stats[level.level.value] = {
                'requests': level.stats.total_requests,
                'hits': level.stats.cache_hits,
                'misses': level.stats.cache_misses,
                'hit_rate': level.stats.hit_rate,
                'evictions': level.stats.evictions,
                'size_bytes': level.stats.size_bytes,
                'entry_count': level.stats.entry_count,
                'avg_response_time': level.stats.avg_response_time
            }

        overall_hit_rate = 0.0
        if self._stats['total_requests'] > 0:
            overall_hit_rate = self._stats['total_hits'] / self._stats['total_requests']

        return {
            'enabled': self._enabled,
            'total_requests': self._stats['total_requests'],
            'total_hits': self._stats['total_hits'],
            'total_misses': self._stats['total_misses'],
            'overall_hit_rate': overall_hit_rate,
            'levels': level_stats,
            'level_count': len(self._levels)
        }

    async def cleanup_expired(self) -> None:
        """清理过期缓存"""
        logger.info("开始清理过期缓存")

        for level in self._levels:
            try:
                # 对于内存缓存，需要在get时检查过期
                if level.level == CacheLevel.L1_MEMORY:
                    continue

                # 其他后端的清理逻辑需要具体实现
                logger.debug(f"清理 {level.level.value} 缓存完成")
            except Exception as e:
                logger.warning(f"清理 {level.level.value} 缓存失败: {e}")

        logger.info("过期缓存清理完成")


# 全局缓存管理器实例
cache_manager = CacheManager()