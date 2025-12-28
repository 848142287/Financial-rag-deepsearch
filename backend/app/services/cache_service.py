"""
缓存策略和Redis同步服务
"""

import json
import logging
import pickle
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
from enum import Enum

from app.core.redis_client import redis_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """缓存策略"""
    WRITE_THROUGH = "write_through"      # 写穿透：同时更新缓存和数据库
    WRITE_BEHIND = "write_behind"        # 写回：先更新缓存，异步更新数据库
    REFRESH_AHEAD = "refresh_ahead"      # 预刷新：在过期前主动刷新
    CACHE_ASIDE = "cache_aside"          # 旁路缓存：应用管理缓存


class CacheLevel(Enum):
    """缓存级别"""
    L1 = "l1"  # 内存缓存（应用内）
    L2 = "l2"  # Redis缓存（分布式）
    L3 = "l3"  # 持久化缓存


class CacheConfig:
    """缓存配置"""

    def __init__(self, ttl: int = 3600, strategy: CacheStrategy = CacheStrategy.CACHE_ASIDE,
                 max_size: Optional[int] = None, serialize_method: str = "json"):
        self.ttl = ttl  # 生存时间（秒）
        self.strategy = strategy
        self.max_size = max_size  # 最大缓存项数
        self.serialize_method = serialize_method  # 序列化方法


class CacheService:
    """缓存服务"""

    def __init__(self):
        # 预定义的缓存配置
        self.configs = {
            "query_result": CacheConfig(ttl=1800, strategy=CacheStrategy.REFRESH_AHEAD),  # 30分钟
            "document_metadata": CacheConfig(ttl=3600, strategy=CacheStrategy.WRITE_THROUGH),  # 1小时
            "user_session": CacheConfig(ttl=7200, strategy=CacheStrategy.WRITE_BEHIND),  # 2小时
            "search_history": CacheConfig(ttl=86400, strategy=CacheStrategy.CACHE_ASIDE),  # 24小时
            "hot_documents": CacheConfig(ttl=1800, strategy=CacheStrategy.REFRESH_AHEAD),  # 30分钟
            "system_stats": CacheConfig(ttl=300, strategy=CacheStrategy.CACHE_ASIDE),  # 5分钟
            "entity_cache": CacheConfig(ttl=3600, strategy=CacheStrategy.WRITE_THROUGH),  # 1小时
            "index_stats": CacheConfig(ttl=600, strategy=CacheStrategy.CACHE_ASIDE),  # 10分钟
        }

        # 内存缓存（L1）
        self._memory_cache: Dict[str, Any] = {}
        self._memory_cache_access: Dict[str, datetime] = {}

    def get_config(self, cache_type: str) -> CacheConfig:
        """获取缓存配置"""
        return self.configs.get(cache_type, CacheConfig())

    async def get(self, key: str, cache_type: str = "default") -> Optional[Any]:
        """获取缓存值"""
        config = self.get_config(cache_type)

        # 先查内存缓存（L1）
        if key in self._memory_cache:
            self._memory_cache_access[key] = datetime.utcnow()
            return self._memory_cache[key]

        # 查Redis缓存（L2）
        try:
            data = redis_client.get(key)
            if data:
                # 反序列化
                value = self._deserialize(data, config.serialize_method)

                # 写入内存缓存
                self._memory_cache[key] = value
                self._memory_cache_access[key] = datetime.utcnow()

                logger.debug(f"Cache hit (L2): {key}")
                return value
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")

        logger.debug(f"Cache miss: {key}")
        return None

    async def set(self, key: str, value: Any, cache_type: str = "default", ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        config = self.get_config(cache_type)
        cache_ttl = ttl or config.ttl

        try:
            # 序列化
            serialized = self._serialize(value, config.serialize_method)

            # 设置Redis缓存（L2）
            success = redis_client.setex(key, cache_ttl, serialized)

            if success:
                # 设置内存缓存（L1）
                self._memory_cache[key] = value
                self._memory_cache_access[key] = datetime.utcnow()

                # 检查内存缓存大小
                await self._check_memory_cache_size()

                logger.debug(f"Cache set: {key}")
                return True

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")

        return False

    async def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            # 删除内存缓存
            if key in self._memory_cache:
                del self._memory_cache[key]
            if key in self._memory_cache_access:
                del self._memory_cache_access[key]

            # 删除Redis缓存
            result = redis_client.delete(key)

            logger.debug(f"Cache deleted: {key}")
            return result > 0

        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """按模式清除缓存"""
        try:
            # 清除内存缓存
            memory_keys = [k for k in self._memory_cache.keys() if pattern in k]
            for key in memory_keys:
                del self._memory_cache[key]
                if key in self._memory_cache_access:
                    del self._memory_cache_access[key]

            # 清除Redis缓存
            redis_keys = redis_client.keys(pattern)
            if redis_keys:
                result = redis_client.delete(*redis_keys)
                logger.info(f"Cleared {result} cache entries matching pattern: {pattern}")
                return result

        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")

        return 0

    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        if key in self._memory_cache:
            return True

        try:
            return redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """设置缓存过期时间"""
        try:
            return redis_client.expire(key, ttl)
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """获取缓存剩余生存时间"""
        try:
            return redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Cache TTL error for key {key}: {e}")
            return -1

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """递增缓存值"""
        try:
            return redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return None

    def _serialize(self, value: Any, method: str) -> bytes:
        """序列化值"""
        if method == "json":
            return json.dumps(value, ensure_ascii=False).encode('utf-8')
        elif method == "pickle":
            return pickle.dumps(value)
        else:
            return str(value).encode('utf-8')

    def _deserialize(self, data: bytes, method: str) -> Any:
        """反序列化值"""
        if method == "json":
            return json.loads(data.decode('utf-8'))
        elif method == "pickle":
            return pickle.loads(data)
        else:
            return data.decode('utf-8')

    async def _check_memory_cache_size(self):
        """检查内存缓存大小"""
        max_size = 1000  # 默认最大1000项
        if len(self._memory_cache) > max_size:
            # LRU淘汰
            sorted_items = sorted(
                self._memory_cache_access.items(),
                key=lambda x: x[1]
            )
            # 删除最旧的100项
            for key, _ in sorted_items[:100]:
                del self._memory_cache[key]
                del self._memory_cache_access[key]

    async def get_memory_cache_stats(self) -> Dict[str, Any]:
        """获取内存缓存统计"""
        return {
            "size": len(self._memory_cache),
            "max_size": 1000,
            "utilization": len(self._memory_cache) / 1000
        }


# 全局缓存服务实例
cache_service = CacheService()


# 缓存装饰器
def cached(cache_type: str, key_prefix: str = "", ttl: Optional[int] = None):
    """缓存装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = _generate_cache_key(key_prefix, func.__name__, args, kwargs)

            # 尝试从缓存获取
            cached_result = await cache_service.get(cache_key, cache_type)
            if cached_result is not None:
                logger.debug(f"Cache hit for function {func.__name__}")
                return cached_result

            # 执行函数
            result = await func(*args, **kwargs)

            # 存入缓存
            await cache_service.set(cache_key, result, cache_type, ttl)
            logger.debug(f"Cache set for function {func.__name__}")

            return result

        return wrapper
    return decorator


def _generate_cache_key(prefix: str, func_name: str, args: tuple, kwargs: dict) -> str:
    """生成缓存键"""
    # 将参数序列化为字符串
    args_str = json.dumps([str(arg) for arg in args], sort_keys=True)
    kwargs_str = json.dumps({k: str(v) for k, v in sorted(kwargs.items())}, sort_keys=True)

    # 生成哈希
    key_content = f"{prefix}:{func_name}:{args_str}:{kwargs_str}"
    key_hash = hashlib.md5(key_content.encode()).hexdigest()

    return f"cache:{prefix}:{func_name}:{key_hash}"


class CacheSyncService:
    """缓存同步服务"""

    def __init__(self):
        self.cache_service = cache_service
        self._sync_handlers: Dict[str, Callable] = {}

    def register_sync_handler(self, event_type: str, handler: Callable):
        """注册同步处理器"""
        self._sync_handlers[event_type] = handler

    async def sync_document_update(self, document_id: str, update_data: Dict[str, Any]):
        """同步文档更新"""
        # 清除相关缓存
        patterns = [
            f"*document*:{document_id}*",
            f"*search*:*{document_id}*",
            f"*metadata*:{document_id}*"
        ]

        for pattern in patterns:
            await self.cache_service.clear_pattern(pattern)

        # 更新文档元数据缓存
        metadata_key = f"document_metadata:{document_id}"
        await self.cache_service.set(metadata_key, update_data, "document_metadata")

        logger.info(f"Synced document update for {document_id}")

    async def sync_search_index_update(self, index_type: str, affected_documents: List[str]):
        """同步搜索索引更新"""
        # 清除搜索相关缓存
        patterns = [
            "*search_result*",
            "*hot_documents*",
            "*index_stats*"
        ]

        for pattern in patterns:
            await self.cache_service.clear_pattern(pattern)

        # 更新索引统计
        stats_key = f"index_stats:{index_type}"
        stats_data = {
            "last_update": datetime.utcnow().isoformat(),
            "affected_documents": affected_documents,
            "document_count": len(affected_documents)
        }
        await self.cache_service.set(stats_key, stats_data, "index_stats")

        logger.info(f"Synced search index update for {index_type}")

    async def sync_user_activity(self, user_id: str, activity_type: str, data: Dict[str, Any]):
        """同步用户活动"""
        # 更新用户活动缓存
        activity_key = f"user_activity:{user_id}:{activity_type}"
        await self.cache_service.set(activity_key, data, "user_session", ttl=7200)

        # 清除搜索历史缓存（强制刷新）
        if activity_type == "search":
            history_key = f"search_history:{user_id}"
            await self.cache_service.delete(history_key)

        logger.info(f"Synced user activity for {user_id}")

    async def refresh_ahead_cache(self, cache_type: str):
        """预刷新缓存"""
        if cache_type == "query_result":
            await self._refresh_hot_queries()
        elif cache_type == "hot_documents":
            await self._refresh_hot_documents()
        elif cache_type == "system_stats":
            await self._refresh_system_stats()

    async def _refresh_hot_queries(self):
        """刷新热点查询"""
        # 从Redis获取热点查询
        hot_queries = redis_client.zrevrange("hot_queries", 0, 9, withscores=True)

        for query, score in hot_queries:
            query = query.decode('utf-8')
            # 预执行查询并缓存结果
            cache_key = f"query_result:{hashlib.md5(query.encode()).hexdigest()}"
            if not await self.cache_service.exists(cache_key):
                # 这里可以触发异步搜索任务
                pass

    async def _refresh_hot_documents(self):
        """刷新热点文档"""
        # 从Redis获取热点文档
        hot_docs = redis_client.zrevrange("hot_documents", 0, 9)

        for doc_id in hot_docs:
            doc_id = doc_id.decode('utf-8')
            # 预加载文档元数据
            metadata_key = f"document_metadata:{doc_id}"
            if not await self.cache_service.exists(metadata_key):
                # 从数据库加载并缓存
                pass

    async def _refresh_system_stats(self):
        """刷新系统统计"""
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            # 这里可以收集各种系统指标
        }
        await self.cache_service.set("system_stats:current", stats, "system_stats")


# 全局缓存同步服务实例
cache_sync_service = CacheSyncService()