"""
查询缓存服务
缓存检索结果以提升性能，减少重复查询的响应时间
"""

import json
import hashlib
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import redis.asyncio as aioredis
from app.core.structured_logging import get_structured_logger
from app.core.config import settings

logger = get_structured_logger(__name__)


@dataclass
class CachedQueryResult:
    """缓存的查询结果"""
    query: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: float
    ttl: int  # 缓存生存时间（秒）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedQueryResult':
        """从字典创建实例"""
        return cls(**data)


class QueryCacheService:
    """查询缓存服务"""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,  # 默认1小时
        prefix: str = "query_cache:"
    ):
        self.redis_url = redis_url or settings.redis_url
        self.default_ttl = default_ttl
        self.prefix = prefix
        self._redis: Optional[aioredis.Redis] = None
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }

    async def _get_redis(self) -> aioredis.Redis:
        """获取Redis连接"""
        if self._redis is None:
            try:
                self._redis = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                logger.info(f"Connected to Redis for query caching: {self.redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._redis

    def _generate_cache_key(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        生成缓存键

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            缓存键
        """
        # 创建包含查询参数的标准化字符串
        cache_data = {
            'query': query.strip().lower(),
            'top_k': top_k,
            'filters': filters or {}
        }

        # 序列化并哈希
        cache_str = json.dumps(cache_data, sort_keys=True)
        hash_value = hashlib.md5(cache_str.encode()).hexdigest()

        return f"{self.prefix}{hash_value}"

    async def get(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[CachedQueryResult]:
        """
        从缓存获取查询结果

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            缓存的查询结果，如果未命中则返回None
        """
        try:
            cache_key = self._generate_cache_key(query, top_k, filters)
            redis = await self._get_redis()

            # 从Redis获取
            cached_data = await redis.get(cache_key)

            if cached_data:
                # 解析缓存数据
                data = json.loads(cached_data)
                result = CachedQueryResult.from_dict(data)

                # 检查是否过期（双重检查）
                age = time.time() - result.timestamp
                if age < result.ttl:
                    self._stats['hits'] += 1
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return result
                else:
                    # 缓存已过期，删除
                    await redis.delete(cache_key)
                    self._stats['misses'] += 1
                    logger.debug(f"Cache expired for query: {query[:50]}...")
                    return None
            else:
                self._stats['misses'] += 1
                logger.debug(f"Cache miss for query: {query[:50]}...")
                return None

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            self._stats['misses'] += 1
            return None

    async def set(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        缓存查询结果

        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回结果数量
            filters: 过滤条件
            metadata: 元数据（如检索方法、耗时等）
            ttl: 缓存生存时间（秒），默认使用default_ttl

        Returns:
            是否成功缓存
        """
        try:
            cache_key = self._generate_cache_key(query, top_k, filters)
            redis = await self._get_redis()

            # 创建缓存对象
            cached_result = CachedQueryResult(
                query=query,
                results=results,
                metadata=metadata or {},
                timestamp=time.time(),
                ttl=ttl or self.default_ttl
            )

            # 序列化并存储
            cached_data = json.dumps(cached_result.to_dict(), ensure_ascii=False)
            actual_ttl = ttl or self.default_ttl

            await redis.setex(cache_key, actual_ttl, cached_data)

            self._stats['sets'] += 1
            logger.debug(f"Cached query result: {query[:50]}... (TTL: {actual_ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    async def delete(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        删除特定查询的缓存

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            是否成功删除
        """
        try:
            cache_key = self._generate_cache_key(query, top_k, filters)
            redis = await self._get_redis()

            await redis.delete(cache_key)

            self._stats['deletes'] += 1
            logger.debug(f"Deleted cache for query: {query[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Error deleting cache: {e}")
            return False

    async def clear_all(self) -> bool:
        """
        清空所有查询缓存

        Returns:
            是否成功清空
        """
        try:
            redis = await self._get_redis()

            # 扫描所有缓存键
            keys = []
            async for key in redis.scan_iter(match=f"{self.prefix}*"):
                keys.append(key)

            if keys:
                await redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} cached queries")

            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        try:
            redis = await self._get_redis()

            # 统计缓存键数量
            key_count = 0
            async for key in redis.scan_iter(match=f"{self.prefix}*"):
                key_count += 1

            # 计算命中率
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0

            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'sets': self._stats['sets'],
                'deletes': self._stats['deletes'],
                'key_count': key_count,
                'hit_rate': f"{hit_rate:.2%}",
                'total_requests': total_requests
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    async def cleanup_expired(self) -> int:
        """
        清理过期的缓存条目（通常由Redis自动处理，此方法用于手动触发）

        Returns:
            清理的条目数量
        """
        logger.info("Redis automatically handles expired keys, manual cleanup not needed")
        return 0

    async def close(self):
        """关闭Redis连接"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis connection closed")


# 全局服务实例
_query_cache_service: Optional[QueryCacheService] = None


def get_query_cache_service() -> QueryCacheService:
    """获取查询缓存服务单例"""
    global _query_cache_service
    if _query_cache_service is None:
        _query_cache_service = QueryCacheService()
    return _query_cache_service


# 便捷函数
async def cached_query(
    query: str,
    retrieval_func,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    ttl: Optional[int] = None
) -> Dict[str, Any]:
    """
    带缓存的查询执行

    Args:
        query: 查询文本
        retrieval_func: 检索函数（异步，接收query和top_k）
        top_k: 返回结果数量
        filters: 过滤条件
        ttl: 缓存生存时间

    Returns:
        包含results、metadata、cached等字段的字典
    """
    cache_service = get_query_cache_service()

    # 尝试从缓存获取
    cached = await cache_service.get(query, top_k, filters)

    if cached is not None:
        return {
            'results': cached.results,
            'metadata': {
                **cached.metadata,
                'cached': True,
                'cache_age': time.time() - cached.timestamp
            },
            'cached': True
        }

    # 缓存未命中，执行检索
    start_time = time.time()
    results = await retrieval_func(query, top_k)
    elapsed = time.time() - start_time

    # 缓存结果
    metadata = {
        'retrieval_time': elapsed,
        'cached': False,
        'result_count': len(results)
    }

    await cache_service.set(
        query=query,
        results=results,
        top_k=top_k,
        filters=filters,
        metadata=metadata,
        ttl=ttl
    )

    return {
        'results': results,
        'metadata': metadata,
        'cached': False
    }
