"""
评估缓存管理器
缓存评估结果以提高性能
"""

from hashlib import md5
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

@dataclass
class CachedEvaluation:
    """缓存的评估结果"""
    query: str
    context: str
    answer: str
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    ttl_hours: int = 24

    def is_expired(self) -> bool:
        """检查缓存是否过期"""
        return datetime.now() - self.timestamp > timedelta(hours=self.ttl_hours)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'query': self.query,
            'context': self.context,
            'answer': self.answer,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
            'ttl_hours': self.ttl_hours
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedEvaluation':
        """从字典创建"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class EvaluationCacheManager:
    """评估缓存管理器"""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, CachedEvaluation] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _generate_key(self, query: str, context: str = "") -> str:
        """生成缓存键"""
        content = f"{query}:{context}"
        return md5(content.encode()).hexdigest()

    async def get(self, query: str, context: str = "") -> Optional[CachedEvaluation]:
        """获取缓存的评估结果"""
        key = self._generate_key(query, context)

        if key in self._cache:
            cached = self._cache[key]
            if not cached.is_expired():
                self._hits += 1
                logger.debug(f"缓存命中: {query[:50]}...")
                return cached
            else:
                # 删除过期缓存
                del self._cache[key]

        self._misses += 1
        return None

    async def set(
        self,
        query: str,
        context: str,
        answer: str,
        metrics: Dict[str, float],
        ttl_hours: int = 24
    ):
        """设置缓存"""
        # 如果缓存已满，删除最旧的条目
        if len(self._cache) >= self._max_size:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].timestamp
            )
            del self._cache[oldest_key]
            logger.debug("缓存已满，删除最旧条目")

        key = self._generate_key(query, context)
        cached = CachedEvaluation(
            query=query,
            context=context,
            answer=answer,
            metrics=metrics,
            ttl_hours=ttl_hours
        )
        self._cache[key] = cached
        logger.debug(f"缓存评估结果: {query[:50]}...")

    async def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("评估缓存已清空")

    async def cleanup_expired(self):
        """清理过期缓存"""
        expired_keys = [
            key for key, cached in self._cache.items()
            if cached.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.info(f"清理了 {len(expired_keys)} 个过期缓存")

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            'cache_size': len(self._cache),
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

# 全局缓存管理器实例
evaluation_cache_manager = EvaluationCacheManager()

# 导出
__all__ = [
    'CachedEvaluation',
    'EvaluationCacheManager',
    'evaluation_cache_manager'
]
