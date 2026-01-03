"""
统一嵌入服务基类
定义所有嵌入提供者的公共接口和数据结构

优化点：
- 清晰的抽象接口
- 统一的配置管理
- 标准化的数据结构
- 完善的错误处理
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum
import time
import hashlib

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

class EmbeddingProviderType(Enum):
    """嵌入提供者类型"""
    BGE_LOCAL = "bge_local"          # 本地BGE模型
    QWEN_API = "qwen_api"            # Qwen API
    OPENAI_API = "openai_api"        # OpenAI API
    AUTO = "auto"                    # 自动选择

@dataclass
class EmbeddingConfig:
    """统一嵌入配置"""
    # 提供者选择
    provider_type: EmbeddingProviderType = EmbeddingProviderType.BGE_LOCAL

    # 通用配置
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 0.5

    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 秒

    # 向量质量验证
    enable_validation: bool = True
    skip_invalid_embeddings: bool = True

    # BGE本地配置
    bge_model_path: str = "backend/models/bge-large-zh-v1.5"
    bge_device: str = "cpu"

    # Qwen API配置
    qwen_api_key: Optional[str] = None
    qwen_base_url: Optional[str] = None
    qwen_model: str = "text-embedding-v4"

    # OpenAI API配置
    openai_api_key: Optional[str] = None
    openai_model: str = "text-embedding-3-small"

@dataclass
class EmbeddingResult:
    """嵌入结果（统一格式）"""
    embedding: List[float]
    dimension: int
    model: str
    cache_hit: bool = False
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "embedding": self.embedding,
            "dimension": self.dimension,
            "model": self.model,
            "cache_hit": self.cache_hit,
            "processing_time": self.processing_time
        }

@dataclass
class EmbeddingStats:
    """嵌入服务统计"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_errors: int = 0
    avg_processing_time: float = 0.0
    total_embeddings: int = 0

    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{self.get_cache_hit_rate():.2%}",
            "total_errors": self.total_errors,
            "avg_processing_time": f"{self.avg_processing_time:.3f}s",
            "total_embeddings": self.total_embeddings
        }

class EmbeddingCache:
    """统一嵌入缓存"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._cache: Dict[str, tuple] = {}  # (embedding, timestamp)
        self._access_count: Dict[str, int] = {}
        self._hits = 0
        self._misses = 0

    def _generate_key(self, text: str, provider: str) -> str:
        """生成缓存键"""
        content = f"{provider}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, provider: str) -> Optional[List[float]]:
        """获取缓存的嵌入"""
        if not self.config.enable_cache:
            return None

        key = self._generate_key(text, provider)

        if key in self._cache:
            embedding, timestamp = self._cache[key]

            # 检查TTL
            if time.time() - timestamp < self.config.cache_ttl:
                self._hits += 1
                self._access_count[key] = self._access_count.get(key, 0) + 1
                return embedding.copy()
            else:
                # 过期,删除
                del self._cache[key]
                if key in self._access_count:
                    del self._access_count[key]

        self._misses += 1
        return None

    def set(self, text: str, provider: str, embedding: List[float]):
        """缓存嵌入"""
        if not self.config.enable_cache:
            return

        key = self._generate_key(text, provider)

        # LRU淘汰
        if len(self._cache) >= self.config.cache_size and key not in self._cache:
            if self._access_count:
                lru_key = min(self._access_count, key=self._access_count.get)
                del self._cache[lru_key]
                del self._access_count[lru_key]

        self._cache[key] = (embedding.copy(), time.time())
        self._access_count[key] = 0

    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._access_count.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0

        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': f"{hit_rate:.2%}",
            'size': len(self._cache),
            'capacity': self.config.cache_size
        }

class BaseEmbeddingProvider(ABC):
    """
    嵌入提供者抽象基类

    定义所有嵌入提供者必须实现的接口
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.stats = EmbeddingStats()

    # ========================================================================
    # 核心接口
    # ========================================================================

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        生成单个文本的嵌入

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成嵌入

        Args:
            texts: 输入文本列表

        Returns:
            嵌入向量列表
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        获取嵌入维度

        Returns:
            向量维度
        """
        pass

    # ========================================================================
    # 工具方法
    # ========================================================================

    def _normalize(self, embedding: List[float]) -> List[float]:
        """归一化嵌入向量"""
        if not self.config.normalize_embeddings:
            return embedding

        import numpy as np
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            return [x / norm for x in embedding]
        return embedding

    def _validate_embedding(self, embedding: List[float]) -> bool:
        """验证嵌入向量质量"""
        if not self.config.enable_validation:
            return True

        import numpy as np
        arr = np.array(embedding)

        # 检查NaN
        if np.any(np.isnan(arr)):
            logger.warning("Embedding contains NaN values")
            return False

        # 检查Inf
        if np.any(np.isinf(arr)):
            logger.warning("Embedding contains Inf values")
            return False

        # 检查零向量
        if np.all(arr == 0):
            logger.warning("Embedding is zero vector")
            return False

        return True

    def _update_stats(self, count: int, processing_time: float, cache_hit: bool = False):
        """更新统计信息"""
        self.stats.total_requests += 1
        self.stats.total_embeddings += count

        if cache_hit:
            self.stats.cache_hits += 1
        else:
            self.stats.cache_misses += 1

        # 更新平均处理时间
        total = self.stats.total_requests
        self.stats.avg_processing_time = (
            (self.stats.avg_processing_time * (total - 1) + processing_time) / total
        )

# 导出
__all__ = [
    'EmbeddingProviderType',
    'EmbeddingConfig',
    'EmbeddingResult',
    'EmbeddingStats',
    'EmbeddingCache',
    'BaseEmbeddingProvider'
]
