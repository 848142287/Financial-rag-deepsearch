"""
统一嵌入服务 - 消除重复代码
整合所有嵌入服务提供者,提供统一的接口

优化版本 - 集成优化的embedding策略
- 文档类型自适应chunking
- 元数据增强（券商、日期、报告类型等）
- 金融术语上下文增强
- 表格和图表特殊处理
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
import asyncio
import hashlib
import time
import concurrent.futures

import numpy as np

from app.core.structured_logging import get_structured_logger
from app.core.config import settings

logger = get_structured_logger(__name__)

# 导入优化的embedding策略
try:
    OPTIMIZED_EMBEDDING_AVAILABLE = True
except ImportError:
    OPTIMIZED_EMBEDDING_AVAILABLE = False
    logger.warning("Optimized embedding strategy not available, using baseline")

class EmbeddingProviderType(Enum):
    """嵌入提供者类型"""
    BGE_LOCAL = "bge_local"          # 本地BGE模型（唯一选项）
    AUTO = "auto"                    # 自动选择（自动选择BGE）

@dataclass
class EmbeddingConfig:
    """统一嵌入配置"""
    # 提供者选择
    primary_provider: EmbeddingProviderType = EmbeddingProviderType.BGE_LOCAL
    fallback_provider: Optional[EmbeddingProviderType] = None  # 不使用fallback

    # 通用配置
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    timeout: int = 60

    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 秒

    # 降级配置
    enable_fallback: bool = False  # 不启用降级
    max_retries: int = 3
    retry_delay: float = 0.5

    # BGE本地配置
    bge_model_path: str = "backend/models/bge-large-zh-v1.5"
    bge_device: str = "cpu"

    # 向量质量验证
    enable_validation: bool = True
    skip_invalid_embeddings: bool = True

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

    def get(self, text: str, provider: str) -> Optional[np.ndarray]:
        """获取缓存的嵌入"""
        # 显式转换为bool，避免NumPy数组布尔值歧义
        enable_cache = bool(self.config.enable_cache)
        if not enable_cache:
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

    def set(self, text: str, provider: str, embedding: np.ndarray):
        """缓存嵌入"""
        # 显式转换为bool，避免NumPy数组布尔值歧义
        enable_cache = bool(self.config.enable_cache)
        if not enable_cache:
            return

        key = self._generate_key(text, provider)

        # LRU淘汰
        if len(self._cache) >= self.config.cache_size and key not in self._cache:
            # 淘汰访问次数最少的
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

class EmbeddingProvider(ABC):
    """嵌入提供者抽象基类"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """生成单个文本的嵌入"""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量生成嵌入"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """获取嵌入维度"""
        pass

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """归一化嵌入向量"""
        # 显式转换为bool，避免NumPy数组布尔值歧义
        normalize_embeddings = bool(self.config.normalize_embeddings)
        if not normalize_embeddings:
            return embedding

        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _validate_embedding(self, embedding: np.ndarray) -> bool:
        """验证嵌入向量质量"""
        # 显式转换为bool，避免NumPy数组布尔值歧义
        enable_validation = bool(self.config.enable_validation)
        if not enable_validation:
            return True

        # 检查NaN
        if np.any(np.isnan(embedding)):
            logger.warning("Embedding contains NaN values")
            return False

        # 检查Inf
        if np.any(np.isinf(embedding)):
            logger.warning("Embedding contains Inf values")
            return False

        # 检查零向量
        if np.all(embedding == 0):
            logger.warning("Embedding is zero vector")
            return False

        return True

class QwenEmbeddingProvider(EmbeddingProvider):
    """Qwen API嵌入提供者"""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.api_key = config.qwen_api_key or settings.qwen_api_key
        self.base_url = config.qwen_base_url or settings.qwen_base_url
        self.model = config.qwen_model

        if not self.api_key:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("QWEN_API_KEY not configured. Qwen embedding service will not be functional.")
            self._available = False
            self._client = None
            return

        self._available = True
        # 延迟初始化HTTP客户端
        self._client = None

    async def _get_client(self):
        """获取HTTP客户端"""
        if not self._available:
            raise ValueError("Qwen embedding service is not available (QWEN_API_KEY not configured)")
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.config.timeout
            )
        return self._client

    async def embed(self, text: str) -> np.ndarray:
        """生成单个文本的嵌入"""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量生成嵌入"""
        try:
            client = await self._get_client()

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            results = []
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]

                payload = {
                    "model": self.model,
                    "input": {
                        "texts": batch
                    }
                }

                response = await client.post(
                    "/embeddings",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()

                # 提取嵌入
                batch_embeddings = [
                    np.array(emb["embedding"], dtype=np.float32)
                    for emb in data["output"]["embeddings"]
                ]

                # 归一化
                batch_embeddings = [
                    self._normalize(emb) if self._validate_embedding(emb) else emb
                    for emb in batch_embeddings
                ]

                results.extend(batch_embeddings)

                # 批次间延迟
                if i + self.config.batch_size < len(texts):
                    await asyncio.sleep(self.config.retry_delay)

            return results

        except Exception as e:
            logger.error(f"Qwen embedding error: {e}")
            raise

    def get_dimension(self) -> int:
        """Qwen text-embedding-v4 维度是1536"""
        return 1536

class BGELocalEmbeddingProvider(EmbeddingProvider):
    """BGE本地模型嵌入提供者"""

    def __init__(self, config: EmbeddingConfig, eager_load: bool = False):
        super().__init__(config)
        self.model_path = config.bge_model_path
        self.device = config.bge_device
        self._model = None
        # 创建专用线程池
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # 如果启用预加载，立即加载模型
        if eager_load:
            self._load_model()

    def _load_model(self):
        """加载模型"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import os

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            logger.info(f"Loading BGE embedding model from {self.model_path}")

            self._model = SentenceTransformer(
                self.model_path,
                device=self.device,
                local_files_only=True
            )

            logger.info(f"BGE embedding model loaded successfully on {self.device}")

        except ImportError as e:
            logger.error(f"sentence_transformers not installed: {e}")
            raise RuntimeError("Please install sentence-transformers")

    def warmup(self):
        """预热模型（预加载）"""
        logger.info("Warming up BGE embedding model...")
        self._load_model()

        # 使用测试查询进行预热
        test_query = "测试查询"
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已经在事件循环中，使用线程池
                embedding = loop.run_in_executor(self._executor, self._sync_embed, test_query)
            else:
                # 如果不在事件循环中，直接调用
                embedding = self._sync_embed(test_query)
            logger.info("BGE embedding model warmed up successfully")
        except Exception as e:
            logger.warning(f"BGE embedding warmup test failed: {e}")

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model is not None

    async def embed(self, text: str) -> np.ndarray:
        """生成单个文本的嵌入"""
        # 使用专用线程池执行
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_embed, text)

    def _sync_embed(self, text: str) -> np.ndarray:
        """同步嵌入"""
        self._load_model()

        embedding = self._model.encode(
            text,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False
        )

        if self._validate_embedding(embedding):
            return embedding
        else:
            if self.config.skip_invalid_embeddings:
                return np.zeros(self.get_dimension(), dtype=np.float32)
            else:
                raise ValueError("Invalid embedding generated")

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量生成嵌入"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_embed_batch, texts)

    def _sync_embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """同步批量嵌入"""
        self._load_model()

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.config.normalize_embeddings,
            batch_size=self.config.batch_size,
            show_progress_bar=False
        )

        # 确保embeddings是2D数组格式
        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            elif embeddings.ndim > 2:
                # 如果维度太高，展平到2D
                embeddings = embeddings.reshape(len(texts), -1)

        results = []
        for i, emb in enumerate(embeddings):
            # 确保每个embedding都是1D数组
            if isinstance(emb, np.ndarray):
                if emb.ndim != 1:
                    emb = emb.flatten()

            if self._validate_embedding(emb):
                results.append(emb)
            elif self.config.skip_invalid_embeddings:
                results.append(np.zeros(self.get_dimension(), dtype=np.float32))
            else:
                raise ValueError(f"Invalid embedding in batch at index {i}")

        return results

    def get_dimension(self) -> int:
        """BGE-large-zh-v1.5 维度是1024"""
        return 1024

class UnifiedEmbeddingService:
    """
    统一嵌入服务（优化版）

    特性:
    - 统一的接口
    - 自动提供者选择
    - 统一的缓存管理
    - 自动降级
    - 向量质量验证
    - 文档类型自适应chunking（优化版新增）
    - 元数据增强（优化版新增）
    - 金融术语上下文增强（优化版新增）
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.cache = EmbeddingCache(self.config)

        # 只创建BGE提供者
        self.primary_provider = self._create_provider(self.config.primary_provider)

        # 不使用fallback
        self.fallback_provider = None

        # 初始化优化策略（如果可用）
        self.embedding_strategy = None
        if OPTIMIZED_EMBEDDING_AVAILABLE:
            try:
                self.embedding_strategy = create_embedding_strategy()
                logger.info("Optimized embedding strategy enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize optimized embedding strategy: {e}")

        logger.info(f"UnifiedEmbeddingService initialized with {self.config.primary_provider.value} (BGE only, optimized: {OPTIMIZED_EMBEDDING_AVAILABLE})")

    def _create_provider(self, provider_type: EmbeddingProviderType, eager_load: bool = False) -> EmbeddingProvider:
        """创建提供者实例"""
        if provider_type == EmbeddingProviderType.BGE_LOCAL:
            return BGELocalEmbeddingProvider(self.config, eager_load=eager_load)
        else:
            # 只支持BGE本地模型
            logger.warning(f"Unknown provider type: {provider_type}, falling back to BGE_LOCAL")
            return BGELocalEmbeddingProvider(self.config, eager_load=eager_load)

    async def embed(
        self,
        text: str,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        生成单个文本的嵌入

        Args:
            text: 输入文本
            use_cache: 是否使用缓存

        Returns:
            嵌入向量
        """
        # 尝试从缓存获取
        if use_cache:
            cached = self.cache.get(text, self.config.primary_provider.value)
            if cached is not None:
                return cached

        # 生成嵌入
        embedding = await self._embed_with_fallback(text)

        # 缓存结果
        if use_cache:
            self.cache.set(text, self.config.primary_provider.value, embedding)

        return embedding

    async def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[np.ndarray]:
        """
        批量生成嵌入

        Args:
            texts: 输入文本列表
            use_cache: 是否使用缓存

        Returns:
            嵌入向量列表
        """
        results = []

        # 尝试从缓存获取
        if use_cache:
            for text in texts:
                cached = self.cache.get(text, self.config.primary_provider.value)
                if cached is not None:
                    results.append(cached)
                else:
                    results.append(None)

            # 过滤出未缓存的文本
            uncached_indices = [i for i, emb in enumerate(results) if emb is None]
            uncached_texts = [texts[i] for i in uncached_indices]

            if uncached_texts:
                # 批量生成未缓存的嵌入
                new_embeddings = await self.primary_provider.embed_batch(uncached_texts)

                # 缓存并填充结果
                for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                    results[idx] = emb
                    self.cache.set(text, self.config.primary_provider.value, emb)

            # 移除None占位符
            results = [emb for emb in results if emb is not None]

        else:
            # 直接批量生成
            results = await self.primary_provider.embed_batch(texts)

        return results

    async def embed_with_metadata(
        self,
        text: str,
        metadata: Dict[str, Any],
        use_cache: bool = True
    ) -> np.ndarray:
        """
        使用元数据增强生成嵌入（优化版）

        优化内容：
        - 添加文档类型、券商、日期等元数据前缀
        - 增强金融术语上下文
        - 特殊处理表格和图表内容

        Args:
            text: 输入文本
            metadata: 元数据字典，可包含：
                - document_type: 文档类型
                - broker: 券商名称
                - date: 日期
                - section: 章节
                - title: 标题
                - table: 表格数据
                - chart: 图表数据
            use_cache: 是否使用缓存

        Returns:
            嵌入向量
        """
        # 如果优化策略可用，使用增强后的文本
        if self.embedding_strategy:
            try:
                enhanced_text = self.embedding_strategy.build_enhanced_chunk(text, metadata)
                logger.debug(f"Using enhanced text (len: {len(enhanced_text)} vs original: {len(text)})")
                text = enhanced_text
            except Exception as e:
                logger.warning(f"Failed to enhance text: {e}, using original")

        return await self.embed(text, use_cache=use_cache)

    async def embed_batch_with_metadata(
        self,
        texts: List[str],
        metadata_list: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> List[np.ndarray]:
        """
        批量使用元数据增强生成嵌入（优化版）

        Args:
            texts: 输入文本列表
            metadata_list: 元数据列表，与texts一一对应
            use_cache: 是否使用缓存

        Returns:
            嵌入向量列表
        """
        if len(texts) != len(metadata_list):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of metadata ({len(metadata_list)})")

        # 如果优化策略可用，增强所有文本
        if self.embedding_strategy:
            try:
                enhanced_texts = []
                for text, metadata in zip(texts, metadata_list):
                    enhanced = self.embedding_strategy.build_enhanced_chunk(text, metadata)
                    enhanced_texts.append(enhanced)
                texts = enhanced_texts
                logger.debug(f"Enhanced {len(texts)} texts with metadata")
            except Exception as e:
                logger.warning(f"Failed to enhance batch texts: {e}, using originals")

        return await self.embed_batch(texts, use_cache=use_cache)

    def detect_document_type(
        self,
        filename: str,
        content_preview: str = ""
    ) -> Optional[str]:
        """
        检测文档类型（优化版）

        Args:
            filename: 文件名
            content_preview: 内容预览（可选）

        Returns:
            文档类型字符串或None
        """
        if not self.embedding_strategy:
            return None

        try:
            doc_type = self.embedding_strategy.detect_document_type(filename, content_preview)
            return doc_type.value
        except Exception as e:
            logger.warning(f"Failed to detect document type: {e}")
            return None

    def get_adaptive_chunk_size(
        self,
        document_type: str
    ) -> tuple:
        """
        获取自适应chunk大小（优化版）

        Args:
            document_type: 文档类型字符串

        Returns:
            (chunk_size, chunk_overlap) 元组
        """
        if not self.embedding_strategy:
            return (512, 50)  # 默认值

        try:
            # 转换字符串为DocumentType枚举
            from enum import Enum
            doc_type = DocumentType(document_type)
            return self.embedding_strategy.get_adaptive_chunk_size(doc_type)
        except Exception as e:
            logger.warning(f"Failed to get adaptive chunk size: {e}, using defaults")
            return (512, 50)

    async def _embed_with_fallback(self, text: str) -> np.ndarray:
        """带降级的嵌入生成"""
        last_error = None

        # 尝试主提供者
        for attempt in range(self.config.max_retries):
            try:
                return await self.primary_provider.embed(text)
            except Exception as e:
                last_error = e
                logger.warning(f"Primary provider attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)

        # 尝试降级提供者
        if self.fallback_provider:
            try:
                logger.info("Falling back to fallback provider")
                return await self.fallback_provider.embed(text)
            except Exception as e:
                logger.error(f"Fallback provider also failed: {e}")

        # 所有尝试都失败
        raise RuntimeError(f"All embedding providers failed. Last error: {last_error}")

    def get_dimension(self) -> int:
        """获取嵌入维度"""
        return self.primary_provider.get_dimension()

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache.get_stats()

    def warmup(self):
        """预热所有模型（预加载）"""
        logger.info("="*80)
        logger.info("  Warming up embedding models...")
        logger.info("="*80)

        # 预热embedding模型
        if hasattr(self.primary_provider, 'warmup'):
            self.primary_provider.warmup()
        else:
            logger.warning("Primary provider does not support warmup")

        logger.info("="*80)
        logger.info("  Model warmup completed")
        logger.info("="*80)

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        if hasattr(self.primary_provider, 'is_model_loaded'):
            return self.primary_provider.is_model_loaded()
        return False

# 全局服务实例
_global_embedding_service: Optional[UnifiedEmbeddingService] = None

def get_embedding_service() -> UnifiedEmbeddingService:
    """获取全局嵌入服务实例"""
    global _global_embedding_service
    if _global_embedding_service is None:
        # 从settings创建配置 - 只使用BGE本地模型
        config = EmbeddingConfig(
            primary_provider=EmbeddingProviderType.BGE_LOCAL,
            fallback_provider=None,  # 不使用fallback
            bge_model_path=settings.bge_embedding_model_path,
            bge_device=settings.bge_embedding_device,
            batch_size=settings.bge_embedding_batch_size,
            enable_cache=True,
            enable_fallback=False  # 不启用降级
        )
        _global_embedding_service = UnifiedEmbeddingService(config)
    return _global_embedding_service

# 便捷函数
async def get_embedding(text: str) -> np.ndarray:
    """获取文本嵌入(便捷函数)"""
    service = get_embedding_service()
    return await service.embed(text)

async def get_embeddings(texts: List[str]) -> List[np.ndarray]:
    """批量获取嵌入(便捷函数)"""
    service = get_embedding_service()
    return await service.embed_batch(texts)
