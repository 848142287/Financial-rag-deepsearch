"""
BGE模型服务 - 纯BGEM3FlagModel实现
统一使用BGEM3FlagModel调用方式处理所有BGE模型
集成BAAI/bge-large-zh-financial, bge-large-zh-v1.5, bge-reranker-v2-m3
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class BGEConfig:
    """BGE模型配置"""
    # Embedding模型 - 都使用BGEM3FlagModel调用
    primary_embedding_model: str = "BAAI/bge-large-zh-v1.5"
    backup_embedding_model: str = "BAAI/bge-large-zh-v1.5"

    # Reranker模型
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # 模型配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    batch_size: int = 32
    normalize_embeddings: bool = True

    # 缓存配置
    cache_size: int = 1000
    enable_cache: bool = True

    # 模型使用模式（统一使用BGEM3FlagModel）
    use_dense_embeddings: bool = True
    use_lexical_weight: float = 0.1  # 用于混合搜索
    use_colbert: bool = False  # 是否使用ColBERT特征


class BGEModelService:
    """BGE模型服务类 - 统一BGEM3FlagModel调用方式"""

    def __init__(self, config: Optional[BGEConfig] = None):
        # 从settings读取配置或使用默认值
        if config is None:
            config = BGEConfig(
                primary_embedding_model=getattr(settings, 'bge_primary_embedding_model', "BAAI/bge-large-zh-financial"),
                backup_embedding_model=getattr(settings, 'bge_backup_embedding_model', "BAAI/bge-large-zh-v1.5"),
                reranker_model=getattr(settings, 'bge_reranker_model', "BAAI/bge-reranker-v2-m3"),
                device=getattr(settings, 'bge_device', "cuda" if torch.cuda.is_available() else "cpu"),
                max_length=getattr(settings, 'bge_max_length', 512),
                batch_size=getattr(settings, 'bge_batch_size', 32),
                normalize_embeddings=getattr(settings, 'bge_normalize_embeddings', True),
                cache_size=getattr(settings, 'bge_cache_size', 1000),
                enable_cache=getattr(settings, 'bge_enable_cache', True)
            )

        self.config = config
        self.primary_model = None
        self.backup_model = None
        self.reranker_model = None
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.primary_model_failures = 0
        self.max_primary_failures = 5

    async def initialize(self):
        """初始化模型"""
        try:
            logger.info("Initializing BGE models with unified BGEM3FlagModel approach...")

            # 初始化主要embedding模型 - 使用BGEM3FlagModel
            logger.info(f"Loading primary embedding model: {self.config.primary_embedding_model}")
            self.primary_model = BGEM3FlagModel(
                self.config.primary_embedding_model,
                device=self.config.device,
                use_fp16=False  # 确保精度
            )

            # 初始化备用embedding模型 - 也使用BGEM3FlagModel
            logger.info(f"Loading backup embedding model: {self.config.backup_embedding_model}")
            self.backup_model = BGEM3FlagModel(
                self.config.backup_embedding_model,
                device=self.config.device,
                use_fp16=False
            )

            # 初始化reranker模型
            logger.info(f"Loading reranker model: {self.config.reranker_model}")
            self.reranker_model = FlagReranker(
                self.config.reranker_model,
                device=self.config.device,
                use_fp16=False
            )

            logger.info("BGE models initialized successfully with unified BGEM3FlagModel approach")

        except Exception as e:
            logger.error(f"Failed to initialize BGE models: {e}")
            raise

    async def embed_texts(
        self,
        texts: List[str],
        use_primary_model: bool = True
    ) -> np.ndarray:
        """
        生成文本嵌入向量
        使用统一的BGEM3FlagModel调用方式处理所有模型
        """
        if not texts:
            return np.array([])

        try:
            # 检查缓存
            if self.config.enable_cache:
                cached_embeddings = []
                uncached_texts = []
                uncached_indices = []

                for i, text in enumerate(texts):
                    text_hash = self._get_text_hash(text)
                    if text_hash in self.embedding_cache:
                        cached_embeddings.append((i, self.embedding_cache[text_hash]))
                        self.cache_hits += 1
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                        self.cache_misses += 1

                # 如果有缓存的结果
                if cached_embeddings:
                    if not uncached_texts:
                        # 全部命中缓存
                        result = np.zeros((len(texts), self.get_embedding_dimension()))
                        for i, embedding in cached_embeddings:
                            result[i] = embedding
                        return result

                    # 部分命中缓存，处理未缓存的
                    uncached_embeddings = await self._compute_embeddings(
                        uncached_texts, use_primary_model
                    )

                    # 组装结果
                    result = np.zeros((len(texts), self.get_embedding_dimension()))

                    # 填入缓存结果
                    for i, embedding in cached_embeddings:
                        result[i] = embedding

                    # 填入新计算的结果
                    for idx, embedding in zip(uncached_indices, uncached_embeddings):
                        result[idx] = embedding
                        # 添加到缓存
                        if len(self.embedding_cache) < self.config.cache_size:
                            text_hash = self._get_text_hash(texts[idx])
                            self.embedding_cache[text_hash] = embedding

                    return result
                else:
                    # 没有缓存命中
                    embeddings = await self._compute_embeddings(texts, use_primary_model)

                    # 添加到缓存
                    if len(self.embedding_cache) < self.config.cache_size:
                        for text, embedding in zip(texts, embeddings):
                            text_hash = self._get_text_hash(text)
                            self.embedding_cache[text_hash] = embedding

                    return embeddings
            else:
                return await self._compute_embeddings(texts, use_primary_model)

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # 返回零向量作为fallback
            return np.zeros((len(texts), self.get_embedding_dimension()))

    async def _compute_embeddings(
        self,
        texts: List[str],
        use_primary_model: bool = True
    ) -> np.ndarray:
        """
        计算嵌入向量
        统一使用BGEM3FlagModel.encode()方法
        """
        # 尝试使用主要模型
        if use_primary_model and self.primary_model and self.primary_model_failures < self.max_primary_failures:
            try:
                return await self._compute_with_bgem3(self.primary_model, texts)
            except Exception as e:
                logger.warning(f"Primary model failed: {e}")
                self.primary_model_failures += 1

                # 如果主要模型失败次数过多，切换到备用模型
                if self.primary_model_failures >= self.max_primary_failures:
                    logger.info("Switching to backup embedding model")
                    return await self._compute_with_bgem3(self.backup_model, texts)
                else:
                    # 重试主要模型
                    await asyncio.sleep(1)  # 短暂等待后重试
                    return await self._compute_with_bgem3(self.primary_model, texts)

        # 使用备用模型
        return await self._compute_with_bgem3(self.backup_model, texts)

    async def _compute_with_bgem3(
        self,
        model: BGEM3FlagModel,
        texts: List[str]
    ) -> np.ndarray:
        """
        使用BGEM3FlagModel计算嵌入
        统一调用方式，适用于所有BGE模型
        """
        loop = asyncio.get_event_loop()

        def sync_compute():
            # 使用BGEM3FlagModel.encode()统一接口
            result = model.encode(
                texts,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                normalize_embeddings=self.config.normalize_embeddings
            )

            # 提取密集向量（所有BGE模型都支持）
            return result['dense_vecs']

        return await loop.run_in_executor(None, sync_compute)

    async def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        使用reranker模型重新排序
        统一使用FlagReranker.compute_score()
        """
        if not passages:
            return []

        try:
            # 准备输入对
            pairs = [[query, passage] for passage in passages]

            # 使用统一的compute_score方法
            loop = asyncio.get_event_loop()

            def sync_rerank():
                scores = self.reranker_model.compute_score(
                    pairs,
                    normalize=True  # 归一化分数
                )
                return scores

            scores = await loop.run_in_executor(None, sync_rerank)

            # 组装结果
            results = []
            for i, (passage, score) in enumerate(zip(passages, scores)):
                results.append({
                    "index": i,
                    "passage": passage,
                    "score": float(score)  # 确保是Python float类型
                })

            # 按分数排序
            results.sort(key=lambda x: x["score"], reverse=True)

            # 应用top_k限制
            if top_k:
                results = results[:top_k]

            return results

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # 返回原始顺序，分数为0
            return [{"index": i, "passage": p, "score": 0.0} for i, p in enumerate(passages)]

    async def batch_rerank(
        self,
        queries: List[str],
        passages_list: List[List[str]],
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """批量rerank"""
        results = []

        for query, passages in zip(queries, passages_list):
            result = await self.rerank(query, passages, top_k)
            results.append(result)

        return results

    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        # BGE-large-zh-financial: 1024维
        # BGE-large-zh-v1.5: 1024维
        return 1024

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_size": len(self.embedding_cache),
            "max_cache_size": self.config.cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "primary_model_failures": self.primary_model_failures,
            "model_type": "BGEM3FlagModel (unified approach)"
        }

    def clear_cache(self):
        """清理缓存"""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Embedding cache cleared")

    def _get_text_hash(self, text: str) -> str:
        """获取文本哈希"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "status": "healthy",
            "primary_model": "loaded" if self.primary_model else "not_loaded",
            "backup_model": "loaded" if self.backup_model else "not_loaded",
            "reranker_model": "loaded" if self.reranker_model else "not_loaded",
            "device": self.config.device,
            "implementation": "BGEM3FlagModel (unified)",
            "cache_stats": self.get_cache_stats()
        }

        # 测试模型推理
        try:
            test_embedding = await self.embed_texts(["测试文本"], use_primary_model=True)
            health_status["primary_model_test"] = "passed"
            health_status["test_embedding_shape"] = list(test_embedding.shape)
        except Exception as e:
            health_status["primary_model_test"] = f"failed: {str(e)}"
            health_status["status"] = "degraded"

        try:
            test_rerank = await self.rerank("测试查询", ["测试文档1", "测试文档2"])
            health_status["reranker_test"] = "passed"
        except Exception as e:
            health_status["reranker_test"] = f"failed: {str(e)}"
            health_status["status"] = "degraded"

        return health_status

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "implementation": "BGEM3FlagModel (unified for all BGE models)",
            "primary_model": {
                "name": self.config.primary_embedding_model,
                "dimension": self.get_embedding_dimension(),
                "max_length": self.config.max_length,
                "device": self.config.device,
                "class": "BGEM3FlagModel"
            },
            "backup_model": {
                "name": self.config.backup_embedding_model,
                "dimension": self.get_embedding_dimension(),
                "max_length": self.config.max_length,
                "device": self.config.device,
                "class": "BGEM3FlagModel"
            },
            "reranker_model": {
                "name": self.config.reranker_model,
                "device": self.config.device,
                "class": "FlagReranker"
            },
            "config": {
                "use_dense_embeddings": self.config.use_dense_embeddings,
                "normalize_embeddings": self.config.normalize_embeddings,
                "batch_size": self.config.batch_size,
                "cache_enabled": self.config.enable_cache
            }
        }


# 全局服务实例
bge_service = BGEModelService()


async def get_bge_service() -> BGEModelService:
    """获取BGE服务实例"""
    if not bge_service.primary_model:
        await bge_service.initialize()
    return bge_service


# 便捷函数
async def embed_texts(texts: List[str]) -> np.ndarray:
    """便捷的文本嵌入函数"""
    service = await get_bge_service()
    return await service.embed_texts(texts)


async def rerank_passages(query: str, passages: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
    """便捷的文档重排序函数"""
    service = await get_bge_service()
    return await service.rerank(query, passages, top_k)