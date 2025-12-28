"""
Embedding服务兼容层
提供向后兼容的接口，底层使用Qwen2.5-VL-Embedding
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

from app.services.qwen_embedding_service import (
    QwenEmbeddingService,
    qwen_embedding_service
)

logger = logging.getLogger(__name__)


class EmbeddingServiceWrapper:
    """
    Embedding服务包装器
    提供兼容旧代码的接口，同时使用Qwen2.5-VL-Embedding
    """

    def __init__(self):
        self._service = qwen_embedding_service
        self._initialized = False

    async def initialize(self):
        """初始化服务"""
        if not self._initialized:
            self._initialized = True
            logger.info("EmbeddingService initialized")

    async def get_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量列表
        """
        embedding = await self._service.encode_single(text)
        return embedding.tolist()

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的嵌入向量

        Args:
            texts: 输入文本列表

        Returns:
            嵌入向量列表
        """
        embeddings = await self._service.encode(texts)
        return [emb.tolist() for emb in embeddings]

    async def encode_text(self, text: str) -> List[float]:
        """
        编码文本（兼容旧接口）

        Args:
            text: 输入文本

        Returns:
            嵌入向量列表
        """
        return await self.get_embedding(text)

    async def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量编码文本（兼容旧接口）

        Args:
            texts: 输入文本列表

        Returns:
            嵌入向量列表
        """
        return await self.get_embeddings(texts)

    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量维度

        Returns:
            向量维度
        """
        return self._service.get_embedding_dimension()

    async def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        计算两个文本的相似度

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            相似度分数
        """
        return await self._service.compute_similarity(text1, text2)

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[tuple]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前K个结果

        Returns:
            重排序后的结果列表 [(index, score), ...]
        """
        return await self._service.rerank(query, documents, top_k)

    async def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[str]]
    ) -> List[List[tuple]]:
        """
        批量重排序

        Args:
            queries: 查询文本列表
            documents_list: 文档列表的列表

        Returns:
            重排序结果列表
        """
        return await self._service.batch_rerank(queries, documents_list)

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            健康状态信息
        """
        return await self._service.health_check()

    async def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return await self._service.get_model_info()


# 创建全局服务实例
embedding_service = EmbeddingServiceWrapper()

# 导出类供其他模块使用
RerankService = QwenEmbeddingService
EmbeddingService = QwenEmbeddingService
