"""
向量嵌入生成器
为文档处理流水线提供向量生成功能
"""

from app.core.structured_logging import get_structured_logger
from typing import List
import numpy as np

from app.services.qwen_embedding_service import qwen_embedding_service as embedding_service

logger = get_structured_logger(__name__)


class EmbeddingGenerator:
    """向量嵌入生成器"""

    def __init__(self):
        self.embedding_service = embedding_service
        self._initialized = False

    async def ensure_initialized(self):
        """确保服务已初始化"""
        if not self._initialized:
            await self.embedding_service.initialize()
            self._initialized = True
            logger.info("EmbeddingGenerator initialized successfully")

    def generate_embedding(self, text: str) -> List[float]:
        """
        生成文本的向量嵌入

        Args:
            text: 输入文本

        Returns:
            向量嵌入
        """
        try:
            # 使用同步方式调用异步服务（在Celery任务中）
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                embedding = loop.run_until_complete(
                    self.embedding_service.get_embedding(text)
                )
                return embedding
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"生成嵌入失败: {e}")
            # 返回零向量作为fallback
            dimension = 1024  # Qwen2.5-VL-Embedding的维度
            return np.zeros(dimension).tolist()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本向量嵌入

        Args:
            texts: 输入文本列表

        Returns:
            向量嵌入列表
        """
        try:
            # 使用同步方式调用异步服务（在Celery任务中）
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                embeddings = loop.run_until_complete(
                    self.embedding_service.get_embeddings(texts)
                )
                return embeddings
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"批量生成嵌入失败: {e}")
            # 返回零向量作为fallback
            dimension = 1024  # Qwen2.5-VL-Embedding的维度
            return [np.zeros(dimension).tolist() for _ in texts]

    async def generate_embedding_async(self, text: str) -> List[float]:
        """
        异步生成文本的向量嵌入

        Args:
            text: 输入文本

        Returns:
            向量嵌入
        """
        await self.ensure_initialized()
        return await self.embedding_service.get_embedding(text)

    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """
        异步批量生成文本向量嵌入

        Args:
            texts: 输入文本列表

        Returns:
            向量嵌入列表
        """
        await self.ensure_initialized()
        return await self.embedding_service.get_embeddings(texts)

    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        return 1024  # Qwen2.5-VL-Embedding的维度

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        计算两个嵌入向量的余弦相似度

        Args:
            embedding1: 第一个向量
            embedding2: 第二个向量

        Returns:
            相似度分数
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return 0.0