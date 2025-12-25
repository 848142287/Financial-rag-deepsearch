"""
重排序服务
使用BGE-reranker模型对搜索结果进行重排序，提高检索准确性
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from FlagEmbedding import FlagReranker

from app.core.config import settings
from app.services.bge_service import BGEModelService

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """重排序结果"""
    original_indices: List[int]
    reranked_indices: List[int]
    scores: List[float]
    original_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = None


class RerankService:
    """重排序服务"""

    def __init__(self):
        self.reranker = None
        self.bge_service = None
        self.max_length = 512
        self.batch_size = 16

    async def initialize(self):
        """初始化重排序服务"""
        try:
            # 初始化BGE reranker
            logger.info(f"Loading BGE reranker model: {settings.bge_reranker_model}")
            self.reranker = FlagReranker(
                settings.bge_reranker_model,
                device=settings.bge_device,
                use_fp16=False
            )
            logger.info("BGE reranker loaded successfully")

            # 初始化BGE服务（用于生成查询嵌入）
            from app.services.bge_service import BGEModelService, BGEConfig
            bge_config = BGEConfig(
                primary_embedding_model=settings.bge_primary_embedding_model,
                device=settings.bge_device,
                max_length=self.max_length,
                batch_size=self.batch_size
            )
            self.bge_service = BGEModelService(bge_config)
            await self.bge_service.initialize()

        except Exception as e:
            logger.error(f"Failed to initialize rerank service: {e}")
            raise

    async def rerank_documents(
        self,
        query: str,
        documents: List[str],
        original_scores: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        return_scores: bool = True
    ) -> RerankResult:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 候选文档列表
            original_scores: 原始相似度分数（可选）
            top_k: 返回前k个结果
            return_scores: 是否返回标准化分数

        Returns:
            RerankResult: 重排序结果
        """
        try:
            if not documents:
                return RerankResult(
                    original_indices=[],
                    reranked_indices=[],
                    scores=[]
                )

            logger.info(f"Reranking {len(documents)} documents for query: {query[:50]}...")

            # 准备查询-文档对
            pairs = [[query, doc] for doc in documents]

            # 使用BGE reranker计算分数
            loop = asyncio.get_event_loop()

            def sync_rerank():
                scores = self.reranker.compute_score(
                    pairs,
                    normalize=return_scores
                )
                return scores

            scores = await loop.run_in_executor(None, sync_rerank)

            # 获取原始索引
            original_indices = list(range(len(documents)))

            # 根据分数排序
            sorted_indices = sorted(
                range(len(documents)),
                key=lambda i: scores[i],
                reverse=True
            )

            # 应用top_k限制
            if top_k:
                sorted_indices = sorted_indices[:top_k]

            # 提取对应分数
            reranked_scores = [scores[i] for i in sorted_indices]

            result = RerankResult(
                original_indices=original_indices,
                reranked_indices=sorted_indices,
                scores=reranked_scores,
                original_scores=original_scores,
                metadata={
                    "query": query,
                    "document_count": len(documents),
                    "reranker_model": settings.bge_reranker_model
                }
            )

            logger.info(f"Reranking completed. Top score: {max(reranked_scores):.4f}, "
                       f"Average score: {np.mean(reranked_scores):.4f}")

            return result

        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # 返回原始顺序作为回退
            return RerankResult(
                original_indices=list(range(len(documents))),
                reranked_indices=list(range(len(documents))),
                scores=[0.0] * len(documents),
                original_scores=original_scores,
                metadata={"error": str(e)}
            )

    async def hybrid_rerank(
        self,
        query: str,
        documents: List[str],
        vector_scores: List[float],
        alpha: float = 0.7,  # 向量分数权重
        top_k: Optional[int] = None
    ) -> RerankResult:
        """
        混合重排序（向量分数 + 重排序分数）

        Args:
            query: 查询文本
            documents: 候选文档列表
            vector_scores: 向量相似度分数
            alpha: 向量分数权重 (0-1)
            top_k: 返回前k个结果

        Returns:
            RerankResult: 重排序结果
        """
        try:
            logger.info(f"Hybrid reranking {len(documents)} documents with alpha={alpha}")

            # 获取重排序分数
            rerank_result = await self.rerank_documents(
                query, documents, top_k=len(documents)
            )

            # 标准化分数
            if len(vector_scores) == len(rerank_result.scores):
                vector_normalized = np.array(vector_scores)
                rerank_normalized = np.array(rerank_result.scores)

                # Min-max标准化
                vector_normalized = (vector_normalized - vector_normalized.min()) / (vector_normalized.max() - vector_normalized.min() + 1e-8)
                rerank_normalized = (rerank_normalized - rerank_normalized.min()) / (rerank_normalized.max() - rerank_normalized.min() + 1e-8)

                # 混合分数
                hybrid_scores = alpha * vector_normalized + (1 - alpha) * rerank_normalized

                # 根据混合分数排序
                hybrid_indices = sorted(
                    range(len(documents)),
                    key=lambda i: hybrid_scores[i],
                    reverse=True
                )

                # 应用top_k限制
                if top_k:
                    hybrid_indices = hybrid_indices[:top_k]

                hybrid_result_scores = [hybrid_scores[i] for i in hybrid_indices]

                logger.info(f"Hybrid reranking completed. Top score: {max(hybrid_result_scores):.4f}")

                return RerankResult(
                    original_indices=rerank_result.original_indices,
                    reranked_indices=hybrid_indices,
                    scores=hybrid_result_scores,
                    original_scores=vector_scores,
                    metadata={
                        **rerank_result.metadata,
                        "hybrid_alpha": alpha,
                        "vector_max": vector_normalized.max(),
                        "rerank_max": rerank_normalized.max()
                    }
                )
            else:
                logger.warning("Vector scores and rerank scores length mismatch, using rerank only")
                return rerank_result

        except Exception as e:
            logger.error(f"Error in hybrid reranking: {e}")
            return await self.rerank_documents(query, documents, vector_scores, top_k)

    async def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        批量重排序

        Args:
            queries: 查询列表
            documents_list: 文档列表的列表
            top_k: 每个查询返回前k个结果

        Returns:
            List[RerankResult]: 重排序结果列表
        """
        try:
            logger.info(f"Batch reranking {len(queries)} queries")

            results = []
            for query, documents in zip(queries, documents_list):
                result = await self.rerank_documents(query, documents, top_k=top_k)
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in batch reranking: {e}")
            # 返回原始顺序
            return [
                RerankResult(
                    original_indices=list(range(len(docs))),
                    reranked_indices=list(range(len(docs))),
                    scores=[0.0] * len(docs)
                )
                for docs in documents_list
            ]

    async def validate_reranking(
        self,
        query: str,
        documents: List[str],
        relevant_indices: List[int],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        验证重排序效果

        Args:
            query: 查询文本
            documents: 文档列表
            relevant_indices: 已知相关文档的索引
            threshold: 相关性阈值

        Returns:
            Dict: 验证结果
        """
        try:
            rerank_result = await self.rerank_documents(query, documents)

            # 计算MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for i, idx in enumerate(rerank_result.reranked_indices[:10]):
                if idx in relevant_indices:
                    mrr = 1.0 / (i + 1)
                    break

            # 计算precision@k
            precision_at_1 = 1.0 if rerank_result.reranked_indices[0] in relevant_indices else 0.0
            precision_at_5 = len(set(rerank_result.reranked_indices[:5]) & set(relevant_indices)) / 5.0
            precision_at_10 = len(set(rerank_result.reranked_indices[:10]) & set(relevant_indices)) / 10.0

            return {
                "mrr": mrr,
                "precision_at_1": precision_at_1,
                "precision_at_5": precision_at_5,
                "precision_at_10": precision_at_10,
                "rerank_successful": True,
                "threshold_met": mrr >= threshold
            }

        except Exception as e:
            logger.error(f"Error in reranking validation: {e}")
            return {
                "mrr": 0.0,
                "precision_at_1": 0.0,
                "precision_at_5": 0.0,
                "precision_at_10": 0.0,
                "rerank_successful": False,
                "threshold_met": False,
                "error": str(e)
            }

    def get_rerank_stats(self) -> Dict[str, Any]:
        """获取重排序统计信息"""
        try:
            return {
                "reranker_model": settings.bge_reranker_model,
                "device": settings.bge_device,
                "max_length": self.max_length,
                "batch_size": self.batch_size,
                "status": "initialized" if self.reranker else "not_initialized"
            }
        except Exception as e:
            logger.error(f"Error getting rerank stats: {e}")
            return {"status": "error", "error": str(e)}


# 全局实例
rerank_service = RerankService()


async def get_rerank_service() -> RerankService:
    """获取重排序服务实例"""
    if not rerank_service.reranker:
        await rerank_service.initialize()
    return rerank_service


# 便捷函数
async def rerank_search_results(
    query: str,
    documents: List[str],
    original_scores: List[float] = None,
    top_k: int = 10
) -> RerankResult:
    """便捷的重排序函数"""
    service = await get_rerank_service()
    return await service.rerank_documents(
        query, documents, original_scores, top_k
    )