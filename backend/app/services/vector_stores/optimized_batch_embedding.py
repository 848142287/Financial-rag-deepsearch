"""
优化的批量向量嵌入服务
集成了 Multimodal_RAG 的批处理优化，使用线程池并行处理
显著提升嵌入性能，同时保持与现有系统的兼容性
"""

import time
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.config import settings
from app.services.qwen_embedding_service import qwen_embedding_service

logger = logging.getLogger(__name__)


class OptimizedBatchEmbedding:
    """
    优化的批量嵌入服务

    功能：
    1. 高效批处理：使用线程池并行处理多个批次
    2. 性能监控：实时统计速率和进度
    3. 顺序保证：确保结果与输入顺序一致
    4. 错误处理：批次失败时终止所有任务
    """

    def __init__(self, default_chunk_size: int = 20):
        """
        初始化优化批量嵌入服务

        Args:
            default_chunk_size: 默认批次大小
        """
        self.default_chunk_size = default_chunk_size
        logger.info(f"初始化优化批量嵌入服务，默认批次大小: {default_chunk_size}")

    def embed_batch_documents(
        self,
        texts: List[str],
        chunk_size: int = None,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        批量嵌入文档，使用线程池并行处理

        Args:
            texts: 要嵌入的文本列表
            chunk_size: 每批处理的文本数量（默认使用 default_chunk_size）
            show_progress: 是否显示进度信息

        Returns:
            List[List[float]]: 嵌入向量列表，顺序与输入文本列表相同

        Raises:
            RuntimeError: 部分文本嵌入失败
        """
        if not texts:
            return []

        chunk_size = chunk_size or self.default_chunk_size
        total_texts = len(texts)

        if show_progress:
            logger.info(f"开始处理总计 {total_texts} 个文本的向量嵌入，批次大小: {chunk_size}")

        # 预分配结果列表，确保结果顺序与输入顺序一致
        all_embeddings = [None] * total_texts
        start_time_total = time.time()

        # 定义批次处理任务
        def process_batch(batch_idx, batch_texts):
            """处理单个批次"""
            start_idx = batch_idx * chunk_size
            end_idx = min(start_idx + len(batch_texts), total_texts)
            batch_indices = list(range(start_idx, end_idx))

            start_time_batch = time.time()
            if show_progress:
                logger.info(f"开始处理批次 {batch_idx+1}/{len(batches)}，文本索引 {start_idx} 到 {end_idx-1} (共 {len(batch_texts)} 个)")

            try:
                # 调用现有的嵌入服务
                batch_embeddings = qwen_embedding_service.embed_documents(batch_texts)

                # 验证返回的向量数量是否正确
                if len(batch_embeddings) != len(batch_texts):
                    raise ValueError(
                        f"模型返回的向量数量 ({len(batch_embeddings)}) 与请求数量 ({len(batch_texts)}) 不匹配"
                    )

                # 计算批次性能
                elapsed_batch = time.time() - start_time_batch
                rate = len(batch_texts) / elapsed_batch if elapsed_batch > 0 else 0
                if show_progress:
                    logger.info(
                        f"批次 {batch_idx+1} 完成，耗时: {elapsed_batch:.2f}秒, "
                        f"速率: {rate:.2f} 文本/秒"
                    )

                # 返回批次索引和结果
                return batch_idx, batch_indices, batch_embeddings

            except Exception as e:
                logger.error(f"批次 {batch_idx+1} 处理失败: {str(e)}")
                raise RuntimeError(f"批次 {batch_idx+1} 处理失败: {str(e)}") from e

        # 准备批次
        batches = []
        for i in range(0, total_texts, chunk_size):
            batch_end = min(i + chunk_size, total_texts)
            batches.append((i // chunk_size, texts[i:batch_end]))

        # 使用线程池并行处理
        with ThreadPoolExecutor() as executor:
            # 提交所有任务
            future_to_batch = {
                executor.submit(process_batch, batch_idx, batch_texts): batch_idx
                for batch_idx, batch_texts in batches
            }

            try:
                # 收集结果
                for future in as_completed(future_to_batch):
                    batch_idx, batch_indices, batch_embeddings = future.result()

                    for idx, embedding in zip(batch_indices, batch_embeddings):
                        all_embeddings[idx] = embedding

                    if show_progress:
                        logger.info(f"已完成批次 {batch_idx+1}/{len(batches)} 的结果收集")

            except Exception as e:
                # 取消所有未完成的任务
                for f in future_to_batch:
                    f.cancel()
                raise RuntimeError(f"部分文本嵌入失败,终止所有任务: {str(e)}") from e

        # 检查是否所有文本都成功处理
        if None in all_embeddings:
            failed_indices = [i for i, emb in enumerate(all_embeddings) if emb is None]
            raise RuntimeError(
                f"部分文本嵌入失败，失败的索引: {failed_indices}"
            )

        # 计算总体性能
        total_elapsed = time.time() - start_time_total
        total_rate = total_texts / total_elapsed if total_elapsed > 0 else 0

        if show_progress:
            logger.info(
                f"全部 {total_texts} 个文本嵌入完成，总耗时: {total_elapsed:.2f}秒, "
                f"平均速率: {total_rate:.2f} 文本/秒"
            )

        return all_embeddings

    async def embed_batch_documents_async(
        self,
        texts: List[str],
        chunk_size: int = None,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        异步批量嵌入文档

        Args:
            texts: 要嵌入的文本列表
            chunk_size: 每批处理的文本数量
            show_progress: 是否显示进度信息

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        # 对于异步场景，在线程池中执行同步方法
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.embed_batch_documents(texts, chunk_size, show_progress)
        )

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本

        Args:
            text: 查询文本

        Returns:
            List[float]: 嵌入向量
        """
        return qwen_embedding_service.embed_query(text)

    async def embed_query_async(self, text: str) -> List[float]:
        """
        异步嵌入单个查询文本

        Args:
            text: 查询文本

        Returns:
            List[float]: 嵌入向量
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)


# 全局实例
optimized_batch_embedding = OptimizedBatchEmbedding(default_chunk_size=20)


# 便捷函数
def embed_batch_documents(
    texts: List[str],
    chunk_size: int = 20,
    show_progress: bool = True
) -> List[List[float]]:
    """
    便捷函数：批量嵌入文档

    Args:
        texts: 要嵌入的文本列表
        chunk_size: 每批处理的文本数量
        show_progress: 是否显示进度信息

    Returns:
        List[List[float]]: 嵌入向量列表
    """
    return optimized_batch_embedding.embed_batch_documents(texts, chunk_size, show_progress)


async def embed_batch_documents_async(
    texts: List[str],
    chunk_size: int = 20,
    show_progress: bool = True
) -> List[List[float]]:
    """
    便捷函数：异步批量嵌入文档

    Args:
        texts: 要嵌入的文本列表
        chunk_size: 每批处理的文本数量
        show_progress: 是否显示进度信息

    Returns:
        List[List[float]]: 嵌入向量列表
    """
    return await optimized_batch_embedding.embed_batch_documents_async(texts, chunk_size, show_progress)
