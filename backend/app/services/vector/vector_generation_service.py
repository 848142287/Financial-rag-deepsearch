"""
独立的向量生成服务 - 从CoreServiceIntegrator拆分
负责批量并行向量生成
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class VectorGenerationService:
    """
    向量生成服务

    功能：
    - 批量并行向量生成
    - 向量质量验证
    - 性能监控
    """

    def __init__(self, embedding_service, batch_size: int = 50, max_concurrent: int = 10):
        self.embedding_service = embedding_service
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent

    async def generate_vectors_batch(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        批量并行生成向量

        Args:
            chunks: chunk列表
            document_id: 文档ID

        Returns:
            带有向量的chunk列表
        """
        if not chunks:
            return []

        total_chunks = len(chunks)
        logger.info(f"🔢 批量并行向量生成: {total_chunks} chunks, batch_size={self.batch_size}")

        start_time = datetime.now()

        # 分批
        batches = [
            chunks[i:i + self.batch_size]
            for i in range(0, total_chunks, self.batch_size)
        ]

        logger.info(f"📦 分为 {len(batches)} 个batch")

        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_batch(batch: List[Dict], batch_idx: int) -> List[Dict]:
            """处理一个batch"""
            async with semaphore:
                try:
                    texts = [chunk['content'] for chunk in batch]
                    embeddings = await self.embedding_service.embed_batch(texts)

                    result_batch = []
                    for chunk, embedding in zip(batch, embeddings):
                        result_batch.append({
                            **chunk,
                            'embedding': embedding if embedding else None
                        })

                    logger.info(f"✅ Batch {batch_idx + 1}/{len(batches)} 完成: {len(batch)} chunks")
                    return result_batch

                except Exception as e:
                    logger.error(f"❌ Batch {batch_idx + 1} 失败: {e}")
                    return [{**chunk, 'embedding': None} for chunk in batch]

        # 并发执行所有batch
        batch_tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # 合并结果
        chunks_with_vectors = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch异常: {result}")
                continue
            chunks_with_vectors.extend(result)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ 向量生成完成: {len(chunks_with_vectors)} chunks, 耗时 {duration:.2f}秒")

        return chunks_with_vectors

    def validate_embeddings(self, chunks_with_vectors: List[Dict]) -> Dict[str, Any]:
        """验证向量质量"""
        valid_count = 0
        invalid_count = 0
        total_dim = 0

        for chunk in chunks_with_vectors:
            embedding = chunk.get('embedding')
            if embedding is not None and hasattr(embedding, '__len__') and len(embedding) > 0:
                valid_count += 1
                if total_dim == 0:
                    total_dim = len(embedding)
            else:
                invalid_count += 1

        return {
            'total': len(chunks_with_vectors),
            'valid': valid_count,
            'invalid': invalid_count,
            'dimension': total_dim,
            'valid_rate': valid_count / len(chunks_with_vectors) if chunks_with_vectors else 0
        }
