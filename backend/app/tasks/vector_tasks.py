"""
向量化与检索任务
使用BGE-Financial模型进行文本嵌入，存储到Milvus向量数据库
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from celery import current_task
from celery.exceptions import Retry

from app.core.async_tasks.celery_app import celery_app
from app.services.qwen_embedding_service import QwenEmbeddingService as EmbeddingService, RerankService
from app.services.milvus_service import MilvusService
from app.core.config import settings
from app.core.database import get_db
from app.models.document import Document, DocumentChunk, VectorStorage

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,)
)
def vectorize_content(self, document_id: str) -> Dict[str, Any]:
    """
    向量化文档内容任务
    """
    try:
        logger.info(f"Starting vectorization for document_id: {document_id}")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Initializing vectorization"}
        )

        # 异步执行向量化
        result = asyncio.run(_vectorize_content_async(document_id, self))

        logger.info(f"Content vectorization completed for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Content vectorization failed for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=60)


async def _vectorize_content_async(document_id: str, task) -> Dict[str, Any]:
    """异步执行内容向量化"""
    embedding_service = EmbeddingService()
    milvus_service = MilvusService()

    # 获取文档块
    async with get_db() as db:
        from sqlalchemy import select
        result = await db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        chunks = result.scalars().all()

    if not chunks:
        raise ValueError(f"No chunks found for document {document_id}")

    # 批量向量化
    task.update_state(
        state="PROGRESS",
        meta={"current": 20, "total": 100, "status": "Generating embeddings"}
    )

    embeddings = []
    chunk_ids = []
    contents = []

    # 分批处理，避免内存溢出
    batch_size = 32
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]

        task.update_state(
            state="PROGRESS",
            meta={
                "current": 20 + (i / total_chunks) * 40,
                "total": 100,
                "status": f"Processing batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}"
            }
        )

        # 生成嵌入向量
        batch_texts = [chunk.content for chunk in batch_chunks]
        batch_embeddings = await embedding_service.embed_batch(batch_texts)

        embeddings.extend(batch_embeddings)
        chunk_ids.extend([chunk.id for chunk in batch_chunks])
        contents.extend(batch_texts)

    # 存储到Milvus
    task.update_state(
        state="PROGRESS",
        meta={"current": 70, "total": 100, "status": "Storing vectors in Milvus"}
    )

    milvus_ids = await milvus_service.insert_vectors(
        document_id=document_id,
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        contents=contents,
        metadatas=[{
            "chunk_index": chunk.chunk_index,
            "chunk_type": chunk.chunk_type,
            "page_number": chunk.page_number,
            "chapter_title": chunk.chapter_title
        } for chunk in chunks]
    )

    # 保存向量存储记录
    task.update_state(
        state="PROGRESS",
        meta={"current": 90, "total": 100, "status": "Saving vector records"}
    )

    await _save_vector_records(document_id, chunk_ids, milvus_ids)

    return {
        "status": "completed",
        "document_id": document_id,
        "vectorized_chunks": len(embeddings),
        "embedding_dimension": len(embeddings[0]) if embeddings else 0,
        "milvus_ids": milvus_ids,
        "vectorization_time": datetime.utcnow().isoformat()
    }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30
)
def update_document_vectors(self, document_id: str, chunk_ids: List[str]) -> Dict[str, Any]:
    """
    更新文档向量（重新向量化指定的块）
    """
    try:
        logger.info(f"Updating vectors for document_id: {document_id}, chunks: {len(chunk_ids)}")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Updating vectors"}
        )

        # 异步执行更新
        result = asyncio.run(_update_vectors_async(document_id, chunk_ids, self))

        logger.info(f"Vector update completed for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Vector update failed for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=30)


async def _update_vectors_async(document_id: str, chunk_ids: List[str], task) -> Dict[str, Any]:
    """异步更新向量"""
    embedding_service = EmbeddingService()
    milvus_service = MilvusService()

    # 获取要更新的块
    async with get_db() as db:
        from sqlalchemy import select
        result = await db.execute(
            select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.id.in_(chunk_ids)
            )
        )
        chunks = result.scalars().all()

    if not chunks:
        raise ValueError(f"No chunks found for update in document {document_id}")

    # 生成新的嵌入向量
    task.update_state(
        state="PROGRESS",
        meta={"current": 30, "total": 100, "status": "Generating new embeddings"}
    )

    texts = [chunk.content for chunk in chunks]
    new_embeddings = await embedding_service.embed_batch(texts)

    # 更新Milvus中的向量
    task.update_state(
        state="PROGRESS",
        meta={"current": 70, "total": 100, "status": "Updating vectors in Milvus"}
    )

    # 先删除旧向量
    await milvus_service.delete_chunks(document_id, chunk_ids)

    # 插入新向量
    new_milvus_ids = await milvus_service.insert_vectors(
        document_id=document_id,
        chunk_ids=[chunk.id for chunk in chunks],
        embeddings=new_embeddings,
        contents=texts,
        metadatas=[{
            "chunk_index": chunk.chunk_index,
            "chunk_type": chunk.chunk_type,
            "page_number": chunk.page_number,
            "chapter_title": chunk.chapter_title,
            "updated_at": datetime.utcnow().isoformat()
        } for chunk in chunks]
    )

    # 更新数据库记录
    await _update_vector_records(document_id, [chunk.id for chunk in chunks], new_milvus_ids)

    return {
        "status": "completed",
        "document_id": document_id,
        "updated_chunks": len(new_embeddings),
        "milvus_ids": new_milvus_ids,
        "update_time": datetime.utcnow().isoformat()
    }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30
)
def delete_document_vectors(self, document_id: str) -> Dict[str, Any]:
    """
    删除文档的所有向量
    """
    try:
        logger.info(f"Deleting vectors for document_id: {document_id}")

        # 异步执行删除
        result = asyncio.run(_delete_vectors_async(document_id))

        logger.info(f"Vector deletion completed for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Vector deletion failed for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=30)


async def _delete_vectors_async(document_id: str) -> Dict[str, Any]:
    """异步删除向量"""
    milvus_service = MilvusService()

    # 从Milvus删除
    deleted_count = await milvus_service.delete_document(document_id)

    # 从数据库删除记录
    async with get_db() as db:
        from sqlalchemy import delete
        await db.execute(
            delete(VectorStorage).where(VectorStorage.document_id == document_id)
        )
        await db.commit()

    return {
        "status": "completed",
        "document_id": document_id,
        "deleted_vectors": deleted_count,
        "deletion_time": datetime.utcnow().isoformat()
    }


@celery_app.task(bind=True)
def build_vector_index(self, collection_name: str = None) -> Dict[str, Any]:
    """
    构建向量索引
    """
    try:
        logger.info(f"Building vector index for collection: {collection_name}")

        # 异步执行索引构建
        result = asyncio.run(_build_index_async(collection_name))

        logger.info(f"Vector index built for collection: {collection_name}")
        return result

    except Exception as e:
        logger.error(f"Vector index build failed for collection: {collection_name}, error: {e}")
        raise


async def _build_index_async(collection_name: str = None) -> Dict[str, Any]:
    """异步构建索引"""
    milvus_service = MilvusService()

    # 创建或获取集合
    if collection_name:
        await milvus_service.create_collection(collection_name)
    else:
        collection_name = settings.MILVUS_COLLECTION

    # 构建索引
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }

    index_result = await milvus_service.create_index(collection_name, index_params)

    return {
        "status": "completed",
        "collection_name": collection_name,
        "index_params": index_params,
        "index_result": index_result,
        "build_time": datetime.utcnow().isoformat()
    }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30
)
def optimize_vector_storage(self) -> Dict[str, Any]:
    """
    优化向量存储（压缩、清理等）
    """
    try:
        logger.info("Starting vector storage optimization")

        # 异步执行优化
        result = asyncio.run(_optimize_storage_async(self))

        logger.info("Vector storage optimization completed")
        return result

    except Exception as e:
        logger.error(f"Vector storage optimization failed: {e}")
        raise self.retry(exc=e, countdown=30)


async def _optimize_storage_async(task) -> Dict[str, Any]:
    """异步优化存储"""
    milvus_service = MilvusService()

    # 获取集合统计信息
    task.update_state(
        state="PROGRESS",
        meta={"current": 20, "total": 100, "status": "Analyzing storage"}
    )

    stats = await milvus_service.get_collection_stats()

    # 压缩向量（如果支持）
    task.update_state(
        state="PROGRESS",
        meta={"current": 50, "total": 100, "status": "Compressing vectors"}
    )

    compression_result = await milvus_service.compress_vectors()

    # 清理无效向量
    task.update_state(
        state="PROGRESS",
        meta={"current": 80, "total": 100, "status": "Cleaning invalid vectors"}
    )

    cleanup_result = await milvus_service.cleanup_invalid_vectors()

    return {
        "status": "completed",
        "original_stats": stats,
        "compression_result": compression_result,
        "cleanup_result": cleanup_result,
        "optimization_time": datetime.utcnow().isoformat()
    }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30
)
def batch_vectorize_documents(self, document_ids: List[str]) -> Dict[str, Any]:
    """
    批量向量化文档
    """
    try:
        logger.info(f"Starting batch vectorization for {len(document_ids)} documents")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": len(document_ids), "status": "Starting batch vectorization"}
        )

        results = []
        failed_documents = []

        for i, doc_id in enumerate(document_ids):
            try:
                # 调用单个文档向量化任务
                result = vectorize_content.delay(doc_id)
                results.append({
                    "document_id": doc_id,
                    "task_id": result.id,
                    "status": "started"
                })

                # 更新进度
                progress = int((i + 1) / len(document_ids) * 100)
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": i + 1,
                        "total": len(document_ids),
                        "progress": progress,
                        "status": f"Processing document {i + 1}/{len(document_ids)}"
                    }
                )

            except Exception as e:
                logger.error(f"Failed to start vectorization for document {doc_id}: {str(e)}")
                failed_documents.append({
                    "document_id": doc_id,
                    "error": str(e)
                })

        return {
            "task_id": self.request.id,
            "total_documents": len(document_ids),
            "successful": len(results),
            "failed": len(failed_documents),
            "results": results,
            "failed_documents": failed_documents
        }

    except Exception as e:
        logger.error(f"Batch vectorization failed: {e}")
        raise


async def _save_vector_records(document_id: str, chunk_ids: List[str], milvus_ids: List[str]):
    """保存向量存储记录"""
    async with get_db() as db:
        for chunk_id, milvus_id in zip(chunk_ids, milvus_ids):
            vector_record = VectorStorage(
                document_id=document_id,
                chunk_id=chunk_id,
                milvus_id=milvus_id,
                created_at=datetime.utcnow()
            )
            db.add(vector_record)

        await db.commit()


async def _update_vector_records(document_id: str, chunk_ids: List[str], milvus_ids: List[str]):
    """更新向量存储记录"""
    async with get_db() as db:
        from sqlalchemy import update

        for chunk_id, milvus_id in zip(chunk_ids, milvus_ids):
            await db.execute(
                update(VectorStorage)
                .where(VectorStorage.chunk_id == chunk_id)
                .values(
                    milvus_id=milvus_id,
                    updated_at=datetime.utcnow()
                )
            )

        await db.commit()


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30
)
def calculate_similarity_matrix(self, document_id: str) -> Dict[str, Any]:
    """
    计算文档内相似度矩阵
    """
    try:
        logger.info(f"Calculating similarity matrix for document_id: {document_id}")

        # 异步执行计算
        result = asyncio.run(_calculate_similarity_matrix_async(document_id, self))

        logger.info(f"Similarity matrix calculation completed for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Similarity matrix calculation failed for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=30)


async def _calculate_similarity_matrix_async(document_id: str, task) -> Dict[str, Any]:
    """异步计算相似度矩阵"""
    milvus_service = MilvusService()
    embedding_service = EmbeddingService()

    # 获取文档的所有向量
    task.update_state(
        state="PROGRESS",
        meta={"current": 20, "total": 100, "status": "Loading document vectors"}
    )

    vectors = await milvus_service.get_document_vectors(document_id)

    if len(vectors) < 2:
        return {
            "status": "skipped",
            "reason": "Not enough chunks for similarity calculation",
            "chunk_count": len(vectors)
        }

    # 计算相似度矩阵
    task.update_state(
        state="PROGRESS",
        meta={"current": 60, "total": 100, "status": "Computing similarities"}
    )

    similarity_matrix = await embedding_service.calculate_similarity_matrix(vectors)

    # 保存相似度结果
    task.update_state(
        state="PROGRESS",
        meta={"current": 90, "total": 100, "status": "Saving similarity results"}
    )

    await _save_similarity_results(document_id, similarity_matrix)

    return {
        "status": "completed",
        "document_id": document_id,
        "matrix_shape": similarity_matrix.shape,
        "average_similarity": float(np.mean(similarity_matrix)),
        "max_similarity": float(np.max(similarity_matrix)),
        "min_similarity": float(np.min(similarity_matrix)),
        "calculation_time": datetime.utcnow().isoformat()
    }


async def _save_similarity_results(document_id: str, similarity_matrix: np.ndarray):
    """保存相似度结果"""
    async with get_db() as db:
        document = await db.get(Document, document_id)
        if document:
            document.doc_metadata = {
                **document.doc_metadata,
                "similarity_matrix": {
                    "shape": similarity_matrix.shape.tolist(),
                    "average": float(np.mean(similarity_matrix)),
                    "max": float(np.max(similarity_matrix)),
                    "min": float(np.min(similarity_matrix)),
                    "computed_at": datetime.utcnow().isoformat()
                }
            }
            await db.commit()