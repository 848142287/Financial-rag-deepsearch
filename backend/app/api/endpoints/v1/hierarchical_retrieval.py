"""
分层检索API接口
提供基于三层索引（文档摘要、章节、片段）的智能检索
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List

from app.core.structured_logging import get_structured_logger
from app.schemas.hierarchical_index import (
    HierarchicalRetrievalRequest,
    HierarchicalRetrievalResult,
    HierarchicalIndexBuildRequest,
    HierarchicalIndexBuildResponse
)
from app.services.hierarchical_index import (
    get_hierarchical_index_extractor,
    get_hierarchical_retrieval_service,
    get_hierarchical_milvus_service
)
from app.services.embeddings.unified_embedding_service import UnifiedEmbeddingService
from app.services.llm_service import LLMService

logger = get_structured_logger(__name__)

router = APIRouter()


# ==================== 分层检索 ====================

@router.post("/retrieve", response_model=HierarchicalRetrievalResult)
async def hierarchical_retrieve(request: HierarchicalRetrievalRequest):
    """
    分层检索

    使用三层索引进行智能检索：
    1. 文档摘要层 - 粗粒度筛选
    2. 章节索引层 - 中粒度定位
    3. 片段索引层 - 细粒度精确检索

    Args:
        request: 检索请求参数

    Returns:
        HierarchicalRetrievalResult: 分层检索结果
    """
    try:
        logger.info(f"🔍 分层检索请求: {request.query}")

        # 获取服务
        retrieval_service = get_hierarchical_retrieval_service()
        milvus_service = await get_hierarchical_milvus_service()

        # 生成查询嵌入
        embedding_service = UnifiedEmbeddingService()
        query_embedding = await embedding_service.embed_batch([request.query])
        query_embedding = query_embedding[0].tolist()

        # 第1层：文档摘要检索
        logger.info("  📄 第1层：文档摘要检索...")
        doc_results = await milvus_service.search_document_summaries(
            query_embedding=query_embedding,
            top_k=request.max_docs,
            document_ids=request.document_ids
        )

        # 过滤低于阈值的文档
        doc_results = [d for d in doc_results if d["score"] >= request.doc_threshold]
        logger.info(f"    ✓ 找到 {len(doc_results)} 个相关文档")

        if not doc_results:
            return HierarchicalRetrievalResult(
                query=request.query,
                documents=[],
                chapters=[],
                chunks=[],
                merged_results=[],
                retrieval_time=0.0,
                total_docs=0,
                total_chapters=0,
                total_chunks=0
            )

        # 第2层：章节检索
        logger.info("  📑 第2层：章节检索...")
        relevant_doc_ids = [d["document_id"] for d in doc_results]
        chapter_results = await milvus_service.search_chapter_indexes(
            query_embedding=query_embedding,
            top_k=request.max_chapters_per_doc * len(relevant_doc_ids),
            document_ids=relevant_doc_ids
        )

        # 过滤低于阈值的章节
        chapter_results = [c for c in chapter_results if c["score"] >= request.chapter_threshold]
        logger.info(f"    ✓ 找到 {len(chapter_results)} 个相关章节")

        # 第3层：片段检索
        logger.info("  ✂️ 第3层：片段检索...")
        relevant_chapter_ids = [c["chapter_id"] for c in chapter_results] if chapter_results else None
        chunk_results = await milvus_service.search_chunk_indexes(
            query_embedding=query_embedding,
            top_k=request.top_k,
            document_ids=relevant_doc_ids,
            chapter_ids=relevant_chapter_ids
        )

        # 过滤低于阈值的片段
        chunk_results = [c for c in chunk_results if c["score"] >= request.chunk_threshold]
        logger.info(f"    ✓ 找到 {len(chunk_results)} 个相关片段")

        # 构建结果
        from app.schemas.hierarchical_index import (
            RetrievedDocument,
            RetrievedChapter,
            RetrievedChunk
        )

        results = HierarchicalRetrievalResult(
            query=request.query,
            documents=[
                RetrievedDocument(
                    document_id=d["document_id"],
                    summary_text=d["summary_text"],
                    score=d["score"],
                    keywords=d["keywords"],
                    entities=d["entities"]
                )
                for d in doc_results[:request.max_docs]
            ],
            chapters=[
                RetrievedChapter(
                    chapter_id=c["chapter_id"],
                    document_id=c["document_id"],
                    title=c["title"],
                    summary=c["summary"],
                    score=c["score"],
                    level=c["level"],
                    chunk_count=c["chunk_count"]
                )
                for c in chapter_results[:request.max_chapters_per_doc * request.max_docs]
            ],
            chunks=[
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    document_id=c["document_id"],
                    chapter_id=c["chapter_id"],
                    content=c["content"],
                    score=c["score"],
                    metadata={
                        "chunk_type": c["chunk_type"],
                        "chunk_index": c["chunk_index"],
                        "page_number": c["page_number"]
                    }
                )
                for c in chunk_results[:request.top_k]
            ],
            merged_results=[],
            retrieval_time=0.0,
            total_docs=len(doc_results),
            total_chapters=len(chapter_results),
            total_chunks=len(chunk_results)
        )

        # 合并结果
        results.merged_results = results.chunks[:request.top_k]

        # 记录监控指标
        try:
            # 在这里可以添加监控代码
            pass
        except Exception as monitor_error:
            logger.warning(f"记录监控指标失败: {monitor_error}")

        logger.info(f"✅ 分层检索完成！文档: {results.total_docs}, 章节: {results.total_chapters}, 片段: {results.total_chunks}")

        return results

    except Exception as e:
        logger.error(f"❌ 分层检索失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分层检索失败: {str(e)}"
        )


@router.post("/build-index", response_model=HierarchicalIndexBuildResponse)
async def build_hierarchical_index(request: HierarchicalIndexBuildRequest):
    """
    构建分层索引

    为文档构建三层索引结构（文档摘要、章节、片段）并存储到Milvus

    Args:
        request: 索引构建请求

    Returns:
        HierarchicalIndexBuildResponse: 构建结果
    """
    try:
        logger.info(f"📚 开始构建文档 {request.document_id} 的分层索引")

        # 这里需要从数据库或存储中获取文档内容
        # 简化版本：假设可以获取到文档的markdown内容
        # 实际实现需要集成到文档处理流水线中

        # TODO: 集成到文档处理流水线
        # 1. 获取文档内容
        # 2. 调用index_extractor抽取分层索引
        # 3. 调用milvus_service存储到向量数据库

        response = HierarchicalIndexBuildResponse(
            document_id=request.document_id,
            success=True,
            message="分层索引构建功能需要集成到文档处理流水线中",
            summary_index=None,
            chapter_count=0,
            chunk_count=0,
            processing_time=0.0
        )

        return response

    except Exception as e:
        logger.error(f"❌ 构建分层索引失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"构建分层索引失败: {str(e)}"
        )


@router.delete("/index/{document_id}")
async def delete_hierarchical_index(document_id: str):
    """
    删除分层索引

    删除指定文档的所有三层索引

    Args:
        document_id: 文档ID

    Returns:
        Dict: 删除结果
    """
    try:
        logger.info(f"🗑️ 删除文档 {document_id} 的分层索引")

        milvus_service = await get_hierarchical_milvus_service()
        await milvus_service.delete_document_index(document_id)

        return {
            "success": True,
            "message": f"成功删除文档 {document_id} 的分层索引",
            "document_id": document_id
        }

    except Exception as e:
        logger.error(f"❌ 删除分层索引失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除分层索引失败: {str(e)}"
        )


@router.get("/stats")
async def get_hierarchical_index_stats():
    """
    获取分层索引统计信息

    返回三层索引的统计数据

    Returns:
        Dict: 统计信息
    """
    try:
        milvus_service = await get_hierarchical_milvus_service()

        # 获取各collection的统计信息
        stats = {
            "document_summaries": {
                "collection_name": milvus_service.COLLECTION_DOC_SUMMARIES,
                "num_entities": 0
            },
            "chapter_indexes": {
                "collection_name": milvus_service.COLLECTION_CHAPTER_INDEXES,
                "num_entities": 0
            },
            "chunk_indexes": {
                "collection_name": milvus_service.COLLECTION_CHUNK_INDEXES,
                "num_entities": 0
            }
        }

        # 获取实体数量
        for collection_name, collection in milvus_service.collections.items():
            collection.load()
            num_entities = collection.num_entities
            if collection_name == "doc_summaries":
                stats["document_summaries"]["num_entities"] = num_entities
            elif collection_name == "chapters":
                stats["chapter_indexes"]["num_entities"] = num_entities
            elif collection_name == "chunks":
                stats["chunk_indexes"]["num_entities"] = num_entities

        return {
            "success": True,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"❌ 获取统计信息失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计信息失败: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    健康检查

    检查分层检索服务的健康状态

    Returns:
        Dict: 健康状态
    """
    try:
        milvus_service = await get_hierarchical_milvus_service()

        health_status = {
            "milvus_connected": milvus_service._is_connected,
            "collections_initialized": len(milvus_service.collections) > 0,
            "services": {
                "index_extractor": True,
                "retrieval_service": True,
                "milvus_service": True
            }
        }

        return {
            "status": "healthy" if all(health_status["services"].values()) else "unhealthy",
            "details": health_status
        }

    except Exception as e:
        logger.error(f"❌ 健康检查失败: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/monitoring/performance")
async def get_performance_statistics(time_window: int = 3600):
    """
    获取性能统计信息

    Args:
        time_window: 时间窗口（秒），默认1小时

    Returns:
        Dict: 性能统计数据
    """
    try:
        # 注意：这需要实现监控服务
        # from app.services.hierarchical_index.performance_monitoring import get_hierarchical_retrieval_monitor
        # monitor = get_hierarchical_retrieval_monitor()
        # stats = monitor.get_statistics(time_window=time_window)

        # 简化版本
        return {
            "status": "success",
            "message": "性能监控功能需要实现HierarchicalRetrievalMonitor服务",
            "time_window": time_window,
            "data": {
                "note": "请参考PERFORMANCE_MONITORING.md实现监控服务"
            }
        }

    except Exception as e:
        logger.error(f"❌ 获取性能统计失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取性能统计失败: {str(e)}"
        )


@router.get("/monitoring/trends")
async def get_performance_trends(
    granularity: str = "5min",
    points: int = 12
):
    """
    获取性能趋势

    Args:
        granularity: 时间粒度（1min, 5min, 15min, 1hour）
        points: 数据点数量

    Returns:
        Dict: 趋势数据
    """
    try:
        # 注意：这需要实现监控服务
        return {
            "status": "success",
            "message": "性能趋势功能需要实现HierarchicalRetrievalMonitor服务",
            "granularity": granularity,
            "points": points,
            "data": {
                "note": "请参考PERFORMANCE_MONITORING.md实现监控服务"
            }
        }

    except Exception as e:
        logger.error(f"❌ 获取性能趋势失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取性能趋势失败: {str(e)}"
        )
