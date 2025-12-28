"""
整合后的RAG API端点

Consolidated RAG API Endpoints - 统一的RAG查询接口
替换原有的多个重复RAG端点，提供一致的API体验
"""

from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import json

from app.services.consolidated_rag_service import (
    ConsolidatedRAGService, RetrievalMode, consolidated_rag_service
)
from app.core.service_registry import service_registry
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])

# 响应模型
class RAGResponse(BaseModel):
    """RAG查询响应"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    mode: str
    execution_time: float
    timestamp: str
    conversation_id: Optional[int] = None
    retrieval_details: Optional[Dict[str, Any]] = None

# 请求模型
class RAGQueryRequest(BaseModel):
    """RAG查询请求"""
    query: str = Field(..., description="查询内容", min_length=1, max_length=1000)
    mode: RetrievalMode = Field(RetrievalMode.ENHANCED, description="检索模式")
    conversation_id: Optional[int] = Field(None, description="对话ID")
    document_ids: Optional[List[int]] = Field(None, description="限定文档ID列表")
    max_results: int = Field(10, description="最大返回结果数", ge=1, le=50)
    options: Optional[Dict[str, Any]] = Field(None, description="额外选项")

class BatchRAGQueryRequest(BaseModel):
    """批量RAG查询请求"""
    queries: List[str] = Field(..., description="查询列表", min_items=1, max_items=10)
    mode: RetrievalMode = Field(RetrievalMode.ENHANCED, description="检索模式")
    conversation_id: Optional[int] = Field(None, description="对话ID")
    document_ids: Optional[List[int]] = Field(None, description="限定文档ID列表")
    max_results: int = Field(10, description="每个查询的最大返回结果数", ge=1, le=50)
    batch_id: Optional[str] = Field(None, description="批次ID")

# 响应模型
class QuerySuggestion(BaseModel):
    """查询建议"""
    id: str
    title: str
    snippet: str
    relevance_score: float
    document_id: Optional[str]
    source_type: Optional[str]

class StrategyInfo(BaseModel):
    """策略信息"""
    mode: str
    name: str
    description: str

class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str
    details: Dict[str, Any]

class StatisticsResponse(BaseModel):
    """统计信息响应"""
    total_queries: int
    successful_queries: int
    strategy_usage: Dict[str, int]
    average_response_time: float
    cache_hit_rate: float
    success_rate: float

# 依赖注入
async def get_rag_service() -> ConsolidatedRAGService:
    """获取RAG服务实例"""
    try:
        service = service_registry.get(ConsolidatedRAGService)
        if service.get_status().value != "running":
            raise HTTPException(status_code=503, detail="RAG服务不可用")
        return service
    except Exception as e:
        logger.error(f"Failed to get RAG service: {e}")
        raise HTTPException(status_code=503, detail="RAG服务不可用")

@router.post("/query", response_model=RAGResponse)
async def query(
    request: RAGQueryRequest,
    rag_service: ConsolidatedRAGService = Depends(get_rag_service)
):
    """
    执行RAG查询

    支持多种检索模式：
    - simple: 简单向量检索
    - enhanced: 向量+知识图谱混合检索
    - deep_search: 多轮深度检索
    - agentic: 智能代理自适应检索
    """
    try:
        logger.info(f"RAG query request: {request.query[:100]}..., mode: {request.mode}")

        result = await rag_service.query(
            question=request.query,
            mode=request.mode,
            conversation_id=request.conversation_id,
            document_ids=request.document_ids,
            max_results=request.max_results,
            **(request.options or {})
        )

        return RAGResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail="查询处理失败")

@router.post("/stream-query")
async def stream_query(
    request: RAGQueryRequest,
    rag_service: ConsolidatedRAGService = Depends(get_rag_service)
):
    """
    流式RAG查询

    返回Server-Sent Events流，实时返回检索过程和结果
    """
    try:
        logger.info(f"Stream RAG query request: {request.query[:100]}..., mode: {request.mode}")

        async def generate_stream():
            try:
                async for chunk in rag_service.stream_query(
                    question=request.query,
                    mode=request.mode,
                    conversation_id=request.conversation_id,
                    document_ids=request.document_ids,
                    max_results=request.max_results,
                    **(request.options or {})
                ):
                    yield chunk

            except Exception as e:
                logger.error(f"Stream query error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        logger.error(f"Stream query setup error: {e}")
        raise HTTPException(status_code=500, detail="流式查询设置失败")

@router.post("/batch-query")
async def batch_query(
    request: BatchRAGQueryRequest,
    background_tasks: BackgroundTasks,
    rag_service: ConsolidatedRAGService = Depends(get_rag_service)
):
    """
    批量RAG查询

    支持同时处理多个查询，返回批量结果
    """
    try:
        logger.info(f"Batch RAG query request: {len(request.queries)} queries, mode: {request.mode}")

        batch_id = request.batch_id or f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        results = []

        for i, query in enumerate(request.queries):
            try:
                result = await rag_service.query(
                    question=query,
                    mode=request.mode,
                    conversation_id=request.conversation_id,
                    document_ids=request.document_ids,
                    max_results=request.max_results,
                    **(request.options or {})
                )

                results.append({
                    "query_index": i,
                    "query": query,
                    "result": result,
                    "status": "success"
                })

            except Exception as e:
                logger.error(f"Batch query {i} failed: {e}")
                results.append({
                    "query_index": i,
                    "query": query,
                    "error": str(e),
                    "status": "failed"
                })

        return {
            "batch_id": batch_id,
            "total_queries": len(request.queries),
            "successful_queries": sum(1 for r in results if r["status"] == "success"),
            "failed_queries": sum(1 for r in results if r["status"] == "failed"),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Batch query error: {e}")
        raise HTTPException(status_code=500, detail="批量查询处理失败")

@router.get("/suggestions", response_model=List[QuerySuggestion])
async def get_query_suggestions(
    q: str = Query(..., description="查询内容", min_length=1),
    limit: int = Query(5, description="建议数量", ge=1, le=20),
    rag_service: ConsolidatedRAGService = Depends(get_rag_service)
):
    """
    获取查询建议

    基于输入查询提供相关的文档建议
    """
    try:
        suggestions = await rag_service.get_query_suggestions(q, limit)
        return [QuerySuggestion(**suggestion) for suggestion in suggestions]

    except Exception as e:
        logger.error(f"Query suggestions error: {e}")
        raise HTTPException(status_code=500, detail="获取查询建议失败")

@router.get("/strategies", response_model=List[StrategyInfo])
async def get_available_strategies(
    rag_service: ConsolidatedRAGService = Depends(get_rag_service)
):
    """
    获取可用的检索策略

    返回所有支持的检索模式及其描述
    """
    try:
        strategies = rag_service.get_available_strategies()
        return [StrategyInfo(**strategy) for strategy in strategies]

    except Exception as e:
        logger.error(f"Get strategies error: {e}")
        raise HTTPException(status_code=500, detail="获取策略信息失败")

@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    rag_service: ConsolidatedRAGService = Depends(get_rag_service)
):
    """
    RAG服务健康检查

    检查RAG服务及其依赖组件的健康状态
    """
    try:
        health_result = await rag_service.health_check()
        return HealthCheckResponse(**health_result)

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            details={"error": str(e)}
        )

@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
    rag_service: ConsolidatedRAGService = Depends(get_rag_service)
):
    """
    获取RAG服务统计信息

    返回查询统计、性能指标等数据
    """
    try:
        stats = rag_service.get_statistics()
        return StatisticsResponse(**stats)

    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail="获取统计信息失败")

# 向后兼容层 - 支持旧端点
@router.post("/simple-query", response_model=RAGResponse)
async def simple_query(
    request: RAGQueryRequest,
    rag_service: ConsolidatedRAGService = Depends(get_rag_service)
):
    """
    简单查询端点 (向后兼容)

    等同于 /query?mode=simple
    """
    request.mode = RetrievalMode.SIMPLE
    return await query(request, rag_service)

@router.post("/enhanced-query", response_model=RAGResponse)
async def enhanced_query(
    request: RAGQueryRequest,
    rag_service: ConsolidatedRAGService = Depends(get_rag_service)
):
    """
    增强查询端点 (向后兼容)

    等同于 /query?mode=enhanced
    """
    request.mode = RetrievalMode.ENHANCED
    return await query(request, rag_service)

@router.post("/deep-query", response_model=RAGResponse)
async def deep_query(
    request: RAGQueryRequest,
    rag_service: ConsolidatedRAGService = Depends(get_rag_service)
):
    """
    深度查询端点 (向后兼容)

    等同于 /query?mode=deep_search
    """
    request.mode = RetrievalMode.DEEP_SEARCH
    return await query(request, rag_service)

@router.post("/agentic-query", response_model=RAGResponse)
async def agentic_query(
    request: RAGQueryRequest,
    rag_service: ConsolidatedRAGService = Depends(get_rag_service)
):
    """
    智能代理查询端点 (向后兼容)

    等同于 /query?mode=agentic
    """
    request.mode = RetrievalMode.AGENTIC
    return await query(request, rag_service)

# 错误处理
@router.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """处理值错误"""
    return HTTPException(
        status_code=400,
        detail={"error": "invalid_parameter", "message": str(exc)}
    )

@router.exception_handler(Exception)
async def general_error_handler(request, exc):
    """处理通用错误"""
    logger.error(f"Unhandled error in RAG API: {exc}")
    return HTTPException(
        status_code=500,
        detail={"error": "internal_error", "message": "内部服务器错误"}
    )