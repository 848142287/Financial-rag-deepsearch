"""
统一系统API接口
整合文档处理、RAG检索、评估、Agent等所有功能的统一入口
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import asyncio
import logging
from datetime import datetime

from app.services.document_processor.unified_processor import unified_document_processor, ProcessingStage
from app.services.retrieval.unified_rag_engine import unified_rag_engine, RetrievalStrategy
from app.services.evaluation.ragas_evaluator import RAGASEvaluator
from app.core.config import settings

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/unified", tags=["unified_system"])

# 请求/响应模型
class DocumentUploadRequest(BaseModel):
    content: str
    file_type: str
    metadata: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str
    strategy: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    session_context: Optional[Dict[str, Any]] = None

class BatchQueryRequest(BaseModel):
    queries: List[str]
    strategy: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class EvaluationRequest(BaseModel):
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

class SystemStatusResponse(BaseModel):
    status: str
    timestamp: datetime
    components: Dict[str, Dict[str, Any]]
    active_sessions: int
    queue_length: int


# 文档处理相关端点
@router.post("/documents/upload", summary="上传并处理文档")
async def upload_document(request: DocumentUploadRequest):
    """上传文档并启动完整处理流程"""
    try:
        result = await unified_document_processor.process_document(
            content=request.content,
            file_type=request.file_type,
            metadata=request.metadata,
            options=request.options
        )

        return {
            "success": result.success,
            "document_id": result.document_id,
            "stage": result.stage.value,
            "processing_time": result.processing_time,
            "data": result.data,
            "metadata": result.metadata,
            "error_message": result.error_message
        }

    except Exception as e:
        logger.error(f"文档上传处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.post("/documents/upload-file", summary="上传文件并处理")
async def upload_file(
    file: UploadFile = File(...),
    metadata: str = Form(default="{}"),
    options: str = Form(default="{}")
):
    """上传文件并启动处理流程"""
    try:
        import json
        metadata_dict = json.loads(metadata) if metadata else {}
        options_dict = json.loads(options) if options else {}

        # 读取文件内容
        content = await file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')

        # 获取文件类型
        file_type = file.filename.split('.')[-1] if file.filename else 'txt'

        result = await unified_document_processor.process_document(
            content=content,
            file_type=file_type,
            metadata={**metadata_dict, "filename": file.filename},
            options=options_dict
        )

        return {
            "success": result.success,
            "document_id": result.document_id,
            "filename": file.filename,
            "stage": result.stage.value,
            "processing_time": result.processing_time,
            "data": result.data,
            "metadata": result.metadata,
            "error_message": result.error_message
        }

    except Exception as e:
        logger.error(f"文件上传处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


@router.get("/documents/{document_id}/status", summary="获取文档处理状态")
async def get_document_status(document_id: str):
    """获取文档处理状态"""
    try:
        status = unified_document_processor.get_processing_status(document_id)

        if not status:
            raise HTTPException(status_code=404, detail="文档处理会话未找到")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


# RAG检索相关端点
@router.post("/rag/query", summary="执行RAG查询")
async def rag_query(request: QueryRequest):
    """执行RAG查询"""
    try:
        # 解析策略
        strategy = None
        if request.strategy:
            try:
                strategy = RetrievalStrategy(request.strategy)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的检索策略: {request.strategy}")

        # 执行查询
        retrieval_result, generation_result, evaluation_result = await unified_rag_engine.process_query(
            query=request.query,
            strategy=strategy,
            options=request.options,
            session_context=request.session_context
        )

        return {
            "query_id": retrieval_result.query_id,
            "query": request.query,
            "strategy_used": retrieval_result.strategy_used.value,
            "retrieval": {
                "documents": retrieval_result.documents,
                "scores": retrieval_result.scores,
                "explanations": retrieval_result.explanations,
                "document_count": len(retrieval_result.documents)
            },
            "generation": {
                "answer": generation_result.answer,
                "confidence": generation_result.confidence,
                "citations": generation_result.citations,
                "source_count": len(generation_result.sources)
            },
            "evaluation": {
                "overall_score": evaluation_result.overall_score if evaluation_result else None,
                "relevance_score": evaluation_result.relevance_score if evaluation_result else None,
                "suggestions": evaluation_result.suggestions if evaluation_result else None
            },
            "processing_time": retrieval_result.processing_time
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post("/rag/batch-query", summary="批量执行RAG查询")
async def batch_rag_query(request: BatchQueryRequest):
    """批量执行RAG查询"""
    try:
        # 解析策略
        strategy = None
        if request.strategy:
            try:
                strategy = RetrievalStrategy(request.strategy)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的检索策略: {request.strategy}")

        results = []
        for query in request.queries:
            try:
                retrieval_result, generation_result, evaluation_result = await unified_rag_engine.process_query(
                    query=query,
                    strategy=strategy,
                    options=request.options
                )

                results.append({
                    "query": query,
                    "success": True,
                    "strategy_used": retrieval_result.strategy_used.value,
                    "answer": generation_result.answer,
                    "confidence": generation_result.confidence,
                    "processing_time": retrieval_result.processing_time,
                    "evaluation_score": evaluation_result.overall_score if evaluation_result else None
                })

            except Exception as e:
                logger.error(f"批量查询中失败: {query}, 错误: {str(e)}")
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })

        return {
            "total_queries": len(request.queries),
            "successful_queries": sum(1 for r in results if r.get("success", False)),
            "failed_queries": sum(1 for r in results if not r.get("success", False)),
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量RAG查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量查询失败: {str(e)}")


@router.post("/rag/evaluate", summary="评估RAG结果")
async def evaluate_rag_result(request: EvaluationRequest):
    """评估RAG结果质量"""
    try:
        evaluator = RAGASEvaluator()
        evaluation_result = await evaluator.evaluate(
            question=request.query,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth
        )

        return {
            "query": request.query,
            "evaluation": evaluation_result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"RAG评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


# 系统管理相关端点
@router.get("/system/status", summary="获取系统状态")
async def get_system_status():
    """获取系统整体状态"""
    try:
        # 检查各组件状态
        status = {
            "document_processor": {
                "status": "healthy",
                "active_sessions": len(unified_document_processor.processing_sessions)
            },
            "rag_engine": {
                "status": "healthy",
                "active_sessions": len(unified_rag_engine.active_sessions)
            },
            "embedding_service": {
                "status": "healthy",
                "available_models": getattr(embedding_service, 'available_models', [])
            },
            "storage": {
                "milvus": "healthy",
                "neo4j": "healthy"
            }
        }

        return SystemStatusResponse(
            status="healthy",
            timestamp=datetime.now(),
            components=status,
            active_sessions=status["document_processor"]["active_sessions"] + status["rag_engine"]["active_sessions"],
            queue_length=0
        )

    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.get("/system/strategies", summary="获取可用的检索策略")
async def get_available_strategies():
    """获取所有可用的检索策略"""
    try:
        strategies = [
            {
                "name": strategy.value,
                "description": _get_strategy_description(strategy),
                "use_cases": _get_strategy_use_cases(strategy)
            }
            for strategy in RetrievalStrategy
        ]

        return {
            "strategies": strategies,
            "total": len(strategies)
        }

    except Exception as e:
        logger.error(f"获取策略列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取策略失败: {str(e)}")


@router.get("/system/capabilities", summary="获取系统能力")
async def get_system_capabilities():
    """获取系统能力概览"""
    try:
        capabilities = {
            "document_processing": {
                "supported_formats": settings.supported_file_types,
                "max_file_size_mb": settings.max_file_size_mb,
                "features": [
                    "金融实体识别",
                    "文档增强处理",
                    "质量评估",
                    "智能分块",
                    "多模态处理"
                ]
            },
            "retrieval": {
                "strategies": [s.value for s in RetrievalStrategy],
                "embedding_models": [
                    "BAAI/bge-large-zh-v1.5",
                    "text-embedding-v4",
                    "text-embedding-v3"
                ],
                "features": [
                    "查询增强",
                    "多级检索",
                    "结果融合",
                    "Agent生成",
                    "迭代优化"
                ]
            },
            "evaluation": {
                "metrics": [
                    "相关性",
                    "忠实性",
                    "答案相关性",
                    "上下文相关性"
                ],
                "frameworks": ["RAGAS"]
            }
        }

        return capabilities

    except Exception as e:
        logger.error(f"获取系统能力失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取能力失败: {str(e)}")


# 辅助函数
def _get_strategy_description(strategy: RetrievalStrategy) -> str:
    """获取策略描述"""
    descriptions = {
        RetrievalStrategy.SIMPLE: "基础向量检索，适合简单查询",
        RetrievalStrategy.ENHANCED: "增强检索，结合向量和实体信息",
        RetrievalStrategy.AGENTIC: "Agent智能检索，适合复杂分析查询",
        RetrievalStrategy.ITERATIVE: "迭代检索，逐步优化结果",
        RetrievalStrategy.HYBRID: "混合检索，组合多种策略"
    }
    return descriptions.get(strategy, "未知策略")


def _get_strategy_use_cases(strategy: RetrievalStrategy) -> List[str]:
    """获取策略适用场景"""
    use_cases = {
        RetrievalStrategy.SIMPLE: ["事实查询", "定义查询", "简单信息检索"],
        RetrievalStrategy.ENHANCED: ["实体相关查询", "中等复杂度分析", "专业领域查询"],
        RetrievalStrategy.AGENTIC: ["比较分析", "影响分析", "原因分析", "复杂推理"],
        RetrievalStrategy.ITERATIVE: ["探索性查询", "多角度分析", "结果优化"],
        RetrievalStrategy.HYBRID: ["综合分析", "高复杂度查询", "多维度检索"]
    }
    return use_cases.get(strategy, [])