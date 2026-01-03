"""
反馈增强RAG API端点

提供带反馈回路的RAG查询接口
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.services.agentic_rag.feedback_loop import get_realtime_feedback_processor
from app.services.agentic_rag.feedback_enhanced_executor import get_feedback_enhanced_executor
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

router = APIRouter(prefix="/feedback-rag", tags=["反馈增强RAG"])


# ========== 请求/响应模型 ==========

class FeedbackRAGRequest(BaseModel):
    """反馈增强RAG请求"""
    query: str = Field(..., description="用户查询")
    retrieval_level: str = Field("enhanced", description="检索级别: fast/enhanced/deep_search")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    enable_feedback: bool = Field(True, description="是否启用反馈优化")
    enable_compression: bool = Field(True, description="是否启用上下文压缩")


class UserInteractions(BaseModel):
    """用户交互数据"""
    clicks: List[Dict[str, Any]] = Field(default_factory=list, description="点击事件")
    dwell_times: Dict[str, float] = Field(default_factory=dict, description="停留时间")
    rating: Optional[int] = Field(None, ge=1, le=5, description="用户评分")
    skipped: bool = Field(False, description="是否跳过结果")


class FeedbackSubmission(BaseModel):
    """反馈提交"""
    query: str = Field(..., description="查询")
    results: List[Dict[str, Any]] = Field(..., description="检索结果")
    interactions: UserInteractions = Field(..., description="用户交互")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: str = Field(..., description="会话ID")


class FeedbackRAGResponse(BaseModel):
    """反馈增强RAG响应"""
    query: str
    optimized_query: str
    results: List[Dict[str, Any]]
    feedback_context: Dict[str, Any]
    optimization_time_ms: float
    total_time_ms: float
    metadata: Dict[str, Any]


# ========== API端点 ==========

@router.post("/query", response_model=FeedbackRAGResponse)
async def query_with_feedback(
    request: FeedbackRAGRequest
):
    """
    使用反馈增强执行查询

    功能:
    1. 基于历史反馈优化查询
    2. 动态调整检索参数
    3. 应用上下文压缩
    4. 返回优化后的结果
    """
    try:
        start_time = datetime.now()

        # 获取执行器
        executor = get_feedback_enhanced_executor(
            enable_feedback=request.enable_feedback,
            enable_compression=request.enable_compression
        )

        # 执行检索
        execution_result = await executor.execute_with_feedback(
            plan=None,  # 如果需要实际plan，从外部传入
            query=request.query,
            retrieval_level=request.retrieval_level,
            user_id=request.user_id,
            session_id=request.session_id
        )

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        return FeedbackRAGResponse(
            query=request.query,
            optimized_query=execution_result.metadata.get("optimized_query", request.query),
            results=execution_result.fused_results,
            feedback_context=execution_result.metadata.get("feedback_context", {}),
            optimization_time_ms=execution_result.metadata.get("optimization_time_ms", 0),
            total_time_ms=total_time,
            metadata=execution_result.metadata
        )

    except Exception as e:
        logger.error(f"查询执行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """
    提交用户反馈

    收集的数据:
    - 点击事件
    - 停留时间
    - 用户评分
    - 跳过行为
    """
    try:
        executor = get_feedback_enhanced_executor()

        # 转换交互数据
        interactions = {
            "clicks": feedback.interactions.clicks,
            "dwell_times": feedback.interactions.dwell_times,
            "rating": feedback.interactions.rating,
            "skipped": feedback.interactions.skipped
        }

        # 收集反馈
        await executor.collect_result_feedback(
            query=feedback.query,
            results=feedback.results,
            user_interactions=interactions,
            user_id=feedback.user_id,
            session_id=feedback.session_id
        )

        return {
            "status": "success",
            "message": "反馈已记录",
            "session_id": feedback.session_id,
            "interaction_count": (
                len(feedback.interactions.clicks) +
                len(feedback.interactions.dwell_times) +
                (1 if feedback.interactions.rating else 0)
            )
        }

    except Exception as e:
        logger.error(f"提交反馈失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights")
async def get_feedback_insights(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    获取反馈洞察

    返回:
    - 总体统计
    - 用户统计 (如果提供user_id)
    - 会话统计 (如果提供session_id)
    """
    try:
        processor = get_realtime_feedback_processor()

        insights = processor.get_insights(
            user_id=user_id,
            session_id=session_id
        )

        return {
            "status": "success",
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"获取洞察失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        processor = get_realtime_feedback_processor()

        return {
            "status": "healthy",
            "feedback_enabled": True,
            "stats": {
                "sessions": len(processor.sessions),
                "users": len(processor.user_preferences),
                "queries": len(processor.query_patterns)
            }
        }

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.post("/cleanup")
async def cleanup_old_sessions(max_age_hours: int = 24):
    """
    清理旧会话

    Args:
        max_age_hours: 保留最近N小时的会话
    """
    try:
        processor = get_realtime_feedback_processor()

        before_count = len(processor.sessions)
        processor.cleanup_old_sessions(max_age_hours)
        after_count = len(processor.sessions)

        return {
            "status": "success",
            "cleaned_sessions": before_count - after_count,
            "remaining_sessions": after_count
        }

    except Exception as e:
        logger.error(f"清理会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== 使用示例 ==========

@router.get("/example")
async def feedback_rag_example():
    """
    反馈增强RAG使用示例

    展示完整的使用流程
    """
    example = {
        "title": "反馈增强RAG使用示例",
        "workflow": [
            {
                "step": 1,
                "action": "执行查询",
                "endpoint": "POST /feedback-rag/query",
                "request": {
                    "query": "盈利能力",
                    "retrieval_level": "enhanced",
                    "user_id": "user_123",
                    "session_id": "session_456"
                },
                "response": {
                    "optimized_query": "净利润",  # 查询被重写
                    "results": [...],
                    "feedback_context": {...}
                }
            },
            {
                "step": 2,
                "action": "用户交互",
                "description": "用户查看结果，点击文档，停留时间等"
            },
            {
                "step": 3,
                "action": "提交反馈",
                "endpoint": "POST /feedback-rag/feedback",
                "request": {
                    "query": "盈利能力",
                    "results": [...],
                    "interactions": {
                        "clicks": [{"doc_id": 1, "position": 1}],
                        "dwell_times": {"1": 45.5},
                        "rating": 4
                    },
                    "session_id": "session_456"
                }
            },
            {
                "step": 4,
                "action": "下次查询自动优化",
                "description": "系统基于反馈自动优化查询和参数"
            }
        ],
        "benefits": [
            "查询自动重写 (盈利能力 → 净利润)",
            "动态参数调整 (top_k, 压缩率)",
            "个性化体验",
            "持续优化"
        ]
    }

    return example
