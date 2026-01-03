"""
RAG问答API端点 - 集成反馈回路

在原有RAG问答流程中集成L1实时反馈回路
"""

from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends

from app.core.database import get_db
from app.core.structured_logging import get_structured_logger
from app.core.errors.unified_errors import handle_errors, ErrorCategory
from app.services.rag_question_answering import get_rag_qa_service
from app.services.agentic_rag.feedback_loop import get_realtime_feedback_processor

logger = get_structured_logger(__name__)

router = APIRouter(prefix="/api/v1/qa-feedback", tags=["问答-反馈增强"])


# ========== 请求/响应模型 ==========

class FeedbackQuestionRequest(BaseModel):
    """带反馈的问题请求"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=500)
    top_k: int = Field(10, description="检索的文档块数量", ge=1, le=50)
    min_confidence: float = Field(0.7, description="最小置信度阈值", ge=0.0, le=1.0)

    # 新增：反馈相关参数
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    enable_feedback: bool = Field(True, description="是否启用反馈优化")


class UserInteraction(BaseModel):
    """用户交互数据"""
    clicks: List[Dict[str, Any]] = Field(default_factory=list, description="点击的来源")
    rating: Optional[int] = Field(None, ge=1, le=5, description="用户评分")
    skipped: bool = Field(False, description="是否跳过结果")
    dwell_time_ms: Optional[int] = Field(None, description="页面停留时间(毫秒)")


class FeedbackSubmissionRequest(BaseModel):
    """反馈提交请求"""
    question: str
    sources: List[Dict[str, Any]]
    interaction: UserInteraction
    user_id: Optional[str] = None
    session_id: str


# ========== API端点 ==========

@router.post("/ask", summary="提问接口-集成反馈回路")
@handle_errors(
    default_return={
        "question": "",
        "optimized_question": "",
        "answer": "问答处理失败，请稍后重试",
        "sources": [],
        "trust_explanation": None,
        "retrieval_path": {},
        "execution_time": 0.0,
        "timestamp": "",
        "feedback_enabled": False,
        "query_optimized": False
    },
    error_category=ErrorCategory.RETRIEVAL
)
async def ask_question_with_feedback(
    request: FeedbackQuestionRequest,
    db: Session = Depends(get_db)
):
    """
    提问接口 - 集成实时反馈回路

    流程:
    1. 应用反馈优化（查询重写、参数调整）
    2. 执行检索和问答
    3. 记录查询事件供反馈学习

    新增功能:
    - 查询自动重写 (盈利能力 → 净利润)
    - 动态参数调整 (根据历史反馈)
    - 个性化体验
    """
    logger.info(
        f"收到问题(反馈增强): {request.question}, "
        f"user={request.user_id}, session={request.session_id}, "
        f"feedback={request.enable_feedback}"
    )

    start_time = datetime.now()
    original_question = request.question
    optimized_question = request.question

    try:
        # ========== 阶段1: 反馈优化 ==========
        feedback_context = {}
        optimization_time = 0.0

        if request.enable_feedback:
            try:
                processor = get_realtime_feedback_processor()

                # 应用反馈优化
                feedback_result = await processor.enhance_query(
                    query=request.question,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    retrieval_level="enhanced"
                )

                optimized_question = feedback_result["optimized_query"]
                feedback_context = feedback_result["feedback_context"]
                optimization_time = feedback_result["optimization_time_ms"]

                # 应用动态参数
                dynamic_params = feedback_result["params"]
                if "top_k" in dynamic_params:
                    request.top_k = dynamic_params["top_k"]

                logger.info(
                    f"查询优化: '{original_question[:30]}...' → "
                    f"'{optimized_question[:30]}...', "
                    f"耗时={optimization_time:.2f}ms"
                )

            except Exception as e:
                logger.warning(f"反馈优化失败，使用原始查询: {e}")

        # ========== 阶段2: 执行RAG问答 ==========
        qa_service = get_rag_qa_service()

        result = await qa_service.answer_question(
            question=optimized_question,  # 使用优化后的查询
            top_k=request.top_k,
            min_confidence=request.min_confidence,
            db=db
        )

        # ========== 阶段3: 添加反馈元数据 ==========
        total_time = (datetime.now() - start_time).total_seconds()

        # 添加反馈相关元数据
        result["feedback_enabled"] = request.enable_feedback
        result["query_optimized"] = (original_question != optimized_question)
        result["original_question"] = original_question
        result["optimized_question"] = optimized_question
        result["optimization_time_ms"] = optimization_time
        result["feedback_context"] = feedback_context
        result["user_id"] = request.user_id
        result["session_id"] = request.session_id

        logger.info(
            f"问题回答完成(反馈增强): question={original_question[:30]}, "
            f"optimized={optimized_question != original_question}, "
            f"耗时={total_time:.2f}s"
        )

        return result

    except Exception as e:
        logger.error(f"问答处理失败: {e}")
        raise


@router.post("/submit-feedback", summary="提交用户反馈")
async def submit_question_feedback(
    request: FeedbackSubmissionRequest
):
    """
    提交问答反馈

    收集:
    - 点击的来源文档
    - 用户评分
    - 是否跳过
    - 停留时间
    """
    try:
        processor = get_realtime_feedback_processor()

        # 构建交互数据
        interactions = {
            "clicks": request.interaction.clicks,
            "rating": request.interaction.rating,
            "skipped": request.interaction.skipped
        }

        if request.interaction.dwell_time_ms:
            interactions["dwell_times"] = {
                str(i): request.interaction.dwell_time_ms / 1000.0
                for i in range(len(request.sources))
            }

        # 转换sources格式
        results = []
        for source in request.sources:
            results.append({
                "content": source.get("content", ""),
                "metadata": {
                    "chunk_id": source.get("chunk_id"),
                    "document_id": source.get("document_id"),
                    "confidence": source.get("confidence")
                }
            })

        # 收集反馈
        await processor.collect_feedback(
            query=request.question,
            results=results,
            user_interactions=interactions,
            user_id=request.user_id,
            session_id=request.session_id
        )

        logger.info(
            f"反馈已记录: question={request.question[:30]}, "
            f"session={request.session_id}, "
            f"rating={request.interaction.rating}"
        )

        return {
            "status": "success",
            "message": "反馈已记录",
            "session_id": request.session_id
        }

    except Exception as e:
        logger.error(f"提交反馈失败: {e}")
        raise


@router.get("/insights", summary="获取反馈洞察")
async def get_feedback_insights(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    获取反馈洞察

    返回:
    - 总体统计
    - 用户偏好 (如果提供user_id)
    - 会话历史 (如果提供session_id)
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
        raise


@router.get("/example", summary="获取反馈增强示例")
async def get_feedback_example():
    """
    获取反馈增强使用示例

    展示完整的使用流程
    """
    return {
        "title": "RAG问答 - 反馈增强使用示例",
        "workflow": [
            {
                "step": 1,
                "action": "提问",
                "endpoint": "POST /api/v1/qa-feedback/ask",
                "request": {
                    "question": "盈利能力怎么样？",
                    "user_id": "user_123",
                    "session_id": "session_456",
                    "enable_feedback": True
                },
                "response": {
                    "optimized_question": "净利润怎么样？",
                    "answer": "根据财务数据...",
                    "sources": [...],
                    "query_optimized": True
                }
            },
            {
                "step": 2,
                "action": "用户查看结果并反馈",
                "description": "用户点击来源、评分等"
            },
            {
                "step": 3,
                "action": "提交反馈",
                "endpoint": "POST /api/v1/qa-feedback/submit-feedback",
                "request": {
                    "question": "盈利能力怎么样？",
                    "interaction": {
                        "clicks": [{"chunk_id": 123, "position": 1}],
                        "rating": 4,
                        "dwell_time_ms": 15000
                    },
                    "session_id": "session_456"
                }
            },
            {
                "step": 4,
                "action": "下次提问自动优化",
                "description": "系统基于反馈自动优化查询"
            }
        ],
        "benefits": [
            "查询自动重写 (盈利能力 → 净利润)",
            "动态参数调整 (top_k等)",
            "个性化体验",
            "持续学习优化"
        ],
        "api_endpoints": {
            "ask": "/api/v1/qa-feedback/ask",
            "submit_feedback": "/api/v1/qa-feedback/submit-feedback",
            "insights": "/api/v1/qa-feedback/insights"
        }
    }


@router.get("/health", summary="健康检查")
async def health_check():
    """健康检查"""
    try:
        processor = get_realtime_feedback_processor()

        return {
            "status": "healthy",
            "service": "RAG Question Answering with Feedback Loop",
            "feedback_enabled": True,
            "features": [
                "Vector Search",
                "In-Context Learning",
                "Real-time Feedback Optimization",
                "Query Rewriting",
                "Dynamic Parameter Adjustment",
                "Personalization"
            ],
            "stats": {
                "sessions": len(processor.sessions),
                "users": len(processor.user_preferences),
                "query_patterns": len(processor.query_patterns)
            }
        }

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
