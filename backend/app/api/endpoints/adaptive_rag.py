"""
自适应RAG API端点

集成查询分类、参数优化、多臂老虎机选择器的智能检索API
"""

from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends

from app.core.database import get_db
from app.core.structured_logging import get_structured_logger
from app.core.errors.unified_errors import handle_errors, ErrorCategory
from app.services.agentic_rag.adaptive_retrieval import get_adaptive_feedback_processor

logger = get_structured_logger(__name__)

router = APIRouter(prefix="/api/v1/adaptive-rag", tags=["自适应RAG"])


# ========== 请求/响应模型 ==========

class AdaptiveQuestionRequest(BaseModel):
    """自适应检索请求"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=500)
    retrieval_level: str = Field("enhanced", description="检索级别: fast/enhanced/deep_search")

    # 用户标识
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")

    # 自适应选项
    enable_adaptive: bool = Field(True, description="是否启用自适应检索")
    enable_classification: bool = Field(True, description="是否启用查询分类")
    enable_optimization: bool = Field(True, description="是否启用参数优化")
    enable_bandit: bool = Field(True, description="是否启用方法选择")

    # 基础参数（可选，会被自适应优化覆盖）
    top_k: Optional[int] = Field(None, description="检索数量（会被自适应优化）")
    similarity_threshold: Optional[float] = Field(None, description="相似度阈值（会被自适应优化）")


class UserInteraction(BaseModel):
    """用户交互数据"""
    clicks: List[Dict[str, Any]] = Field(default_factory=list, description="点击记录")
    dwell_times: Dict[str, int] = Field(default_factory=dict, description="停留时间")
    rating: Optional[int] = Field(None, ge=1, le=5, description="用户评分")
    skipped: bool = Field(False, description="是否跳过")


class AdaptiveFeedbackRequest(BaseModel):
    """自适应反馈提交请求"""
    question: str
    selected_method: str = Field(..., description="使用的检索方法")
    results: List[Dict[str, Any]]
    interaction: UserInteraction
    user_id: Optional[str] = None
    session_id: str


# ========== API端点 ==========

@router.post("/query", summary="自适应查询接口")
@handle_errors(
    default_return={
        "question": "",
        "optimized_question": "",
        "selected_method": "vector",
        "query_features": {},
        "optimized_params": {},
        "recommendations": [],
        "adaptive_enabled": False,
        "timestamp": ""
    },
    error_category=ErrorCategory.RETRIEVAL
)
async def adaptive_query(
    request: AdaptiveQuestionRequest,
    db: Session = Depends(get_db)
):
    """
    自适应查询接口

    功能:
    1. 自动分类查询类型（事实性/分析性/比较性等）
    2. 自动选择最优检索方法（向量/图谱/混合）
    3. 自动优化检索参数（top_k, threshold等）
    4. 基于历史反馈动态调整

    Args:
        request: 自适应查询请求

    Returns:
        包含优化后的查询、选择的方法、参数等
    """
    try:
        # 获取自适应处理器
        processor = get_adaptive_feedback_processor(
            enable_classification=request.enable_classification,
            enable_optimization=request.enable_optimization,
            enable_bandit=request.enable_bandit,
            enable_feedback=True
        )

        # 处理查询
        base_params = {}
        if request.top_k:
            base_params["top_k"] = request.top_k
        if request.similarity_threshold:
            base_params["similarity_threshold"] = request.similarity_threshold

        result = await processor.process_query_with_adaptive_feedback(
            query=request.question,
            user_id=request.user_id,
            session_id=request.session_id or f"session_{datetime.now().timestamp()}",
            retrieval_level=request.retrieval_level,
            base_params=base_params
        )

        # 生成推荐建议
        recommendations = _generate_recommendations(result)

        return {
            "question": request.question,
            "optimized_question": result["query"],
            "selected_method": result["method"],
            "query_features": result.get("features", {}),
            "optimized_params": result["params"],
            "retrieval_level": result["retrieval_level"],
            "recommendations": recommendations,
            "adaptive_enabled": request.enable_adaptive,
            "components_enabled": {
                "classification": request.enable_classification,
                "optimization": request.enable_optimization,
                "bandit": request.enable_bandit
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"自适应查询处理失败: {e}", exc_info=True)
        raise


@router.post("/feedback", summary="提交自适应反馈")
@handle_errors(
    default_return={
        "status": "error",
        "message": "反馈提交失败"
    },
    error_category=ErrorCategory.PROCESSING
)
async def submit_adaptive_feedback(
    request: AdaptiveFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    提交自适应反馈

    收集用户对检索结果的反馈，用于:
    1. 更新多臂老虎机的奖励
    2. 学习最优参数组合
    3. 优化未来的检索策略

    Args:
        request: 反馈请求

    Returns:
        提交状态
    """
    try:
        processor = get_adaptive_feedback_processor()

        # 转换交互数据
        user_interactions = {
            "clicks": request.interaction.clicks,
            "dwell_times": request.interaction.dwell_times,
            "rating": request.interaction.rating,
            "skipped": request.interaction.skipped
        }

        # 提交反馈
        await processor.collect_result_feedback(
            query=request.question,
            user_id=request.user_id,
            session_id=request.session_id,
            method=request.selected_method,
            results=request.results,
            user_interactions=user_interactions
        )

        return {
            "status": "success",
            "message": "反馈已记录",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"反馈提交失败: {e}", exc_info=True)
        raise


@router.get("/insights", summary="获取自适应系统洞察")
@handle_errors(
    default_return={
        "components_enabled": {},
        "statistics": {},
        "recommendations": []
    },
    error_category=ErrorCategory.PROCESSING
)
async def get_adaptive_insights(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    获取自适应系统洞察

    返回:
    - 各组件启用状态
    - 查询统计信息
    - 多臂老虎机性能
    - 参数优化情况
    - 推荐建议

    Args:
        user_id: 用户ID（可选）
        session_id: 会话ID（可选）

    Returns:
        系统洞察数据
    """
    try:
        processor = get_adaptive_feedback_processor()

        insights = processor.get_adaptive_insights(
            user_id=user_id,
            session_id=session_id
        )

        # 生成推荐建议
        recommendations = _generate_system_recommendations(insights)

        return {
            "components_enabled": insights.get("components_enabled", {}),
            "query_stats": insights.get("query_stats", {}),
            "optimizer": insights.get("optimizer", {}),
            "bandit": insights.get("bandit", {}),
            "feedback": insights.get("feedback", {}),
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"获取洞察失败: {e}", exc_info=True)
        raise


@router.post("/reset-stats", summary="重置自适应统计")
@handle_errors(
    default_return={
        "status": "error",
        "message": "重置失败"
    },
    error_category=ErrorCategory.PROCESSING
)
async def reset_adaptive_stats(
    db: Session = Depends(get_db)
):
    """
    重置自适应系统统计

    清空所有学习到的统计数据，用于:
    1. 测试环境重置
    2. 重新开始学习

    Returns:
        重置状态
    """
    try:
        processor = get_adaptive_feedback_processor()
        processor.reset_stats()

        return {
            "status": "success",
            "message": "统计已重置",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"重置统计失败: {e}", exc_info=True)
        raise


# ========== 辅助函数 ==========

def _generate_recommendations(result: Dict[str, Any]) -> List[str]:
    """生成推荐建议"""
    recommendations = []

    features = result.get("features", {})
    params = result.get("params", {})
    method = result.get("method", "vector")

    # 基于查询类型的建议
    query_type = features.get("query_type", "")
    if query_type == "analytical":
        recommendations.append("这是分析性查询，系统已自动启用深度检索")
    elif query_type == "comparative":
        recommendations.append("这是比较性查询，系统可能选择图谱检索以获取关系信息")

    # 基于复杂度的建议
    complexity = features.get("complexity", "")
    if complexity == "complex":
        recommendations.append("这是复杂查询，系统已增加检索数量并降低相似度阈值")

    # 基于方法的建议
    if method == "graph":
        recommendations.append("系统选择了图谱检索，适合获取实体关系和结构化信息")
    elif method == "hybrid":
        recommendations.append("系统选择了混合检索，结合多种方法以提高准确性")

    # 基于参数的建议
    top_k = params.get("top_k", 10)
    if top_k > 15:
        recommendations.append(f"检索数量({top_k})较大，建议关注结果质量")

    return recommendations


def _generate_system_recommendations(insights: Dict[str, Any]) -> List[str]:
    """生成系统级推荐建议"""
    recommendations = []

    # 分析多臂老虎机性能
    bandit = insights.get("bandit", {})
    arm_stats = bandit.get("arm_stats", {})

    if arm_stats:
        # 找出表现最好的方法
        best_method = max(arm_stats.items(), key=lambda x: x[1].get("avg_reward", 0))
        best_name = best_method[1].get("display_name", best_method[0])
        best_reward = best_method[1].get("avg_reward", 0)

        if best_reward > 0.7:
            recommendations.append(f"检索方法'{best_name}'表现优秀(奖励:{best_reward:.2f})")

        # 找出使用最少的方法
        least_used = min(arm_stats.items(), key=lambda x: x[1].get("pulls", 0))
        if least_used[1].get("pulls", 0) < 5:
            recommendations.append(f"方法'{least_used[1].get('display_name')}'使用较少，建议增加探索")

    # 分析参数优化
    optimizer = insights.get("optimizer", {})
    total_usage = optimizer.get("total_usage", 0)

    if total_usage > 100:
        recommendations.append(f"系统已优化{total_usage}次查询，参数学习进展良好")

    # 分析查询统计
    query_stats = insights.get("query_stats", {})
    if len(query_stats) > 10:
        recommendations.append(f"已收集{len(query_stats)}个查询的反馈，系统适应性持续提升")

    return recommendations
