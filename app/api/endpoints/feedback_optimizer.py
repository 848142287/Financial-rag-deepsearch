"""
反馈驱动检索优化API端点
提供RESTful API接口，支持迭代检索优化功能
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import logging

from app.services.feedback_optimizer.feedback_optimizer import feedback_optimizer
from app.core.auth import get_current_user
from app.models.user import User
from app.services.rag.rag_service import rag_service

logger = logging.getLogger(__name__)

router = APIRouter()

# 注入RAG服务
feedback_optimizer.set_rag_service(rag_service)


class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str = Field(..., description="搜索查询")
    user_id: Optional[int] = Field(None, description="用户ID")
    context: Optional[Dict[str, Any]] = Field(default=None, description="搜索上下文")


class FeedbackRequest(BaseModel):
    """反馈请求模型"""
    session_id: str = Field(..., description="会话ID")
    feedback_type: str = Field(..., description="反馈类型")
    rating: Optional[int] = Field(None, ge=1, le=5, description="满意度评分 1-5")
    comments: Optional[str] = Field(None, description="文字反馈")
    rewritten_query: Optional[str] = Field(None, description="用户改写的查询")
    highlighted_items: List[int] = Field(default_factory=list, description="高亮的结果项")
    specific_requirements: List[str] = Field(default_factory=list, description="特定需求")


class SessionInfoRequest(BaseModel):
    """会话信息请求模型"""
    session_id: str = Field(..., description="会话ID")


@router.post("/search/start")
async def start_search_session(
    request: SearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    开始新的搜索会话
    """
    try:
        logger.info(f"用户 {current_user.id} 开始搜索：{request.query}")

        result = await feedback_optimizer.start_search_session(
            user_id=current_user.id,
            initial_query=request.query
        )

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        logger.error(f"开始搜索会话失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"开始搜索失败：{str(e)}")


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    current_user: User = Depends(get_current_user)
):
    """
    提交用户反馈
    """
    try:
        logger.info(f"用户 {current_user.id} 提交反馈：{request.feedback_type}")

        # 验证会话所有权
        session_info = await feedback_optimizer.get_session_info(request.session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="会话不存在")

        if session_info.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="无权访问该会话")

        feedback_data = {
            "feedback_type": request.feedback_type,
            "rating": request.rating,
            "comments": request.comments,
            "rewritten_query": request.rewritten_query,
            "highlighted_items": request.highlighted_items,
            "specific_requirements": request.specific_requirements
        }

        result = await feedback_optimizer.process_feedback(
            session_id=request.session_id,
            feedback_data=feedback_data
        )

        return {
            "success": True,
            "data": result
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"处理反馈失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"处理反馈失败：{str(e)}")


@router.get("/session/{session_id}")
async def get_session_info(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    获取会话信息
    """
    try:
        session_info = await feedback_optimizer.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="会话不存在")

        # 验证会话所有权
        if session_info.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="无权访问该会话")

        return {
            "success": True,
            "data": session_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话信息失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话信息失败：{str(e)}")


@router.get("/session/{session_id}/stats")
async def get_session_stats(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    获取会话统计信息
    """
    try:
        session_stats = await feedback_optimizer.get_session_stats(session_id)
        if not session_stats:
            raise HTTPException(status_code=404, detail="会话不存在")

        # 验证会话所有权
        if session_stats.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="无权访问该会话")

        return {
            "success": True,
            "data": session_stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话统计失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话统计失败：{str(e)}")


@router.post("/session/{session_id}/abandon")
async def abandon_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    放弃会话
    """
    try:
        session_info = await feedback_optimizer.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="会话不存在")

        # 验证会话所有权
        if session_info.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="无权访问该会话")

        success = await feedback_optimizer.abandon_session(session_id)
        if not success:
            raise HTTPException(status_code=400, detail="无法放弃该会话")

        return {
            "success": True,
            "message": "会话已放弃"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"放弃会话失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"放弃会话失败：{str(e)}")


@router.get("/session/{session_id}/suggestions")
async def get_optimization_suggestions(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    获取优化建议
    """
    try:
        session_info = await feedback_optimizer.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="会话不存在")

        # 验证会话所有权
        if session_info.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="无权访问该会话")

        suggestions = await feedback_optimizer.get_optimization_suggestions(session_id)

        return {
            "success": True,
            "data": {
                "suggestions": suggestions,
                "count": len(suggestions)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取优化建议失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"获取优化建议失败：{str(e)}")


@router.get("/sessions/active")
async def get_active_sessions(
    current_user: User = Depends(get_current_user)
):
    """
    获取用户的活跃会话列表
    """
    try:
        # 这里需要实现获取用户所有活跃会话的逻辑
        # 暂时返回模拟数据
        active_sessions = [
            {
                "session_id": "mock_session_1",
                "initial_query": "财务分析报告",
                "current_round": 2,
                "max_rounds": 5,
                "state": "showing",
                "last_activity": "2024-01-15T10:30:00Z"
            }
        ]

        return {
            "success": True,
            "data": {
                "sessions": active_sessions,
                "count": len(active_sessions)
            }
        }

    except Exception as e:
        logger.error(f"获取活跃会话失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"获取活跃会话失败：{str(e)}")


@router.get("/strategies")
async def get_optimization_strategies(
    current_user: User = Depends(get_current_user)
):
    """
    获取可用的优化策略列表
    """
    try:
        strategies = [
            {
                "name": "query_rewrite",
                "label": "查询改写",
                "description": "基于用户反馈重新构建检索查询，提高相关性",
                "applicable_for": ["relevance_low", "general"],
                "typical_improvements": ["相关性提升", "语义匹配优化"]
            },
            {
                "name": "range_expansion",
                "label": "范围扩展",
                "description": "扩大检索范围和深度，获取更全面的信息",
                "applicable_for": ["incomplete"],
                "typical_improvements": ["结果数量增加", "信息完整性提升"]
            },
            {
                "name": "authority_boost",
                "label": "权威性提升",
                "description": "优先显示权威来源，提高信息准确性",
                "applicable_for": ["accuracy_issue"],
                "typical_improvements": ["信息可信度提升", "权威源优先"]
            },
            {
                "name": "sort_optimization",
                "label": "排序优化",
                "description": "调整结果排序算法，优化结果排序",
                "applicable_for": ["sorting_issue"],
                "typical_improvements": ["排序合理性提升", "多样性保证"]
            },
            {
                "name": "comprehensive",
                "label": "综合优化",
                "description": "全面调整检索参数，综合优化各个方面",
                "applicable_for": ["general", "high_severity"],
                "typical_improvements": ["整体质量提升", "多维度优化"]
            }
        ]

        return {
            "success": True,
            "data": {
                "strategies": strategies,
                "count": len(strategies)
            }
        }

    except Exception as e:
        logger.error(f"获取优化策略失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"获取优化策略失败：{str(e)}")


@router.get("/feedback/options")
async def get_feedback_options(
    current_user: User = Depends(get_current_user)
):
    """
    获取反馈选项
    """
    try:
        options = [
            {
                "type": "relevance_low",
                "label": "结果不相关",
                "description": "检索结果与我的问题不匹配",
                "icon": "unlink",
                "priority": "high"
            },
            {
                "type": "incomplete",
                "label": "信息不完整",
                "description": "缺少我需要的信息",
                "icon": "missing",
                "priority": "high"
            },
            {
                "type": "accuracy_issue",
                "label": "准确性有问题",
                "description": "结果中的信息可能不准确或过时",
                "icon": "warning",
                "priority": "high"
            },
            {
                "type": "sorting_issue",
                "label": "排序问题",
                "description": "重要结果的排序不合理",
                "icon": "sort",
                "priority": "medium"
            },
            {
                "type": "general",
                "label": "其他问题",
                "description": "其他类型的问题或建议",
                "icon": "more",
                "priority": "low"
            }
        ]

        return {
            "success": True,
            "data": {
                "options": options,
                "rating_scale": {
                    "min": 1,
                    "max": 5,
                    "labels": {
                        1: "非常不满意",
                        2: "不满意",
                        3: "一般",
                        4: "满意",
                        5: "非常满意"
                    }
                }
            }
        }

    except Exception as e:
        logger.error(f"获取反馈选项失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"获取反馈选项失败：{str(e)}")


# 健康检查端点
@router.get("/health")
async def health_check():
    """
    反馈优化器健康检查
    """
    try:
        # 检查核心组件状态
        active_sessions_count = feedback_optimizer.session_manager.get_active_sessions_count()

        return {
            "status": "healthy",
            "components": {
                "session_manager": "healthy",
                "feedback_analyzer": "healthy",
                "strategy_selector": "healthy",
                "rag_service": "connected" if rag_service else "disconnected"
            },
            "metrics": {
                "active_sessions": active_sessions_count,
                "timestamp": asyncio.get_event_loop().time()
            }
        }

    except Exception as e:
        logger.error(f"健康检查失败：{str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }