"""
Agentic RAG API接口
三级检索体系和三阶段流程的完整API
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from app.core.auth import get_current_user
from app.models.user import User
from app.services.agentic_rag.agentic_rag_system import (
    AgenticRAGSystem, AgenticRAGRequest, RetrievalLevel, ProcessStatus
)

logger = logging.getLogger(__name__)

router = APIRouter()

# 初始化系统
agentic_rag_system = AgenticRAGSystem()


# 请求模型
class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., description="用户查询", min_length=1, max_length=1000)
    retrieval_level: str = Field("fast", description="检索级别: fast/enhanced/deep_search")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")
    session_id: Optional[str] = Field(None, description="会话ID")
    async_mode: bool = Field(False, description="是否异步处理")


class AsyncTaskRequest(BaseModel):
    """异步任务请求"""
    query: str = Field(..., description="用户查询", min_length=1, max_length=1000)
    retrieval_level: str = Field("deep_search", description="检索级别")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")
    session_id: Optional[str] = Field(None, description="会话ID")


@router.post("/query")
async def process_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    处理查询请求（同步模式）
    """
    try:
        logger.info(f"用户 {current_user.id} 发起查询: {request.query[:100]}...")

        # 验证检索级别
        try:
            retrieval_level = RetrievalLevel(request.retrieval_level)
        except ValueError:
            raise HTTPException(status_code=400, detail="无效的检索级别")

        # 构建请求
        agentic_request = AgenticRAGRequest(
            query=request.query,
            retrieval_level=retrieval_level,
            context=request.context,
            session_id=request.session_id,
            user_id=str(current_user.id),
            async_mode=request.async_mode
        )

        # 处理查询
        response = await agentic_rag_system.process_query(agentic_request)

        # 构建返回数据
        return_data = {
            "request_id": response.request_id,
            "query": response.query,
            "answer": response.answer,
            "retrieval_level": response.retrieval_level.value,
            "sources": response.sources,
            "quality_score": response.quality_score,
            "processing_time": response.processing_time,
            "metadata": response.metadata
        }

        # 如果需要详细信息
        if request.context and request.context.get("include_details", False):
            return_data.update({
                "plan": {
                    "plan_id": response.plan.plan_id,
                    "query_type": response.plan.query_analysis.query_type.value,
                    "complexity": response.plan.query_analysis.complexity.value,
                    "strategies": [
                        {
                            "primary_method": s.primary_method,
                            "secondary_methods": s.secondary_methods,
                            "quality_threshold": s.quality_threshold
                        }
                        for s in response.plan.strategies
                    ]
                },
                "execution_stats": response.execution_result.execution_stats,
                "generation_metadata": response.generation_result.metadata
            })

        return {
            "success": True,
            "data": return_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@router.post("/async_query")
async def create_async_task(
    request: AsyncTaskRequest,
    current_user: User = Depends(get_current_user)
):
    """
    创建异步查询任务（DeepSearch模式）
    """
    try:
        logger.info(f"用户 {current_user.id} 创建异步任务: {request.query[:100]}...")

        # 验证检索级别
        try:
            retrieval_level = RetrievalLevel(request.retrieval_level)
        except ValueError:
            raise HTTPException(status_code=400, detail="无效的检索级别")

        # 强制异步模式用于深度检索
        if retrieval_level != RetrievalLevel.DEEP_SEARCH:
            raise HTTPException(status_code=400, detail="异步查询仅支持deep_search级别")

        # 构建请求
        agentic_request = AgenticRAGRequest(
            query=request.query,
            retrieval_level=retrieval_level,
            context=request.context,
            session_id=request.session_id,
            user_id=str(current_user.id),
            async_mode=True
        )

        # 创建异步任务
        task_id = await agentic_rag_system._create_async_task(agentic_request)

        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "message": "异步任务已创建",
                "query": request.query,
                "retrieval_level": retrieval_level.value
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"异步任务创建失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"异步任务创建失败: {str(e)}")


@router.get("/task/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    获取异步任务状态
    """
    try:
        task_info = await agentic_rag_system.get_task_status(task_id)

        if not task_info:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 构建返回数据
        status_data = {
            "task_id": task_info.task_id,
            "status": task_info.status.value,
            "progress": task_info.progress,
            "current_stage": task_info.current_stage,
            "created_at": task_info.created_at.isoformat(),
            "updated_at": task_info.updated_at.isoformat()
        }

        # 如果任务完成，包含结果
        if task_info.status == ProcessStatus.COMPLETED and task_info.result:
            status_data["result"] = {
                "answer": task_info.result.answer,
                "sources": task_info.result.sources,
                "quality_score": task_info.result.quality_score,
                "processing_time": task_info.result.processing_time,
                "retrieval_level": task_info.result.retrieval_level.value
            }

        # 如果任务失败，包含错误信息
        if task_info.status == ProcessStatus.FAILED:
            status_data["error_message"] = task_info.error_message

        return {
            "success": True,
            "data": status_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")


@router.delete("/task/{task_id}")
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    取消异步任务
    """
    try:
        success = await agentic_rag_system.cancel_task(task_id)

        if not success:
            raise HTTPException(status_code=404, detail="任务不存在或无法取消")

        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "message": "任务已取消"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")


@router.get("/tasks")
async def get_user_tasks(
    status: Optional[str] = Query(None, description="任务状态过滤"),
    limit: int = Query(20, description="返回数量限制"),
    current_user: User = Depends(get_current_user)
):
    """
    获取用户的任务列表
    """
    try:
        # 获取系统所有任务（简化实现）
        system_stats = agentic_rag_system.get_system_stats()

        # 这里应该实现用户任务过滤
        # 简化返回系统统计
        return {
            "success": True,
            "data": {
                "system_stats": system_stats,
                "note": "详细用户任务列表需要数据库支持"
            }
        }

    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@router.get("/stats")
async def get_system_stats(
    current_user: User = Depends(get_current_user)
):
    """
    获取系统统计信息
    """
    try:
        stats = agentic_rag_system.get_system_stats()

        # 添加级别配置信息
        level_configs = {
            "fast": {
                "description": "快速检索：简单查询，直接向量检索",
                "p95_response": "≤ 3秒",
                "max_results": 5,
                "quality_threshold": 0.6
            },
            "enhanced": {
                "description": "增强检索：复杂查询，多策略并行执行",
                "p95_response": "≤ 8秒",
                "max_results": 15,
                "quality_threshold": 0.75
            },
            "deep_search": {
                "description": "深度检索：研究型查询，多轮迭代优化",
                "p95_response": "支持异步处理",
                "max_results": 30,
                "quality_threshold": 0.8
            }
        }

        return {
            "success": True,
            "data": {
                "task_statistics": stats,
                "level_configurations": level_configs,
                "system_status": "running"
            }
        }

    except Exception as e:
        logger.error(f"获取系统统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统统计失败: {str(e)}")


@router.get("/health")
async def health_check(
    current_user: User = Depends(get_current_user)
):
    """
    健康检查
    """
    try:
        # 简单的健康检查
        system_stats = agentic_rag_system.get_system_stats()

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "planner": "ok",
                "executor": "ok",
                "generator": "ok"
            },
            "metrics": system_stats
        }

        return {
            "success": True,
            "data": health_status
        }

    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {
            "success": False,
            "data": {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        }


@router.get("/config")
async def get_system_config(
    current_user: User = Depends(get_current_user)
):
    """
    获取系统配置信息
    """
    try:
        config = {
            "retrieval_levels": {
                "fast": {
                    "name": "快速检索",
                    "description": "适用于简单事实查询",
                    "max_time": 10,
                    "max_results": 5,
                    "quality_threshold": 0.6
                },
                "enhanced": {
                    "name": "增强检索",
                    "description": "适用于复杂查询",
                    "max_time": 20,
                    "max_results": 15,
                    "quality_threshold": 0.75
                },
                "deep_search": {
                    "name": "深度检索",
                    "description": "适用于研究型查询",
                    "max_time": None,  # 异步处理
                    "max_results": 30,
                    "quality_threshold": 0.8
                }
            },
            "processing_stages": {
                "planning": {
                    "name": "计划阶段",
                    "description": "理解意图，制定检索策略"
                },
                "execution": {
                    "name": "执行阶段",
                    "description": "并行执行多路检索"
                },
                "generation": {
                    "name": "生成阶段",
                    "description": "基于结果生成答案"
                }
            },
            "features": {
                "async_processing": True,
                "multi_strategy_retrieval": True,
                "content_filtering": True,
                "fact_checking": True,
                "quality_assessment": True
            }
        }

        return {
            "success": True,
            "data": config
        }

    except Exception as e:
        logger.error(f"获取系统配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统配置失败: {str(e)}")


# 导出路由器
__all__ = ["router"]