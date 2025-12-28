"""
文档重试API端点

提供文档处理失败时的重试功能
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_db
from app.services.retry_service import (
    document_retry_service, ErrorType
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["文档重试"])

# 请求模型
class RetryRequest(BaseModel):
    """重试请求"""
    document_id: int = Field(..., description="文档ID")
    force_retry: bool = Field(False, description="是否强制重试")

class BatchRetryRequest(BaseModel):
    """批量重试请求"""
    limit: int = Field(10, description="批量重试数量限制", ge=1, le=50)
    error_types: Optional[List[str]] = Field(None, description="指定重试的错误类型")

# 响应模型
class RetryResponse(BaseModel):
    """重试响应"""
    success: bool
    document_id: Optional[int] = None
    task_id: Optional[str] = None
    retry_count: Optional[int] = None
    error_type: Optional[str] = None
    retry_strategy: Optional[str] = None
    retry_delay: Optional[int] = None
    error: Optional[str] = None

class BatchRetryResponse(BaseModel):
    """批量重试响应"""
    success: bool
    total: int
    retried: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    error: Optional[str] = None

@router.post("/documents/{document_id}/retry", response_model=RetryResponse)
async def retry_document(
    document_id: int,
    request: RetryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_db)
):
    """
    重试单个文档的处理

    Args:
        document_id: 文档ID
        request: 重试请求
        background_tasks: 后台任务
        db: 数据库会话
    """
    try:
        logger.info(f"收到文档重试请求: document_id={document_id}, force_retry={request.force_retry}")

        # 执行重试
        result = await document_retry_service.retry_document_processing(
            document_id=document_id,
            db=db,
            force_retry=request.force_retry
        )

        return RetryResponse(**result)

    except Exception as e:
        logger.error(f"文档重试失败: {e}")
        raise HTTPException(status_code=500, detail=f"重试失败: {str(e)}")

@router.post("/documents/batch-retry", response_model=BatchRetryResponse)
async def batch_retry_documents(
    request: BatchRetryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_db)
):
    """
    批量重试失败的文档

    Args:
        request: 批量重试请求
        background_tasks: 后台任务
        db: 数据库会话
    """
    try:
        logger.info(f"收到批量重试请求: limit={request.limit}, error_types={request.error_types}")

        # 转换错误类型
        error_types = None
        if request.error_types:
            error_types = []
            for error_type_str in request.error_types:
                try:
                    error_type = ErrorType(error_type_str)
                    error_types.append(error_type)
                except ValueError:
                    logger.warning(f"无效的错误类型: {error_type_str}")

        # 执行批量重试
        result = await document_retry_service.batch_retry_failed_documents(
            db=db,
            limit=request.limit,
            error_types=error_types
        )

        # 检查是否成功
        if not result.get("success", False):
            error_msg = result.get("error", "批量重试失败")
            logger.error(f"批量重试失败: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        return BatchRetryResponse(**result)

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise
    except Exception as e:
        logger.error(f"批量重试失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量重试失败: {str(e)}")

@router.get("/documents/failed-summary")
async def get_failed_documents_summary(
    limit: int = Query(50, ge=1, le=100, description="查询限制"),
    db: AsyncSession = Depends(get_async_db)
):
    """
    获取失败文档的汇总信息

    Args:
        limit: 查询限制
        db: 数据库会话
    """
    try:
        logger.info(f"获取失败文档汇总: limit={limit}")

        # 获取汇总信息
        summary = await document_retry_service.get_failed_documents_summary(
            db=db,
            limit=limit
        )

        return summary

    except Exception as e:
        logger.error(f"获取失败文档汇总失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取汇总信息失败: {str(e)}")

@router.get("/documents/retry-strategies")
async def get_retry_strategies():
    """
    获取可用的重试策略信息
    """
    try:
        from app.services.document_retry_service import RetryStrategy, ErrorType

        return {
            "error_types": [
                {
                    "value": error_type.value,
                    "name": error_type.name,
                    "description": {
                        ErrorType.NETWORK_ERROR: "网络连接错误",
                        ErrorType.FILE_CORRUPTION: "文件损坏或格式错误",
                        ErrorType.PARSING_ERROR: "文档解析错误",
                        ErrorType.MEMORY_ERROR: "内存不足错误",
                        ErrorType.TIMEOUT_ERROR: "处理超时错误",
                        ErrorType.UNKNOWN_ERROR: "未知错误"
                    }.get(error_type, "未知错误类型")
                }
                for error_type in ErrorType
            ],
            "retry_strategies": [
                {
                    "value": strategy.value,
                    "name": strategy.name,
                    "description": {
                        RetryStrategy.IMMEDIATE: "立即重试",
                        RetryStrategy.FIXED_INTERVAL: "固定间隔重试",
                        RetryStrategy.EXPONENTIAL_BACKOFF: "指数退避重试",
                        RetryStrategy.SMART_RETRY: "智能重试（基于错误类型）"
                    }.get(strategy, "未知策略")
                }
                for strategy in RetryStrategy
            ],
            "max_retries": document_retry_service.max_retries
        }

    except Exception as e:
        logger.error(f"获取重试策略信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略信息失败: {str(e)}")

@router.post("/documents/cleanup-old-retries")
async def cleanup_old_retries(
    days: int = Query(7, ge=1, le=30, description="保留天数"),
    db: AsyncSession = Depends(get_async_db)
):
    """
    清理旧的重试记录

    Args:
        days: 保留天数
        db: 数据库会话
    """
    try:
        logger.info(f"清理旧重试记录: days={days}")

        # 执行清理
        cleaned_count = await document_retry_service.cleanup_old_retries(
            db=db,
            days=days
        )

        return {
            "success": True,
            "cleaned_count": cleaned_count,
            "message": f"已清理 {cleaned_count} 条旧重试记录"
        }

    except Exception as e:
        logger.error(f"清理旧重试记录失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")

# 错误处理（注：exception_handler应该在FastAPI应用级别定义，这里移除router级别的处理）