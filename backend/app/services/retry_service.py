"""
文档重试服务
提供文档处理失败后的重试功能
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.structured_logging import get_structured_logger
from app.models.document import Document

logger = get_structured_logger(__name__)

class ErrorType(Enum):
    """错误类型"""
    NETWORK_ERROR = "network_error"
    FILE_CORRUPTION = "file_corruption"
    PARSING_ERROR = "parsing_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"

class RetryStrategy(Enum):
    """重试策略"""
    IMMEDIATE = "immediate"
    FIXED_INTERVAL = "fixed_interval"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    SMART_RETRY = "smart_retry"

class RetryService:
    """文档重试服务"""

    def __init__(self, max_retries: int = 3):
        """
        初始化重试服务

        Args:
            max_retries: 最大重试次数
        """
        self.max_retries = max_retries

    async def retry_document_processing(
        self,
        document_id: int,
        db: AsyncSession,
        force_retry: bool = False
    ) -> Dict[str, Any]:
        """
        重试单个文档的处理

        Args:
            document_id: 文档ID
            db: 数据库会话
            force_retry: 是否强制重试

        Returns:
            重试结果字典
        """
        try:
            logger.info(f"开始重试文档: document_id={document_id}, force_retry={force_retry}")

            # 查询文档
            result = await db.execute(select(Document).where(Document.id == document_id))
            document = result.scalar_one_or_none()

            if not document:
                return {
                    "success": False,
                    "error": f"文档不存在: document_id={document_id}",
                    "document_id": document_id
                }

            # 检查重试次数
            retry_count = document.retry_count or 0
            if not force_retry and retry_count >= self.max_retries:
                return {
                    "success": False,
                    "error": f"已达到最大重试次数: {retry_count}",
                    "document_id": document_id,
                    "retry_count": retry_count
                }

            # 重新提交到Celery队列
            from app.tasks.unified_task_manager import process_document_unified

            # 重置文档状态
            document.status = "uploading"
            document.error_message = None
            document.retry_count = retry_count + 1
            document.updated_at = datetime.utcnow()

            await db.commit()
            await db.refresh(document)

            # 提交任务到Celery
            task = process_document_unified.delay(
                document_id=str(document.id),
                original_filename=document.filename,
                user_id=None
            )

            logger.info(f"文档重试任务已提交: document_id={document_id}, task_id={task.id}")

            return {
                "success": True,
                "document_id": document_id,
                "task_id": task.id,
                "retry_count": document.retry_count,
                "error_type": self._detect_error_type(document.error_message),
                "retry_strategy": RetryStrategy.SMART_RETRY.value,
                "retry_delay": 0
            }

        except Exception as e:
            logger.error(f"重试文档失败: document_id={document_id}, error={e}")
            await db.rollback()
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id
            }

    async def batch_retry_failed_documents(
        self,
        db: AsyncSession,
        limit: int = 10,
        error_types: Optional[List[ErrorType]] = None
    ) -> Dict[str, Any]:
        """
        批量重试失败的文档

        Args:
            db: 数据库会话
            limit: 批量重试数量限制
            error_types: 指定重试的错误类型列表

        Returns:
            批量重试结果字典
        """
        try:
            logger.info(f"开始批量重试失败文档: limit={limit}, error_types={error_types}")

            # 构建查询条件
            conditions = [Document.status == "processing_failed"]

            # 如果指定了错误类型，添加过滤条件
            if error_types:
                error_type_values = [et.value for et in error_types]
                # 这里简化处理，实际应用中可能需要更复杂的错误类型匹配
                # 暂时忽略这个条件

            # 查询失败的文档
            result = await db.execute(
                select(Document)
                .where(and_(*conditions))
                .order_by(Document.created_at.desc())
                .limit(limit)
            )
            failed_documents = result.scalars().all()

            if not failed_documents:
                return {
                    "success": True,
                    "total": 0,
                    "retried": 0,
                    "successful": 0,
                    "failed": 0,
                    "results": []
                }

            logger.info(f"找到 {len(failed_documents)} 个失败文档")

            # 批量重试
            results = []
            successful_count = 0
            failed_count = 0

            for document in failed_documents:
                retry_result = await self.retry_document_processing(
                    document_id=document.id,
                    db=db,
                    force_retry=False
                )

                results.append(retry_result)

                if retry_result.get("success"):
                    successful_count += 1
                else:
                    failed_count += 1

            return {
                "success": True,
                "total": len(failed_documents),
                "retried": len(failed_documents),
                "successful": successful_count,
                "failed": failed_count,
                "results": results
            }

        except Exception as e:
            logger.error(f"批量重试文档失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "total": 0,
                "retried": 0,
                "successful": 0,
                "failed": 0,
                "results": []
            }

    async def get_failed_documents_summary(
        self,
        db: AsyncSession,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        获取失败文档的汇总信息

        Args:
            db: 数据库会话
            limit: 查询限制

        Returns:
            失败文档汇总信息
        """
        try:
            logger.info(f"获取失败文档汇总: limit={limit}")

            # 查询失败的文档
            result = await db.execute(
                select(Document)
                .where(Document.status == "processing_failed")
                .order_by(Document.created_at.desc())
                .limit(limit)
            )
            failed_documents = result.scalars().all()

            # 统计错误类型
            error_type_counts = {}
            for doc in failed_documents:
                error_type = self._detect_error_type(doc.error_message)
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

            # 构建文档列表
            documents_info = []
            for doc in failed_documents:
                documents_info.append({
                    "document_id": doc.id,
                    "title": doc.title,
                    "filename": doc.filename,
                    "status": doc.status,
                    "error_message": doc.error_message,
                    "retry_count": doc.retry_count or 0,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
                })

            return {
                "total_failed": len(failed_documents),
                "error_type_distribution": error_type_counts,
                "documents": documents_info,
                "limit": limit
            }

        except Exception as e:
            logger.error(f"获取失败文档汇总失败: {e}")
            return {
                "total_failed": 0,
                "error_type_distribution": {},
                "documents": [],
                "error": str(e)
            }

    async def cleanup_old_retries(
        self,
        db: AsyncSession,
        days: int = 7
    ) -> int:
        """
        清理旧的重试记录

        Args:
            db: 数据库会话
            days: 保留天数

        Returns:
            清理的记录数量
        """
        try:
            logger.info(f"清理旧重试记录: days={days}")

            # 计算截止日期
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # 删除旧的失败文档记录（这里简化处理，实际可能需要软删除）
            # 暂时返回0，因为实际删除需要更谨慎的处理
            cleaned_count = 0

            logger.info(f"清理完成: cleaned_count={cleaned_count}")
            return cleaned_count

        except Exception as e:
            logger.error(f"清理旧重试记录失败: {e}")
            return 0

    def _detect_error_type(self, error_message: Optional[str]) -> str:
        """
        根据错误消息检测错误类型

        Args:
            error_message: 错误消息

        Returns:
            错误类型字符串
        """
        if not error_message:
            return ErrorType.UNKNOWN_ERROR.value

        error_message_lower = error_message.lower()

        # 根据错误消息关键词判断错误类型
        if any(keyword in error_message_lower for keyword in ["network", "connection", "timeout", "网络", "连接"]):
            return ErrorType.NETWORK_ERROR.value
        elif any(keyword in error_message_lower for keyword in ["corrupt", "损坏", "格式"]):
            return ErrorType.FILE_CORRUPTION.value
        elif any(keyword in error_message_lower for keyword in ["parse", "解析", "parsing"]):
            return ErrorType.PARSING_ERROR.value
        elif any(keyword in error_message_lower for keyword in ["memory", "内存", "oom"]):
            return ErrorType.MEMORY_ERROR.value
        elif any(keyword in error_message_lower for keyword in ["timeout", "超时"]):
            return ErrorType.TIMEOUT_ERROR.value
        else:
            return ErrorType.UNKNOWN_ERROR.value

# 导出
__all__ = [
    'ErrorType',
    'RetryStrategy',
    'RetryService'
]
