"""
文档重试服务

提供文档上传和解析失败时的重试机制
支持智能重试策略、错误分类和自动恢复
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import aiofiles
import tempfile
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from app.models.document import Document, DocumentStatus
from app.core.celery import celery_app
from app.tasks.complete_pipeline_task import process_document_complete
from app.tasks.complete_pipeline_task import process_document_complete
from app.services.minio_service import MinIOService
from app.services.document_deduplication import document_deduplication_service

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """错误类型"""
    NETWORK_ERROR = "network_error"          # 网络错误
    FILE_CORRUPTION = "file_corruption"      # 文件损坏
    PARSING_ERROR = "parsing_error"          # 解析错误
    MEMORY_ERROR = "memory_error"            # 内存不足
    TIMEOUT_ERROR = "timeout_error"          # 超时错误
    UNKNOWN_ERROR = "unknown_error"          # 未知错误

class RetryStrategy(Enum):
    """重试策略"""
    IMMEDIATE = "immediate"                  # 立即重试
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # 指数退避
    FIXED_INTERVAL = "fixed_interval"        # 固定间隔
    SMART_RETRY = "smart_retry"              # 智能重试

class DocumentRetryService:
    """文档重试服务"""

    def __init__(self):
        self.minio_service = MinIOService()
        self.max_retries = 3
        self.retry_intervals = {
            RetryStrategy.IMMEDIATE: 0,
            RetryStrategy.FIXED_INTERVAL: 60,  # 60秒
            RetryStrategy.EXPONENTIAL_BACKOFF: lambda retry_count: min(300, 60 * (2 ** retry_count)),  # 最大5分钟
            RetryStrategy.SMART_RETRY: lambda retry_count: min(600, 120 * (retry_count + 1))  # 最大10分钟
        }

    async def classify_error(self, error_message: str) -> Tuple[ErrorType, RetryStrategy]:
        """
        分类错误并推荐重试策略

        Args:
            error_message: 错误消息

        Returns:
            (错误类型, 推荐重试策略)
        """
        error_msg_lower = error_message.lower()

        # 网络错误
        if any(keyword in error_msg_lower for keyword in [
            'connection', 'network', 'timeout', 'socket', 'dns'
        ]):
            return ErrorType.NETWORK_ERROR, RetryStrategy.EXPONENTIAL_BACKOFF

        # 文件损坏
        if any(keyword in error_msg_lower for keyword in [
            'corrupt', 'invalid format', 'damaged', 'truncated'
        ]):
            return ErrorType.FILE_CORRUPTION, RetryStrategy.SMART_RETRY

        # 解析错误
        if any(keyword in error_msg_lower for keyword in [
            'parse', 'pdf', 'extract', 'ocr', 'format'
        ]):
            return ErrorType.PARSING_ERROR, RetryStrategy.SMART_RETRY

        # 内存错误
        if any(keyword in error_msg_lower for keyword in [
            'memory', 'out of memory', 'oom'
        ]):
            return ErrorType.MEMORY_ERROR, RetryStrategy.EXPONENTIAL_BACKOFF

        # 超时错误
        if any(keyword in error_msg_lower for keyword in [
            'timeout', 'time out', 'deadline'
        ]):
            return ErrorType.TIMEOUT_ERROR, RetryStrategy.EXPONENTIAL_BACKOFF

        return ErrorType.UNKNOWN_ERROR, RetryStrategy.SMART_RETRY

    async def get_retry_delay(self, strategy: RetryStrategy, retry_count: int) -> int:
        """获取重试延迟时间（秒）"""
        if strategy == RetryStrategy.IMMEDIATE:
            return 0
        elif strategy == RetryStrategy.FIXED_INTERVAL:
            return self.retry_intervals[strategy]
        elif strategy in [RetryStrategy.EXPONENTIAL_BACKOFF, RetryStrategy.SMART_RETRY]:
            delay_func = self.retry_intervals[strategy]
            return delay_func(retry_count)
        return 60  # 默认60秒

    async def can_retry(self, document: Document) -> bool:
        """
        检查文档是否可以重试

        Args:
            document: 文档对象

        Returns:
            是否可以重试
        """
        # 只有失败状态的文档可以重试
        if document.status != DocumentStatus.FAILED:
            return False

        # 检查重试次数限制
        retry_count = document.retry_count or 0
        if retry_count >= self.max_retries:
            logger.warning(f"文档 {document.id} 已达到最大重试次数 ({self.max_retries})")
            return False

        # 检查下次重试时间
        if document.next_retry_at:
            if datetime.utcnow() < document.next_retry_at:
                logger.warning(f"文档 {document.id} 尚未到重试时间")
                return False

        return True

    async def prepare_document_for_retry(self, document: Document) -> Optional[bytes]:
        """
        为重试准备文档内容

        Args:
            document: 文档对象

        Returns:
            文档内容，如果准备失败返回None
        """
        try:
            # 尝试从MinIO获取文件
            if document.file_path:
                content = await self.minio_service.get_file(document.file_path)
                if content:
                    return content

            # 如果MinIO中没有，尝试重新上传（需要原始文件）
            logger.warning(f"无法从MinIO获取文档 {document.id} 的内容")
            return None

        except Exception as e:
            logger.error(f"准备文档 {document.id} 重试失败: {e}")
            return None

    async def retry_document_processing(
        self,
        document_id: int,
        db: AsyncSession,
        force_retry: bool = False
    ) -> Dict[str, Any]:
        """
        重试文档处理

        Args:
            document_id: 文档ID
            db: 数据库会话
            force_retry: 是否强制重试（忽略重试次数限制）

        Returns:
            重试结果
        """
        try:
            # 获取文档
            result = await db.execute(select(Document).where(Document.id == document_id))
            document = result.scalar_one_or_none()

            if not document:
                return {
                    "success": False,
                    "error": f"文档 {document_id} 不存在"
                }

            # 检查是否可以重试
            if not force_retry and not await self.can_retry(document):
                retry_count = getattr(document, 'retry_count', 0)
                return {
                    "success": False,
                    "error": f"文档无法重试，当前重试次数: {retry_count}/{self.max_retries}"
                }

            # 准备文档内容
            file_content = await self.prepare_document_for_retry(document)
            if not file_content:
                return {
                    "success": False,
                    "error": "无法准备文档内容进行重试"
                }

            # 分类错误
            error_type, retry_strategy = await self.classify_error(document.error_message or "")

            # 更新重试信息
            retry_count = (document.retry_count or 0) + 1
            document.retry_count = retry_count
            document.status = DocumentStatus.PROCESSING
            document.error_message = None

            await db.commit()

            # 计算重试延迟
            retry_delay = await self.get_retry_delay(retry_strategy, retry_count)

            # 更新下次重试时间
            if retry_delay > 0:
                document.next_retry_at = datetime.utcnow() + timedelta(seconds=retry_delay)
                await db.commit()

            # 延迟执行重试
            if retry_delay > 0:
                await asyncio.sleep(retry_delay)

            # 提交重试任务
            try:
                task = process_document_complete.delay(
                    document_id=str(document.id),
                    original_filename=document.filename,
                    user_id="retry_service"
                )

                # 更新任务ID
                document.task_id = task.id
                await db.commit()

                logger.info(f"文档 {document.id} 重试任务已提交: {task.id} (第 {retry_count} 次重试)")

                return {
                    "success": True,
                    "document_id": document.id,
                    "task_id": task.id,
                    "retry_count": retry_count,
                    "error_type": error_type.value,
                    "retry_strategy": retry_strategy.value,
                    "retry_delay": retry_delay
                }

            except Exception as task_error:
                logger.error(f"提交重试任务失败: {task_error}")

                # 更新状态为失败
                document.status = DocumentStatus.FAILED
                document.error_message = f"重试任务提交失败: {str(task_error)}"
                await db.commit()

                return {
                    "success": False,
                    "error": f"重试任务提交失败: {str(task_error)}"
                }

        except Exception as e:
            logger.error(f"重试文档 {document_id} 处理失败: {e}")
            return {
                "success": False,
                "error": f"重试处理失败: {str(e)}"
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
            error_types: 指定重试的错误类型，None表示重试所有类型

        Returns:
            批量重试结果
        """
        try:
            # 构建查询条件
            conditions = [Document.status == DocumentStatus.FAILED]

            # 添加重试次数限制
            conditions.append(
                or_(
                    Document.retry_count < self.max_retries,
                    Document.retry_count.is_(None)
                )
            )

            # 查询失败的文档
            query = select(Document).where(and_(*conditions)).limit(limit)
            result = await db.execute(query)
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

            logger.info(f"找到 {len(failed_documents)} 个可重试的失败文档")

            # 批量重试
            results = []
            successful_count = 0
            failed_count = 0

            for document in failed_documents:
                # 如果指定了错误类型，检查是否匹配
                if error_types:
                    doc_error_type, _ = await self.classify_error(document.error_message or "")
                    if doc_error_type not in error_types:
                        continue

                retry_result = await self.retry_document_processing(document.id, db)
                results.append({
                    "document_id": document.id,
                    "filename": document.filename,
                    "result": retry_result
                })

                if retry_result["success"]:
                    successful_count += 1
                else:
                    failed_count += 1

                # 避免同时提交太多任务
                await asyncio.sleep(1)

            return {
                "success": True,
                "total": len(failed_documents),
                "retried": len(results),
                "successful": successful_count,
                "failed": failed_count,
                "results": results
            }

        except Exception as e:
            logger.error(f"批量重试失败文档出错: {e}")
            return {
                "success": False,
                "error": f"批量重试失败: {str(e)}"
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
            失败文档汇总
        """
        try:
            # 查询失败文档
            result = await db.execute(
                select(Document)
                .where(Document.status == DocumentStatus.FAILED)
                .order_by(Document.updated_at.desc())
                .limit(limit)
            )
            failed_documents = result.scalars().all()

            # 统计错误类型
            error_type_counts = {}
            retry_count_distribution = {i: 0 for i in range(self.max_retries + 1)}

            for doc in failed_documents:
                # 分类错误
                error_type, _ = await self.classify_error(doc.error_message or "")
                error_type_counts[error_type.value] = error_type_counts.get(error_type.value, 0) + 1

                # 统计重试次数
                retry_count = doc.retry_count or 0
                retry_count_distribution[retry_count] = retry_count_distribution.get(retry_count, 0) + 1

            # 计算可重试的文档数
            retryable_count = 0
            for doc in failed_documents:
                if await self.can_retry(doc):
                    retryable_count += 1

            return {
                "total_failed": len(failed_documents),
                "retryable": retryable_count,
                "error_type_distribution": error_type_counts,
                "retry_count_distribution": retry_count_distribution,
                "max_retries": self.max_retries,
                "sample_errors": [
                    {
                        "document_id": doc.id,
                        "filename": doc.filename,
                        "error_message": doc.error_message,
                        "retry_count": doc.retry_count or 0,
                        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
                    }
                    for doc in failed_documents[:10]
                ]
            }

        except Exception as e:
            logger.error(f"获取失败文档汇总失败: {e}")
            return {
                "error": f"获取汇总信息失败: {str(e)}"
            }

    async def cleanup_old_retries(self, db: AsyncSession, days: int = 7) -> int:
        """
        清理旧的重试记录

        Args:
            db: 数据库会话
            days: 保留天数

        Returns:
            清理的文档数量
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # 这里可以添加清理逻辑，比如重置很久以前的重试计数
            # 目前只是记录日志
            logger.info(f"清理 {cutoff_date} 之前 {days} 天的重试记录")

            return 0

        except Exception as e:
            logger.error(f"清理旧重试记录失败: {e}")
            return 0

# 创建全局重试服务实例
document_retry_service = DocumentRetryService()