"""
优化文档上传服务
处理MinIO事件，触发文档解析流水线
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import aiofiles
import redis.asyncio as redis
from fastapi import HTTPException, status
from minio import Minio
from minio.error import S3Error
# # import magic  # 禁用magic库，使用更简单的文件类型检测

from app.core.config import settings
from app.core.database import get_db
from app.models.document import Document

# 临时的 DocumentTask 类，后续需要创建正式的模型
class DocumentTask:
    def __init__(self, **kwargs):
        self.task_id = kwargs.get('task_id')
        self.document_id = kwargs.get('document_id')
        self.status = kwargs.get('status', 'pending')
        self.progress = kwargs.get('progress', 0.0)
        self.error_message = kwargs.get('error_message')
        self.created_at = kwargs.get('created_at')
        self.updated_at = kwargs.get('updated_at')

logger = logging.getLogger(__name__)

def get_mime_type(file_path: str) -> str:
    """
    简单的MIME类型检测，替代magic库
    """
    import os
    import mimetypes

    # 初始化mimetypes
    mimetypes.init()

    # 基于文件扩展名的MIME类型检测
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type:
        return mime_type

    # 基于文件扩展名的简单检测
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.pdf': 'application/pdf',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xls': 'application/vnd.ms-excel',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.ppt': 'application/vnd.ms-powerpoint',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.txt': 'text/plain',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.xml': 'application/xml',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.zip': 'application/zip',
        '.rar': 'application/x-rar-compressed',
        '.7z': 'application/x-7z-compressed'
    }

    return mime_types.get(ext, 'application/octet-stream')


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class UploadEvent:
    """上传事件数据结构"""
    bucket_name: str
    object_name: str
    event_type: str
    event_time: datetime
    file_size: int
    content_type: str
    etag: str

    @classmethod
    def from_minio_event(cls, event_data: dict) -> "UploadEvent":
        """从MinIO事件数据创建UploadEvent"""
        records = event_data.get("Records", [])
        if not records:
            raise ValueError("Invalid MinIO event format")

        record = records[0]
        s3 = record.get("s3", {})
        bucket = s3.get("bucket", {})
        object_info = s3.get("object", {})

        return cls(
            bucket_name=bucket.get("name"),
            object_name=object_info.get("key"),
            event_type=record.get("eventName"),
            event_time=datetime.fromisoformat(
                record.get("eventTime").replace("Z", "+00:00")
            ),
            file_size=object_info.get("size", 0),
            content_type=object_info.get("contentType", ""),
            etag=object_info.get("eTag", "").strip('"')
        )


class DocumentValidator:
    """文档验证器"""

    # 支持的文件类型
    SUPPORTED_TYPES = {
        "application/pdf": [".pdf"],
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
        "application/msword": [".doc"],  # Legacy Word format
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
        "application/vnd.ms-excel": [".xls"],  # Legacy Excel format
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
        "application/vnd.ms-powerpoint": [".ppt"],  # Legacy PowerPoint format
        "text/plain": [".txt"],
        "text/markdown": [".md"],
        "image/jpeg": [".jpg", ".jpeg"],
        "image/png": [".png"],
        "image/tiff": [".tiff", ".tif"],
        "image/bmp": [".bmp"]
    }

    # 最大文件大小 (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024

    @classmethod
    async def validate_file(cls, file_path: str) -> Tuple[bool, str]:
        """验证文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return False, "File not found"

            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size > cls.MAX_FILE_SIZE:
                return False, f"File too large: {file_size} bytes (max: {cls.MAX_FILE_SIZE})"

            if file_size == 0:
                return False, "Empty file"

            # 检查文件类型
            mime_type = get_mime_type(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()

            supported = False
            for supported_mime, extensions in cls.SUPPORTED_TYPES.items():
                if mime_type == supported_mime or file_ext in extensions:
                    supported = True
                    break

            if not supported:
                return False, f"Unsupported file type: {mime_type} ({file_ext})"

            # 病毒扫描 (如果配置了ClamAV)
            if settings.VIRUS_SCAN_ENABLED:
                scan_result = await cls._virus_scan(file_path)
                if not scan_result:
                    return False, "Virus detected"

            return True, "Valid file"

        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False, f"Validation error: {str(e)}"

    @staticmethod
    async def _virus_scan(file_path: str) -> bool:
        """病毒扫描"""
        try:
            import pyclamd
            cd = pyclamd.ClamdUnixSocket()
            scan_result = cd.scan_file(file_path)
            return scan_result is None
        except ImportError:
            logger.warning("ClamAV not installed, skipping virus scan")
            return True
        except Exception as e:
            logger.error(f"Virus scan error: {e}")
            return True  # 失败时允许通过，避免阻塞


class TaskQueueManager:
    """任务队列管理器"""

    def __init__(self):
        self.redis_client = None

    async def initialize(self):
        """初始化Redis连接"""
        self.redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True
        )

    async def submit_task(
        self,
        task_data: dict,
        priority: TaskPriority = TaskPriority.NORMAL,
        delay: int = 0
    ) -> str:
        """提交任务到队列"""
        task_id = str(uuid.uuid4())

        # 创建任务对象
        task = {
            "id": task_id,
            "data": task_data,
            "priority": priority,
            "status": TaskStatus.PENDING,
            "created_at": datetime.utcnow().isoformat(),
            "retry_count": 0,
            "max_retries": 5,
            "delay": delay
        }

        # 根据优先级选择队列
        queue_name = f"document_parse:{priority.name.lower()}"

        # 存储任务详情
        await self.redis_client.hset(
            f"task:{task_id}",
            mapping=task
        )

        # 添加到队列
        if delay > 0:
            # 延迟队列
            await self.redis_client.zadd(
                "delayed_tasks",
                {task_id: datetime.utcnow().timestamp() + delay}
            )
        else:
            # 立即队列
            await self.redis_client.lpush(queue_name, task_id)

        logger.info(f"Task {task_id} submitted to queue {queue_name}")
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[dict]:
        """获取任务状态"""
        task_data = await self.redis_client.hgetall(f"task:{task_id}")
        if not task_data:
            return None

        # 转换类型
        if "priority" in task_data:
            task_data["priority"] = TaskPriority(int(task_data["priority"]))
        if "status" in task_data:
            task_data["status"] = TaskStatus(task_data["status"])

        return task_data

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        error_message: str = None
    ):
        """更新任务状态"""
        update_data = {
            "status": status.value,
            "updated_at": datetime.utcnow().isoformat()
        }

        if error_message:
            update_data["error_message"] = error_message

        await self.redis_client.hset(
            f"task:{task_id}",
            mapping=update_data
        )

        # 如果任务完成或失败，设置过期时间
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            await self.redis_client.expire(f"task:{task_id}", 86400)  # 24小时后过期


class UploadService:
    """文档上传服务"""

    def __init__(self):
        self.minio_client = None
        self.task_manager = TaskQueueManager()
        self.validator = DocumentValidator()

    async def initialize(self):
        """初始化服务"""
        # 初始化MinIO客户端
        self.minio_client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )

        # 初始化任务队列管理器
        await self.task_manager.initialize()

        logger.info("Upload service initialized")

    async def handle_minio_event(self, event_data: dict) -> dict:
        """处理MinIO上传事件"""
        try:
            # 解析事件
            event = UploadEvent.from_minio_event(event_data)

            # 只处理新文件上传事件
            if event.event_type != "s3:ObjectCreated:Put":
                return {"status": "ignored", "reason": "Not a create event"}

            logger.info(f"Processing upload event: {event.object_name}")

            # 下载文件进行验证
            file_path = await self._download_file(event)

            # 验证文件
            is_valid, message = await self.validator.validate_file(file_path)
            if not is_valid:
                os.remove(file_path)  # 删除临时文件
                raise ValueError(f"File validation failed: {message}")

            # 计算文件哈希
            file_hash = await self._calculate_file_hash(file_path)

            # 检查是否重复
            existing_doc = await self._check_duplicate(file_hash)
            if existing_doc:
                os.remove(file_path)
                logger.info(f"Duplicate file detected: {existing_doc.id}")
                return {
                    "status": "duplicate",
                    "document_id": existing_doc.id,
                    "message": "File already processed"
                }

            # 创建文档记录
            document = await self._create_document_record(event, file_hash, file_path)

            # 提交解析任务
            task_id = await self._submit_parse_task(document, event)

            # 清理临时文件
            os.remove(file_path)

            return {
                "status": "accepted",
                "document_id": document.id,
                "task_id": task_id,
                "message": "Document upload accepted for processing"
            }

        except Exception as e:
            logger.error(f"Error processing upload event: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to process upload"
            }

    async def _download_file(self, event: UploadEvent) -> str:
        """下载文件到临时目录"""
        # 创建临时目录
        temp_dir = "/tmp/document_uploads"
        os.makedirs(temp_dir, exist_ok=True)

        # 生成临时文件路径
        file_name = os.path.basename(event.object_name)
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file_name}")

        # 下载文件
        try:
            self.minio_client.fget_object(
                event.bucket_name,
                event.object_name,
                temp_file_path
            )
            logger.info(f"File downloaded to: {temp_file_path}")
            return temp_file_path
        except S3Error as e:
            logger.error(f"MinIO download error: {e}")
            raise

    async def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        sha256_hash = hashlib.sha256()

        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    async def _check_duplicate(self, file_hash: str) -> Optional[Document]:
        """检查文件是否重复"""
        async with get_db() as db:
            from sqlalchemy import select
            result = await db.execute(
                select(Document).where(Document.content_hash == file_hash)
            )
            return result.scalar_one_or_none()

    async def _create_document_record(
        self,
        event: UploadEvent,
        file_hash: str,
        file_path: str
    ) -> Document:
        """创建文档记录"""
        async with get_db() as db:
            # 获取文件元数据
            file_ext = os.path.splitext(event.object_name)[1].lower()
            mime_type = get_mime_type(file_path)

            # 创建文档记录
            document = Document(
                title=os.path.basename(event.object_name),
                filename=event.object_name,
                file_path=f"{event.bucket_name}/{event.object_name}",
                file_size=event.file_size,
                file_type=file_ext,
                mime_type=mime_type,
                content_hash=file_hash,
                status="uploaded",
                doc_metadata={
                    "upload_time": event.event_time.isoformat(),
                    "etag": event.etag,
                    "content_type": event.content_type,
                    "bucket": event.bucket_name
                }
            )

            db.add(document)
            await db.commit()
            await db.refresh(document)

            logger.info(f"Document record created: {document.id}")
            return document

    async def _submit_parse_task(
        self,
        document: Document,
        event: UploadEvent
    ) -> str:
        """提交解析任务"""
        task_data = {
            "document_id": document.id,
            "file_path": document.file_path,
            "file_type": document.file_type,
            "mime_type": document.mime_type,
            "file_size": document.file_size,
            "bucket": event.bucket_name,
            "object_name": event.object_name
        }

        # 根据文件大小设置优先级
        if document.file_size > 50 * 1024 * 1024:  # > 50MB
            priority = TaskPriority.LOW
        elif document.file_size > 10 * 1024 * 1024:  # > 10MB
            priority = TaskPriority.NORMAL
        else:
            priority = TaskPriority.HIGH

        task_id = await self.task_manager.submit_task(
            task_data=task_data,
            priority=priority
        )

        # 创建任务记录
        async with get_db() as db:
            doc_task = DocumentTask(
                document_id=document.id,
                task_id=task_id,
                task_type="document_parse",
                status=TaskStatus.PENDING,
                priority=priority,
                task_data=task_data
            )

            db.add(doc_task)
            await db.commit()

        logger.info(f"Parse task submitted: {task_id} for document {document.id}")
        return task_id

    async def save_file(self, upload_file, temp_file_path: str) -> str:
        """保存上传的文件到MinIO存储"""
        try:
            # 生成唯一文件名
            file_name = upload_file.filename
            if not file_name:
                file_name = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 确保文件名安全
            file_name = "".join(c for c in file_name if c.isalnum() or c in (' ', '.', '_','-')).rstrip()

            # 生成对象名称（包含日期路径，不重复bucket名称）
            date_path = datetime.now().strftime("%Y/%m/%d")
            object_name = f"{date_path}/{file_name}"  # 移除 "documents/" 前缀，因为bucket已经是documents

            # 确保bucket存在
            bucket_name = "documents"
            try:
                if not self.minio_client.bucket_exists(bucket_name):
                    self.minio_client.make_bucket(bucket_name)
            except Exception as e:
                logger.warning(f"Bucket check failed: {e}")

            # 上传文件到MinIO
            self.minio_client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=temp_file_path,
                content_type=upload_file.content_type or "application/octet-stream"
            )

            logger.info(f"File saved to MinIO: {bucket_name}/{object_name}")
            return f"{bucket_name}/{object_name}"

        except Exception as e:
            logger.error(f"Failed to save file to MinIO: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"文件保存失败: {str(e)}"
            )

    async def get_upload_status(self, task_id: str) -> dict:
        """获取上传/处理状态"""
        # 从Redis获取任务状态
        task_status = await self.task_manager.get_task_status(task_id)
        if not task_status:
            return {"status": "not_found", "message": "Task not found"}

        # 从数据库获取文档信息
        async with get_db() as db:
            from sqlalchemy import select
            result = await db.execute(
                select(DocumentTask).where(DocumentTask.task_id == task_id)
            )
            doc_task = result.scalar_one_or_none()

            if doc_task:
                return {
                    "task_id": task_id,
                    "document_id": doc_task.document_id,
                    "status": task_status["status"].value,
                    "created_at": task_status["created_at"],
                    "updated_at": task_status.get("updated_at"),
                    "error_message": task_status.get("error_message"),
                    "progress": self._calculate_progress(doc_task)
                }

            return task_status

    def _calculate_progress(self, doc_task: DocumentTask) -> dict:
        """计算处理进度"""
        # 这里可以根据任务的具体阶段计算进度
        # 简化实现，返回预估进度
        base_progress = {
            "uploaded": 10,
            "parsing": 30,
            "analyzing": 60,
            "vectorizing": 80,
            "completed": 100
        }

        status = doc_task.status
        return {
            "percentage": base_progress.get(status, 0),
            "current_stage": status,
            "estimated_remaining": self._estimate_time_remaining(status)
        }

    def _estimate_time_remaining(self, status: str) -> str:
        """估算剩余时间"""
        time_estimates = {
            "pending": "< 1 minute",
            "processing": "2-10 minutes",
            "analyzing": "5-15 minutes",
            "vectorizing": "1-5 minutes",
            "completed": "0 minutes"
        }
        return time_estimates.get(status, "Unknown")

    async def get_file_content(self, file_path: str) -> bytes:
        """从MinIO获取文件内容"""
        try:
            # 解析文件路径，格式通常是 "bucket/object_name"
            if '/' not in file_path:
                raise ValueError(f"Invalid file path format: {file_path}")

            bucket_name, object_name = file_path.split('/', 1)

            # 从MinIO获取文件
            response = self.minio_client.get_object(bucket_name, object_name)
            file_content = response.read()
            response.close()
            response.release_conn()

            logger.info(f"File content retrieved from MinIO: {bucket_name}/{object_name}")
            return file_content

        except Exception as e:
            logger.error(f"Failed to get file content from MinIO: {e}")
            raise


# 全局实例
upload_service = UploadService()


async def get_upload_service() -> UploadService:
    """获取上传服务实例"""
    if not upload_service.minio_client:
        await upload_service.initialize()
    return upload_service