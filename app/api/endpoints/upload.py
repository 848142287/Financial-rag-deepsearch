"""
文档上传API端点
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services.upload_service import get_upload_service, UploadService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class WebhookEvent(BaseModel):
    """MinIO Webhook事件模型"""
    Records: list = []


class UploadResponse(BaseModel):
    """上传响应模型"""
    status: str
    document_id: str = None
    task_id: str = None
    message: str
    error: str = None


class TaskStatusResponse(BaseModel):
    """任务状态响应模型"""
    task_id: str
    document_id: str = None
    status: str
    created_at: str
    updated_at: str = None
    error_message: str = None
    progress: Dict[str, Any] = None


@router.post("/webhook/minio", response_model=UploadResponse)
async def minio_webhook(
    event: WebhookEvent,
    upload_service: UploadService = Depends(get_upload_service)
):
    """
    MinIO上传事件Webhook端点
    接收MinIO的事件通知，触发文档解析流程
    """
    try:
        logger.info(f"Received MinIO webhook event with {len(event.Records)} records")

        if not event.Records:
            raise HTTPException(
                status_code=400,
                detail="No records in webhook event"
            )

        # 处理事件
        result = await upload_service.handle_minio_event(event.dict())

        # 验证事件来源 (可选)
        # if settings.MINIO_WEBHOOK_SECRET:
        #     # 验证签名
        #     pass

        return UploadResponse(**result)

    except Exception as e:
        logger.error(f"Error processing MinIO webhook: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process upload event: {str(e)}"
        )


@router.post("/direct", response_model=UploadResponse)
async def direct_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    upload_service: UploadService = Depends(get_upload_service)
):
    """
    直接文件上传端点
    允许客户端直接上传文件到系统
    """
    try:
        # 验证文件大小
        if file.size and file.size > 100 * 1024 * 1024:  # 100MB
            raise HTTPException(
                status_code=413,
                detail="File too large (max 100MB)"
            )

        # 验证文件类型
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "text/plain",
            "text/markdown",
            "image/jpeg",
            "image/png"
        ]

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )

        # 这里应该实现文件上传到MinIO的逻辑
        # 为了简化，我们返回一个模拟响应
        task_id = "direct_upload_" + str(hash(file.filename))

        # 在后台任务中处理文件
        background_tasks.add_task(
            process_direct_upload,
            file,
            upload_service
        )

        return UploadResponse(
            status="accepted",
            task_id=task_id,
            message="File uploaded successfully, processing started"
        )

    except Exception as e:
        logger.error(f"Error in direct upload: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


async def process_direct_upload(
    file: UploadFile,
    upload_service: UploadService
):
    """后台任务：处理直接上传的文件"""
    try:
        # 这里应该实现：
        # 1. 将文件保存到MinIO
        # 2. 创建相应的MinIO事件
        # 3. 调用handle_minio_event处理

        logger.info(f"Processing direct upload: {file.filename}")

        # 模拟处理
        import asyncio
        await asyncio.sleep(1)

        logger.info(f"Direct upload processed: {file.filename}")

    except Exception as e:
        logger.error(f"Error processing direct upload: {e}")


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    upload_service: UploadService = Depends(get_upload_service)
):
    """
    获取任务处理状态
    """
    try:
        status = await upload_service.get_upload_status(task_id)

        if not status or status.get("status") == "not_found":
            raise HTTPException(
                status_code=404,
                detail="Task not found"
            )

        return TaskStatusResponse(**status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "upload_service",
        "timestamp": "2024-01-01T00:00:00Z"
    }


@router.post("/test-webhook")
async def test_webhook():
    """
    测试Webhook端点
    用于测试MinIO事件处理
    """
    test_event = {
        "Records": [{
            "eventVersion": "2.1",
            "eventSource": "minio:s3",
            "awsRegion": "",
            "eventTime": "2024-01-01T12:00:00.000Z",
            "eventName": "s3:ObjectCreated:Put",
            "s3": {
                "bucket": {
                    "name": "raw-documents"
                },
                "object": {
                    "key": "test-document.pdf",
                    "size": 1024000,
                    "contentType": "application/pdf",
                    "eTag": "\"test-etag-123\""
                }
            }
        }]
    }

    try:
        upload_service = await get_upload_service()
        result = await upload_service.handle_minio_event(test_event)
        return result
    except Exception as e:
        logger.error(f"Error in test webhook: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


# 错误处理
@router.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc) if settings.DEBUG else None
        }
    )