"""
任务进度跟踪API端点
"""

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import asyncio
import logging

from app.core.database import get_db
from app.schemas.admin import TaskProgress, TaskStatus, ProgressUpdate
from app.models.admin import TaskQueue
from app.services.progress_tracker import ProgressTracker
from app.services.websocket_service import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter()

# 进度跟踪服务实例
progress_service = ProgressTracker()
# WebSocket管理器
websocket_manager = ConnectionManager()


@router.get("/tasks", response_model=List[TaskProgress])
async def get_active_tasks(
    task_type: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取活跃任务列表"""
    try:
        tasks = progress_service.get_active_tasks(
            db,
            task_type=task_type,
            status=status,
            skip=skip,
            limit=limit
        )
        return tasks
    except Exception as e:
        logger.error(f"Failed to get active tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active tasks"
        )


@router.get("/tasks/{task_id}", response_model=TaskProgress)
async def get_task_progress(
    task_id: str,
    db: Session = Depends(get_db)
):
    """获取特定任务的进度"""
    try:
        task = progress_service.get_task_progress(db, task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task progress for {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve task progress"
        )


@router.put("/tasks/{task_id}/progress", response_model=TaskProgress)
async def update_task_progress(
    task_id: str,
    progress_update: ProgressUpdate,
    db: Session = Depends(get_db)
):
    """更新任务进度"""
    try:
        task = progress_service.update_progress(
            db,
            task_id=task_id,
            progress=progress_update.progress,
            message=progress_update.message,
            metadata=progress_update.metadata
        )

        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )

        # 通过WebSocket广播更新
        await websocket_manager.broadcast_task_update(task_id, {
            "type": "progress_update",
            "task_id": task_id,
            "progress": progress_update.progress,
            "message": progress_update.message,
            "timestamp": datetime.utcnow().isoformat()
        })

        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update task progress for {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update task progress"
        )


@router.put("/tasks/{task_id}/status", response_model=TaskProgress)
async def update_task_status(
    task_id: str,
    task_status: TaskStatus,
    db: Session = Depends(get_db)
):
    """更新任务状态"""
    try:
        task = progress_service.update_status(
            db,
            task_id=task_id,
            status=task_status.status,
            error_message=task_status.error_message,
            result=task_status.result
        )

        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )

        # 通过WebSocket广播状态更新
        await websocket_manager.broadcast_task_update(task_id, {
            "type": "status_update",
            "task_id": task_id,
            "status": task_status.status,
            "error_message": task_status.error_message,
            "timestamp": datetime.utcnow().isoformat()
        })

        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update task status for {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update task status"
        )


@router.get("/tasks/{task_id}/logs")
async def get_task_logs(
    task_id: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取任务日志"""
    try:
        logs = progress_service.get_task_logs(
            db,
            task_id=task_id,
            skip=skip,
            limit=limit
        )
        return {
            "task_id": task_id,
            "logs": logs,
            "total": len(logs)
        }
    except Exception as e:
        logger.error(f"Failed to get task logs for {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve task logs"
        )


@router.get("/statistics")
async def get_progress_statistics(
    time_range: Optional[str] = "24h",
    task_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """获取任务进度统计"""
    try:
        stats = progress_service.get_statistics(
            db,
            time_range=time_range,
            task_type=task_type
        )
        return stats
    except Exception as e:
        logger.error(f"Failed to get progress statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve progress statistics"
        )


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    db: Session = Depends(get_db)
):
    """取消任务"""
    try:
        success = progress_service.cancel_task(db, task_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found or cannot be cancelled"
            )

        # 通过WebSocket广播取消通知
        await websocket_manager.broadcast_task_update(task_id, {
            "type": "task_cancelled",
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat()
        })

        return {"message": "Task cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel task"
        )


@router.get("/queue/status")
async def get_queue_status(db: Session = Depends(get_db)):
    """获取任务队列状态"""
    try:
        queue_status = progress_service.get_queue_status(db)
        return queue_status
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve queue status"
        )


@router.websocket("/ws/tasks/{task_id}")
async def websocket_task_progress(websocket: WebSocket, task_id: str):
    """WebSocket端点用于实时任务进度更新"""
    await websocket_manager.connect(websocket, task_id)

    try:
        # 发送当前任务状态
        # 这里需要从数据库获取当前任务状态
        # 简化实现
        await websocket.send_text(json.dumps({
            "type": "connected",
            "task_id": task_id,
            "message": "Connected to task progress updates"
        }))

        while True:
            # 等待客户端消息（心跳等）
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                data = json.loads(message)

                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))

            except asyncio.TimeoutError:
                # 发送心跳
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                }))

    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket, task_id)
        logger.info(f"WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
        await websocket_manager.disconnect(websocket, task_id)


@router.post("/cleanup")
async def cleanup_completed_tasks(
    older_than_hours: int = 24,
    db: Session = Depends(get_db)
):
    """清理已完成的旧任务"""
    try:
        result = progress_service.cleanup_completed_tasks(
            db,
            older_than_hours=older_than_hours
        )
        return {
            "message": "Task cleanup completed",
            "deleted_tasks": result.get("deleted_tasks", 0),
            "deleted_logs": result.get("deleted_logs", 0)
        }
    except Exception as e:
        logger.error(f"Failed to cleanup tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup tasks"
        )