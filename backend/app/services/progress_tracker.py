"""
实时进度跟踪服务
用于长时间运行的查询任务的进度管理
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from app.core.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"          # 等待中
    RUNNING = "running"          # 运行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"     # 已取消


class TaskStep(Enum):
    """任务步骤"""
    INITIALIZING = "initializing"
    QUERY_UNDERSTANDING = "query_understanding"
    RETRIEVAL = "retrieval"
    ENHANCED_RETRIEVAL = "enhanced_retrieval"
    DEEP_SEARCH = "deep_search"
    EVIDENCE_COLLECTION = "evidence_collection"
    SYNTHESIS = "synthesis"
    FINALIZING = "finalizing"


@dataclass
class ProgressUpdate:
    """进度更新"""
    task_id: str
    status: TaskStatus
    current_step: TaskStep
    progress_percentage: float
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    estimated_remaining_seconds: Optional[int] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        data['current_step'] = self.current_step.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    query: str
    retrieval_mode: str
    created_at: datetime
    updated_at: datetime
    status: TaskStatus
    current_step: TaskStep
    progress_percentage: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        data['current_step'] = self.current_step.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self):
        self.redis = get_redis_client()
        self.task_ttl = 3600  # 1小时

    async def create_task(
        self,
        query: str,
        retrieval_mode: str,
        estimated_steps: List[TaskStep] = None
    ) -> str:
        """
        创建新任务

        Args:
            query: 查询内容
            retrieval_mode: 检索模式
            estimated_steps: 预估步骤列表

        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())

        now = datetime.utcnow()
        task_info = TaskInfo(
            task_id=task_id,
            query=query,
            retrieval_mode=retrieval_mode,
            created_at=now,
            updated_at=now,
            status=TaskStatus.PENDING,
            current_step=TaskStep.INITIALIZING
        )

        # 存储任务信息
        await self._store_task_info(task_info)

        # 存储预估步骤（用于计算进度）
        if estimated_steps:
            steps_data = [step.value for step in estimated_steps]
            await self.redis.set(
                f"task:steps:{task_id}",
                json.dumps(steps_data),
                ex=self.task_ttl
            )

        logger.info(f"Created task {task_id} for query: {query[:50]}...")
        return task_id

    async def update_progress(
        self,
        task_id: str,
        status: TaskStatus,
        current_step: TaskStep,
        progress_percentage: float,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        estimated_remaining: Optional[int] = None
    ):
        """
        更新任务进度

        Args:
            task_id: 任务ID
            status: 任务状态
            current_step: 当前步骤
            progress_percentage: 进度百分比(0-100)
            message: 进度消息
            details: 详细信息
            estimated_remaining: 预估剩余时间（秒）
        """
        try:
            # 更新任务信息
            task_info = await self._get_task_info(task_id)
            if task_info:
                task_info.status = status
                task_info.current_step = current_step
                task_info.progress_percentage = progress_percentage
                task_info.message = message
                task_info.updated_at = datetime.utcnow()
                await self._store_task_info(task_info)

            # 创建进度更新
            progress_update = ProgressUpdate(
                task_id=task_id,
                status=status,
                current_step=current_step,
                progress_percentage=progress_percentage,
                message=message,
                details=details,
                estimated_remaining_seconds=estimated_remaining
            )

            # 存储进度更新
            await self._store_progress_update(progress_update)

            # 如果任务完成或失败，清理资源
            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                await self._cleanup_task(task_id)

            logger.debug(f"Updated progress for task {task_id}: {progress_percentage}% - {message}")

        except Exception as e:
            logger.error(f"Error updating progress for task {task_id}: {str(e)}")

    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        """完成任务"""
        await self.update_progress(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            current_step=TaskStep.FINALIZING,
            progress_percentage=100.0,
            message="任务完成",
            details={"result": result}
        )

        # 更新任务结果
        task_info = await self._get_task_info(task_id)
        if task_info:
            task_info.result = result
            await self._store_task_info(task_info)

    async def fail_task(self, task_id: str, error: str):
        """任务失败"""
        await self.update_progress(
            task_id=task_id,
            status=TaskStatus.FAILED,
            current_step=TaskStep.FINALIZING,
            progress_percentage=100.0,
            message=f"任务失败: {error}",
            details={"error": error}
        )

        # 更新任务错误信息
        task_info = await self._get_task_info(task_id)
        if task_info:
            task_info.error = error
            await self._store_task_info(task_info)

    async def cancel_task(self, task_id: str):
        """取消任务"""
        await self.update_progress(
            task_id=task_id,
            status=TaskStatus.CANCELLED,
            current_step=TaskStep.FINALIZING,
            progress_percentage=100.0,
            message="任务已取消"
        )

    async def get_progress(self, task_id: str) -> Optional[ProgressUpdate]:
        """获取最新进度"""
        try:
            key = f"task:progress:latest:{task_id}"
            data = await self.redis.get(key)
            if data:
                progress_dict = json.loads(data)
                progress_dict['status'] = TaskStatus(progress_dict['status'])
                progress_dict['current_step'] = TaskStep(progress_dict['current_step'])
                progress_dict['timestamp'] = datetime.fromisoformat(progress_dict['timestamp'])
                return ProgressUpdate(**progress_dict)
        except Exception as e:
            logger.error(f"Error getting progress for task {task_id}: {str(e)}")
        return None

    async def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        return await self._get_task_info(task_id)

    async def get_progress_history(
        self,
        task_id: str,
        limit: int = 100
    ) -> List[ProgressUpdate]:
        """获取进度历史"""
        try:
            key = f"task:progress:history:{task_id}"
            data_list = await self.redis.lrange(key, 0, limit - 1)

            history = []
            for data in data_list:
                progress_dict = json.loads(data)
                progress_dict['status'] = TaskStatus(progress_dict['status'])
                progress_dict['current_step'] = TaskStep(progress_dict['current_step'])
                progress_dict['timestamp'] = datetime.fromisoformat(progress_dict['timestamp'])
                history.append(ProgressUpdate(**progress_dict))

            return history

        except Exception as e:
            logger.error(f"Error getting progress history for task {task_id}: {str(e)}")
            return []

    async def get_active_tasks(self) -> List[TaskInfo]:
        """获取所有活跃任务"""
        try:
            # 获取所有任务键
            pattern = "task:info:*"
            keys = await self.redis.keys(pattern)

            active_tasks = []
            for key in keys:
                data = await self.redis.get(key)
                if data:
                    task_dict = json.loads(data)
                    status = TaskStatus(task_dict['status'])
                    if status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                        task_dict['status'] = status
                        task_dict['current_step'] = TaskStep(task_dict['current_step'])
                        task_dict['created_at'] = datetime.fromisoformat(task_dict['created_at'])
                        task_dict['updated_at'] = datetime.fromisoformat(task_dict['updated_at'])
                        active_tasks.append(TaskInfo(**task_dict))

            return active_tasks

        except Exception as e:
            logger.error(f"Error getting active tasks: {str(e)}")
            return []

    async def cleanup_expired_tasks(self):
        """清理过期任务"""
        try:
            pattern = "task:info:*"
            keys = await self.redis.keys(pattern)

            now = datetime.utcnow()
            for key in keys:
                data = await self.redis.get(key)
                if data:
                    task_dict = json.loads(data)
                    updated_at = datetime.fromisoformat(task_dict['updated_at'])

                    # 如果任务超过1小时未更新，清理
                    if (now - updated_at).total_seconds() > self.task_ttl:
                        task_id = key.split(':')[-1]
                        await self._cleanup_task(task_id)

        except Exception as e:
            logger.error(f"Error cleaning up expired tasks: {str(e)}")

    async def _store_task_info(self, task_info: TaskInfo):
        """存储任务信息"""
        key = f"task:info:{task_info.task_id}"
        await self.redis.set(
            key,
            json.dumps(task_info.to_dict()),
            ex=self.task_ttl
        )

    async def _store_progress_update(self, progress: ProgressUpdate):
        """存储进度更新"""
        # 存储最新进度
        latest_key = f"task:progress:latest:{progress.task_id}"
        await self.redis.set(
            latest_key,
            json.dumps(progress.to_dict()),
            ex=self.task_ttl
        )

        # 存储历史记录
        history_key = f"task:progress:history:{progress.task_id}"
        await self.redis.lpush(
            history_key,
            json.dumps(progress.to_dict())
        )
        await self.redis.expire(history_key, self.task_ttl)

    async def _get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        try:
            key = f"task:info:{task_id}"
            data = await self.redis.get(key)
            if data:
                task_dict = json.loads(data)
                task_dict['status'] = TaskStatus(task_dict['status'])
                task_dict['current_step'] = TaskStep(task_dict['current_step'])
                task_dict['created_at'] = datetime.fromisoformat(task_dict['created_at'])
                task_dict['updated_at'] = datetime.fromisoformat(task_dict['updated_at'])
                return TaskInfo(**task_dict)
        except Exception as e:
            logger.error(f"Error getting task info for {task_id}: {str(e)}")
        return None

    async def _cleanup_task(self, task_id: str):
        """清理任务相关数据"""
        try:
            keys_to_delete = [
                f"task:info:{task_id}",
                f"task:progress:latest:{task_id}",
                f"task:progress:history:{task_id}",
                f"task:steps:{task_id}"
            ]

            for key in keys_to_delete:
                await self.redis.delete(key)

            logger.info(f"Cleaned up task {task_id}")

        except Exception as e:
            logger.error(f"Error cleaning up task {task_id}: {str(e)}")


# 全局实例
progress_tracker = ProgressTracker()


# 装饰器：自动跟踪任务进度
def track_progress(
    query: str,
    retrieval_mode: str,
    estimated_steps: List[TaskStep] = None
):
    """任务进度跟踪装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 创建任务
            task_id = await progress_tracker.create_task(
                query=query,
                retrieval_mode=retrieval_mode,
                estimated_steps=estimated_steps
            )

            # 将task_id添加到kwargs
            kwargs['task_id'] = task_id

            try:
                # 执行原函数
                result = await func(*args, **kwargs)

                # 完成任务
                await progress_tracker.complete_task(task_id, result)

                return result

            except Exception as e:
                # 任务失败
                await progress_tracker.fail_task(task_id, str(e))
                raise

        return wrapper
    return decorator