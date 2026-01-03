"""
统一的任务相关数据模型

包括：TaskStatus（任务状态）、TaskResult（任务结果）
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict
from datetime import datetime
from enum import Enum


# TODO: TaskStatus → core.TaskStatus
class TaskStatus(Enum):
    """
    统一的任务状态枚举

    替代在6个文件中重复定义的TaskStatus类
    """
    PENDING = "pending"  # 等待执行
    RUNNING = "running"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消
    TIMEOUT = "timeout"  # 超时

    def is_terminal(self) -> bool:
        """是否为终止状态"""
        return self in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT
        }

    def is_active(self) -> bool:
        """是否为活跃状态"""
        return self in {
            TaskStatus.PENDING,
            TaskStatus.RUNNING
        }


@dataclass
class TaskResult:
    """
    统一的任务结果类

    封装任务执行的结果
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "error_code": self.error_code,
            "metadata": self.metadata
        }

    @classmethod
    def success_result(cls, data: Any = None, **metadata) -> 'TaskResult':
        """创建成功结果"""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def failure_result(cls, error: str, error_code: str = None, **metadata) -> 'TaskResult':
        """创建失败结果"""
        return cls(success=False, error=error, error_code=error_code, metadata=metadata)


@dataclass
class TaskProgress:
    """任务进度"""
    current: int = 0
    total: int = 0
    message: str = ""
    percent: float = 0.0

    def update(self, current: int, total: int, message: str = ""):
        """更新进度"""
        self.current = current
        self.total = total
        self.message = message
        self.percent = (current / total * 100) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "percent": f"{self.percent:.1f}%"
        }


@dataclass
class TaskInfo:
    """
    统一的任务信息类

    包含任务的所有相关信息
    """
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: TaskProgress = field(default_factory=TaskProgress)
    result: Optional[TaskResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # 任务类型和参数
    task_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def mark_started(self):
        """标记任务开始"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
        self.updated_at = datetime.now()

    def mark_completed(self, result: TaskResult):
        """标记任务完成"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        self.result = result

    def mark_failed(self, error: str):
        """标记任务失败"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        self.error_message = error

    def get_duration(self) -> Optional[float]:
        """获取任务执行时长（秒）"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress.to_dict(),
            "result": self.result.to_dict() if self.result else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "duration": self.get_duration()
        }


__all__ = [
    'TaskStatus',
    'TaskResult',
    'TaskProgress',
    'TaskInfo',
]
