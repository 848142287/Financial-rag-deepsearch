"""
异步任务处理层
基于Celery和Redis实现后台任务处理
"""

from .celery_app import celery_app
from .tasks import (
    DocumentProcessingTask,
    SearchTask,
    DeepSearchTask,
    EvaluationTask,
    CacheWarmingTask,
    SystemMaintenanceTask
)

__all__ = [
    'celery_app',
    'DocumentProcessingTask',
    'SearchTask',
    'DeepSearchTask',
    'EvaluationTask',
    'CacheWarmingTask',
    'SystemMaintenanceTask'
]