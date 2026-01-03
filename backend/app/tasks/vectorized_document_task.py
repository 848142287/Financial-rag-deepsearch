"""
向量化文档任务 - 占位符文件
用于避免Celery自动发现任务时的导入错误
"""

from .vector_tasks import *

__all__ = ['placeholder_task']
