"""
知识图谱文档任务 - 占位符文件
用于避免Celery自动发现任务时的导入错误
"""

from app.tasks.unified_task_manager import celery_app

# 占位符任务，避免导入错误
@celery_app.task(name='app.tasks.kg_document_task.placeholder')
def placeholder_task():
    """占位符任务"""
    pass
