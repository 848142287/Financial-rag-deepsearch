"""
Celery配置
"""

from celery import Celery
from app.core.config import settings

# 创建Celery实例
celery_app = Celery(
    "financial_rag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.document_processing", "app.tasks.complete_pipeline_task"]
)

# Celery配置
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_send_sent_event=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_compression="gzip",
    result_compression="gzip",
    result_expires=3600,
    result_backend_transport_options={
        "master_name": "financial_rag",
    },
)

# 定时任务配置
celery_app.conf.beat_schedule = {
    "cleanup-temp-files": {
        "task": "app.services.tasks.cleanup_temp_files",
        "schedule": 3600.0,  # 每小时执行一次
    },
    "health-check": {
        "task": "app.services.tasks.health_check",
        "schedule": 300.0,  # 每5分钟执行一次
    },
}

if __name__ == "__main__":
    celery_app.start()