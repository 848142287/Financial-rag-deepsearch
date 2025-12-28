"""
Celery配置
"""

import os
from celery import Celery
from app.core.config import settings

# 使用环境变量构建Redis URL（与async_tasks/celery_app.py保持一致）
redis_host = os.getenv('REDIS_HOST', 'redis')
redis_port = os.getenv('REDIS_PORT', '6379')
redis_password = os.getenv('REDIS_PASSWORD', 'redis123456')
redis_db = os.getenv('REDIS_DB', '0')

# 构建Redis URL
if redis_password:
    broker_url = f'redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}'
    backend_url = f'redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}'
else:
    broker_url = f'redis://{redis_host}:{redis_port}/{redis_db}'
    backend_url = f'redis://{redis_host}:{redis_port}/{redis_db}'

# 创建Celery实例
celery_app = Celery(
    "financial_rag",
    broker=broker_url,
    backend=backend_url,
    include=[
        "app.tasks.complete_pipeline_task",
        "app.tasks.unified_task_manager"  # 添加统一任务管理器
    ]
)

# 设置默认队列
celery_app.conf.task_default_queue = 'default'

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
    # "cleanup-temp-files": {
    #     "task": "app.services.tasks.cleanup_temp_files",
    #     "schedule": 3600.0,  # 每小时执行一次
    # },
    # "health-check": {
    #     "task": "app.services.tasks.health_check",
    #     "schedule": 300.0,  # 每5分钟执行一次
    # },
}

if __name__ == "__main__":
    celery_app.start()