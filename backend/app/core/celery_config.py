"""
Celery统一配置
整合所有Celery配置，提供单一的配置入口
"""

import os
from app.core.structured_logging import get_structured_logger
from celery import Celery
from celery.schedules import crontab
from kombu import Exchange, Queue
from app.core.config import settings

logger = get_structured_logger(__name__)

# Celery配置 - 使用环境变量构建Redis URL
redis_host = os.getenv('REDIS_HOST', 'redis')
redis_port = os.getenv('REDIS_PORT', '6379')
redis_password = os.getenv('REDIS_PASSWORD', 'redis123456')
redis_db = os.getenv('REDIS_DB', '0')

# 构建Redis URL，支持密码认证
if redis_password:
    celery_broker = f'redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}'
    celery_result_backend = f'redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}'
else:
    celery_broker = f'redis://{redis_host}:{redis_port}/{redis_db}'
    celery_result_backend = f'redis://{redis_host}:{redis_port}/{redis_db}'

# 创建Celery应用
celery_app = Celery(
    'financial_rag_system',
    broker=celery_broker,
    backend=celery_result_backend,
    include=[
        'app.core.async_tasks.tasks',
        'app.tasks.vectorized_document_task',
        'app.tasks.kg_document_task',
        'app.tasks.complete_pipeline_task',
        'app.tasks.unified_task_manager',
        'app.tasks.async_vector_kg_tasks'  # 新增异步向量化和知识图谱任务
    ]
)

# Celery基础配置
celery_app.conf.update(
    # 任务序列化
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,

    # 任务路由
    task_routes={
        'app.core.async_tasks.tasks.DocumentProcessingTask.*': {'queue': 'document_processing'},
        'app.core.async_tasks.tasks.SearchTask.*': {'queue': 'search_tasks'},
        'app.core.async_tasks.tasks.DeepSearchTask.*': {'queue': 'deep_search'},
        'app.core.async_tasks.tasks.EvaluationTask.*': {'queue': 'evaluation'},
        'app.core.async_tasks.tasks.CacheWarmingTask.*': {'queue': 'maintenance'},
        'app.core.async_tasks.tasks.SystemMaintenanceTask.*': {'queue': 'maintenance'},
    },

    # 任务优先级
    task_inherit_parent_priority=True,
    task_default_priority=5,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_send_task_events=True,

    # 结果过期
    result_expires=3600,  # 1小时
    result_backend_max_retries=10,

    # 任务重试
    task_reject_on_worker_lost=True,
    task_track_started=True,
    task_send_sent_event=True,

    # 并发控制
    worker_concurrency=4,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,

    # 任务超时
    task_soft_time_limit=300,  # 5分钟软超时
    task_time_limit=600,      # 10分钟硬超时

    # 消息传输协议
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=5,

    # 心跳和健康检查
    worker_health_check_interval=60,

    # 队列配置
    task_default_queue='default',
    task_queues={
        'default': {
            'exchange': 'default',
            'routing_key': 'default',
        },
        'document_processing': {
            'exchange': 'document_processing',
            'routing_key': 'document_processing',
        },
        'search_tasks': {
            'exchange': 'search_tasks',
            'routing_key': 'search_tasks',
        },
        'deep_search': {
            'exchange': 'deep_search',
            'routing_key': 'deep_search',
        },
        'evaluation': {
            'exchange': 'evaluation',
            'routing_key': 'evaluation',
        },
        'maintenance': {
            'exchange': 'maintenance',
            'routing_key': 'maintenance',
        }
    },

    # 交换机配置
    task_default_exchange='default',
    task_default_exchange_type='direct',
    task_default_routing_key='default',

    # 日志配置
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_log_color=False,

    # 压缩配置
    task_compression='gzip',
    result_compression='gzip',
)

# 定义交换机和队列
default_exchange = Exchange('default', type='direct')
document_processing_exchange = Exchange('document_processing', type='direct')
search_tasks_exchange = Exchange('search_tasks', type='direct')
deep_search_exchange = Exchange('deep_search', type='direct')
evaluation_exchange = Exchange('evaluation', type='direct')
maintenance_exchange = Exchange('maintenance', type='direct')

# 定义队列
default_queue = Queue('default', default_exchange, routing_key='default')
document_processing_queue = Queue('document_processing', document_processing_exchange, routing_key='document_processing')
search_tasks_queue = Queue('search_tasks', search_tasks_exchange, routing_key='search_tasks')
deep_search_queue = Queue('deep_search', deep_search_exchange, routing_key='deep_search')
evaluation_queue = Queue('evaluation', evaluation_exchange, routing_key='evaluation')
maintenance_queue = Queue('maintenance', maintenance_exchange, routing_key='maintenance')

# 配置定期任务（合并自 celerybeat_schedule.py）
celery_app.conf.beat_schedule = {
    # 每5分钟清理过期任务结果
    'cleanup-expired-tasks': {
        'task': 'app.core.async_tasks.tasks.cleanup_expired_results',
        'schedule': crontab(minute='*/5'),
    },
    # 每小时进行系统健康检查
    'system-health-check': {
        'task': 'app.core.async_tasks.tasks.system_health_check',
        'schedule': crontab(minute='0', hour='*'),
    },
    # 每天凌晨预热缓存
    'cache-warmup': {
        'task': 'app.core.async_tasks.tasks.daily_cache_warmup',
        'schedule': crontab(hour=2, minute='0'),
    },
    # 每周清理日志
    'log-cleanup': {
        'task': 'app.core.async_tasks.tasks.weekly_log_cleanup',
        'schedule': crontab(day_of_week='1', hour='3', minute='0'),
    },

    # 数据同步监控 - 每15分钟执行一次
    'data-sync-monitor': {
        'task': 'app.tasks.monitoring_tasks.data_sync_monitor',
        'schedule': 15.0 * 60,
        'options': {'queue': 'monitoring'}
    },
    # 系统健康检查 - 每30分钟执行一次
    'health-check': {
        'task': 'app.tasks.monitoring_tasks.health_check',
        'schedule': 30.0 * 60,
        'options': {'queue': 'monitoring'}
    },
    # 每日数据同步报告 - 每天凌晨1点执行
    'daily-sync-report': {
        'task': 'app.tasks.monitoring_tasks.daily_sync_report',
        'schedule': crontab(hour=1, minute=0),
        'options': {'queue': 'monitoring'}
    },
    # 数据完整性深度检查 - 每小时执行一次
    'data-integrity-check': {
        'task': 'app.tasks.monitoring_tasks.data_integrity_check',
        'schedule': 60.0 * 60,
        'options': {'queue': 'monitoring'}
    },
    # 清理过期日志 - 每天凌晨2点执行
    'cleanup-logs': {
        'task': 'app.tasks.maintenance.cleanup_old_logs',
        'schedule': crontab(hour=2, minute=0),
        'options': {'queue': 'maintenance'}
    }
}

# 配置任务自动发现
celery_app.autodiscover_tasks(['app.core.async_tasks'])

# 信号处理
from celery.signals import task_prerun, task_postrun, task_failure, task_success
from datetime import datetime


@celery_app.task(bind=True)
def debug_task(self):
    """调试任务"""
    print(f'Request: {self.request!r}')
    return f'Hello from Celery worker!'


# 信号处理器
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """任务开始前的处理"""
    logger.info(f'Task {task_id} ({task.name if task else "unknown"}) is about to start')


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """任务完成后的处理"""
    logger.info(f'Task {task_id} ({task.name if task else "unknown"}) finished with state: {state}')


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """任务失败处理"""
    logger.error(f'Task {task_id} failed: {exception}')


@task_success.connect
def task_success_handler(sender=None, result=None, args=None, kwargs=None, **kwds):
    """任务成功处理"""
    logger.info(f'Task succeeded with result: {result}')


# Worker配置
def configure_worker():
    """配置Celery Worker"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
    )

    # 禁用Celery的重启检测
    os.environ['FORKED_BY_MULTIPROCESSING'] = '1'

    return {
        'loglevel': 'info',
        'concurrency': 4,
        'prefetch_multiplier': 1,
        'max_tasks_per_child': 1000,
    }


# Flower监控配置（可选）
if hasattr(settings, 'ENABLE_FLOWER') and settings.ENABLE_FLOWER:
    celery_app.conf.update(
        flower_enable=True,
        flower_host='0.0.0.0',
        flower_port=5555,
        flower_basic_auth=['admin:password'],
    )


# 健康检查任务
@celery_app.task
def health_check():
    """健康检查任务"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'celery_status': 'running'
    }


# 获取Celery应用实例
def get_celery_app():
    """获取Celery应用实例"""
    return celery_app


# Worker启动钩子
@celery_app.on_after_configure.connect
def setup_directories(sender, **kwargs):
    """设置必要的目录"""
    os.makedirs('logs', exist_ok=True)
    os.makedirs('temp', exist_ok=True)


# 错误处理
@celery_app.task(bind=True)
def handle_task_error(self, exc, task_id, args, kwargs, einfo):
    """处理任务错误"""
    logger.error(f'Task {task_id} raised an exception: {exc}')
    logger.error(f'Traceback: {einfo}')

    # 记录错误到数据库或监控系统
    try:
        from app.core.events import event_publisher
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(event_publisher.system_error(
            component='celery_worker',
            error=exc,
            task_id=task_id,
            args=args,
            kwargs=kwargs
        ))
        loop.close()
    except Exception as e:
        logger.error(f'Failed to report error to event system: {e}')
