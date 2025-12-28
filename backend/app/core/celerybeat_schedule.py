"""
Celery Beat 定时任务配置
定义数据同步监控的定期执行任务
"""

from celery.schedules import crontab

# Celery Beat 定时任务配置
beat_schedule = {
    # 数据同步监控 - 每15分钟执行一次
    'data-sync-monitor': {
        'task': 'app.tasks.monitoring_tasks.data_sync_monitor',
        'schedule': 15.0 * 60,  # 15分钟
        'options': {'queue': 'monitoring'}
    },

    # 系统健康检查 - 每30分钟执行一次
    'health-check': {
        'task': 'app.tasks.monitoring_tasks.health_check',
        'schedule': 30.0 * 60,  # 30分钟
        'options': {'queue': 'monitoring'}
    },

    # 每日数据同步报告 - 每天凌晨1点执行
    'daily-sync-report': {
        'task': 'app.tasks.monitoring_tasks.daily_sync_report',
        'schedule': crontab(hour=1, minute=0),  # 每天凌晨1:00
        'options': {'queue': 'monitoring'}
    },

    # 数据完整性深度检查 - 每小时执行一次
    'data-integrity-check': {
        'task': 'app.tasks.monitoring_tasks.data_integrity_check',
        'schedule': 60.0 * 60,  # 1小时
        'options': {'queue': 'monitoring'}
    },

    # 清理过期日志 - 每天凌晨2点执行
    'cleanup-logs': {
        'task': 'app.tasks.maintenance.cleanup_old_logs',
        'schedule': crontab(hour=2, minute=0),  # 每天凌晨2:00
        'options': {'queue': 'maintenance'}
    }
}