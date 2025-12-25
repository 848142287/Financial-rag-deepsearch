"""
监控任务
使用Celery Beat定期执行数据同步监控
"""

import logging
import json
from datetime import datetime
from celery import current_task
from app.core.async_tasks.celery_app import celery_app
from app.tasks.sync_monitoring import run_sync_monitoring

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='app.tasks.monitoring_tasks.data_sync_monitor')
def data_sync_monitor(self):
    """
    数据同步监控任务
    每15分钟执行一次
    """
    try:
        logger.info("开始执行数据同步监控任务")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"status": "Checking data synchronization", "progress": 25}
        )

        # 运行监控检查
        result = run_sync_monitoring()

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"status": "Analyzing monitoring results", "progress": 75}
        )

        # 处理告警
        alerts = result.get('alerts', [])
        if alerts:
            logger.warning(f"发现 {len(alerts)} 个告警")
            for alert in alerts:
                if alert['type'] == 'critical':
                    logger.error(f"严重告警: {alert['message']}")
                    # 这里可以添加发送邮件/短信告警的逻辑
                elif alert['type'] == 'warning':
                    logger.warning(f"警告告警: {alert['message']}")
                    # 这里可以添加发送邮件/Slack通知的逻辑

        # 更新任务状态
        self.update_state(
            state="SUCCESS",
            meta={
                "status": "Monitoring completed",
                "progress": 100,
                "result": result
            }
        )

        logger.info("数据同步监控任务完成")
        return {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'monitoring_result': result
        }

    except Exception as e:
        logger.error(f"数据同步监控任务失败: {e}")
        self.update_state(
            state="FAILURE",
            meta={"status": "Monitoring failed", "error": str(e)}
        )
        raise

@celery_app.task(bind=True, name='app.tasks.monitoring_tasks.daily_sync_report')
def daily_sync_report(self):
    """
    每日数据同步报告任务
    每天凌晨1点执行
    """
    try:
        logger.info("开始生成每日数据同步报告")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"status": "Generating daily sync report", "progress": 50}
        )

        # 运行监控检查
        result = run_sync_monitoring()

        # 生成报告
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'mysql_vectors': result['sync_status']['data_sources']['mysql'].get('vectors', 0),
                'milvus_vectors': result['sync_status']['data_sources']['milvus'].get('vectors', 0),
                'mysql_entities': result['sync_status']['data_sources']['mysql'].get('entities', 0),
                'neo4j_entities': result['sync_status']['data_sources']['neo4j'].get('entities', 0),
                'vector_sync_rate': result['sync_status'].get('vector_sync_rate', 0),
                'entity_sync_rate': result['sync_status'].get('entity_sync_rate', 0),
                'overall_status': result['sync_status'].get('overall_status', 'unknown'),
                'alert_count': len(result.get('alerts', []))
            },
            'alerts': result.get('alerts', []),
            'detailed_stats': result['sync_status']['data_sources']
        }

        # 这里可以添加保存报告到数据库或发送邮件的逻辑
        logger.info(f"每日数据同步报告生成完成 - 状态: {report['summary']['overall_status']}")

        # 更新任务状态
        self.update_state(
            state="SUCCESS",
            meta={
                "status": "Daily report generated",
                "progress": 100,
                "report": report
            }
        )

        return {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'daily_report': report
        }

    except Exception as e:
        logger.error(f"每日数据同步报告任务失败: {e}")
        self.update_state(
            state="FAILURE",
            meta={"status": "Daily report failed", "error": str(e)}
        )
        raise

@celery_app.task(bind=True, name='app.tasks.monitoring_tasks.health_check')
def health_check(self):
    """
    系统健康检查任务
    每30分钟执行一次
    """
    try:
        logger.info("开始执行系统健康检查")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"status": "Performing health check", "progress": 33}
        )

        # 运行监控检查
        result = run_sync_monitoring()

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"status": "Analyzing health status", "progress": 66}
        )

        # 评估系统健康状态
        health_score = result['summary']['data_integrity_score']
        alert_count = result['summary']['alert_count']

        if health_score >= 95 and alert_count == 0:
            health_status = "excellent"
        elif health_score >= 90 and alert_count <= 1:
            health_status = "good"
        elif health_score >= 80 and alert_count <= 3:
            health_status = "fair"
        else:
            health_status = "poor"

        # 更新任务状态
        self.update_state(
            state="SUCCESS",
            meta={
                "status": "Health check completed",
                "progress": 100,
                "health_status": health_status,
                "health_score": health_score,
                "alert_count": alert_count
            }
        )

        logger.info(f"系统健康检查完成 - 状态: {health_status}, 得分: {health_score}")

        return {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'health_status': health_status,
            'health_score': health_score,
            'monitoring_result': result
        }

    except Exception as e:
        logger.error(f"系统健康检查任务失败: {e}")
        self.update_state(
            state="FAILURE",
            meta={"status": "Health check failed", "error": str(e)}
        )
        raise