"""
数据完整性检查任务
结合同步监控、增量同步和告警系统
"""

import logging
from datetime import datetime
from celery import current_task
from app.core.async_tasks.celery_app import celery_app
from app.tasks.sync_monitoring import run_sync_monitoring
from app.services.incremental_sync_service import IncrementalSyncService
from app.services.alert_service import AlertService

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='app.tasks.data_integrity_check.run_comprehensive_check')
def run_comprehensive_check(self):
    """
    运行全面的数据完整性检查
    每小时执行一次
    """
    try:
        logger.info("开始全面数据完整性检查")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"status": "Starting comprehensive check", "progress": 10}
        )

        # 1. 运行同步监控
        self.update_state(
            state="PROGRESS",
            meta={"status": "Running sync monitoring", "progress": 20}
        )

        sync_status = run_sync_monitoring()
        sync_result = sync_status['sync_status']
        health_result = sync_status['health_status']

        # 2. 运行增量同步
        self.update_state(
            state="PROGRESS",
            meta={"status": "Running incremental sync", "progress": 50}
        )

        sync_service = IncrementalSyncService()
        incremental_result = sync_service.run_incremental_sync()

        # 3. 检查告警
        self.update_state(
            state="PROGRESS",
            meta={"status": "Checking for alerts", "progress": 70}
        )

        alert_service = AlertService()
        all_alerts = []

        # 检查同步状态告警
        sync_alerts = alert_service.create_sync_alerts(sync_result)
        all_alerts.extend(sync_alerts)

        # 检查健康状态告警
        health_alerts = alert_service.create_health_alerts(health_result)
        all_alerts.extend(health_alerts)

        # 发送告警
        self.update_state(
            state="PROGRESS",
            meta={"status": "Sending alerts if needed", "progress": 85}
        )

        sent_alerts = 0
        for alert in all_alerts:
            if alert_service.process_alert(alert, channels=['email', 'slack']):
                sent_alerts += 1

        # 4. 生成报告
        self.update_state(
            state="PROGRESS",
            meta={"status": "Generating report", "progress": 95}
        )

        report = {
            'timestamp': datetime.now().isoformat(),
            'check_id': self.request.id,
            'overall_status': sync_result.get('overall_status', 'unknown'),
            'data_integrity_score': sync_status['summary']['data_integrity_score'],
            'sync_status': {
                'vector_sync_rate': sync_result.get('vector_sync_rate', 0),
                'entity_sync_rate': sync_result.get('entity_sync_rate', 0),
                'relation_sync_rate': sync_result.get('relation_sync_rate', 0),
                'mysql_data': sync_result['data_sources']['mysql'],
                'milvus_data': sync_result['data_sources']['milvus'],
                'neo4j_data': sync_result['data_sources']['neo4j']
            },
            'health_status': {
                'recent_completed': health_result.get('recent_completed', 0),
                'failed_documents': health_result.get('failed_documents', 0),
                'processing_documents': health_result.get('processing_documents', 0),
                'health_score': health_result.get('health_score', 'unknown')
            },
            'incremental_sync_result': incremental_result,
            'alerts': {
                'total_generated': len(all_alerts),
                'total_sent': sent_alerts,
                'critical_alerts': len([a for a in all_alerts if a.type == 'critical']),
                'warning_alerts': len([a for a in all_alerts if a.type == 'warning']),
                'info_alerts': len([a for a in all_alerts if a.type == 'info'])
            },
            'summary': {
                'status': 'healthy' if sync_result.get('overall_status') == 'good' and len(all_alerts) == 0 else 'attention_needed',
                'data_synced_successfully': incremental_result.get('vectors', {}).get('synced_count', 0) + incremental_result.get('entities', {}).get('synced_count', 0),
                'alerts_generated': len(all_alerts),
                'alerts_sent': sent_alerts
            }
        }

        # 更新任务状态
        self.update_state(
            state="SUCCESS",
            meta={
                "status": "Comprehensive check completed",
                "progress": 100,
                "report": report
            }
        )

        logger.info(f"全面数据完整性检查完成 - 状态: {report['summary']['status']}")
        return {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'comprehensive_report': report
        }

    except Exception as e:
        logger.error(f"全面数据完整性检查失败: {e}")
        self.update_state(
            state="FAILURE",
            meta={"status": "Comprehensive check failed", "error": str(e)}
        )
        raise

@celery_app.task(bind=True, name='app.tasks.data_integrity_check.quick_health_check')
def quick_health_check(self):
    """
    快速健康检查
    每15分钟执行一次
    """
    try:
        logger.info("开始快速健康检查")

        # 运行基本监控
        sync_status = run_sync_monitoring()

        # 快速评估
        status = 'healthy'
        issues = []

        sync_score = sync_status['summary']['data_integrity_score']
        alert_count = sync_status['summary']['alert_count']

        if sync_score < 80:
            status = 'critical'
            issues.append(f'数据同步率过低: {sync_score}%')
        elif sync_score < 95 or alert_count > 0:
            status = 'warning'
            if sync_score < 95:
                issues.append(f'数据同步率需要关注: {sync_score}%')
            if alert_count > 0:
                issues.append(f'存在 {alert_count} 个告警')

        result = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'sync_score': sync_score,
            'alert_count': alert_count,
            'issues': issues,
            'quick_summary': '系统正常运行' if status == 'healthy' else f'系统需要关注: {", ".join(issues)}'
        }

        logger.info(f"快速健康检查完成 - 状态: {status}")
        return result

    except Exception as e:
        logger.error(f"快速健康检查失败: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        }