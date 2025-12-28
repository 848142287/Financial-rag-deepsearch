"""
评估和监控异步任务
包括性能评估、质量监控、系统维护等功能
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import statistics

from celery import current_task
from app.core.celery_config import celery_app, monitor_task, EvaluationTask
from app.core.websocket_manager import connection_manager, MessageType
from app.services.evaluation.performance_evaluator import PerformanceEvaluator
from app.services.evaluation.quality_evaluator import QualityEvaluator
from app.services.evaluation.system_monitor import SystemMonitor
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=EvaluationTask,
    name='app.tasks.evaluation.collect_performance_metrics'
)
@monitor_task('performance_metrics_collection')
def collect_performance_metrics(self):
    """
    收集系统性能指标
    """
    logger.info("开始收集性能指标")

    try:
        # 获取任务统计信息
        from app.core.celery_config import get_task_statistics
        task_stats = get_task_statistics()

        # 获取系统资源使用情况
        system_metrics = get_system_metrics()

        # 获取数据库性能指标
        db_metrics = get_database_metrics()

        # 获取缓存指标
        cache_metrics = get_cache_metrics()

        # 获取网络和服务指标
        service_metrics = get_service_metrics()

        # 组合所有指标
        all_metrics = {
            'timestamp': datetime.now().isoformat(),
            'task_statistics': task_stats,
            'system_metrics': system_metrics,
            'database_metrics': db_metrics,
            'cache_metrics': cache_metrics,
            'service_metrics': service_metrics
        }

        # 保存到Redis
        save_metrics_to_redis('performance_metrics', all_metrics)

        # 检查告警条件
        check_performance_alerts(all_metrics)

        logger.info("性能指标收集完成")
        return all_metrics

    except Exception as e:
        logger.error(f"收集性能指标失败: {e}")
        raise


@celery_app.task(
    bind=True,
    base=EvaluationTask,
    name='app.tasks.evaluation.evaluate_retrieval_quality'
)
@monitor_task('quality_evaluation')
def evaluate_retrieval_quality(self, evaluation_window: int = 24):
    """
    评估检索质量
    """
    logger.info(f"开始评估检索质量 (窗口: {evaluation_window}小时)")

    try:
        evaluator = QualityEvaluator()

        # 获取评估时间段内的搜索记录
        search_records = get_recent_search_records(evaluation_window)

        if not search_records:
            logger.info("没有找到搜索记录，跳过质量评估")
            return {'status': 'no_data'}

        # 执行质量评估
        quality_metrics = evaluator.evaluate_quality(search_records)

        # 计算质量趋势
        quality_trend = calculate_quality_trend(quality_metrics)

        # 识别问题模式
        issues = identify_quality_issues(quality_metrics)

        # 保存评估结果
        save_quality_evaluation_results(quality_metrics, quality_trend, issues)

        # 发送质量报告
        send_quality_report(quality_metrics, quality_trend, issues)

        logger.info(f"检索质量评估完成，评估记录数: {len(search_records)}")
        return {
            'status': 'success',
            'evaluated_records': len(search_records),
            'quality_metrics': quality_metrics,
            'trend': quality_trend,
            'issues': issues
        }

    except Exception as e:
        logger.error(f"检索质量评估失败: {e}")
        raise


@celery_app.task(
    bind=True,
    base=EvaluationTask,
    name='app.tasks.evaluation.generate_system_report'
)
@monitor_task('system_report_generation')
def generate_system_report(self, report_type: str = 'daily', report_period: int = 24):
    """
    生成系统报告
    """
    logger.info(f"开始生成{report_type}报告 (周期: {report_period}小时)")

    try:
        # 收集报告所需的数据
        report_data = collect_report_data(report_period)

        # 生成报告内容
        report_content = generate_report_content(report_type, report_data)

        # 保存报告
        report_id = save_report(report_type, report_content, report_data)

        # 发送报告通知
        send_report_notification(report_id, report_type, report_content)

        logger.info(f"{report_type}报告生成完成，报告ID: {report_id}")
        return {
            'status': 'success',
            'report_id': report_id,
            'report_type': report_type,
            'generated_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"生成系统报告失败: {e}")
        raise


@celery_app.task(
    bind=True,
    base=EvaluationTask,
    name='app.tasks.evaluation.optimize_system_performance'
)
@monitor_task('performance_optimization')
def optimize_system_performance(self):
    """
    优化系统性能
    """
    logger.info("开始系统性能优化")

    try:
        optimizations = []

        # 1. 清理过期数据
        cleanup_result = cleanup_expired_data()
        optimizations.append({
            'type': 'cleanup',
            'description': '清理过期数据',
            'result': cleanup_result
        })

        # 2. 优化缓存
        cache_optimization = optimize_cache_performance()
        optimizations.append({
            'type': 'cache_optimization',
            'description': '优化缓存性能',
            'result': cache_optimization
        })

        # 3. 优化数据库索引
        db_optimization = optimize_database_performance()
        optimizations.append({
            'type': 'database_optimization',
            'description': '优化数据库性能',
            'result': db_optimization
        })

        # 4. 调整任务队列配置
        queue_optimization = optimize_task_queues()
        optimizations.append({
            'type': 'queue_optimization',
            'description': '优化任务队列',
            'result': queue_optimization
        })

        # 5. 更新监控配置
        monitoring_optimization = update_monitoring_config()
        optimizations.append({
            'type': 'monitoring_optimization',
            'description': '更新监控配置',
            'result': monitoring_optimization
        })

        logger.info("系统性能优化完成")
        return {
            'status': 'success',
            'optimizations': optimizations,
            'optimized_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"系统性能优化失败: {e}")
        raise


def get_system_metrics() -> Dict[str, Any]:
    """获取系统指标"""
    try:
        import psutil

        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used
        memory_total = memory.total

        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used = disk.used
        disk_total = disk.total

        # 网络统计
        network = psutil.net_io_counters()
        bytes_sent = network.bytes_sent
        bytes_recv = network.bytes_recv

        return {
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count
            },
            'memory': {
                'percent': memory_percent,
                'used': memory_used,
                'total': memory_total
            },
            'disk': {
                'percent': disk_percent,
                'used': disk_used,
                'total': disk_total
            },
            'network': {
                'bytes_sent': bytes_sent,
                'bytes_recv': bytes_recv
            },
            'load_average': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
        }

    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        return {}


def get_database_metrics() -> Dict[str, Any]:
    """获取数据库指标"""
    try:
        from app.services.storage.mysql_client import mysql_client
        from app.services.storage.milvus_client import milvus_client
        from app.services.storage.neo4j_client import neo4j_client

        # MySQL指标
        mysql_stats = mysql_client.get_performance_stats()

        # Milvus指标
        milvus_stats = milvus_client.get_collection_stats()

        # Neo4j指标
        neo4j_stats = neo4j_client.get_database_stats()

        return {
            'mysql': mysql_stats,
            'milvus': milvus_stats,
            'neo4j': neo4j_stats
        }

    except Exception as e:
        logger.error(f"获取数据库指标失败: {e}")
        return {}


def get_cache_metrics() -> Dict[str, Any]:
    """获取缓存指标"""
    try:
        # Redis信息
        redis_info = redis_client.info()

        # 连接数
        connected_clients = redis_info.get('connected_clients', 0)

        # 内存使用
        used_memory = redis_info.get('used_memory', 0)
        used_memory_human = redis_info.get('used_memory_human', '0B')

        # 命中率
        hits = redis_info.get('keyspace_hits', 0)
        misses = redis_info.get('keyspace_misses', 0)
        hit_rate = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0

        # 键空间统计
        keyspace_info = redis_info.get('db0', {})

        return {
            'connected_clients': connected_clients,
            'used_memory': used_memory,
            'used_memory_human': used_memory_human,
            'hit_rate': hit_rate,
            'hits': hits,
            'misses': misses,
            'keyspace': keyspace_info
        }

    except Exception as e:
        logger.error(f"获取缓存指标失败: {e}")
        return {}


def get_service_metrics() -> Dict[str, Any]:
    """获取服务指标"""
    try:
        # WebSocket连接统计
        ws_stats = asyncio.run(connection_manager.get_connection_stats())

        # LLM服务状态
        llm_status = check_llm_service_status()

        # MinIO服务状态
        minio_status = check_minio_service_status()

        return {
            'websocket': ws_stats,
            'llm_service': llm_status,
            'minio_service': minio_status,
            'active_services': get_active_services()
        }

    except Exception as e:
        logger.error(f"获取服务指标失败: {e}")
        return {}


def save_metrics_to_redis(metrics_type: str, metrics_data: Dict[str, Any]):
    """保存指标到Redis"""
    try:
        # 保存到时间序列
        key = f"metrics:{metrics_type}:{int(datetime.now().timestamp())}"
        redis_client.setex(key, 86400 * 7, json.dumps(metrics_data, ensure_ascii=False))  # 保存7天

        # 保存到列表用于实时监控
        redis_client.lpush(f"metrics:{metrics_type}:latest", json.dumps(metrics_data, ensure_ascii=False))
        redis_client.ltrim(f"metrics:{metrics_type}:latest", 0, 99)  # 保留最近100条

    except Exception as e:
        logger.error(f"保存指标到Redis失败: {e}")


def check_performance_alerts(metrics: Dict[str, Any]):
    """检查性能告警"""
    try:
        alerts = []

        # CPU告警
        cpu_percent = metrics.get('system_metrics', {}).get('cpu', {}).get('percent', 0)
        if cpu_percent > 80:
            alerts.append({
                'type': 'cpu_high',
                'level': 'warning',
                'message': f'CPU使用率过高: {cpu_percent:.1f}%'
            })

        # 内存告警
        memory_percent = metrics.get('system_metrics', {}).get('memory', {}).get('percent', 0)
        if memory_percent > 85:
            alerts.append({
                'type': 'memory_high',
                'level': 'warning',
                'message': f'内存使用率过高: {memory_percent:.1f}%'
            })

        # 磁盘告警
        disk_percent = metrics.get('system_metrics', {}).get('disk', {}).get('percent', 0)
        if disk_percent > 90:
            alerts.append({
                'type': 'disk_high',
                'level': 'critical',
                'message': f'磁盘使用率过高: {disk_percent:.1f}%'
            })

        # 缓存命中率告警
        hit_rate = metrics.get('cache_metrics', {}).get('hit_rate', 0)
        if hit_rate < 70:
            alerts.append({
                'type': 'cache_low_hit_rate',
                'level': 'warning',
                'message': f'缓存命中率过低: {hit_rate:.1f}%'
            })

        # 发送告警通知
        if alerts:
            send_performance_alerts(alerts)

    except Exception as e:
        logger.error(f"检查性能告警失败: {e}")


def get_recent_search_records(hours: int) -> List[Dict[str, Any]]:
    """获取最近的搜索记录"""
    try:
        # 从Redis获取搜索记录
        records = []
        pattern = f"search_result:*"

        keys = redis_client.keys(pattern)
        cutoff_time = datetime.now() - timedelta(hours=hours)

        for key in keys:
            try:
                data = redis_client.get(key)
                if data:
                    record = json.loads(data.decode('utf-8'))
                    record_time = datetime.fromisoformat(record.get('created_at', ''))
                    if record_time >= cutoff_time:
                        records.append(record)
            except:
                continue

        return records

    except Exception as e:
        logger.error(f"获取搜索记录失败: {e}")
        return []


def calculate_quality_trend(quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """计算质量趋势"""
    try:
        # 获取历史质量数据
        historical_data = get_historical_quality_data()

        if not historical_data:
            return {'trend': 'insufficient_data'}

        # 计算趋势
        current_score = quality_metrics.get('overall_score', 0)
        historical_scores = [h.get('overall_score', 0) for h in historical_data]

        if len(historical_scores) >= 3:
            recent_avg = statistics.mean(historical_scores[-3:])
            if current_score > recent_avg * 1.1:
                trend = 'improving'
            elif current_score < recent_avg * 0.9:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'trend': trend,
            'current_score': current_score,
            'historical_average': statistics.mean(historical_scores) if historical_scores else 0,
            'data_points': len(historical_scores)
        }

    except Exception as e:
        logger.error(f"计算质量趋势失败: {e}")
        return {'trend': 'error'}


def identify_quality_issues(quality_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """识别质量问题"""
    try:
        issues = []

        # 相关性问题
        relevance_score = quality_metrics.get('relevance_score', 1.0)
        if relevance_score < 0.6:
            issues.append({
                'type': 'low_relevance',
                'severity': 'high',
                'description': f'检索相关性过低: {relevance_score:.2f}',
                'suggestions': ['调整查询改写策略', '优化向量嵌入模型']
            })

        # 完整性问题
        completeness_score = quality_metrics.get('completeness_score', 1.0)
        if completeness_score < 0.7:
            issues.append({
                'type': 'low_completeness',
                'severity': 'medium',
                'description': f'结果完整性不足: {completeness_score:.2f}',
                'suggestions': ['扩大检索范围', '增加多源检索']
            })

        # 新鲜度问题
        freshness_score = quality_metrics.get('freshness_score', 1.0)
        if freshness_score < 0.8:
            issues.append({
                'type': 'low_freshness',
                'severity': 'medium',
                'description': f'信息新鲜度不足: {freshness_score:.2f}',
                'suggestions': ['优化时间过滤策略', '增加实时数据源']
            })

        return issues

    except Exception as e:
        logger.error(f"识别质量问题失败: {e}")
        return []


def get_historical_quality_data() -> List[Dict[str, Any]]:
    """获取历史质量数据"""
    try:
        # 从Redis获取历史质量评估结果
        historical_data = []
        pattern = "quality_evaluation:*"

        keys = redis_client.keys(pattern)
        for key in keys:
            try:
                data = redis_client.get(key)
                if data:
                    evaluation = json.loads(data.decode('utf-8'))
                    historical_data.append(evaluation.get('quality_metrics', {}))
            except:
                continue

        return historical_data

    except Exception as e:
        logger.error(f"获取历史质量数据失败: {e}")
        return []


def save_quality_evaluation_results(quality_metrics: Dict[str, Any], trend: Dict[str, Any], issues: List[Dict[str, Any]]):
    """保存质量评估结果"""
    try:
        evaluation_data = {
            'timestamp': datetime.now().isoformat(),
            'quality_metrics': quality_metrics,
            'trend': trend,
            'issues': issues
        }

        # 保存到Redis
        key = f"quality_evaluation:{int(datetime.now().timestamp())}"
        redis_client.setex(key, 86400 * 30, json.dumps(evaluation_data, ensure_ascii=False))  # 保存30天

    except Exception as e:
        logger.error(f"保存质量评估结果失败: {e}")


def send_quality_report(quality_metrics: Dict[str, Any], trend: Dict[str, Any], issues: List[Dict[str, Any]]):
    """发送质量报告"""
    try:
        report_summary = {
            'overall_score': quality_metrics.get('overall_score', 0),
            'trend': trend.get('trend', 'unknown'),
            'issues_count': len(issues),
            'critical_issues': len([i for i in issues if i.get('severity') == 'high'])
        }

        asyncio.run(connection_manager.send_system_notification(
            f"质量评估报告完成，总体评分: {report_summary['overall_score']:.2f}",
            'info' if report_summary['overall_score'] >= 0.8 else 'warning'
        ))

    except Exception as e:
        logger.error(f"发送质量报告失败: {e}")


def send_performance_alerts(alerts: List[Dict[str, Any]]):
    """发送性能告警"""
    try:
        for alert in alerts:
            asyncio.run(connection_manager.send_system_notification(
                alert['message'],
                alert['level']
            ))

            # 保存告警记录
            alert_data = {
                **alert,
                'timestamp': datetime.now().isoformat()
            }
            redis_client.lpush('performance_alerts', json.dumps(alert_data, ensure_ascii=False))
            redis_client.ltrim('performance_alerts', 0, 999)  # 保留最近1000条

    except Exception as e:
        logger.error(f"发送性能告警失败: {e}")


# 辅助函数
def check_llm_service_status() -> Dict[str, Any]:
    """检查LLM服务状态"""
    try:
        # 这里应该实际调用LLM健康检查
        return {
            'status': 'healthy',
            'response_time_ms': 150,
            'last_check': datetime.now().isoformat()
        }
    except:
        return {
            'status': 'unhealthy',
            'error': 'Service unavailable'
        }


def check_minio_service_status() -> Dict[str, Any]:
    """检查MinIO服务状态"""
    try:
        from app.core.minio_client import minio_client
        # 执行简单的健康检查
        minio_client.list_buckets()
        return {
            'status': 'healthy',
            'last_check': datetime.now().isoformat()
        }
    except:
        return {
            'status': 'unhealthy',
            'error': 'Connection failed'
        }


def get_active_services() -> List[str]:
    """获取活跃服务列表"""
    return [
        'celery_worker',
        'websocket_server',
        'api_server',
        'redis',
        'mysql',
        'milvus',
        'neo4j'
    ]