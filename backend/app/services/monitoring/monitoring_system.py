"""
监控与运维系统
实现完整的日志、指标监控与告警体系
"""

import asyncio
import logging
import json
import psutil
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading

from app.core.redis_client import redis_client
from app.core.websocket_manager import connection_manager

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"      # 计数器
    GAUGE = "gauge"          # 仪表盘
    HISTOGRAM = "histogram"  # 直方图
    TIMER = "timer"         # 计时器


@dataclass
class Alert:
    """告警信息"""
    id: str
    level: AlertLevel
    source: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class Metric:
    """监控指标"""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str]
    timestamp: datetime


class MonitoringSystem:
    """监控系统"""

    def __init__(self):
        self.metrics_store = defaultdict(list)
        self.alerts = deque(maxlen=1000)  # 保留最近1000条告警
        self.alert_rules = []
        self.monitors = {}
        self.collectors = {}
        self.running = False
        self.collection_interval = 60  # 60秒

        # 预定义告警规则
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self):
        """设置默认告警规则"""
        default_rules = [
            {
                'name': 'high_cpu_usage',
                'condition': lambda metrics: self._get_metric_value(metrics, 'system_cpu_usage') > 80,
                'level': AlertLevel.WARNING,
                'message': 'CPU使用率过高',
                'threshold': 80
            },
            {
                'name': 'high_memory_usage',
                'condition': lambda metrics: self._get_metric_value(metrics, 'system_memory_usage') > 85,
                'level': AlertLevel.WARNING,
                'message': '内存使用率过高',
                'threshold': 85
            },
            {
                'name': 'high_error_rate',
                'condition': lambda metrics: self._get_metric_value(metrics, 'api_error_rate') > 5,
                'level': AlertLevel.ERROR,
                'message': 'API错误率过高',
                'threshold': 5
            },
            {
                'name': 'slow_response_time',
                'condition': lambda metrics: self._get_metric_value(metrics, 'api_response_time_p95') > 5000,
                'level': AlertLevel.WARNING,
                'message': 'API响应时间过慢',
                'threshold': 5000
            },
            {
                'name': 'low_rag_score',
                'condition': lambda metrics: self._get_metric_value(metrics, 'rag_overall_score') < 0.7,
                'level': AlertLevel.WARNING,
                'message': 'RAG整体评分偏低',
                'threshold': 0.7
            },
            {
                'name': 'disk_space_low',
                'condition': lambda metrics: self._get_metric_value(metrics, 'disk_usage_percent') > 90,
                'level': AlertLevel.CRITICAL,
                'message': '磁盘空间不足',
                'threshold': 90
            }
        ]

        self.alert_rules = default_rules

    def start_monitoring(self):
        """启动监控"""
        if self.running:
            logger.warning("监控系统已在运行")
            return

        self.running = True

        # 启动指标收集任务
        asyncio.create_task(self._metric_collection_loop())

        # 启动告警检查任务
        asyncio.create_task(self._alert_check_loop())

        logger.info("监控系统已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        logger.info("监控系统已停止")

    async def _metric_collection_loop(self):
        """指标收集循环"""
        while self.running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"指标收集失败: {e}")
                await asyncio.sleep(10)  # 错误后等待10秒

    async def _alert_check_loop(self):
        """告警检查循环"""
        while self.running:
            try:
                await self._check_alerts()
                await asyncio.sleep(30)  # 每30秒检查一次告警
            except Exception as e:
                logger.error(f"告警检查失败: {e}")
                await asyncio.sleep(30)

    async def _collect_all_metrics(self):
        """收集所有指标"""
        try:
            timestamp = datetime.now()

            # 系统指标
            await self._collect_system_metrics(timestamp)

            # 应用指标
            await self._collect_application_metrics(timestamp)

            # 业务指标
            await self._collect_business_metrics(timestamp)

            # 清理过期指标（保留最近1小时）
            self._cleanup_old_metrics()

        except Exception as e:
            logger.error(f"收集指标失败: {e}")

    async def _collect_system_metrics(self, timestamp: datetime):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric(Metric(
                name='system_cpu_usage',
                type=MetricType.GAUGE,
                value=cpu_percent,
                labels={'unit': 'percent'},
                timestamp=timestamp
            ))

            # 内存使用率
            memory = psutil.virtual_memory()
            self._add_metric(Metric(
                name='system_memory_usage',
                type=MetricType.GAUGE,
                value=memory.percent,
                labels={'unit': 'percent'},
                timestamp=timestamp
            ))

            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._add_metric(Metric(
                name='disk_usage_percent',
                type=MetricType.GAUGE,
                value=disk_percent,
                labels={'unit': 'percent'},
                timestamp=timestamp
            ))

            # 网络IO
            network = psutil.net_io_counters()
            self._add_metric(Metric(
                name='network_bytes_sent',
                type=MetricType.COUNTER,
                value=network.bytes_sent,
                labels={'direction': 'out'},
                timestamp=timestamp
            ))
            self._add_metric(Metric(
                name='network_bytes_received',
                type=MetricType.COUNTER,
                value=network.bytes_recv,
                labels={'direction': 'in'},
                timestamp=timestamp
            ))

        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")

    async def _collect_application_metrics(self, timestamp: datetime):
        """收集应用指标"""
        try:
            # API请求统计
            api_metrics = await self._get_api_metrics()
            for name, value in api_metrics.items():
                self._add_metric(Metric(
                    name=f'api_{name}',
                    type=MetricType.GAUGE,
                    value=value,
                    labels={},
                    timestamp=timestamp
                ))

            # Celery任务统计
            celery_metrics = await self._get_celery_metrics()
            for name, value in celery_metrics.items():
                self._add_metric(Metric(
                    name=f'celery_{name}',
                    type=MetricType.GAUGE,
                    value=value,
                    labels={},
                    timestamp=timestamp
                ))

            # Redis缓存统计
            redis_metrics = await self._get_redis_metrics()
            for name, value in redis_metrics.items():
                self._add_metric(Metric(
                    name=f'redis_{name}',
                    type=MetricType.GAUGE,
                    value=value,
                    labels={},
                    timestamp=timestamp
                ))

        except Exception as e:
            logger.error(f"收集应用指标失败: {e}")

    async def _collect_business_metrics(self, timestamp: datetime):
        """收集业务指标"""
        try:
            # RAG性能指标
            rag_metrics = await self._get_rag_metrics()
            for name, value in rag_metrics.items():
                self._add_metric(Metric(
                    name=f'rag_{name}',
                    type=MetricType.GAUGE,
                    value=value,
                    labels={},
                    timestamp=timestamp
                ))

            # 用户行为指标
            user_metrics = await self._get_user_metrics()
            for name, value in user_metrics.items():
                self._add_metric(Metric(
                    name=f'user_{name}',
                    type=MetricType.COUNTER,
                    value=value,
                    labels={},
                    timestamp=timestamp
                ))

        except Exception as e:
            logger.error(f"收集业务指标失败: {e}")

    async def _get_api_metrics(self) -> Dict[str, float]:
        """获取API指标"""
        try:
            # 从Redis获取API统计
            stats = redis_client.hgetall('api_statistics')

            return {
                'request_count': float(stats.get('request_count', 0)),
                'error_rate': float(stats.get('error_rate', 0)),
                'response_time_avg': float(stats.get('response_time_avg', 0)),
                'response_time_p95': float(stats.get('response_time_p95', 0))
            }

        except Exception as e:
            logger.error(f"获取API指标失败: {e}")
            return {}

    async def _get_celery_metrics(self) -> Dict[str, float]:
        """获取Celery指标"""
        try:
            from app.core.celery_config import celery_app

            inspect = celery_app.control.inspect()
            stats = inspect.stats()

            if not stats:
                return {}

            total_tasks = sum(worker.get('total', 0) for worker in stats.values())
            active_tasks = sum(len(tasks) for tasks in inspect.active().values()) if inspect.active() else 0

            return {
                'total_tasks': float(total_tasks),
                'active_tasks': float(active_tasks),
                'worker_count': float(len(stats))
            }

        except Exception as e:
            logger.error(f"获取Celery指标失败: {e}")
            return {}

    async def _get_redis_metrics(self) -> Dict[str, float]:
        """获取Redis指标"""
        try:
            info = redis_client.info()

            return {
                'connected_clients': float(info.get('connected_clients', 0)),
                'used_memory_mb': float(info.get('used_memory', 0)) / (1024 * 1024),
                'hit_rate': self._calculate_hit_rate(info),
                'commands_per_sec': float(info.get('instantaneous_ops_per_sec', 0))
            }

        except Exception as e:
            logger.error(f"获取Redis指标失败: {e}")
            return {}

    def _calculate_hit_rate(self, redis_info: Dict) -> float:
        """计算Redis命中率"""
        try:
            hits = redis_info.get('keyspace_hits', 0)
            misses = redis_info.get('keyspace_misses', 0)
            total = hits + misses

            return (hits / total * 100) if total > 0 else 0.0

        except:
            return 0.0

    async def _get_rag_metrics(self) -> Dict[str, float]:
        """获取RAG指标"""
        try:
            # 从Redis获取RAG评估结果
            recent_evaluations = redis_client.lrange('evaluation_results', 0, 99)  # 最近100条

            if not recent_evaluations:
                return {'overall_score': 0.0, 'faithfulness': 0.0}

            scores = []
            faithfulness_scores = []

            for eval_data in recent_evaluations:
                try:
                    data = json.loads(eval_data.decode('utf-8'))
                    if isinstance(data, dict) and 'overall_score' in data:
                        scores.append(data['overall_score'])
                    if isinstance(data, dict) and 'ragas_metrics' in data:
                        faithfulness_scores.append(data['ragas_metrics'].get('faithfulness', 0))
                except:
                    continue

            return {
                'overall_score': sum(scores) / len(scores) if scores else 0.0,
                'faithfulness': sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
                'evaluation_count': float(len(scores))
            }

        except Exception as e:
            logger.error(f"获取RAG指标失败: {e}")
            return {}

    async def _get_user_metrics(self) -> Dict[str, float]:
        """获取用户指标"""
        try:
            # 从Redis获取用户行为统计
            stats = redis_client.hgetall('user_statistics')

            return {
                'active_users_24h': float(stats.get('active_users_24h', 0)),
                'daily_queries': float(stats.get('daily_queries', 0)),
                'avg_session_duration': float(stats.get('avg_session_duration', 0)),
                'user_satisfaction': float(stats.get('user_satisfaction', 0))
            }

        except Exception as e:
            logger.error(f"获取用户指标失败: {e}")
            return {}

    def _add_metric(self, metric: Metric):
        """添加指标"""
        self.metrics_store[metric.name].append(metric)

    def _cleanup_old_metrics(self):
        """清理过期指标"""
        cutoff_time = datetime.now() - timedelta(hours=1)

        for name, metrics in self.metrics_store.items():
            # 保留最近的指标
            self.metrics_store[name] = [
                m for m in metrics if m.timestamp >= cutoff_time
            ]

    async def _check_alerts(self):
        """检查告警条件"""
        try:
            # 获取最新指标
            current_metrics = {}
            for name, metrics in self.metrics_store.items():
                if metrics:
                    current_metrics[name] = metrics[-1]  # 最新指标

            # 检查每个告警规则
            for rule in self.alert_rules:
                try:
                    if rule['condition'](current_metrics):
                        await self._trigger_alert(rule, current_metrics)
                except Exception as e:
                    logger.error(f"检查告警规则失败 {rule['name']}: {e}")

        except Exception as e:
            logger.error(f"告警检查失败: {e}")

    async def _trigger_alert(self, rule: Dict[str, Any], metrics: Dict[str, Metric]):
        """触发告警"""
        try:
            alert = Alert(
                id=f"{rule['name']}_{int(time.time())}",
                level=rule['level'],
                source='monitoring_system',
                message=rule['message'],
                details={
                    'rule_name': rule['name'],
                    'threshold': rule['threshold'],
                    'current_value': self._get_metric_value(metrics, self._get_rule_metric_name(rule['name']))
                },
                timestamp=datetime.now()
            )

            # 检查是否已经存在相同的告警
            existing_alerts = [a for a in self.alerts if not a.resolved and a.source == rule['name']]
            if existing_alerts:
                return  # 避免重复告警

            # 添加告警
            self.alerts.append(alert)

            # 发送通知
            await self._send_alert_notification(alert)

            # 保存告警到Redis
            alert_data = asdict(alert)
            alert_data['timestamp'] = alert_data['timestamp'].isoformat()
            if alert_data['resolved_at']:
                alert_data['resolved_at'] = alert_data['resolved_at'].isoformat()

            redis_client.lpush('system_alerts', json.dumps(alert_data, default=str))
            redis_client.ltrim('system_alerts', 0, 999)  # 保留最近1000条

            logger.warning(f"触发告警: {rule['message']}")

        except Exception as e:
            logger.error(f"触发告警失败: {e}")

    def _get_rule_metric_name(self, rule_name: str) -> str:
        """获取规则对应的指标名称"""
        mapping = {
            'high_cpu_usage': 'system_cpu_usage',
            'high_memory_usage': 'system_memory_usage',
            'high_error_rate': 'api_error_rate',
            'slow_response_time': 'api_response_time_p95',
            'low_rag_score': 'rag_overall_score',
            'disk_space_low': 'disk_usage_percent'
        }
        return mapping.get(rule_name, rule_name)

    def _get_metric_value(self, metrics: Dict[str, Metric], metric_name: str) -> float:
        """获取指标值"""
        metric = metrics.get(metric_name)
        return metric.value if metric else 0.0

    async def _send_alert_notification(self, alert: Alert):
        """发送告警通知"""
        try:
            # 通过WebSocket发送实时通知
            level_map = {
                AlertLevel.INFO: 'info',
                AlertLevel.WARNING: 'warning',
                AlertLevel.ERROR: 'error',
                AlertLevel.CRITICAL: 'critical'
            }

            await connection_manager.send_system_notification(
                f"系统告警: {alert.message}",
                level_map[alert.level]
            )

            # 可以集成其他通知方式：邮件、短信、钉钉等
            # await self._send_email_notification(alert)
            # await self._send_sms_notification(alert)

        except Exception as e:
            logger.error(f"发送告警通知失败: {e}")

    def add_custom_metric(self, name: str, value: float, labels: Dict[str, str] = None, metric_type: MetricType = MetricType.GAUGE):
        """添加自定义指标"""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            labels=labels or {},
            timestamp=datetime.now()
        )
        self._add_metric(metric)

    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """递增计数器指标"""
        try:
            # 获取当前值
            current_metrics = self.metrics_store.get(name, [])
            current_value = current_metrics[-1].value if current_metrics else 0.0

            # 添加新值
            self.add_metric(name, current_value + value, labels, MetricType.COUNTER)

        except Exception as e:
            logger.error(f"递增计数器失败: {e}")

    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """记录计时器指标"""
        self.add_metric(name, duration, labels, MetricType.TIMER)

    async def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """获取指标摘要"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            summary = {}

            for name, metrics in self.metrics_store.items():
                recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]

                if recent_metrics:
                    values = [m.value for m in recent_metrics]
                    summary[name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'latest': values[-1]
                    }

            return {
                'time_range_minutes': minutes,
                'metrics': summary,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取指标摘要失败: {e}")
            return {}

    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        try:
            active_alerts = [alert for alert in self.alerts if not alert.resolved]

            return [
                {
                    'id': alert.id,
                    'level': alert.level.value,
                    'source': alert.source,
                    'message': alert.message,
                    'details': alert.details,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in active_alerts
            ]

        except Exception as e:
            logger.error(f"获取活跃告警失败: {e}")
            return []

    async def resolve_alert(self, alert_id: str):
        """解决告警"""
        try:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()

                    # 更新Redis中的告警状态
                    redis_alerts = redis_client.lrange('system_alerts', 0, -1)
                    for i, alert_data in enumerate(redis_alerts):
                        data = json.loads(alert_data.decode('utf-8'))
                        if data.get('id') == alert_id:
                            data['resolved'] = True
                            data['resolved_at'] = alert.resolved_at.isoformat()
                            redis_client.lset('system_alerts', i, json.dumps(data, default=str))
                            break

                    logger.info(f"告警已解决: {alert_id}")
                    return True

            return False

        except Exception as e:
            logger.error(f"解决告警失败: {e}")
            return False

    def register_collector(self, name: str, collector_func: Callable):
        """注册指标收集器"""
        self.collectors[name] = collector_func

    async def run_custom_collector(self, name: str):
        """运行自定义收集器"""
        try:
            if name in self.collectors:
                timestamp = datetime.now()
                metrics = await self.collectors[name](timestamp)

                if isinstance(metrics, list):
                    for metric in metrics:
                        self._add_metric(metric)
                elif isinstance(metrics, dict):
                    for name, value in metrics.items():
                        self.add_metric(f"custom_{name}_{name}", value)

                logger.info(f"自定义收集器 {name} 执行完成")
                return True
            else:
                logger.warning(f"自定义收集器 {name} 不存在")
                return False

        except Exception as e:
            logger.error(f"运行自定义收集器失败 {name}: {e}")
            return False

    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        try:
            # 获取最新的系统指标
            system_metrics = {
                'cpu_usage': self._get_latest_metric('system_cpu_usage'),
                'memory_usage': self._get_latest_metric('system_memory_usage'),
                'disk_usage': self._get_latest_metric('disk_usage_percent')
            }

            # 评估健康状态
            health_score = 100
            issues = []

            if system_metrics['cpu_usage'] > 90:
                health_score -= 25
                issues.append("CPU使用率过高")
            elif system_metrics['cpu_usage'] > 80:
                health_score -= 10
                issues.append("CPU使用率偏高")

            if system_metrics['memory_usage'] > 90:
                health_score -= 25
                issues.append("内存使用率过高")
            elif system_metrics['memory_usage'] > 80:
                health_score -= 10
                issues.append("内存使用率偏高")

            if system_metrics['disk_usage'] > 95:
                health_score -= 30
                issues.append("磁盘空间严重不足")
            elif system_metrics['disk_usage'] > 85:
                health_score -= 15
                issues.append("磁盘空间不足")

            # 获取活跃告警数量
            active_alerts_count = len([a for a in self.alerts if not a.resolved])
            if active_alerts_count > 0:
                health_score -= min(active_alerts_count * 5, 20)

            health_status = "healthy"
            if health_score < 60:
                health_status = "unhealthy"
            elif health_score < 80:
                health_status = "warning"

            return {
                'status': health_status,
                'score': health_score,
                'issues': issues,
                'metrics': system_metrics,
                'active_alerts': active_alerts_count,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取系统健康状态失败: {e}")
            return {
                'status': 'unknown',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# 全局监控系统实例
monitoring_system = MonitoringSystem()