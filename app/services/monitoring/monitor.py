"""
系统监控服务
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
import psutil
import time
from datetime import datetime, timedelta
import json
import statistics
from collections import defaultdict, deque

from app.core.redis_client import redis_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class MetricType(Enum):
    """监控指标类型"""
    SYSTEM_CPU = "system_cpu"
    SYSTEM_MEMORY = "system_memory"
    SYSTEM_DISK = "system_disk"
    SYSTEM_NETWORK = "system_network"
    APPLICATION_RESPONSE_TIME = "app_response_time"
    APPLICATION_ERROR_RATE = "app_error_rate"
    APPLICATION_THROUGHPUT = "app_throughput"
    DATABASE_CONNECTIONS = "db_connections"
    DATABASE_QUERIES = "db_queries"
    REDIS_CONNECTIONS = "redis_connections"
    CELERY_QUEUE_SIZE = "celery_queue_size"
    CELERY_WORKER_STATUS = "celery_worker_status"
    BUSINESS_DAILY_QUERIES = "business_daily_queries"
    BUSINESS_USER_SATISFACTION = "business_user_satisfaction"
    BUSINESS_DOCUMENT_PROCESSING = "business_document_processing"


@dataclass
class MetricValue:
    """指标值"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['metric_type'] = self.metric_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class Alert:
    """告警"""
    id: str
    metric_type: MetricType
    severity: AlertSeverity
    title: str
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['metric_type'] = self.metric_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data


@dataclass
class AlertRule:
    """告警规则"""
    id: str
    name: str
    metric_type: MetricType
    condition: str  # gt, lt, eq, gte, lte
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    duration: int = 300  # 持续时间（秒）
    description: str = ""
    tags: Dict[str, str] = None

    def evaluate(self, value: float) -> bool:
        """评估告警条件"""
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "gte":
            return value >= self.threshold
        elif self.condition == "lte":
            return value <= self.threshold
        elif self.condition == "eq":
            return abs(value - self.threshold) < 0.001
        return False


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics_buffer = deque(maxlen=10000)  # 缓冲最近10000个指标
        self.collection_interval = 30  # 30秒收集一次

    async def collect_system_metrics(self) -> List[MetricValue]:
        """收集系统指标"""
        metrics = []
        timestamp = datetime.utcnow()

        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricValue(
                metric_type=MetricType.SYSTEM_CPU,
                value=cpu_percent,
                timestamp=timestamp,
                tags={"host": "main"}
            ))

            # 内存使用率
            memory = psutil.virtual_memory()
            metrics.append(MetricValue(
                metric_type=MetricType.SYSTEM_MEMORY,
                value=memory.percent,
                timestamp=timestamp,
                tags={"host": "main"},
                metadata={
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used
                }
            ))

            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(MetricValue(
                metric_type=MetricType.SYSTEM_DISK,
                value=disk_percent,
                timestamp=timestamp,
                tags={"host": "main", "mount": "/"},
                metadata={
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free
                }
            ))

            # 网络IO
            net_io = psutil.net_io_counters()
            metrics.append(MetricValue(
                metric_type=MetricType.SYSTEM_NETWORK,
                value=net_io.bytes_sent + net_io.bytes_recv,
                timestamp=timestamp,
                tags={"host": "main"},
                metadata={
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            ))

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

        return metrics

    async def collect_application_metrics(self) -> List[MetricValue]:
        """收集应用指标"""
        metrics = []
        timestamp = datetime.utcnow()

        try:
            # 响应时间（从Redis获取最近统计）
            response_times = self._get_response_times_from_redis()
            if response_times:
                p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
                metrics.append(MetricValue(
                    metric_type=MetricType.APPLICATION_RESPONSE_TIME,
                    value=p95_response_time,
                    timestamp=timestamp,
                    tags={"percentile": "p95"}
                ))

            # 错误率
            error_rate = self._calculate_error_rate()
            metrics.append(MetricValue(
                metric_type=MetricType.APPLICATION_ERROR_RATE,
                value=error_rate,
                timestamp=timestamp
            ))

            # 吞吐量
            throughput = self._calculate_throughput()
            metrics.append(MetricValue(
                metric_type=MetricType.APPLICATION_THROUGHPUT,
                value=throughput,
                timestamp=timestamp,
                tags={"period": "1m"}
            ))

        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")

        return metrics

    async def collect_database_metrics(self) -> List[MetricValue]:
        """收集数据库指标"""
        metrics = []
        timestamp = datetime.utcnow()

        try:
            # 数据库连接数（需要连接数据库获取）
            db_connections = await self._get_db_connections()
            metrics.append(MetricValue(
                metric_type=MetricType.DATABASE_CONNECTIONS,
                value=db_connections,
                timestamp=timestamp
            ))

            # 查询性能
            query_stats = await self._get_query_stats()
            if query_stats:
                avg_query_time = query_stats.get("avg_time", 0)
                metrics.append(MetricValue(
                    metric_type=MetricType.DATABASE_QUERIES,
                    value=avg_query_time,
                    timestamp=timestamp,
                    tags={"type": "avg_time"}
                ))

        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")

        return metrics

    async def collect_business_metrics(self) -> List[MetricValue]:
        """收集业务指标"""
        metrics = []
        timestamp = datetime.utcnow()

        try:
            # 每日查询数
            daily_queries = await self._get_daily_query_count()
            metrics.append(MetricValue(
                metric_type=MetricType.BUSINESS_DAILY_QUERIES,
                value=daily_queries,
                timestamp=timestamp,
                tags={"period": "daily"}
            ))

            # 用户满意度
            user_satisfaction = await self._get_user_satisfaction()
            metrics.append(MetricValue(
                metric_type=MetricType.BUSINESS_USER_SATISFACTION,
                value=user_satisfaction,
                timestamp=timestamp
            ))

            # 文档处理统计
            doc_stats = await self._get_document_processing_stats()
            metrics.append(MetricValue(
                metric_type=MetricType.BUSINESS_DOCUMENT_PROCESSING,
                value=doc_stats.get("success_rate", 0),
                timestamp=timestamp,
                tags={"type": "success_rate"}
            ))

        except Exception as e:
            logger.error(f"Failed to collect business metrics: {e}")

        return metrics

    def _get_response_times_from_redis(self) -> List[float]:
        """从Redis获取响应时间数据"""
        try:
            # 获取最近1分钟的响应时间数据
            key_pattern = "response_time:*"
            keys = redis_client.keys(key_pattern)
            response_times = []

            for key in keys:
                data = redis_client.get(key)
                if data:
                    response_times.append(float(data))

            return response_times
        except Exception:
            return []

    def _calculate_error_rate(self) -> float:
        """计算错误率"""
        try:
            # 从统计日志计算错误率
            total_requests = redis_client.get("stats:total_requests") or 0
            error_requests = redis_client.get("stats:error_requests") or 0

            total = int(total_requests)
            errors = int(error_requests)

            if total > 0:
                return (errors / total) * 100
            return 0.0
        except Exception:
            return 0.0

    def _calculate_throughput(self) -> float:
        """计算吞吐量（每分钟请求数）"""
        try:
            # 获取最近1分钟的请求数
            current_time = int(time.time())
            one_minute_ago = current_time - 60

            # 使用Redis的有序集合存储时间序列数据
            request_count = redis_client.zcount("requests:timeline", one_minute_ago, current_time)
            return float(request_count)
        except Exception:
            return 0.0

    async def _get_db_connections(self) -> float:
        """获取数据库连接数"""
        try:
            # 这里应该查询实际的数据库连接池状态
            # 暂时返回模拟数据
            return 10.0
        except Exception:
            return 0.0

    async def _get_query_stats(self) -> Optional[Dict[str, float]]:
        """获取查询统计"""
        try:
            # 从查询日志统计
            return {"avg_time": 150.0}  # 模拟数据
        except Exception:
            return None

    async def _get_daily_query_count(self) -> float:
        """获取每日查询数"""
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            key = f"stats:daily_queries:{today}"
            count = redis_client.get(key)
            return float(count) if count else 0.0
        except Exception:
            return 0.0

    async def _get_user_satisfaction(self) -> float:
        """获取用户满意度"""
        try:
            # 从用户反馈数据计算满意度
            satisfaction_scores = redis_client.lrange("user_satisfaction", 0, -1)
            if satisfaction_scores:
                scores = [float(s) for s in satisfaction_scores]
                return statistics.mean(scores)
            return 0.0
        except Exception:
            return 0.0

    async def _get_document_processing_stats(self) -> Dict[str, float]:
        """获取文档处理统计"""
        try:
            total_docs = redis_client.get("stats:total_documents") or 0
            successful_docs = redis_client.get("stats:successful_documents") or 0

            total = int(total_docs)
            successful = int(successful_docs)

            if total > 0:
                return {"success_rate": (successful / total) * 100}
            return {"success_rate": 0.0}
        except Exception:
            return {"success_rate": 0.0}


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []

        # 初始化默认告警规则
        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认告警规则"""
        default_rules = [
            AlertRule(
                id="cpu_high",
                name="CPU使用率过高",
                metric_type=MetricType.SYSTEM_CPU,
                condition="gt",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                description="CPU使用率超过80%"
            ),
            AlertRule(
                id="cpu_critical",
                name="CPU使用率严重过高",
                metric_type=MetricType.SYSTEM_CPU,
                condition="gt",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                description="CPU使用率超过95%"
            ),
            AlertRule(
                id="memory_high",
                name="内存使用率过高",
                metric_type=MetricType.SYSTEM_MEMORY,
                condition="gt",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                description="内存使用率超过85%"
            ),
            AlertRule(
                id="disk_high",
                name="磁盘使用率过高",
                metric_type=MetricType.SYSTEM_DISK,
                condition="gt",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                description="磁盘使用率超过90%"
            ),
            AlertRule(
                id="response_time_high",
                name="响应时间过长",
                metric_type=MetricType.APPLICATION_RESPONSE_TIME,
                condition="gt",
                threshold=5000.0,
                severity=AlertSeverity.WARNING,
                description="P95响应时间超过5秒"
            ),
            AlertRule(
                id="error_rate_high",
                name="错误率过高",
                metric_type=MetricType.APPLICATION_ERROR_RATE,
                condition="gt",
                threshold=5.0,
                severity=AlertSeverity.CRITICAL,
                description="错误率超过5%"
            ),
            AlertRule(
                id="throughput_low",
                name="吞吐量过低",
                metric_type=MetricType.APPLICATION_THROUGHPUT,
                condition="lt",
                threshold=10.0,
                severity=AlertSeverity.WARNING,
                description="吞吐量低于10 req/min"
            )
        ]

        for rule in default_rules:
            self.rules[rule.id] = rule

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules[rule.id] = rule

    def remove_rule(self, rule_id: str):
        """删除告警规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]

    def evaluate_metric(self, metric: MetricValue) -> List[Alert]:
        """评估指标是否触发告警"""
        alerts = []

        for rule in self.rules.values():
            if not rule.enabled or rule.metric_type != metric.metric_type:
                continue

            if rule.evaluate(metric.value):
                alert_key = f"{rule.id}_{hash(str(metric.tags))}"
                current_time = datetime.utcnow()

                # 检查是否已有活跃告警
                if alert_key in self.active_alerts:
                    # 更新现有告警
                    alert = self.active_alerts[alert_key]
                    alert.current_value = metric.value
                    alert.timestamp = current_time
                else:
                    # 创建新告警
                    alert = Alert(
                        id=alert_key,
                        metric_type=metric.metric_type,
                        severity=rule.severity,
                        title=rule.name,
                        message=f"{rule.description}. 当前值: {metric.value:.2f}, 阈值: {rule.threshold}",
                        current_value=metric.value,
                        threshold=rule.threshold,
                        timestamp=current_time
                    )
                    self.active_alerts[alert_key] = alert
                    alerts.append(alert)

        return alerts

    def resolve_alert(self, alert_id: str):
        """解决告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()

            # 移动到历史记录
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]

            logger.info(f"Alert resolved: {alert.title}")

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        return sorted(self.alert_history, key=lambda x: x.timestamp, reverse=True)[:limit]

    def add_notification_handler(self, handler: Callable):
        """添加通知处理器"""
        self.notification_handlers.append(handler)

    async def send_notification(self, alert: Alert):
        """发送告警通知"""
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")


class MonitoringService:
    """监控服务"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.is_running = False
        self.monitoring_task = None

    async def start(self):
        """启动监控服务"""
        if self.is_running:
            return

        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        # 添加通知处理器
        self.alert_manager.add_notification_handler(self._websocket_notification_handler)
        self.alert_manager.add_notification_handler(self._log_notification_handler)

        logger.info("Monitoring service started")

    async def stop(self):
        """停止监控服务"""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Monitoring service stopped")

    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集所有类型的指标
                all_metrics = []

                # 系统指标
                system_metrics = await self.metrics_collector.collect_system_metrics()
                all_metrics.extend(system_metrics)

                # 应用指标
                app_metrics = await self.metrics_collector.collect_application_metrics()
                all_metrics.extend(app_metrics)

                # 数据库指标
                db_metrics = await self.metrics_collector.collect_database_metrics()
                all_metrics.extend(db_metrics)

                # 业务指标
                business_metrics = await self.metrics_collector.collect_business_metrics()
                all_metrics.extend(business_metrics)

                # 评估告警
                for metric in all_metrics:
                    alerts = self.alert_manager.evaluate_metric(metric)
                    for alert in alerts:
                        await self.alert_manager.send_notification(alert)

                # 存储指标到Redis
                await self._store_metrics(all_metrics)

                # 清理过期数据
                await self._cleanup_old_data()

                # 等待下次收集
                await asyncio.sleep(self.metrics_collector.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # 错误后短暂等待

    async def _store_metrics(self, metrics: List[MetricValue]):
        """存储指标到Redis"""
        try:
            for metric in metrics:
                # 存储到时间序列
                timestamp = int(metric.timestamp.timestamp())
                key = f"metrics:{metric.metric_type.value}"
                redis_client.zadd(key, {str(metric.value): timestamp})

                # 设置过期时间（7天）
                redis_client.expire(key, 86400 * 7)

                # 添加到缓冲区
                self.metrics_collector.metrics_buffer.append(metric)

        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")

    async def _cleanup_old_data(self):
        """清理过期数据"""
        try:
            # 清理7天前的指标数据
            cutoff_time = int((datetime.utcnow() - timedelta(days=7)).timestamp())

            for metric_type in MetricType:
                key = f"metrics:{metric_type.value}"
                redis_client.zremrangebyscore(key, 0, cutoff_time)

            # 清理已解决的告警（保留30天）
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            self.alert_manager.alert_history = [
                alert for alert in self.alert_manager.alert_history
                if alert.timestamp > cutoff_date
            ]

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    async def _websocket_notification_handler(self, alert: Alert):
        """WebSocket通知处理器"""
        try:
            from app.services.websocket_service import websocket_service
            await websocket_service.send_message(
                "system_alert",
                {
                    "alert_id": alert.id,
                    "title": alert.title,
                    "message": alert.message,
                    "severity": alert.severity.value,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to send WebSocket notification: {e}")

    def _log_notification_handler(self, alert: Alert):
        """日志通知处理器"""
        level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.FATAL: logging.CRITICAL
        }.get(alert.severity, logging.INFO)

        logger.log(level, f"ALERT: {alert.title} - {alert.message}")

    async def get_metrics_summary(self, metric_type: str, hours: int = 1) -> Dict[str, Any]:
        """获取指标摘要"""
        try:
            cutoff_time = int((datetime.utcnow() - timedelta(hours=hours)).timestamp())
            key = f"metrics:{metric_type}"

            # 获取最近N小时的指标
            metrics_data = redis_client.zrangebyscore(key, cutoff_time, "+inf", withscores=True)

            if not metrics_data:
                return {}

            values = [float(m[0]) for m in metrics_data]

            return {
                "metric_type": metric_type,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else 0,
                "time_range_hours": hours
            }

        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        active_alerts = self.alert_manager.get_active_alerts()

        # 按严重程度统计
        alert_counts = {
            "info": 0,
            "warning": 0,
            "critical": 0,
            "fatal": 0
        }

        for alert in active_alerts:
            alert_counts[alert.severity.value] += 1

        # 系统健康度评分
        health_score = max(0, 100 - (alert_counts["warning"] * 10 + alert_counts["critical"] * 25 + alert_counts["fatal"] * 50))

        return {
            "status": "healthy" if health_score >= 80 else "warning" if health_score >= 60 else "critical",
            "health_score": health_score,
            "active_alerts": len(active_alerts),
            "alert_counts": alert_counts,
            "monitoring_active": self.is_running,
            "last_update": datetime.utcnow().isoformat()
        }


# 全局监控服务实例
monitoring_service = MonitoringService()