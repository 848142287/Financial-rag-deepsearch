"""
缓存统计和分析
提供缓存性能监控、分析和优化建议
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
import time

logger = logging.getLogger(__name__)


class StatsPeriod(Enum):
    """统计周期"""
    REALTIME = "realtime"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CacheMetrics:
    """缓存指标"""
    timestamp: datetime
    level: str
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_response_time: float = 0.0
    size_bytes: int = 0
    entry_count: int = 0
    evictions: int = 0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.hit_rate,
            'avg_response_time': self.avg_response_time,
            'size_bytes': self.size_bytes,
            'entry_count': self.entry_count,
            'evictions': self.evictions,
            'errors': self.errors
        }


@dataclass
class PerformanceAlert:
    """性能告警"""
    id: str
    level: AlertLevel
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class OptimizationSuggestion:
    """优化建议"""
    category: str
    priority: str
    title: str
    description: str
    impact: str
    effort: str
    metrics: List[str]
    action_items: List[str]


class CacheStatsCollector:
    """缓存统计收集器"""

    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self._metrics_history: List[CacheMetrics] = []
        self._realtime_metrics: CacheMetrics = None
        self._alerts: List[PerformanceAlert] = []
        self._lock = asyncio.Lock()
        self._thresholds = self._get_default_thresholds()

    def _get_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """获取默认阈值"""
        return {
            'hit_rate': {
                'warning': 0.7,
                'error': 0.5,
                'critical': 0.3
            },
            'avg_response_time': {
                'warning': 0.1,
                'error': 0.5,
                'critical': 1.0
            },
            'evictions': {
                'warning': 100,
                'error': 500,
                'critical': 1000
            },
            'errors': {
                'warning': 10,
                'error': 50,
                'critical': 100
            }
        }

    async def collect_metrics(self, cache_stats: Dict[str, Any]) -> None:
        """收集缓存指标"""
        try:
            async with self._lock:
                timestamp = datetime.now()

                # 收集整体指标
                overall_metrics = CacheMetrics(
                    timestamp=timestamp,
                    level="overall",
                    total_requests=cache_stats.get('total_requests', 0),
                    cache_hits=cache_stats.get('total_hits', 0),
                    cache_misses=cache_stats.get('total_misses', 0),
                    hit_rate=cache_stats.get('overall_hit_rate', 0.0),
                    avg_response_time=cache_stats.get('avg_response_time', 0.0),
                    size_bytes=cache_stats.get('total_size_bytes', 0),
                    entry_count=cache_stats.get('total_entries', 0),
                    evictions=cache_stats.get('total_evictions', 0),
                    errors=cache_stats.get('total_errors', 0)
                )

                # 收集各级指标
                level_metrics = []
                level_stats = cache_stats.get('levels', {})
                for level_name, level_data in level_stats.items():
                    metrics = CacheMetrics(
                        timestamp=timestamp,
                        level=level_name,
                        total_requests=level_data.get('requests', 0),
                        cache_hits=level_data.get('hits', 0),
                        cache_misses=level_data.get('misses', 0),
                        hit_rate=level_data.get('hit_rate', 0.0),
                        avg_response_time=level_data.get('avg_response_time', 0.0),
                        size_bytes=level_data.get('size_bytes', 0),
                        entry_count=level_data.get('entry_count', 0),
                        evictions=level_data.get('evictions', 0),
                        errors=level_data.get('errors', 0)
                    )
                    level_metrics.append(metrics)

                # 更新实时指标
                self._realtime_metrics = overall_metrics

                # 添加到历史记录
                self._metrics_history.append(overall_metrics)
                self._metrics_history.extend(level_metrics)

                # 清理过期数据
                await self._cleanup_old_metrics()

                # 检查告警
                await self._check_alerts(overall_metrics, level_metrics)

                logger.debug(f"缓存指标收集完成: {overall_metrics.to_dict()}")

        except Exception as e:
            logger.error(f"收集缓存指标失败: {e}")

    async def _cleanup_old_metrics(self) -> None:
        """清理过期指标"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        self._metrics_history = [
            m for m in self._metrics_history if m.timestamp > cutoff_date
        ]

    async def _check_alerts(self, overall_metrics: CacheMetrics,
                           level_metrics: List[CacheMetrics]) -> None:
        """检查告警条件"""
        all_metrics = [overall_metrics] + level_metrics

        for metrics in all_metrics:
            await self._check_metric_alerts(metrics)

    async def _check_metric_alerts(self, metrics: CacheMetrics) -> None:
        """检查单个指标的告警"""
        # 检查命中率
        await self._check_threshold_alert(
            metrics, 'hit_rate', metrics.hit_rate, self._thresholds['hit_rate']
        )

        # 检查响应时间
        await self._check_threshold_alert(
            metrics, 'avg_response_time', metrics.avg_response_time,
            self._thresholds['avg_response_time'], reverse=True
        )

        # 检查淘汰次数
        await self._check_threshold_alert(
            metrics, 'evictions', metrics.evictions, self._thresholds['evictions']
        )

        # 检查错误次数
        await self._check_threshold_alert(
            metrics, 'errors', metrics.errors, self._thresholds['errors']
        )

    async def _check_threshold_alert(self, metrics: CacheMetrics, metric_name: str,
                                   current_value: float, thresholds: Dict[str, float],
                                   reverse: bool = False) -> None:
        """检查阈值告警"""
        # 确定是否超过阈值
        def exceeds_threshold(value, threshold):
            return value < threshold if reverse else value > threshold

        level = None
        threshold = None

        if exceeds_threshold(current_value, thresholds['critical']):
            level = AlertLevel.CRITICAL
            threshold = thresholds['critical']
        elif exceeds_threshold(current_value, thresholds['error']):
            level = AlertLevel.ERROR
            threshold = thresholds['error']
        elif exceeds_threshold(current_value, thresholds['warning']):
            level = AlertLevel.WARNING
            threshold = thresholds['warning']

        if level:
            await self._create_alert(metrics, metric_name, current_value, threshold, level)

    async def _create_alert(self, metrics: CacheMetrics, metric_name: str,
                          current_value: float, threshold: float,
                          level: AlertLevel) -> None:
        """创建告警"""
        alert_id = f"{metrics.level}_{metric_name}_{int(time.time())}"

        # 检查是否已有相同的未解决告警
        existing_alert = next(
            (a for a in self._alerts
             if a.metric_name == metric_name and
             a.level == level and
             not a.resolved and
             a.level_name == metrics.level),
            None
        )

        if existing_alert:
            # 更新现有告警
            existing_alert.current_value = current_value
            existing_alert.timestamp = datetime.now()
        else:
            # 创建新告警
            title = f"{metric_name} {level.value.title()} Alert"
            message = f"{metrics.level.title()} cache {metric_name} is {current_value:.2f}, " \
                     f"threshold is {threshold:.2f}"

            alert = PerformanceAlert(
                id=alert_id,
                level=level,
                title=title,
                message=message,
                metric_name=metric_name,
                current_value=current_value,
                threshold=threshold,
                timestamp=datetime.now(),
                level_name=metrics.level
            )

            self._alerts.append(alert)
            logger.warning(f"创建缓存告警: {title} - {message}")

    def get_realtime_metrics(self) -> Optional[CacheMetrics]:
        """获取实时指标"""
        return self._realtime_metrics

    async def get_metrics_history(self, period: StatsPeriod = StatsPeriod.HOUR,
                                level: Optional[str] = None) -> List[CacheMetrics]:
        """获取历史指标"""
        async with self._lock:
            # 过滤时间段
            cutoff_time = self._get_cutoff_time(period)
            filtered_metrics = [
                m for m in self._metrics_history
                if m.timestamp > cutoff_time and (level is None or m.level == level)
            ]

            return sorted(filtered_metrics, key=lambda x: x.timestamp)

    def _get_cutoff_time(self, period: StatsPeriod) -> datetime:
        """获取截止时间"""
        now = datetime.now()
        if period == StatsPeriod.MINUTE:
            return now - timedelta(minutes=1)
        elif period == StatsPeriod.HOUR:
            return now - timedelta(hours=1)
        elif period == StatsPeriod.DAY:
            return now - timedelta(days=1)
        elif period == StatsPeriod.WEEK:
            return now - timedelta(weeks=1)
        elif period == StatsPeriod.MONTH:
            return now - timedelta(days=30)
        else:
            return now - timedelta(minutes=5)  # 默认5分钟

    async def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[PerformanceAlert]:
        """获取活跃告警"""
        async with self._lock:
            alerts = [a for a in self._alerts if not a.resolved]
            if level:
                alerts = [a for a in alerts if a.level == level]
            return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    async def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        async with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    logger.info(f"解决告警: {alert.title}")
                    return True
            return False

    async def get_performance_report(self, period: StatsPeriod = StatsPeriod.DAY) -> Dict[str, Any]:
        """获取性能报告"""
        metrics = await self.get_metrics_history(period)
        if not metrics:
            return {'error': 'No metrics available'}

        # 按级别分组
        level_groups = {}
        for metric in metrics:
            if metric.level not in level_groups:
                level_groups[metric.level] = []
            level_groups[metric.level].append(metric)

        report = {
            'period': period.value,
            'time_range': {
                'start': min(m.timestamp for m in metrics).isoformat(),
                'end': max(m.timestamp for m in metrics).isoformat()
            },
            'overall': self._calculate_level_summary(level_groups.get('overall', [])),
            'levels': {
                level: self._calculate_level_summary(level_metrics)
                for level, level_metrics in level_groups.items()
                if level != 'overall'
            },
            'alerts': {
                'total': len([a for a in self._alerts if not a.resolved]),
                'critical': len([a for a in self._alerts if not a.resolved and a.level == AlertLevel.CRITICAL]),
                'error': len([a for a in self._alerts if not a.resolved and a.level == AlertLevel.ERROR]),
                'warning': len([a for a in self._alerts if not a.resolved and a.level == AlertLevel.WARNING])
            }
        }

        return report

    def _calculate_level_summary(self, metrics: List[CacheMetrics]) -> Dict[str, Any]:
        """计算级别摘要"""
        if not metrics:
            return {}

        hit_rates = [m.hit_rate for m in metrics]
        response_times = [m.avg_response_time for m in metrics]
        total_requests = sum(m.total_requests for m in metrics)

        return {
            'hit_rate': {
                'avg': statistics.mean(hit_rates),
                'min': min(hit_rates),
                'max': max(hit_rates),
                'trend': self._calculate_trend(hit_rates[-10:]) if len(hit_rates) > 1 else 0.0
            },
            'avg_response_time': {
                'avg': statistics.mean(response_times),
                'min': min(response_times),
                'max': max(response_times),
                'trend': self._calculate_trend(response_times[-10:]) if len(response_times) > 1 else 0.0
            },
            'total_requests': total_requests,
            'total_evictions': sum(m.evictions for m in metrics),
            'total_errors': sum(m.errors for m in metrics),
            'current_entry_count': metrics[-1].entry_count if metrics else 0,
            'current_size_bytes': metrics[-1].size_bytes if metrics else 0
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势（简单线性回归斜率）"""
        if len(values) < 2:
            return 0.0

        x = list(range(len(values)))
        n = len(values)

        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        # 计算斜率
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        return slope

    async def get_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """获取优化建议"""
        suggestions = []
        realtime_metrics = self.get_realtime_metrics()

        if not realtime_metrics:
            return suggestions

        # 命中率优化建议
        if realtime_metrics.hit_rate < 0.7:
            suggestions.append(OptimizationSuggestion(
                category="hit_rate",
                priority="high",
                title="提高缓存命中率",
                description="当前缓存命中率较低，建议优化缓存策略",
                impact="提高响应速度，减少后端负载",
                effort="medium",
                metrics=["hit_rate", "cache_misses"],
                action_items=[
                    "分析缓存未命中的原因",
                    "优化缓存键的设计",
                    "调整TTL策略",
                    "增加缓存预热"
                ]
            ))

        # 响应时间优化建议
        if realtime_metrics.avg_response_time > 0.1:
            suggestions.append(OptimizationSuggestion(
                category="response_time",
                priority="medium",
                title="降低缓存响应时间",
                description="缓存响应时间偏高，建议进行性能优化",
                impact="提高用户体验",
                effort="low",
                metrics=["avg_response_time"],
                action_items=[
                    "检查缓存后端性能",
                    "优化序列化方式",
                    "考虑使用更快的存储",
                    "增加内存缓存比例"
                ]
            ))

        # 淘汰频率优化建议
        if realtime_metrics.evictions > 100:
            suggestions.append(OptimizationSuggestion(
                category="evictions",
                priority="high",
                title="减少缓存淘汰频率",
                description="缓存淘汰频繁，建议增加缓存容量或优化策略",
                impact="提高缓存效率",
                effort="medium",
                metrics=["evictions", "size_bytes"],
                action_items=[
                    "增加缓存容量",
                    "优化LRU策略",
                    "分析热点数据",
                    "考虑分层缓存"
                ]
            ))

        # 错误率优化建议
        if realtime_metrics.errors > 10:
            suggestions.append(OptimizationSuggestion(
                category="errors",
                priority="critical",
                title="降低缓存错误率",
                description="缓存错误较多，需要立即处理",
                impact="提高系统稳定性",
                effort="high",
                metrics=["errors"],
                action_items=[
                    "检查缓存后端连接",
                    "增加错误处理机制",
                    "实施缓存降级策略",
                    "加强监控告警"
                ]
            ))

        return suggestions

    def set_threshold(self, metric_name: str, level: AlertLevel, threshold: float) -> None:
        """设置告警阈值"""
        if metric_name not in self._thresholds:
            self._thresholds[metric_name] = {}

        self._thresholds[metric_name][level.value] = threshold
        logger.info(f"设置告警阈值: {metric_name} {level.value} = {threshold}")

    def get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """获取所有阈值"""
        return self._thresholds.copy()

    async def export_metrics(self, file_path: str, period: StatsPeriod = StatsPeriod.DAY) -> bool:
        """导出指标数据"""
        try:
            metrics = await self.get_metrics_history(period)
            alerts = await self.get_active_alerts()

            export_data = {
                'export_time': datetime.now().isoformat(),
                'period': period.value,
                'metrics': [m.to_dict() for m in metrics],
                'alerts': [
                    {
                        'id': a.id,
                        'level': a.level.value,
                        'title': a.title,
                        'message': a.message,
                        'metric_name': a.metric_name,
                        'current_value': a.current_value,
                        'threshold': a.threshold,
                        'timestamp': a.timestamp.isoformat(),
                        'resolved': a.resolved,
                        'resolved_at': a.resolved_at.isoformat() if a.resolved_at else None
                    }
                    for a in alerts
                ],
                'thresholds': self._thresholds
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"指标数据导出成功: {file_path}")
            return True

        except Exception as e:
            logger.error(f"指标数据导出失败: {e}")
            return False


# 全局统计收集器实例
cache_stats_collector: Optional[CacheStatsCollector] = None


def get_cache_stats_collector(retention_days: int = 30) -> CacheStatsCollector:
    """获取缓存统计收集器实例"""
    global cache_stats_collector

    if cache_stats_collector is None:
        cache_stats_collector = CacheStatsCollector(retention_days=retention_days)

    return cache_stats_collector