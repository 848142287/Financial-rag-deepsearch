"""
多维评估指标计算器
实现准确度、效率、可用性等维度的指标计算
"""

import asyncio
import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import math

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """指标类别"""
    ACCURACY = "accuracy"       # 准确度维度
    EFFICIENCY = "efficiency"   # 效率维度
    AVAILABILITY = "availability"  # 可用性维度
    QUALITY = "quality"         # 质量维度


@dataclass
class MetricResult:
    """指标结果"""
    name: str
    category: MetricCategory
    value: float
    unit: str
    description: str
    threshold: Optional[float] = None
    status: str = "normal"  # normal, warning, critical


class MetricsCalculator:
    """多维指标计算器"""

    def __init__(self):
        self.metrics_history = []

    async def calculate_accuracy_metrics(self, test_results: List[Dict[str, Any]]) -> List[MetricResult]:
        """
        计算准确度维度指标
        """
        try:
            metrics = []

            # 1. 召回准确率 (Recall Accuracy)
            recall_accuracy = await self._calculate_recall_accuracy(test_results)
            metrics.append(MetricResult(
                name="recall_accuracy",
                category=MetricCategory.ACCURACY,
                value=recall_accuracy,
                unit="percentage",
                description="召回准确率：检索到的相关文档占所有相关文档的比例",
                threshold=0.85
            ))

            # 2. 答案相关性 (Answer Relevance)
            answer_relevance = await self._calculate_answer_relevance(test_results)
            metrics.append(MetricResult(
                name="answer_relevance",
                category=MetricCategory.ACCURACY,
                value=answer_relevance,
                unit="score",
                description="答案相关性：生成答案与问题的相关程度",
                threshold=0.80
            ))

            # 3. 事实一致性 (Factual Consistency)
            factual_consistency = await self._calculate_factual_consistency(test_results)
            metrics.append(MetricResult(
                name="factual_consistency",
                category=MetricCategory.ACCURACY,
                value=factual_consistency,
                unit="score",
                description="事实一致性：答案中的事实与源文档的一致程度",
                threshold=0.90
            ))

            # 4. 引用准确率 (Citation Accuracy)
            citation_accuracy = await self._calculate_citation_accuracy(test_results)
            metrics.append(MetricResult(
                name="citation_accuracy",
                category=MetricCategory.ACCURACY,
                value=citation_accuracy,
                unit="percentage",
                description="引用准确率：答案中的引用与源文档的匹配程度",
                threshold=0.85
            ))

            # 5. 知识覆盖率 (Knowledge Coverage)
            knowledge_coverage = await self._calculate_knowledge_coverage(test_results)
            metrics.append(MetricResult(
                name="knowledge_coverage",
                category=MetricCategory.ACCURACY,
                value=knowledge_coverage,
                unit="percentage",
                description="知识覆盖率：答案覆盖问题所需知识的程度",
                threshold=0.75
            ))

            # 设置指标状态
            for metric in metrics:
                metric.status = self._get_metric_status(metric.value, metric.threshold)

            return metrics

        except Exception as e:
            logger.error(f"计算准确度指标失败: {e}")
            return []

    async def calculate_efficiency_metrics(self, performance_data: List[Dict[str, Any]]) -> List[MetricResult]:
        """
        计算效率维度指标
        """
        try:
            metrics = []

            # 1. 响应时间百分位数 (Response Time Percentiles)
            response_times = [d.get('response_time', 0) for d in performance_data]

            if response_times:
                p50 = statistics.median(response_times)
                p90 = self._calculate_percentile(response_times, 0.9)
                p99 = self._calculate_percentile(response_times, 0.99)

                metrics.append(MetricResult(
                    name="response_time_p50",
                    category=MetricCategory.EFFICIENCY,
                    value=p50,
                    unit="ms",
                    description="P50响应时间：50%请求的响应时间",
                    threshold=2000
                ))

                metrics.append(MetricResult(
                    name="response_time_p90",
                    category=MetricCategory.EFFICIENCY,
                    value=p90,
                    unit="ms",
                    description="P90响应时间：90%请求的响应时间",
                    threshold=5000
                ))

                metrics.append(MetricResult(
                    name="response_time_p99",
                    category=MetricCategory.EFFICIENCY,
                    value=p99,
                    unit="ms",
                    description="P99响应时间：99%请求的响应时间",
                    threshold=10000
                ))

            # 2. 吞吐量 (Throughput)
            throughput = await self._calculate_throughput(performance_data)
            metrics.append(MetricResult(
                name="throughput",
                category=MetricCategory.EFFICIENCY,
                value=throughput,
                unit="requests/minute",
                description="系统吞吐量：每分钟处理的请求数",
                threshold=100
            ))

            # 3. 资源利用率 (Resource Utilization)
            resource_utilization = await self._calculate_resource_utilization(performance_data)
            metrics.append(MetricResult(
                name="resource_utilization",
                category=MetricCategory.EFFICIENCY,
                value=resource_utilization,
                unit="percentage",
                description="资源利用率：CPU、内存等资源的使用率",
                threshold=80.0
            ))

            # 4. 任务队列效率 (Queue Efficiency)
            queue_efficiency = await self._calculate_queue_efficiency(performance_data)
            metrics.append(MetricResult(
                name="queue_efficiency",
                category=MetricCategory.EFFICIENCY,
                value=queue_efficiency,
                unit="percentage",
                description="任务队列效率：任务在队列中的平均等待时间",
                threshold=90.0
            ))

            # 5. 缓存命中率 (Cache Hit Rate)
            cache_hit_rate = await self._calculate_cache_hit_rate(performance_data)
            metrics.append(MetricResult(
                name="cache_hit_rate",
                category=MetricCategory.EFFICIENCY,
                value=cache_hit_rate,
                unit="percentage",
                description="缓存命中率：从缓存获取数据的比例",
                threshold=70.0
            ))

            # 设置指标状态
            for metric in metrics:
                # 对于效率指标，越低越好（响应时间）或越高越好（吞吐量）
                if "response_time" in metric.name:
                    metric.status = self._get_metric_status_efficiency(
                        metric.value, metric.threshold, lower_is_better=True
                    )
                else:
                    metric.status = self._get_metric_status(metric.value, metric.threshold)

            return metrics

        except Exception as e:
            logger.error(f"计算效率指标失败: {e}")
            return []

    async def calculate_availability_metrics(self, system_logs: List[Dict[str, Any]]) -> List[MetricResult]:
        """
        计算可用性维度指标
        """
        try:
            metrics = []

            # 1. 系统可用性 (System Availability)
            availability = await self._calculate_availability(system_logs)
            metrics.append(MetricResult(
                name="system_availability",
                category=MetricCategory.AVAILABILITY,
                value=availability,
                unit="percentage",
                description="系统可用性：系统可正常提供服务的时间比例",
                threshold=99.9
            ))

            # 2. 错误率 (Error Rate)
            error_rate = await self._calculate_error_rate(system_logs)
            metrics.append(MetricResult(
                name="error_rate",
                category=MetricCategory.AVAILABILITY,
                value=error_rate,
                unit="percentage",
                description="错误率：请求失败的比例",
                threshold=5.0
            ))

            # 3. 平均故障恢复时间 (MTTR)
            mttr = await self._calculate_mttr(system_logs)
            metrics.append(MetricResult(
                name="mttr",
                category=MetricCategory.AVAILABILITY,
                value=mttr,
                unit="minutes",
                description="平均故障恢复时间：系统从故障到恢复的平均时间",
                threshold=30.0
            ))

            # 4. 平均无故障时间 (MTBF)
            mtbf = await self._calculate_mtbf(system_logs)
            metrics.append(MetricResult(
                name="mtbf",
                category=MetricCategory.AVAILABILITY,
                value=mtbf,
                unit="hours",
                description="平均无故障时间：系统平均连续无故障运行时间",
                threshold=168.0  # 7天
            ))

            # 5. 用户满意度 (User Satisfaction)
            user_satisfaction = await self._calculate_user_satisfaction(system_logs)
            metrics.append(MetricResult(
                name="user_satisfaction",
                category=MetricCategory.AVAILABILITY,
                value=user_satisfaction,
                unit="score",
                description="用户满意度：用户对系统服务的满意程度",
                threshold=4.0
            ))

            # 设置指标状态
            for metric in metrics:
                if "error_rate" in metric.name:
                    metric.status = self._get_metric_status_efficiency(
                        metric.value, metric.threshold, lower_is_better=True
                    )
                elif metric.name == "mttr":
                    metric.status = self._get_metric_status_efficiency(
                        metric.value, metric.threshold, lower_is_better=True
                    )
                else:
                    metric.status = self._get_metric_status(metric.value, metric.threshold)

            return metrics

        except Exception as e:
            logger.error(f"计算可用性指标失败: {e}")
            return []

    async def _calculate_recall_accuracy(self, test_results: List[Dict[str, Any]]) -> float:
        """计算召回准确率"""
        try:
            total_relevant = 0
            total_retrieved_relevant = 0

            for result in test_results:
                relevant_docs = result.get('relevant_documents', [])
                retrieved_docs = result.get('retrieved_documents', [])

                total_relevant += len(relevant_docs)

                # 计算检索到的相关文档数
                retrieved_relevant += len(set(relevant_docs) & set(retrieved_docs))

            return total_retrieved_relevant / total_relevant if total_relevant > 0 else 0.0

        except Exception as e:
            logger.error(f"计算召回准确率失败: {e}")
            return 0.0

    async def _calculate_answer_relevance(self, test_results: List[Dict[str, Any]]) -> float:
        """计算答案相关性"""
        try:
            relevance_scores = []

            for result in test_results:
                # 使用模型或规则计算答案相关性
                # 这里简化实现，实际应该调用LLM进行相关性评分
                score = result.get('relevance_score', 0.8)  # 默认值
                if score is not None:
                    relevance_scores.append(score)

            return statistics.mean(relevance_scores) if relevance_scores else 0.0

        except Exception as e:
            logger.error(f"计算答案相关性失败: {e}")
            return 0.0

    async def _calculate_factual_consistency(self, test_results: List[Dict[str, Any]]) -> float:
        """计算事实一致性"""
        try:
            consistency_scores = []

            for result in test_results:
                # 检查答案中的事实是否与源文档一致
                consistency_score = result.get('consistency_score', 0.9)
                if consistency_score is not None:
                    consistency_scores.append(consistency_score)

            return statistics.mean(consistency_scores) if consistency_scores else 0.0

        except Exception as e:
            logger.error(f"计算事实一致性失败: {e}")
            return 0.0

    async def _calculate_citation_accuracy(self, test_results: List[Dict[str, Any]]) -> float:
        """计算引用准确率"""
        try:
            total_citations = 0
            correct_citations = 0

            for result in test_results:
                answer = result.get('answer', '')
                citations = result.get('citations', [])

                # 检查引用是否准确匹配
                for citation in citations:
                    total_citations += 1
                    if self._validate_citation(answer, citation):
                        correct_citations += 1

            return correct_citations / total_citations if total_citations > 0 else 0.0

        except Exception as e:
            logger.error(f"计算引用准确率失败: {e}")
            return 0.0

    def _validate_citation(self, answer: str, citation: Dict[str, Any]) -> bool:
        """验证引用准确性"""
        # 简化实现，实际应该检查引用内容是否在答案中
        citation_text = citation.get('text', '')
        return citation_text in answer

    async def _calculate_knowledge_coverage(self, test_results: List[Dict[str, Any]]) -> float:
        """计算知识覆盖率"""
        try:
            coverage_scores = []

            for result in test_results:
                required_knowledge = result.get('required_knowledge_points', [])
                covered_knowledge = result.get('covered_knowledge_points', [])

                if required_knowledge:
                    coverage = len(set(required_knowledge) & set(covered_knowledge)) / len(required_knowledge)
                    coverage_scores.append(coverage)

            return statistics.mean(coverage_scores) if coverage_scores else 0.0

        except Exception as e:
            logger.error(f"计算知识覆盖率失败: {e}")
            return 0.0

    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = math.ceil(len(sorted_data) * percentile / 100) - 1
        index = max(0, min(index, len(sorted_data) - 1))
        return sorted_data[index]

    async def _calculate_throughput(self, performance_data: List[Dict[str, Any]]) -> float:
        """计算吞吐量"""
        try:
            if not performance_data:
                return 0.0

            # 计算每分钟处理的请求数
            time_range = self._get_time_range(performance_data)
            total_requests = len(performance_data)

            if time_range.total_seconds() > 0:
                throughput = total_requests / (time_range.total_seconds() / 60)
                return round(throughput, 2)

            return 0.0

        except Exception as e:
            logger.error(f"计算吞吐量失败: {e}")
            return 0.0

    def _get_time_range(self, data: List[Dict[str, Any]]) -> timedelta:
        """获取数据的时间范围"""
        timestamps = [datetime.fromisoformat(d.get('timestamp', datetime.now().isoformat()))
                     for d in data]

        if timestamps:
            return max(timestamps) - min(timestamps)
        return timedelta(0)

    async def _calculate_resource_utilization(self, performance_data: List[Dict[str, Any]]) -> float:
        """计算资源利用率"""
        try:
            cpu_usage = [d.get('cpu_usage', 0) for d in performance_data]
            memory_usage = [d.get('memory_usage', 0) for d in performance_data]

            if cpu_usage and memory_usage:
                avg_cpu = statistics.mean(cpu_usage)
                avg_memory = statistics.mean(memory_usage)
                return max(avg_cpu, avg_memory)

            return 0.0

        except Exception as e:
            logger.error(f"计算资源利用率失败: {e}")
            return 0.0

    async def _calculate_queue_efficiency(self, performance_data: List[Dict[str, Any]]) -> float:
        """计算任务队列效率"""
        try:
            wait_times = [d.get('queue_wait_time', 0) for d in performance_data]

            if wait_times:
                avg_wait_time = statistics.mean(wait_times)
                # 效率 = 1 - (平均等待时间 / 最大可接受等待时间)
                max_acceptable_wait = 30.0  # 30秒
                efficiency = max(0, 1 - (avg_wait_time / max_acceptable_wait))
                return efficiency * 100

            return 0.0

        except Exception as e:
            logger.error(f"计算队列效率失败: {e}")
            return 0.0

    async def _calculate_cache_hit_rate(self, performance_data: List[Dict[str, Any]]) -> float:
        """计算缓存命中率"""
        try:
            cache_hits = sum(1 for d in performance_data if d.get('cache_hit', False))
            total_requests = len(performance_data)

            if total_requests > 0:
                return (cache_hits / total_requests) * 100

            return 0.0

        except Exception as e:
            logger.error(f"计算缓存命中率失败: {e}")
            return 0.0

    async def _calculate_availability(self, system_logs: List[Dict[str, Any]]) -> float:
        """计算系统可用性"""
        try:
            total_time = 0.0
            downtime = 0.0

            for log_entry in system_logs:
                duration = log_entry.get('duration', 0)
                total_time += duration

                if log_entry.get('status') == 'down':
                    downtime += duration

            if total_time > 0:
                return ((total_time - downtime) / total_time) * 100

            return 100.0

        except Exception as e:
            logger.error(f"计算系统可用性失败: {e}")
            return 0.0

    async def _calculate_error_rate(self, system_logs: List[Dict[str, Any]]) -> float:
        """计算错误率"""
        try:
            total_requests = len(system_logs)
            error_requests = sum(1 for log in system_logs if log.get('status') == 'error')

            if total_requests > 0:
                return (error_requests / total_requests) * 100

            return 0.0

        except Exception as e:
            logger.error(f"计算错误率失败: {e}")
            return 0.0

    async def _calculate_mttr(self, system_logs: List[Dict[str, Any]]) -> float:
        """计算平均故障恢复时间"""
        try:
            recovery_times = []

            for i in range(1, len(system_logs)):
                prev_log = system_logs[i-1]
                curr_log = system_logs[i]

                if prev_log.get('status') == 'down' and curr_log.get('status') == 'up':
                    # 计算恢复时间（分钟）
                    downtime = curr_log.get('timestamp', datetime.now()) - prev_log.get('timestamp', datetime.now())
                    recovery_times.append(downtime.total_seconds() / 60)

            return statistics.mean(recovery_times) if recovery_times else 0.0

        except Exception as e:
            logger.error(f"计算MTTR失败: {e}")
            return 0.0

    async def _calculate_mtbf(self, system_logs: List[Dict[str, Any]]) -> float:
        """计算平均无故障时间"""
        try:
            uptime_periods = []

            for i in range(1, len(system_logs)):
                prev_log = system_logs[i-1]
                curr_log = system_logs[i]

                if prev_log.get('status') == 'up' and curr_log.get('status') == 'down':
                    # 计算正常运行时间（小时）
                    uptime = curr_log.get('timestamp', datetime.now()) - prev_log.get('timestamp', datetime.now())
                    uptime_periods.append(uptime.total_seconds() / 3600)

            return statistics.mean(uptime_periods) if uptime_periods else 0.0

        except Exception as e:
            logger.error(f"计算MTBF失败: {e}")
            return 0.0

    async def _calculate_user_satisfaction(self, system_logs: List[Dict[str, Any]]) -> float:
        """计算用户满意度"""
        try:
            satisfaction_scores = []

            for log in system_logs:
                score = log.get('user_satisfaction_score')
                if score is not None:
                    satisfaction_scores.append(score)

            return statistics.mean(satisfaction_scores) if satisfaction_scores else 0.0

        except Exception as e:
            logger.error(f"计算用户满意度失败: {e}")
            return 0.0

    def _get_metric_status(self, value: float, threshold: float) -> str:
        """获取指标状态"""
        if value >= threshold:
            return "normal"
        elif value >= threshold * 0.8:
            return "warning"
        else:
            return "critical"

    def _get_metric_status_efficiency(self, value: float, threshold: float, lower_is_better: bool) -> str:
        """获取效率指标状态"""
        if lower_is_better:
            # 值越低越好
            if value <= threshold:
                return "normal"
            elif value <= threshold * 1.2:
                return "warning"
            else:
                return "critical"
        else:
            # 值越高越好
            if value >= threshold:
                return "normal"
            elif value >= threshold * 0.8:
                return "warning"
            else:
                return "critical"

    async def calculate_all_metrics(self,
                                   test_results: List[Dict[str, Any]] = None,
                                   performance_data: List[Dict[str, Any]] = None,
                                   system_logs: List[Dict[str, Any]] = None) -> List[MetricResult]:
        """
        计算所有维度指标
        """
        all_metrics = []

        # 计算准确度指标
        if test_results:
            accuracy_metrics = await self.calculate_accuracy_metrics(test_results)
            all_metrics.extend(accuracy_metrics)

        # 计算效率指标
        if performance_data:
            efficiency_metrics = await self.calculate_efficiency_metrics(performance_data)
            all_metrics.extend(efficiency_metrics)

        # 计算可用性指标
        if system_logs:
            availability_metrics = await self.calculate_availability_metrics(system_logs)
            all_metrics.extend(availability_metrics)

        # 保存指标历史
        self._save_metrics_history(all_metrics)

        return all_metrics

    def _save_metrics_history(self, metrics: List[MetricResult]):
        """保存指标历史"""
        try:
            timestamp = datetime.now()

            for metric in metrics:
                history_entry = {
                    'timestamp': timestamp.isoformat(),
                    'name': metric.name,
                    'category': metric.category.value,
                    'value': metric.value,
                    'status': metric.status
                }

                self.metrics_history.append(history_entry)

            # 保持最近1000条记录
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]

        except Exception as e:
            logger.error(f"保存指标历史失败: {e}")

    def get_metrics_summary(self, metrics: List[MetricResult]) -> Dict[str, Any]:
        """获取指标汇总"""
        try:
            # 按类别分组
            by_category = {}
            for metric in metrics:
                category = metric.category.value
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(metric)

            # 计算各类别平均分
            category_scores = {}
            for category, category_metrics in by_category.items():
                if category_metrics:
                    avg_score = sum(m.value for m in category_metrics) / len(category_metrics)
                    category_scores[category] = round(avg_score, 3)

            # 计算总体状态
            total_metrics = len(metrics)
            normal_count = sum(1 for m in metrics if m.status == 'normal')
            warning_count = sum(1 for m in metrics if m.status == 'warning')
            critical_count = sum(1 for m in metrics if m.status == 'critical')

            overall_status = "normal"
            if critical_count > 0:
                overall_status = "critical"
            elif warning_count > total_metrics * 0.3:
                overall_status = "warning"

            return {
                'total_metrics': total_metrics,
                'category_scores': category_scores,
                'status_counts': {
                    'normal': normal_count,
                    'warning': warning_count,
                    'critical': critical_count
                },
                'overall_status': overall_status,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取指标汇总失败: {e}")
            return {}

    def get_metrics_trend(self, metric_name: str, days: int = 7) -> Dict[str, Any]:
        """获取指标趋势"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)

            # 获取历史数据
            historical_data = [
                entry for entry in self.metrics_history
                if entry['name'] == metric_name and
                   datetime.fromisoformat(entry['timestamp']) >= cutoff_time
            ]

            if not historical_data:
                return {}

            # 计算趋势
            values = [entry['value'] for entry in historical_data]

            if len(values) >= 2:
                recent_avg = statistics.mean(values[-3:])  # 最近3个值的平均
                older_avg = statistics.mean(values[:-3]) if len(values) > 3 else values[0]

                trend = 'stable'
                if recent_avg > older_avg * 1.1:
                    trend = 'improving'
                elif recent_avg < older_avg * 0.9:
                    trend = 'declining'

                return {
                    'trend': trend,
                    'recent_average': recent_avg,
                    'older_average': older_avg,
                    'change_percentage': ((recent_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0,
                    'data_points': len(values)
                }

            return {
                'trend': 'insufficient_data',
                'data_points': len(values)
            }

        except Exception as e:
            logger.error(f"获取指标趋势失败: {e}")
            return {}