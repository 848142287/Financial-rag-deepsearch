"""
评估指标体系
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from app.core.structured_logging import get_structured_logger
from datetime import datetime
import re

logger = get_structured_logger(__name__)


class MetricDimension(Enum):
    """评估维度"""
    ACCURACY = "accuracy"      # 准确度
    EFFICIENCY = "efficiency"  # 效率
    AVAILABILITY = "availability"  # 可用性
    RELEVANCE = "relevance"    # 相关性
    CONSISTENCY = "consistency"  # 一致性


class MetricType(Enum):
    """指标类型"""
    RECALL_PRECISION = "recall_precision"  # 召回准确率
    ANSWER_RELEVANCE = "answer_relevance"  # 答案相关性
    FACTUAL_CONSISTENCY = "factual_consistency"  # 事实一致性
    CITATION_ACCURACY = "citation_accuracy"  # 引用准确率
    RESPONSE_TIME = "response_time"  # 响应时间
    THROUGHPUT = "throughput"  # 吞吐量
    RESOURCE_UTILIZATION = "resource_utilization"  # 资源利用率
    SYSTEM_AVAILABILITY = "system_availability"  # 系统可用性
    ERROR_RATE = "error_rate"  # 错误率
    USER_SATISFACTION = "user_satisfaction"  # 用户满意度


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str
    type: MetricType
    dimension: MetricDimension
    description: str
    unit: str
    higher_is_better: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    thresholds: Optional[Dict[str, float]] = None  # good, warning, critical


@dataclass
class MetricResult:
    """指标结果"""
    metric_name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class EvaluationReport:
    """评估报告"""
    evaluation_id: str
    timestamp: datetime
    metrics: List[MetricResult]
    overall_score: float
    dimension_scores: Dict[str, float]
    summary: str
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['metrics'] = [m.to_dict() for m in self.metrics]
        return data


class MetricsCalculator:
    """指标计算器"""

    def __init__(self):
        # 指标定义
        self.metric_definitions = {
            MetricType.RECALL_PRECISION: MetricDefinition(
                name="Recall-Precision Score",
                type=MetricType.RECALL_PRECISION,
                dimension=MetricDimension.ACCURACY,
                description="检索结果的准确率和召回率综合评分",
                unit="score",
                thresholds={"good": 0.8, "warning": 0.6, "critical": 0.4}
            ),
            MetricType.ANSWER_RELEVANCE: MetricDefinition(
                name="Answer Relevance",
                type=MetricType.ANSWER_RELEVANCE,
                dimension=MetricDimension.RELEVANCE,
                description="生成答案与查询的相关性",
                unit="score",
                thresholds={"good": 0.8, "warning": 0.6, "critical": 0.4}
            ),
            MetricType.FACTUAL_CONSISTENCY: MetricDefinition(
                name="Factual Consistency",
                type=MetricType.FACTUAL_CONSISTENCY,
                dimension=MetricDimension.CONSISTENCY,
                description="答案与源信息的事实一致性",
                unit="score",
                thresholds={"good": 0.9, "warning": 0.7, "critical": 0.5}
            ),
            MetricType.CITATION_ACCURACY: MetricDefinition(
                name="Citation Accuracy",
                type=MetricType.CITATION_ACCURACY,
                dimension=MetricDimension.ACCURACY,
                description="引用的准确性和完整性",
                unit="score",
                thresholds={"good": 0.9, "warning": 0.7, "critical": 0.5}
            ),
            MetricType.RESPONSE_TIME: MetricDefinition(
                name="Response Time",
                type=MetricType.RESPONSE_TIME,
                dimension=MetricDimension.EFFICIENCY,
                description="系统响应时间（P99）",
                unit="ms",
                higher_is_better=False,
                thresholds={"good": 1000, "warning": 3000, "critical": 5000}
            ),
            MetricType.THROUGHPUT: MetricDefinition(
                name="Throughput",
                type=MetricType.THROUGHPUT,
                dimension=MetricDimension.EFFICIENCY,
                description="系统吞吐量（每秒请求数）",
                unit="req/s",
                thresholds={"good": 100, "warning": 50, "critical": 20}
            ),
            MetricType.SYSTEM_AVAILABILITY: MetricDefinition(
                name="System Availability",
                type=MetricType.SYSTEM_AVAILABILITY,
                dimension=MetricDimension.AVAILABILITY,
                description="系统可用性（SLA）",
                unit="%",
                thresholds={"good": 99.9, "warning": 99.5, "critical": 99.0}
            ),
            MetricType.ERROR_RATE: MetricDefinition(
                name="Error Rate",
                type=MetricType.ERROR_RATE,
                dimension=MetricDimension.AVAILABILITY,
                description="系统错误率",
                unit="%",
                higher_is_better=False,
                thresholds={"good": 0.1, "warning": 1.0, "critical": 5.0}
            ),
            MetricType.USER_SATISFACTION: MetricDefinition(
                name="User Satisfaction",
                type=MetricType.USER_SATISFACTION,
                dimension=MetricDimension.AVAILABILITY,
                description="用户满意度评分",
                unit="score",
                thresholds={"good": 4.5, "warning": 3.5, "critical": 2.5}
            )
        }

    def calculate_recall_precision(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """计算召回准确率"""
        if not relevant_docs:
            return 0.0

        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)

        # 计算准确率和召回率
        true_positives = len(retrieved_set & relevant_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0

        # 计算F1分数
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def calculate_answer_relevance(self, query: str, answer: str, context: str = "") -> float:
        """计算答案相关性（简化版，实际应用中可以使用NLP模型）"""
        # 基于关键词重叠度
        query_words = set(self._extract_keywords(query))
        answer_words = set(self._extract_keywords(answer))

        if not query_words:
            return 0.0

        # 计算Jaccard相似度
        intersection = len(query_words & answer_words)
        union = len(query_words | answer_words)

        base_score = intersection / union if union else 0

        # 长度惩罚（答案过长或过短都降低评分）
        answer_length = len(answer.split())
        optimal_length = len(query.split()) * 3  # 理想答案长度
        length_penalty = 1.0 - min(abs(answer_length - optimal_length) / optimal_length, 0.3)

        return base_score * length_penalty

    def calculate_factual_consistency(self, answer: str, source_documents: List[str]) -> float:
        """计算事实一致性"""
        if not source_documents:
            return 0.0

        # 提取答案中的事实陈述
        answer_facts = self._extract_facts(answer)
        if not answer_facts:
            return 0.5  # 中性评分

        # 检查每个事实是否在源文档中得到支持
        supported_facts = 0
        for fact in answer_facts:
            if self._is_fact_supported(fact, source_documents):
                supported_facts += 1

        return supported_facts / len(answer_facts)

    def calculate_citation_accuracy(self, answer: str, citations: List[Dict], source_documents: List[Dict]) -> float:
        """计算引用准确率"""
        if not citations:
            return 0.0

        accurate_citations = 0
        for citation in citations:
            if self._validate_citation(citation, source_documents):
                accurate_citations += 1

        return accurate_citations / len(citations)

    def calculate_response_time_percentiles(self, response_times: List[float]) -> Dict[str, float]:
        """计算响应时间百分位数"""
        if not response_times:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0}

        sorted_times = sorted(response_times)
        n = len(sorted_times)

        return {
            "p50": sorted_times[int(n * 0.5)],
            "p90": sorted_times[int(n * 0.9)],
            "p95": sorted_times[int(n * 0.95)],
            "p99": sorted_times[min(int(n * 0.99), n - 1)]
        }

    def calculate_throughput(self, total_requests: int, time_window_seconds: float) -> float:
        """计算吞吐量"""
        if time_window_seconds <= 0:
            return 0.0
        return total_requests / time_window_seconds

    def calculate_availability(self, total_time: float, downtime: float) -> float:
        """计算系统可用性"""
        if total_time <= 0:
            return 0.0
        uptime = total_time - downtime
        return (uptime / total_time) * 100

    def calculate_error_rate(self, total_requests: int, error_requests: int) -> float:
        """计算错误率"""
        if total_requests <= 0:
            return 0.0
        return (error_requests / total_requests) * 100

    def calculate_user_satisfaction(self, ratings: List[int]) -> float:
        """计算用户满意度"""
        if not ratings:
            return 0.0
        return sum(ratings) / len(ratings)

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取（实际应用中可以使用更复杂的NLP方法）
        # 移除标点符号，转换为小写，过滤停用词
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()

        # 简单的停用词列表
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'
        }

        return [word for word in words if word not in stop_words and len(word) > 1]

    def _extract_facts(self, text: str) -> List[str]:
        """提取事实陈述"""
        # 简化版事实提取（实际应用中应该使用更复杂的NLP方法）
        facts = []
        sentences = re.split(r'[.。！？]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # 过滤太短的句子
                # 检查是否包含数字、日期、专有名词等事实性内容
                if (re.search(r'\d+', sentence) or  # 包含数字
                    any(word in sentence for word in ['公司', '股票', '价格', '营收', '利润', '市值'])):  # 包含金融术语
                    facts.append(sentence)

        return facts

    def _is_fact_supported(self, fact: str, source_documents: List[str]) -> bool:
        """检查事实是否在源文档中得到支持"""
        fact_words = set(self._extract_keywords(fact))

        for doc in source_documents:
            doc_words = set(self._extract_keywords(doc))
            # 计算词汇重叠度
            overlap = len(fact_words & doc_words)
            if overlap >= len(fact_words) * 0.6:  # 60%以上重叠则认为支持
                return True

        return False

    def _validate_citation(self, citation: Dict, source_documents: List[Dict]) -> bool:
        """验证引用的准确性"""
        # 检查引用ID是否存在
        doc_id = citation.get('document_id')
        if not doc_id:
            return False

        # 检查文档是否存在
        return any(doc.get('id') == doc_id for doc in source_documents)


class MetricsAggregator:
    """指标聚合器"""

    def __init__(self):
        self.calculator = MetricsCalculator()

    def aggregate_metrics(self, metric_results: List[MetricResult]) -> Dict[str, Any]:
        """聚合指标结果"""
        if not metric_results:
            return {}

        # 按指标类型分组
        grouped_metrics = {}
        for result in metric_results:
            metric_type = result.metric_name
            if metric_type not in grouped_metrics:
                grouped_metrics[metric_type] = []
            grouped_metrics[metric_type].append(result.value)

        # 计算统计信息
        aggregated = {}
        for metric_type, values in grouped_metrics.items():
            aggregated[metric_type] = {
                "count": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p25": np.percentile(values, 25),
                "p75": np.percentile(values, 75)
            }

        return aggregated

    def calculate_overall_score(self, metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
        """计算总体评分"""
        if not metrics:
            return 0.0

        # 默认权重
        default_weights = {
            "recall_precision": 0.25,
            "answer_relevance": 0.20,
            "factual_consistency": 0.20,
            "citation_accuracy": 0.15,
            "response_time": 0.10,
            "user_satisfaction": 0.10
        }

        if weights:
            default_weights.update(weights)

        # 归一化指标值
        normalized_metrics = {}
        for metric_name, value in metrics.items():
            definition = self.calculator.metric_definitions.get(MetricType(metric_name))
            if definition:
                # 反向指标需要反转
                if not definition.higher_is_better:
                    # 简单的反转（实际应用中需要更复杂的归一化）
                    max_val = definition.thresholds.get("critical", 1.0)
                    normalized_value = max(0, 1 - (value / max_val))
                else:
                    normalized_value = min(1.0, value)

                normalized_metrics[metric_name] = normalized_value

        # 计算加权平均
        total_score = 0.0
        total_weight = 0.0

        for metric_name, normalized_value in normalized_metrics.items():
            weight = default_weights.get(metric_name, 0.1)
            total_score += normalized_value * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def generate_dimension_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """生成分维度评分"""
        dimension_metrics = {
            MetricDimension.ACCURACY.value: [],
            MetricDimension.EFFICIENCY.value: [],
            MetricDimension.AVAILABILITY.value: [],
            MetricDimension.RELEVANCE.value: [],
            MetricDimension.CONSISTENCY.value: []
        }

        # 按维度分组指标
        for metric_type, definition in self.calculator.metric_definitions.items():
            metric_name = metric_type.value
            if metric_name in metrics:
                dimension_metrics[definition.dimension.value].append(metrics[metric_name])

        # 计算每个维度的平均分
        dimension_scores = {}
        for dimension, values in dimension_metrics.items():
            if values:
                dimension_scores[dimension] = np.mean(values)
            else:
                dimension_scores[dimension] = 0.0

        return dimension_scores


class MetricsRegistry:
    """指标注册表"""

    def __init__(self):
        self.metrics = []
        self.evaluations = []

    def register_metric(self, metric_result: MetricResult):
        """注册指标结果"""
        self.metrics.append(metric_result)

    def register_evaluation(self, evaluation_report: EvaluationReport):
        """注册评估报告"""
        self.evaluations.append(evaluation_report)

    def get_metrics_by_type(self, metric_type: str, limit: Optional[int] = None) -> List[MetricResult]:
        """根据类型获取指标"""
        metrics = [m for m in self.metrics if m.metric_name == metric_type]
        if limit:
            return metrics[-limit:]
        return metrics

    def get_latest_metrics(self, limit: int = 100) -> List[MetricResult]:
        """获取最新的指标结果"""
        return sorted(self.metrics, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_evaluation_history(self, limit: int = 50) -> List[EvaluationReport]:
        """获取评估历史"""
        return sorted(self.evaluations, key=lambda x: x.timestamp, reverse=True)[:limit]

    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """清理旧指标数据"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_date]
        self.evaluations = [e for e in self.evaluations if e.timestamp > cutoff_date]


# 全局指标注册表
metrics_registry = MetricsRegistry()