"""
基础评估器抽象类
定义评估器的统一接口和通用方法
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class MetricType(Enum):
    """指标类型枚举"""
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    TOP_K_ACCURACY = "top_k_accuracy"
    MRR = "mrr"
    NDCG = "ndcg"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"


@dataclass
class EvaluationResult:
    """统一评估结果"""
    metric_name: str
    metric_value: float
    metric_type: MetricType
    threshold: Optional[float] = None
    status: str = "unknown"  # excellent, good, warning, critical
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """计算状态"""
        if self.threshold is not None:
            if self.metric_value >= self.threshold:
                self.status = "excellent" if self.metric_value >= self.threshold * 1.1 else "good"
            elif self.metric_value >= self.threshold * 0.8:
                self.status = "warning"
            else:
                self.status = "critical"


@dataclass
class RetrievalTestData:
    """检索测试数据"""
    query_id: str
    query: str
    retrieved_docs: List[Dict[str, Any]]
    retrieved_contexts: List[str]
    ground_truth_docs: List[str]
    ground_truth_answer: str
    retrieval_latency: float
    metadata: Optional[Dict[str, Any]] = None


class BaseEvaluator(ABC):
    """
    基础评估器抽象类
    所有评估器都应该继承这个类并实现抽象方法
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化评估器

        Args:
            config: 评估器配置
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # 默认阈值配置
        self.default_thresholds = {
            MetricType.PRECISION: 0.85,
            MetricType.RECALL: 0.85,
            MetricType.F1_SCORE: 0.80,
            MetricType.TOP_K_ACCURACY: 0.85,
            MetricType.MRR: 0.80,
            MetricType.NDCG: 0.80,
            MetricType.FAITHFULNESS: 0.90,
            MetricType.ANSWER_RELEVANCY: 0.85,
        }

    @abstractmethod
    async def evaluate(
        self,
        test_data: List[RetrievalTestData],
        **kwargs
    ) -> List[EvaluationResult]:
        """
        执行评估

        Args:
            test_data: 测试数据列表
            **kwargs: 其他参数

        Returns:
            评估结果列表
        """
        pass

    @abstractmethod
    def get_supported_metrics(self) -> List[MetricType]:
        """
        获取支持的指标类型

        Returns:
            支持的指标类型列表
        """
        pass

    # ==================== 通用指标计算方法 ====================

    def calculate_precision_recall(
        self,
        retrieved_docs: List[str],
        ground_truth_docs: List[str],
        k: int = 5
    ) -> Tuple[float, float]:
        """
        计算准确率和召回率

        Args:
            retrieved_docs: 检索到的文档ID列表
            ground_truth_docs: 真实相关的文档ID列表
            k: 考虑前K个结果

        Returns:
            (precision, recall)
        """
        if not retrieved_docs or not ground_truth_docs:
            return 0.0, 0.0

        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(ground_truth_docs)

        precision = len(retrieved_k & relevant_set) / len(retrieved_k) if retrieved_k else 0.0
        recall = len(retrieved_k & relevant_set) / len(relevant_set) if relevant_set else 0.0

        return precision, recall

    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        计算F1分数

        Args:
            precision: 准确率
            recall: 召回率

        Returns:
            F1分数
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_top_k_accuracy(
        self,
        retrieved_docs: List[str],
        ground_truth_docs: List[str],
        k: int = 5
    ) -> float:
        """
        计算TopK命中率

        Args:
            retrieved_docs: 检索到的文档ID列表
            ground_truth_docs: 真实相关的文档ID列表
            k: TopK

        Returns:
            TopK命中率 (0或1)
        """
        if not retrieved_docs or not ground_truth_docs:
            return 0.0

        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(ground_truth_docs)

        return 1.0 if (retrieved_k & relevant_set) else 0.0

    def calculate_mrr(
        self,
        retrieved_docs: List[str],
        ground_truth_docs: List[str]
    ) -> float:
        """
        计算平均倒数排名 (MRR)

        Args:
            retrieved_docs: 检索到的文档ID列表
            ground_truth_docs: 真实相关的文档ID列表

        Returns:
            MRR分数
        """
        if not retrieved_docs or not ground_truth_docs:
            return 0.0

        relevant_set = set(ground_truth_docs)

        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_set:
                return 1.0 / i

        return 0.0

    def calculate_ndcg(
        self,
        retrieved_docs: List[str],
        ground_truth_docs: List[str],
        k: int = 5
    ) -> float:
        """
        计算NDCG (Normalized Discounted Cumulative Gain)

        Args:
            retrieved_docs: 检索到的文档ID列表
            ground_truth_docs: 真实相关的文档ID列表
            k: TopK

        Returns:
            NDCG分数
        """
        if not retrieved_docs or not ground_truth_docs:
            return 0.0

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k], 1):
            if doc_id in ground_truth_docs:
                dcg += 1.0 / (i + 1)

        # IDCG
        idcg = 0.0
        for i in range(min(k, len(ground_truth_docs))):
            idcg += 1.0 / (i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    def calculate_average_metrics(
        self,
        metrics_list: List[List[float]]
    ) -> List[float]:
        """
        计算平均指标

        Args:
            metrics_list: 指标列表的列表

        Returns:
            平均指标列表
        """
        if not metrics_list:
            return []

        num_metrics = len(metrics_list[0])
        averages = []

        for i in range(num_metrics):
            values = [m[i] for m in metrics_list if i < len(m)]
            avg = sum(values) / len(values) if values else 0.0
            averages.append(avg)

        return averages

    def get_threshold(self, metric_type: MetricType) -> float:
        """
        获取指标阈值

        Args:
            metric_type: 指标类型

        Returns:
            阈值
        """
        return self.config.get(
            f"{metric_type.value}_threshold",
            self.default_thresholds.get(metric_type, 0.8)
        )

    def validate_test_data(self, test_data: List[RetrievalTestData]) -> bool:
        """
        验证测试数据

        Args:
            test_data: 测试数据

        Returns:
            是否有效
        """
        if not test_data:
            self.logger.warning("测试数据为空")
            return False

        for data in test_data:
            if not data.query:
                self.logger.warning(f"查询为空: {data.query_id}")
                return False
            if not data.retrieved_docs:
                self.logger.warning(f"检索结果为空: {data.query_id}")
                return False

        return True

    def log_evaluation_summary(
        self,
        results: List[EvaluationResult],
        duration: float
    ):
        """
        记录评估摘要

        Args:
            results: 评估结果
            duration: 评估耗时
        """
        summary = {
            "total_metrics": len(results),
            "excellent": sum(1 for r in results if r.status == "excellent"),
            "good": sum(1 for r in results if r.status == "good"),
            "warning": sum(1 for r in results if r.status == "warning"),
            "critical": sum(1 for r in results if r.status == "critical"),
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"评估完成: {summary}")
