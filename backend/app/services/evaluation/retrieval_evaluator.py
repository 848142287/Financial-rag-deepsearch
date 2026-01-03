"""
统一的检索评估器
整合原有的多个评估器,消除重复代码
"""

from typing import List, Dict, Any, Optional
import statistics
from app.core.structured_logging import get_structured_logger
from datetime import datetime

from .base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    RetrievalTestData,
    MetricType
)

logger = get_structured_logger(__name__)


class UnifiedRetrievalEvaluator(BaseEvaluator):
    """
    统一的检索评估器
    整合 ragas_evaluator.py 和 metrics_calculator.py 的功能
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_metrics = [
            MetricType.PRECISION,
            MetricType.RECALL,
            MetricType.F1_SCORE,
            MetricType.TOP_K_ACCURACY,
            MetricType.MRR,
            MetricType.NDCG,
        ]

    def get_supported_metrics(self) -> List[MetricType]:
        """获取支持的指标类型"""
        return self.supported_metrics

    async def evaluate(
        self,
        test_data: List[RetrievalTestData],
        k_list: List[int] = [1, 3, 5, 10],
        **kwargs
    ) -> List[EvaluationResult]:
        """
        执行检索评估

        Args:
            test_data: 测试数据列表
            k_list: 要评估的TopK值列表
            **kwargs: 其他参数

        Returns:
            评估结果列表
        """
        start_time = datetime.now()

        # 验证数据
        if not self.validate_test_data(test_data):
            logger.warning("测试数据验证失败")
            return []

        results = []

        # 对每个k值计算指标
        for k in k_list:
            k_results = await self._evaluate_at_k(test_data, k)
            results.extend(k_results)

        # 记录摘要
        duration = (datetime.now() - start_time).total_seconds()
        self.log_evaluation_summary(results, duration)

        return results

    async def _evaluate_at_k(
        self,
        test_data: List[RetrievalTestData],
        k: int
    ) -> List[EvaluationResult]:
        """
        在特定k值下评估

        Args:
            test_data: 测试数据
            k: TopK值

        Returns:
            评估结果
        """
        precisions = []
        recalls = []
        f1_scores = []
        top_k_accs = []
        mrrs = []
        ndcgs = []

        # 计算每个查询的指标
        for data in test_data:
            retrieved_ids = [doc.get("id", "") for doc in data.retrieved_docs[:k]]
            ground_truth_ids = data.ground_truth_docs

            # 计算指标
            precision, recall = self.calculate_precision_recall(
                retrieved_ids, ground_truth_ids, k
            )
            f1 = self.calculate_f1_score(precision, recall)
            top_k_acc = self.calculate_top_k_accuracy(
                retrieved_ids, ground_truth_ids, k
            )
            mrr = self.calculate_mrr(retrieved_ids, ground_truth_ids)
            ndcg = self.calculate_ndcg(retrieved_ids, ground_truth_ids, k)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            top_k_accs.append(top_k_acc)
            mrrs.append(mrr)
            ndcgs.append(ndcg)

        # 计算平均值
        avg_precision = statistics.mean(precisions) if precisions else 0.0
        avg_recall = statistics.mean(recalls) if recalls else 0.0
        avg_f1 = statistics.mean(f1_scores) if f1_scores else 0.0
        avg_top_k = statistics.mean(top_k_accs) if top_k_accs else 0.0
        avg_mrr = statistics.mean(mrrs) if mrrs else 0.0
        avg_ndcg = statistics.mean(ndcgs) if ndcgs else 0.0

        # 创建结果对象
        results = [
            EvaluationResult(
                metric_name=f"top_{k}_precision",
                metric_value=avg_precision,
                metric_type=MetricType.PRECISION,
                threshold=self.get_threshold(MetricType.PRECISION),
                metadata={"k": k, "count": len(test_data)}
            ),
            EvaluationResult(
                metric_name=f"top_{k}_recall",
                metric_value=avg_recall,
                metric_type=MetricType.RECALL,
                threshold=self.get_threshold(MetricType.RECALL),
                metadata={"k": k, "count": len(test_data)}
            ),
            EvaluationResult(
                metric_name=f"top_{k}_f1",
                metric_value=avg_f1,
                metric_type=MetricType.F1_SCORE,
                threshold=self.get_threshold(MetricType.F1_SCORE),
                metadata={"k": k, "count": len(test_data)}
            ),
            EvaluationResult(
                metric_name=f"top_{k}_accuracy",
                metric_value=avg_top_k,
                metric_type=MetricType.TOP_K_ACCURACY,
                threshold=self.get_threshold(MetricType.TOP_K_ACCURACY),
                metadata={"k": k, "count": len(test_data)}
            ),
            EvaluationResult(
                metric_name=f"top_{k}_mrr",
                metric_value=avg_mrr,
                metric_type=MetricType.MRR,
                threshold=self.get_threshold(MetricType.MRR),
                metadata={"k": k, "count": len(test_data)}
            ),
            EvaluationResult(
                metric_name=f"top_{k}_ndcg",
                metric_value=avg_ndcg,
                metric_type=MetricType.NDCG,
                threshold=self.get_threshold(MetricType.NDCG),
                metadata={"k": k, "count": len(test_data)}
            ),
        ]

        return results

    def get_metrics_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        获取指标摘要

        Args:
            results: 评估结果列表

        Returns:
            摘要字典
        """
        # 按k值分组
        by_k = {}
        for result in results:
            k = result.metadata.get("k", 5) if result.metadata else 5
            if k not in by_k:
                by_k[k] = {}
            by_k[k][result.metric_type] = result

        # 计算整体统计
        total_metrics = len(results)
        excellent = sum(1 for r in results if r.status == "excellent")
        good = sum(1 for r in results if r.status == "good")
        warning = sum(1 for r in results if r.status == "warning")
        critical = sum(1 for r in results if r.status == "critical")

        return {
            "total_metrics": total_metrics,
            "by_k": by_k,
            "status_distribution": {
                "excellent": excellent,
                "good": good,
                "warning": warning,
                "critical": critical
            },
            "success_rate": (excellent + good) / total_metrics if total_metrics > 0 else 0.0
        }


class RagasQualityEvaluator(BaseEvaluator):
    """
    RAGAS质量评估器
    用于评估生成质量(忠实度、答案相关性等)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_metrics = [
            MetricType.FAITHFULNESS,
            MetricType.ANSWER_RELEVANCY,
            MetricType.CONTEXT_PRECISION,
            MetricType.CONTEXT_RECALL,
        ]

        # 尝试导入RAGAS
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
            self.RAGAS_AVAILABLE = True
            self.evaluate_ragas = evaluate
            self.ragas_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            self.Dataset = Dataset
            logger.info("RAGAS库可用")
        except ImportError:
            self.RAGAS_AVAILABLE = False
            logger.warning("RAGAS库未安装,相关评估将跳过")

    def get_supported_metrics(self) -> List[MetricType]:
        """获取支持的指标类型"""
        return self.supported_metrics

    async def evaluate(
        self,
        test_data: List[RetrievalTestData],
        answers: List[str],
        **kwargs
    ) -> List[EvaluationResult]:
        """
        使用RAGAS评估生成质量

        Args:
            test_data: 测试数据列表
            answers: 生成的答案列表
            **kwargs: 其他参数

        Returns:
            评估结果列表
        """
        if not self.RAGAS_AVAILABLE:
            logger.warning("RAGAS不可用,跳过评估")
            return []

        if len(test_data) != len(answers):
            logger.error(f"数据长度不匹配: test_data={len(test_data)}, answers={len(answers)}")
            return []

        start_time = datetime.now()

        try:
            # 准备RAGAS数据集
            data = {
                "question": [d.query for d in test_data],
                "answer": answers,
                "contexts": [d.retrieved_contexts for d in test_data],
                "ground_truth": [d.ground_truth_answer for d in test_data],
            }

            dataset = self.Dataset.from_dict(data)

            # 运行评估
            result = self.evaluate_ragas(
                dataset=dataset,
                metrics=self.ragas_metrics
            )

            # 转换为统一格式
            results = [
                EvaluationResult(
                    metric_name="faithfulness",
                    metric_value=result.get("faithfulness", 0.0),
                    metric_type=MetricType.FAITHFULNESS,
                    threshold=self.get_threshold(MetricType.FAITHFULNESS)
                ),
                EvaluationResult(
                    metric_name="answer_relevancy",
                    metric_value=result.get("answer_relevancy", 0.0),
                    metric_type=MetricType.ANSWER_RELEVANCY,
                    threshold=self.get_threshold(MetricType.ANSWER_RELEVANCY)
                ),
                EvaluationResult(
                    metric_name="context_precision",
                    metric_value=result.get("context_precision", 0.0),
                    metric_type=MetricType.CONTEXT_PRECISION,
                    threshold=self.get_threshold(MetricType.CONTEXT_PRECISION)
                ),
                EvaluationResult(
                    metric_name="context_recall",
                    metric_value=result.get("context_recall", 0.0),
                    metric_type=MetricType.CONTEXT_RECALL,
                    threshold=self.get_threshold(MetricType.CONTEXT_RECALL)
                ),
            ]

            # 记录摘要
            duration = (datetime.now() - start_time).total_seconds()
            self.log_evaluation_summary(results, duration)

            return results

        except Exception as e:
            logger.error(f"RAGAS评估失败: {e}")
            return []
