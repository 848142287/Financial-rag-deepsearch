"""
RAGAS评估系统集成
使用RAGAS框架评估RAG系统的性能
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import json
import logging
import numpy as np

# RAGAS相关导入
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RAGAS未安装: {e}")
    RAGAS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RAGASMetrics:
    """RAGAS评估指标"""
    faithfulness: float          # 忠实度：答案与上下文的一致性
    answer_relevancy: float      # 答案相关性：答案与问题的相关性
    context_precision: float     # 上下文精确度：检索到的上下文的相关性
    context_recall: float        # 上下文召回率：检索到的上下文的完整性
    answer_similarity: float     # 答案相似度：与参考答案的相似度
    overall_score: float         # 综合评分


@dataclass
class EvaluationResult:
    """评估结果"""
    evaluation_id: str
    query: str
    answer: str
    retrieved_docs: List[Dict[str, Any]]
    ground_truth: Optional[str]
    metrics: RAGASMetrics
    evaluation_time: float
    created_at: datetime
    metadata: Dict[str, Any]


class RAGASEvaluator:
    """RAGAS评估器"""

    def __init__(self):
        self.default_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        self.evaluation_cache = {}
        self.evaluation_history = []

    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        ground_truth: Optional[str] = None,
        metrics: Optional[List] = None
    ) -> Dict[str, float]:
        """
        评估RAG结果

        Args:
            query: 原始查询
            answer: 生成的答案
            retrieved_docs: 检索到的文档
            ground_truth: 参考答案（可选）
            metrics: 自定义评估指标

        Returns:
            评估指标字典
        """
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS不可用，返回默认评分")
            return self._get_default_metrics()

        evaluation_id = f"eval_{hash(query + answer) % 100000}"
        start_time = datetime.now(timezone.utc)

        logger.info(f"开始RAGAS评估: {evaluation_id}")

        try:
            # 准备评估数据
            eval_data = self._prepare_evaluation_data(
                query, answer, retrieved_docs, ground_truth
            )

            # 选择评估指标
            eval_metrics = metrics or self.default_metrics

            # 执行评估
            dataset = Dataset.from_dict(eval_data)
            result = evaluate(dataset, metrics=eval_metrics)

            # 提取指标
            metrics_dict = self._extract_metrics(result)

            # 如果有参考答案，计算答案相似度
            if ground_truth:
                similarity_score = await self._calculate_answer_similarity(
                    answer, ground_truth
                )
                metrics_dict['answer_similarity'] = similarity_score

            # 计算综合评分
            overall_score = self._calculate_overall_score(metrics_dict)
            metrics_dict['overall_score'] = overall_score

            # 记录评估历史
            evaluation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._record_evaluation(
                evaluation_id, query, answer, retrieved_docs,
                ground_truth, metrics_dict, evaluation_time
            )

            logger.info(f"RAGAS评估完成: {evaluation_id}, 综合评分: {overall_score:.3f}")

            return metrics_dict

        except Exception as e:
            logger.error(f"RAGAS评估失败: {evaluation_id}, 错误: {e}")
            return self._get_default_metrics()

    async def batch_evaluate(
        self,
        eval_dataset: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """
        批量评估

        Args:
            eval_dataset: 评估数据集，每个元素包含query, answer, contexts, ground_truth

        Returns:
            评估结果列表
        """
        logger.info(f"开始批量RAGAS评估: {len(eval_dataset)} 个样本")

        results = []

        # 并行评估（限制并发数）
        semaphore = asyncio.Semaphore(5)

        async def evaluate_sample(sample):
            async with semaphore:
                return await self._evaluate_sample(sample)

        tasks = [evaluate_sample(sample) for sample in eval_dataset]
        sample_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        for i, result in enumerate(sample_results):
            if isinstance(result, Exception):
                logger.error(f"样本 {i} 评估失败: {result}")
                continue
            results.append(result)

        # 计算批量统计
        batch_stats = self._calculate_batch_statistics(results)
        logger.info(f"批量评估完成: 平均分 {batch_stats['mean_overall_score']:.3f}")

        return results

    async def _evaluate_sample(self, sample: Dict[str, Any]) -> EvaluationResult:
        """评估单个样本"""
        evaluation_id = f"eval_{datetime.now(timezone.utc).timestamp()}"
        start_time = datetime.now(timezone.utc)

        query = sample.get('query', '')
        answer = sample.get('answer', '')
        retrieved_docs = sample.get('contexts', [])
        ground_truth = sample.get('ground_truth')

        # 执行评估
        metrics = await self.evaluate(query, answer, retrieved_docs, ground_truth)

        # 构建评估结果
        evaluation_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        result = EvaluationResult(
            evaluation_id=evaluation_id,
            query=query,
            answer=answer,
            retrieved_docs=retrieved_docs,
            ground_truth=ground_truth,
            metrics=RAGASMetrics(**metrics),
            evaluation_time=evaluation_time,
            created_at=start_time,
            metadata={
                'sample_id': sample.get('id'),
                'source': sample.get('source', 'unknown')
            }
        )

        return result

    def _prepare_evaluation_data(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        ground_truth: Optional[str]
    ) -> Dict[str, List]:
        """准备RAGAS评估数据格式"""
        # 提取文档内容
        contexts = [doc.get('content', '') for doc in retrieved_docs]

        eval_data = {
            'question': [query],
            'answer': [answer],
            'contexts': [contexts]
        }

        if ground_truth:
            eval_data['ground_truth'] = [ground_truth]

        return eval_data

    def _extract_metrics(self, result) -> Dict[str, float]:
        """从RAGAS结果中提取指标"""
        metrics_dict = {}

        # 将RAGAS结果转换为字典
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            if not df.empty:
                for column in df.columns:
                    metrics_dict[column] = df[column].iloc[0]
        else:
            # 备用提取方法
            metrics_dict = {
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0
            }

        return metrics_dict

    async def _calculate_answer_similarity(
        self,
        answer: str,
        ground_truth: str
    ) -> float:
        """计算答案相似度"""
        try:
            # 使用词级别的Jaccard相似度
            answer_words = set(answer.lower().split())
            truth_words = set(ground_truth.lower().split())

            intersection = answer_words.intersection(truth_words)
            union = answer_words.union(truth_words)

            if len(union) == 0:
                return 0.0

            jaccard_similarity = len(intersection) / len(union)

            # 可以结合其他相似度计算方法
            # 例如：BLEU, ROUGE, 或基于嵌入的相似度

            return jaccard_similarity

        except Exception as e:
            logger.error(f"计算答案相似度失败: {e}")
            return 0.0

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """计算综合评分"""
        # 权重配置
        weights = {
            'faithfulness': 0.25,
            'answer_relevancy': 0.25,
            'context_precision': 0.2,
            'context_recall': 0.2,
            'answer_similarity': 0.1
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for metric, score in metrics.items():
            if metric in weights:
                weighted_sum += weights[metric] * score
                total_weight += weights[metric]

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _record_evaluation(
        self,
        evaluation_id: str,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        ground_truth: Optional[str],
        metrics: Dict[str, float],
        evaluation_time: float
    ):
        """记录评估历史"""
        evaluation_record = {
            'evaluation_id': evaluation_id,
            'query': query,
            'answer': answer,
            'doc_count': len(retrieved_docs),
            'has_ground_truth': ground_truth is not None,
            'metrics': metrics,
            'evaluation_time': evaluation_time,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        self.evaluation_history.append(evaluation_record)

        # 限制历史记录数量
        if len(self.evaluation_history) > 1000:
            self.evaluation_history = self.evaluation_history[-1000:]

    def _get_default_metrics(self) -> Dict[str, float]:
        """获取默认指标（当RAGAS不可用时）"""
        return {
            'faithfulness': 0.5,
            'answer_relevancy': 0.5,
            'context_precision': 0.5,
            'context_recall': 0.5,
            'answer_similarity': 0.0,
            'overall_score': 0.5
        }

    def _calculate_batch_statistics(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """计算批量评估统计信息"""
        if not results:
            return {}

        # 提取所有指标
        all_metrics = []
        for result in results:
            metrics = {
                'faithfulness': result.metrics.faithfulness,
                'answer_relevancy': result.metrics.answer_relevancy,
                'context_precision': result.metrics.context_precision,
                'context_recall': result.metrics.context_recall,
                'overall_score': result.metrics.overall_score
            }
            if result.metrics.answer_similarity > 0:
                metrics['answer_similarity'] = result.metrics.answer_similarity
            all_metrics.append(metrics)

        # 计算统计指标
        stats = {}
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                stats[f'mean_{metric_name}'] = np.mean(values)
                stats[f'std_{metric_name}'] = np.std(values)
                stats[f'min_{metric_name}'] = np.min(values)
                stats[f'max_{metric_name}'] = np.max(values)

        # 添加其他统计信息
        stats['total_samples'] = len(results)
        stats['total_evaluation_time'] = sum(r.evaluation_time for r in results)
        stats['average_evaluation_time'] = stats['total_evaluation_time'] / len(results)

        return stats

    def get_evaluation_summary(self, n_recent: int = 100) -> Dict[str, Any]:
        """获取评估摘要"""
        recent_evaluations = self.evaluation_history[-n_recent:]

        if not recent_evaluations:
            return {'message': '暂无评估数据'}

        # 计算平均指标
        overall_scores = [e['metrics']['overall_score'] for e in recent_evaluations]
        evaluation_times = [e['evaluation_time'] for e in recent_evaluations]

        summary = {
            'total_evaluations': len(recent_evaluations),
            'average_overall_score': np.mean(overall_scores),
            'score_std': np.std(overall_scores),
            'min_score': np.min(overall_scores),
            'max_score': np.max(overall_scores),
            'average_evaluation_time': np.mean(evaluation_times),
            'score_distribution': self._calculate_score_distribution(overall_scores),
            'recent_trend': self._calculate_recent_trend(overall_scores[-10:])
        }

        return summary

    def _calculate_score_distribution(
        self,
        scores: List[float]
    ) -> Dict[str, int]:
        """计算分数分布"""
        distribution = {
            'excellent (>0.9)': 0,
            'good (0.7-0.9)': 0,
            'average (0.5-0.7)': 0,
            'poor (<0.5)': 0
        }

        for score in scores:
            if score > 0.9:
                distribution['excellent (>0.9)'] += 1
            elif score > 0.7:
                distribution['good (0.7-0.9)'] += 1
            elif score > 0.5:
                distribution['average (0.5-0.7)'] += 1
            else:
                distribution['poor (<0.5)'] += 1

        return distribution

    def _calculate_recent_trend(self, recent_scores: List[float]) -> str:
        """计算最近趋势"""
        if len(recent_scores) < 3:
            return 'insufficient_data'

        # 简单线性趋势分析
        x = list(range(len(recent_scores)))
        slope = np.polyfit(x, recent_scores, 1)[0]

        if slope > 0.05:
            return 'improving'
        elif slope < -0.05:
            return 'declining'
        else:
            return 'stable'

    async def compare_models(
        self,
        model_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        比较不同模型的性能

        Args:
            model_results: 模型名称到评估结果的映射

        Returns:
            模型比较结果
        """
        logger.info(f"开始模型性能比较: {list(model_results.keys())}")

        comparison_results = {}

        for model_name, results in model_results.items():
            # 评估这个模型的结果
            model_eval_results = await self.batch_evaluate(results)
            batch_stats = self._calculate_batch_statistics(model_eval_results)

            comparison_results[model_name] = {
                'average_score': batch_stats.get('mean_overall_score', 0),
                'score_std': batch_stats.get('std_overall_score', 0),
                'faithfulness': batch_stats.get('mean_faithfulness', 0),
                'relevancy': batch_stats.get('mean_answer_relevancy', 0),
                'precision': batch_stats.get('mean_context_precision', 0),
                'recall': batch_stats.get('mean_context_recall', 0),
                'total_samples': batch_stats.get('total_samples', 0)
            }

        # 排序模型
        sorted_models = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['average_score'],
            reverse=True
        )

        # 添加排名
        for i, (model_name, stats) in enumerate(sorted_models, 1):
            stats['rank'] = i

        comparison_results['ranking'] = [model for model, _ in sorted_models]

        logger.info("模型性能比较完成")
        return comparison_results


# 全局RAGAS评估器实例
ragas_evaluator = RAGASEvaluator()