"""
RAGAS评估框架
用于评估RAG检索质量和生成质量
"""

import os
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: RAGAS not installed. Install with: pip install ragas")


@dataclass
class RetrievalResult:
    """单个检索结果"""
    query_id: str
    query: str
    retrieved_docs: List[Dict[str, Any]]
    retrieved_contexts: List[str]
    ground_truth_docs: List[str]
    ground_truth_answer: str
    retrieval_latency: float


@dataclass
class EvaluationMetrics:
    """评估指标"""
    precision: float  # 准确率
    recall: float  # 召回率
    f1_score: float  # F1分数
    top_k_accuracy: float  # TopK命中率
    mrr: float  # 平均倒数排名
    ndcg: float  # NDCG
    faithfulness: float  # 忠实度
    answer_relevancy: float  # 答案相关性
    context_precision: float  # 上下文精确度
    context_recall: float  # 上下文召回率


class RAGASEvaluator:
    """RAGAS评估器"""

    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 目标指标
        self.target_metrics = {
            "precision": 0.85,
            "recall": 0.85,
            "top_k_accuracy": 0.85,
            "faithfulness": 0.90,
            "answer_relevancy": 0.85,
        }

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

        # 取前K个检索结果
        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(ground_truth_docs)

        # 计算准确率：检索结果中相关文档的比例
        precision = len(retrieved_k & relevant_set) / len(retrieved_k) if retrieved_k else 0.0

        # 计算召回率：相关文档被检索到的比例
        recall = len(retrieved_k & relevant_set) / len(relevant_set) if relevant_set else 0.0

        return precision, recall

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
            ground_truth_docs: 真实相关的文档ID列表 (带相关性分数)
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
                # 相关文档的增益为1
                dcg += 1.0 / (i + 1)

        # IDCG (理想情况：所有相关文档排在最前面)
        idcg = 0.0
        for i in range(min(k, len(ground_truth_docs))):
            idcg += 1.0 / (i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_retrieval(
        self,
        results: List[RetrievalResult],
        k_list: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        评估检索结果

        Args:
            results: 检索结果列表
            k_list: 要评估的TopK值列表

        Returns:
            评估指标字典
        """
        metrics_summary = {}

        for k in k_list:
            precisions = []
            recalls = []
            top_k_accs = []
            mrrs = []
            ndcgs = []

            for result in results:
                # 提取文档ID
                retrieved_ids = [doc.get("id") for doc in result.retrieved_docs[:k]]
                ground_truth_ids = result.ground_truth_docs

                # 计算指标
                precision, recall = self.calculate_precision_recall(
                    retrieved_ids, ground_truth_ids, k
                )
                top_k_acc = self.calculate_top_k_accuracy(
                    retrieved_ids, ground_truth_ids, k
                )
                mrr = self.calculate_mrr(retrieved_ids, ground_truth_ids)
                ndcg = self.calculate_ndcg(retrieved_ids, ground_truth_ids, k)

                precisions.append(precision)
                recalls.append(recall)
                top_k_accs.append(top_k_acc)
                mrrs.append(mrr)
                ndcgs.append(ndcg)

            # 计算平均值
            metrics_summary[f"top_{k}"] = {
                "precision": sum(precisions) / len(precisions),
                "recall": sum(recalls) / len(recalls),
                "f1_score": 2 * (sum(precisions) / len(precisions)) * (sum(recalls) / len(recalls)) /
                           ((sum(precisions) / len(precisions)) + (sum(recalls) / len(recalls)))
                           if (sum(precisions) / len(precisions) + sum(recalls) / len(recalls)) > 0 else 0,
                "top_k_accuracy": sum(top_k_accs) / len(top_k_accs),
                "mrr": sum(mrrs) / len(mrrs),
                "ndcg": sum(ndcgs) / len(ndcgs),
            }

        return metrics_summary

    def evaluate_with_ragas(
        self,
        results: List[RetrievalResult],
        answers: List[str]
    ) -> Dict[str, float]:
        """
        使用RAGAS评估生成质量

        Args:
            results: 检索结果列表
            answers: 生成的答案列表

        Returns:
            RAGAS评估指标
        """
        if not RAGAS_AVAILABLE:
            print("RAGAS not available, skipping RAGAS evaluation")
            return {}

        # 准备RAGAS数据集
        data = {
            "question": [r.query for r in results],
            "answer": answers,
            "contexts": [r.retrieved_contexts for r in results],
            "ground_truth": [r.ground_truth_answer for r in results],
        }

        dataset = Dataset.from_dict(data)

        # 运行评估
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics
            )

            return {
                "faithfulness": result["faithfulness"],
                "answer_relevancy": result["answer_relevancy"],
                "context_precision": result["context_precision"],
                "context_recall": result["context_recall"],
            }
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")
            return {}

    def generate_evaluation_report(
        self,
        retrieval_metrics: Dict[str, Any],
        ragas_metrics: Dict[str, float],
        output_file: str = None
    ) -> str:
        """
        生成评估报告

        Args:
            retrieval_metrics: 检索指标
            ragas_metrics: RAGAS指标
            output_file: 输出文件路径

        Returns:
            报告内容
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_lines = [
            "# RAG检索评估报告",
            f"",
            f"生成时间: {timestamp}",
            f"",
            "=" * 80,
            f"检索质量指标",
            "=" * 80,
            f"",
        ]

        # 检索指标
        for k, metrics in retrieval_metrics.items():
            report_lines.extend([
                f"## {k.upper()}",
                f"",
                f"准确率 (Precision): {metrics['precision']:.2%} {'✅' if metrics['precision'] >= self.target_metrics['precision'] else '❌'}",
                f"召回率 (Recall): {metrics['recall']:.2%} {'✅' if metrics['recall'] >= self.target_metrics['recall'] else '❌'}",
                f"F1分数: {metrics['f1_score']:.4f}",
                f"TopK命中率: {metrics['top_k_accuracy']:.2%} {'✅' if metrics['top_k_accuracy'] >= self.target_metrics['top_k_accuracy'] else '❌'}",
                f"MRR: {metrics['mrr']:.4f}",
                f"NDCG: {metrics['ndcg']:.4f}",
                f"",
            ])

        # RAGAS指标
        if ragas_metrics:
            report_lines.extend([
                "=" * 80,
                "生成质量指标 (RAGAS)",
                "=" * 80,
                f"",
            ])

            report_lines.extend([
                f"忠实度 (Faithfulness): {ragas_metrics.get('faithfulness', 0):.2%} {'✅' if ragas_metrics.get('faithfulness', 0) >= self.target_metrics['faithfulness'] else '❌'}",
                f"答案相关性 (Answer Relevancy): {ragas_metrics.get('answer_relevancy', 0):.2%} {'✅' if ragas_metrics.get('answer_relevancy', 0) >= self.target_metrics['answer_relevancy'] else '❌'}",
                f"上下文精确度 (Context Precision): {ragas_metrics.get('context_precision', 0):.2%}",
                f"上下文召回率 (Context Recall): {ragas_metrics.get('context_recall', 0):.2%}",
                f"",
            ])

        # 总体评估
        report_lines.extend([
            "=" * 80,
            "目标达成情况",
            "=" * 80,
            f"",
        ])

        # 检查目标达成
        target_achieved = True
        for metric_name, target_value in self.target_metrics.items():
            if metric_name in ragas_metrics:
                actual_value = ragas_metrics[metric_name]
            elif metric_name == "top_k_accuracy":
                actual_value = retrieval_metrics.get("top_5", {}).get("top_k_accuracy", 0)
            elif metric_name in ["precision", "recall"]:
                actual_value = retrieval_metrics.get("top_5", {}).get(metric_name, 0)
            else:
                continue

            status = "✅ 达成" if actual_value >= target_value else "❌ 未达成"
            report_lines.append(f"{metric_name}: 目标 {target_value:.0%}, 实际 {actual_value:.2%} - {status}")

            if actual_value < target_value:
                target_achieved = False

        report_lines.extend([
            f"",
            "=" * 80,
            "总体评估",
            "=" * 80,
            f"",
            "✅ 全部指标达标" if target_achieved else "⚠️ 部分指标未达标",
            f"",
        ])

        report_content = "\n".join(report_lines)

        # 保存报告
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"评估报告已保存到: {output_file}")

        return report_content

    def save_detailed_results(
        self,
        results: List[RetrievalResult],
        metrics: Dict[str, Any],
        output_file: str = None
    ):
        """
        保存详细评估结果

        Args:
            results: 检索结果列表
            metrics: 评估指标
            output_file: 输出文件路径
        """
        detailed_results = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(results),
            "metrics": metrics,
            "detailed_results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "retrieved_docs": r.retrieved_docs,
                    "ground_truth_docs": r.ground_truth_docs,
                    "ground_truth_answer": r.ground_truth_answer,
                    "latency_ms": r.retrieval_latency * 1000,
                }
                for r in results
            ]
        }

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(detailed_results, f, ensure_ascii=False, indent=2)
            print(f"详细结果已保存到: {output_file}")

        return detailed_results
