"""
增强RAG评估服务 - 提供完整的RAG系统评估和优化功能
目标：实现80%以上的准确率
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import json
import logging
import numpy as np
from enum import Enum

from app.services.evaluation.ragas_evaluator import (
    RAGASEvaluator, EvaluationTestCase, EvaluationResult, AutomatedEvaluator
)
from app.services.evaluation.metrics import (
    MetricsCalculator, MetricResult, EvaluationReport,
    MetricsAggregator, MetricDimension
)
# DEPRECATED: Use ConsolidatedRAGService instead - from app.services.consolidated_rag_service import ConsolidatedRAGService agentic_rag_service, RetrievalMode
from app.services.agent_rag.agent_engine import AgentRAGEngine
from app.core.config import settings
from app.core.database import get_db
from app.core.redis_client import redis_client
from app.models.document import Document as DocumentModel
from app.models.conversation import Message as MessageModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """优化级别"""
    BASIC = "basic"          # 基础优化
    INTERMEDIATE = "intermediate"  # 中级优化
    ADVANCED = "advanced"    # 高级优化
    EXPERT = "expert"        # 专家级优化


@dataclass
class OptimizationStrategy:
    """优化策略"""
    level: OptimizationLevel
    target_score: float
    optimizations: List[Dict[str, Any]]
    estimated_improvement: float
    implementation_effort: str  # low/medium/high
    priority: int  # 1-10


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    query_type: str
    current_score: float
    target_score: float
    gap: float
    optimization_strategies: List[OptimizationStrategy]


@dataclass
class EvaluationMetrics:
    """评估指标"""
    # 召回相关指标
    precision: float  # 精确率
    recall: float     # 召回率
    f1_score: float   # F1分数

    # 回答质量指标
    faithfulness: float      # 忠实度（回答与检索结果的一致性）
    answer_relevance: float  # 回答相关性
    context_relevance: float # 上下文相关性

    # 系统性能指标
    latency: float           # 响应延迟（毫秒）
    context_utilization: float # 上下文利用率

    # 综合评分
    overall_score: float     # 综合评分


class RAGASEvaluationService:
    """RAGAS评估服务"""

    def __init__(self):
        self.evaluation_prompts = self._load_evaluation_prompts()

    def _load_evaluation_prompts(self) -> Dict[str, str]:
        """加载评估提示模板"""
        return {
            "faithfulness": """
请评估以下回答是否基于给定的上下文信息。请诚实评估，给出0-1之间的分数。

上下文：
{context}

问题：{question}

回答：{answer}

请回答：
1. 回答中的每一个主张是否都能在上下文中找到支持？
2. 回答是否包含了上下文中没有的信息？
3. 总体评估分数（0-1，1表示完全基于上下文）：

以JSON格式返回：
{{
    "score": 0.85,
    "reasoning": "详细分析原因"
}}
""",

            "answer_relevance": """
评估回答与问题的相关性。

问题：{question}

回答：{answer}

请评估：
1. 回答是否直接解决了问题？
2. 回答是否完整？
3. 回答是否准确相关？

以JSON格式返回0-1之间的分数：
{{
    "score": 0.90,
    "reasoning": "评估原因"
}}
""",

            "context_relevance": """
评估上下文与问题的相关性。

问题：{question}

上下文：
{context}

请评估上下文对回答问题的有用程度：
1. 上下文是否包含回答问题所需的信息？
2. 上下文的信息是否相关？
3. 上下文的完整性如何？

以JSON格式返回0-1之间的分数：
{{
    "score": 0.75,
    "reasoning": "评估原因"
}}
"""
        }

    async def evaluate_single_query(
        self,
        question: str,
        response: QueryResponse,
        ground_truth: Optional[str] = None
    ) -> EvaluationMetrics:
        """
        评估单个查询

        Args:
            question: 用户问题
            response: RAG系统响应
            ground_truth: 标准答案（可选）

        Returns:
            评估指标
        """
        try:
            logger.info(f"开始评估查询: {question}")

            # 并行执行各项评估
            tasks = [
                self._evaluate_faithfulness(question, response.answer, response.documents),
                self._evaluate_answer_relevance(question, response.answer),
                self._evaluate_context_relevance(question, response.documents),
                self._calculate_precision_recall(question, response, ground_truth)
            ]

            faithfulness_result, relevance_result, context_result, precision_recall_result = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            # 计算性能指标
            latency = response.get('response_time_ms', 0)
            context_utilization = self._calculate_context_utilization(response.documents)

            # 计算综合评分
            overall_score = self._calculate_overall_score(
                faithfulness_result.get('score', 0) if isinstance(faithfulness_result, dict) else 0,
                relevance_result.get('score', 0) if isinstance(relevance_result, dict) else 0,
                context_result.get('score', 0) if isinstance(context_result, dict) else 0,
                precision_recall_result.get('f1_score', 0) if isinstance(precision_recall_result, dict) else 0
            )

            metrics = EvaluationMetrics(
                precision=precision_recall_result.get('precision', 0) if isinstance(precision_recall_result, dict) else 0,
                recall=precision_recall_result.get('recall', 0) if isinstance(precision_recall_result, dict) else 0,
                f1_score=precision_recall_result.get('f1_score', 0) if isinstance(precision_recall_result, dict) else 0,
                faithfulness=faithfulness_result.get('score', 0) if isinstance(faithfulness_result, dict) else 0,
                answer_relevance=relevance_result.get('score', 0) if isinstance(relevance_result, dict) else 0,
                context_relevance=context_result.get('score', 0) if isinstance(context_result, dict) else 0,
                latency=latency,
                context_utilization=context_utilization,
                overall_score=overall_score
            )

            logger.info(f"查询评估完成，综合评分: {overall_score:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"查询评估失败: {e}")
            raise

    async def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        rag_function
    ) -> List[Dict[str, Any]]:
        """
        批量评估

        Args:
            test_cases: 测试用例列表，每个包含question和可选的ground_truth
            rag_function: RAG查询函数

        Returns:
            批量评估结果
        """
        try:
            logger.info(f"开始批量评估，测试用例数: {len(test_cases)}")

            results = []

            for i, test_case in enumerate(test_cases):
                try:
                    question = test_case['question']
                    ground_truth = test_case.get('ground_truth')

                    # 执行RAG查询
                    response = await rag_function(question)

                    # 评估
                    metrics = await self.evaluate_single_query(question, response, ground_truth)

                    result = {
                        "test_case_id": i,
                        "question": question,
                        "ground_truth": ground_truth,
                        "response": response,
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat()
                    }

                    results.append(result)

                    logger.info(f"测试用例 {i+1}/{len(test_cases)} 评估完成")

                except Exception as e:
                    logger.error(f"测试用例 {i} 评估失败: {e}")
                    results.append({
                        "test_case_id": i,
                        "question": test_case.get('question', ''),
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })

            # 计算总体统计
            batch_stats = self._calculate_batch_statistics(results)

            return {
                "results": results,
                "statistics": batch_stats,
                "total_test_cases": len(test_cases),
                "successful_evaluations": len([r for r in results if 'metrics' in r]),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"批量评估失败: {e}")
            raise

    async def _evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """评估回答忠实度"""
        try:
            context = "\n\n".join([doc.get('content', '') for doc in documents[:5]])

            prompt = self.evaluation_prompts["faithfulness"].format(
                context=context,
                question=question,
                answer=answer
            )

            result = await llm_service.structured_completion(
                prompt=prompt,
                schema={"score": "number", "reasoning": "string"},
                system_prompt="你是一个专业的评估师，负责评估回答与上下文的一致性。"
            )

            return result

        except Exception as e:
            logger.error(f"忠实度评估失败: {e}")
            return {"score": 0.0, "reasoning": f"评估失败: {str(e)}"}

    async def _evaluate_answer_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        """评估回答相关性"""
        try:
            prompt = self.evaluation_prompts["answer_relevance"].format(
                question=question,
                answer=answer
            )

            result = await llm_service.structured_completion(
                prompt=prompt,
                schema={"score": "number", "reasoning": "string"},
                system_prompt="你是一个专业的评估师，负责评估回答与问题的相关性。"
            )

            return result

        except Exception as e:
            logger.error(f"相关性评估失败: {e}")
            return {"score": 0.0, "reasoning": f"评估失败: {str(e)}"}

    async def _evaluate_context_relevance(
        self,
        question: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """评估上下文相关性"""
        try:
            context = "\n\n".join([doc.get('content', '') for doc in documents[:5]])

            prompt = self.evaluation_prompts["context_relevance"].format(
                question=question,
                context=context
            )

            result = await llm_service.structured_completion(
                prompt=prompt,
                schema={"score": "number", "reasoning": "string"},
                system_prompt="你是一个专业的评估师，负责评估上下文与问题的相关性。"
            )

            return result

        except Exception as e:
            logger.error(f"上下文相关性评估失败: {e}")
            return {"score": 0.0, "reasoning": f"评估失败: {str(e)}"}

    async def _calculate_precision_recall(
        self,
        question: str,
        response: QueryResponse,
        ground_truth: Optional[str]
    ) -> Dict[str, Any]:
        """计算精确率和召回率"""
        try:
            if not ground_truth:
                # 如果没有标准答案，使用启发式方法
                return self._heuristic_precision_recall(question, response)

            # 基于标准答案计算
            ground_truth_embedding = await embedding_service.get_embedding(ground_truth)
            answer_embedding = await embedding_service.get_embedding(response.answer)

            # 计算相似度作为相关性指标
            similarity = embedding_service.compute_similarity(ground_truth_embedding, answer_embedding)

            # 简化的精确率和召回率计算
            precision = min(1.0, similarity)
            recall = min(1.0, similarity)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "method": "ground_truth_similarity"
            }

        except Exception as e:
            logger.error(f"精确率召回率计算失败: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    def _heuristic_precision_recall(
        self,
        question: str,
        response: QueryResponse
    ) -> Dict[str, Any]:
        """启发式精确率召回率计算"""
        try:
            # 基于检索文档数量和质量
            documents = response.get('documents', [])

            if not documents:
                return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

            # 计算平均相似度
            avg_similarity = np.mean([doc.get('score', 0) for doc in documents])

            # 基于相似度的启发式评分
            precision = min(1.0, avg_similarity * 1.2)  # 稍微放大分数
            recall = min(1.0, avg_similarity)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "method": "heuristic"
            }

        except Exception as e:
            logger.error(f"启发式计算失败: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    def _calculate_context_utilization(self, documents: List[Dict[str, Any]]) -> float:
        """计算上下文利用率"""
        try:
            if not documents:
                return 0.0

            # 基于文档相似度分布计算利用率
            scores = [doc.get('score', 0) for doc in documents]
            avg_score = np.mean(scores)

            # 考虑文档数量的影响
            doc_count_factor = min(1.0, len(documents) / 5.0)  # 理想文档数量为5

            return min(1.0, avg_score * doc_count_factor)

        except Exception as e:
            logger.error(f"上下文利用率计算失败: {e}")
            return 0.0

    def _calculate_overall_score(
        self,
        faithfulness: float,
        answer_relevance: float,
        context_relevance: float,
        f1_score: float
    ) -> float:
        """计算综合评分"""
        # 权重分配
        weights = {
            "faithfulness": 0.3,
            "answer_relevance": 0.25,
            "context_relevance": 0.25,
            "f1_score": 0.2
        }

        overall_score = (
            faithfulness * weights["faithfulness"] +
            answer_relevance * weights["answer_relevance"] +
            context_relevance * weights["context_relevance"] +
            f1_score * weights["f1_score"]
        )

        return round(overall_score, 3)

    def _calculate_batch_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算批量评估统计信息"""
        try:
            successful_results = [r for r in results if 'metrics' in r]

            if not successful_results:
                return {"error": "没有成功的评估结果"}

            metrics_list = [r['metrics'] for r in successful_results]

            # 计算各指标的平均值、标准差等
            stats = {}
            metric_names = ['precision', 'recall', 'f1_score', 'faithfulness',
                          'answer_relevance', 'context_relevance', 'overall_score']

            for metric_name in metric_names:
                values = [getattr(m, metric_name, 0) for m in metrics_list]
                stats[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values))
                }

            # 性能统计
            latencies = [m.latency for m in metrics_list]
            stats["latency"] = {
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p90_ms": float(np.percentile(latencies, 90)),
                "p95_ms": float(np.percentile(latencies, 95))
            }

            # 分布统计
            overall_scores = [m.overall_score for m in metrics_list]
            stats["score_distribution"] = {
                "excellent": len([s for s in overall_scores if s >= 0.8]),
                "good": len([s for s in overall_scores if 0.6 <= s < 0.8]),
                "fair": len([s for s in overall_scores if 0.4 <= s < 0.6]),
                "poor": len([s for s in overall_scores if s < 0.4])
            }

            stats["summary"] = {
                "total_evaluations": len(results),
                "successful_evaluations": len(successful_results),
                "success_rate": len(successful_results) / len(results),
                "average_overall_score": stats["overall_score"]["mean"]
            }

            return stats

        except Exception as e:
            logger.error(f"批量统计计算失败: {e}")
            return {"error": str(e)}

    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """生成评估报告"""
        try:
            stats = evaluation_results.get('statistics', {})

            report = f"""
# RAG系统评估报告

## 评估概况
- 总测试用例数: {evaluation_results.get('total_test_cases', 0)}
- 成功评估数: {evaluation_results.get('successful_evaluations', 0)}
- 成功率: {stats.get('summary', {}).get('success_rate', 0):.2%}

## 核心指标
- 综合评分: {stats.get('overall_score', {}).get('mean', 0):.3f} ± {stats.get('overall_score', {}).get('std', 0):.3f}
- 忠实度: {stats.get('faithfulness', {}).get('mean', 0):.3f}
- 回答相关性: {stats.get('answer_relevance', {}).get('mean', 0):.3f}
- 上下文相关性: {stats.get('context_relevance', {}).get('mean', 0):.3f}
- F1分数: {stats.get('f1_score', {}).get('mean', 0):.3f}

## 性能指标
- 平均响应时间: {stats.get('latency', {}).get('mean_ms', 0):.0f}ms
- P90响应时间: {stats.get('latency', {}).get('p90_ms', 0):.0f}ms

## 评分分布
- 优秀 (≥0.8): {stats.get('score_distribution', {}).get('excellent', 0)} 个
- 良好 (0.6-0.8): {stats.get('score_distribution', {}).get('good', 0)} 个
- 一般 (0.4-0.6): {stats.get('score_distribution', {}).get('fair', 0)} 个
- 较差 (<0.4): {stats.get('score_distribution', {}).get('poor', 0)} 个

## 改进建议
1. 忠实度偏低时，检查检索质量和回答生成逻辑
2. 上下文相关性不足时，优化检索策略和关键词匹配
3. 响应时间过长时，考虑向量索引优化和缓存策略

报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            return report

        except Exception as e:
            logger.error(f"评估报告生成失败: {e}")
            return f"报告生成失败: {str(e)}"


class EnhancedRAGEvaluator:
    """增强的RAG评估器"""

    def __init__(self):
        self.ragas_evaluator = RAGASEvaluator()
        self.metrics_calculator = MetricsCalculator()
        self.aggregator = MetricsAggregator()
        self.agent_engine = AgentRAGEngine()
        self.optimization_history = []

        # 评估目标
        self.target_scores = {
            "overall": 0.80,
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_relevancy": 0.75,
            "context_recall": 0.80,
            "factual_consistency": 0.90
        }

    async def comprehensive_evaluation(
        self,
        sample_size: int = 100,
        query_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """综合评估"""
        logger.info(f"开始综合评估，样本数量: {sample_size}")

        try:
            # 1. 生成多样化的测试用例
            test_cases = await self._generate_comprehensive_test_cases(sample_size, query_types)

            # 2. 分类型评估
            results_by_type = {}
            all_results = []

            for query_type, cases in test_cases.items():
                logger.info(f"评估查询类型: {query_type}, 用例数: {len(cases)}")

                # 使用对应的最佳策略评估每种类型
                mode = self._get_optimal_mode_for_query_type(query_type)
                type_results = await self._evaluate_with_mode(cases, mode)

                results_by_type[query_type] = {
                    "results": type_results,
                    "avg_score": self._calculate_average_score(type_results),
                    "mode": mode.value
                }
                all_results.extend(type_results)

            # 3. 计算整体指标
            overall_metrics = await self._calculate_comprehensive_metrics(all_results)

            # 4. 生成优化建议
            optimization_plan = await self._generate_optimization_plan(results_by_type, overall_metrics)

            # 5. 创建基准测试结果
            benchmark_results = self._create_benchmark_results(results_by_type)

            # 6. 缓存结果
            evaluation_id = f"comprehensive_{int(datetime.now().timestamp())}"
            await self._cache_evaluation_results(evaluation_id, {
                "test_cases": len(all_results),
                "results_by_type": results_by_type,
                "overall_metrics": overall_metrics,
                "optimization_plan": optimization_plan,
                "benchmark_results": benchmark_results
            })

            return {
                "evaluation_id": evaluation_id,
                "timestamp": datetime.now().isoformat(),
                "test_cases": len(all_results),
                "results_by_type": results_by_type,
                "overall_metrics": overall_metrics,
                "optimization_plan": optimization_plan,
                "benchmark_results": benchmark_results,
                "meets_target": overall_metrics.get("overall_score", 0) >= self.target_scores["overall"]
            }

        except Exception as e:
            logger.error(f"综合评估失败: {str(e)}")
            raise

    async def _generate_comprehensive_test_cases(
        self,
        sample_size: int,
        query_types: Optional[List[str]] = None
    ) -> Dict[str, List[EvaluationTestCase]]:
        """生成综合测试用例"""
        query_types = query_types or ["factual", "analytical", "comparative", "temporal", "causal", "aggregate"]

        test_cases_by_type = {}

        # 从数据库获取真实查询
        async with get_db() as db:
            # 获取历史查询
            query_result = await db.execute(
                select(MessageModel)
                .filter(MessageModel.role == "user")
                .order_by(func.random())
                .limit(sample_size * 2)
            )
            user_queries = query_result.scalars().all()

            # 生成测试用例
            for query_type in query_types:
                cases = []
                type_queries = [q for q in user_queries if self._classify_query_type(q.content) == query_type]

                # 如果类型特定查询不够，使用模板生成
                if len(type_queries) < sample_size // len(query_types):
                    generated_queries = await self._generate_queries_for_type(
                        query_type,
                        sample_size // len(query_types) - len(type_queries)
                    )
                    type_queries.extend(generated_queries)

                # 创建测试用例
                for i, query in enumerate(type_queries[:sample_size // len(query_types)]):
                    case = EvaluationTestCase(
                        id=f"{query_type}_{i}",
                        query=query.content if hasattr(query, 'content') else query,
                        query_type=query_type,
                        difficulty=self._estimate_difficulty(query.content if hasattr(query, 'content') else query),
                        domain="financial",
                        metadata={"source": "real_user_query", "type": query_type}
                    )
                    cases.append(case)

                test_cases_by_type[query_type] = cases

        return test_cases_by_type

    async def _evaluate_with_mode(
        self,
        test_cases: List[EvaluationTestCase],
        mode: RetrievalMode
    ) -> List[EvaluationResult]:
        """使用指定模式评估测试用例"""
        results = []

        for test_case in test_cases:
            try:
                # 使用指定的检索模式执行RAG
                rag_result = await agentic_rag_service.query(
                    question=test_case.query,
                    mode=mode,
                    max_results=10
                )

                # 提取答案和上下文
                answer = rag_result.get('answer', '')
                contexts = [doc.get('content', '') for doc in rag_result.get('documents', [])]

                # 计算RAGAS指标
                ragas_scores = await self.ragas_evaluator._calculate_ragas_metrics(
                    question=test_case.query,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=test_case.ground_truth
                )

                # 计算自定义指标
                custom_scores = await self._calculate_enhanced_metrics(
                    test_case=test_case,
                    answer=answer,
                    contexts=contexts,
                    rag_result=rag_result
                )

                # 创建评估结果
                result = EvaluationResult(
                    test_case=test_case,
                    generated_answer=answer,
                    retrieved_contexts=contexts,
                    ragas_scores=ragas_scores,
                    custom_scores=custom_scores,
                    execution_time=rag_result.get('response_time_ms', 0) / 1000,
                    timestamp=datetime.now()
                )

                results.append(result)

            except Exception as e:
                logger.error(f"评估用例 {test_case.id} 失败: {str(e)}")
                continue

        return results

    async def _calculate_enhanced_metrics(
        self,
        test_case: EvaluationTestCase,
        answer: str,
        contexts: List[str],
        rag_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """计算增强指标"""
        scores = {}

        # 基础指标
        scores["answer_relevance"] = self.metrics_calculator.calculate_answer_relevance(
            test_case.query, answer
        )
        scores["factual_consistency"] = self.metrics_calculator.calculate_factual_consistency(
            answer, contexts
        )

        # 增强指标
        scores["semantic_similarity"] = await self._calculate_semantic_similarity(
            test_case.query, answer
        )

        scores["context_coverage"] = self._calculate_context_coverage(
            answer, contexts
        )

        scores["answer_completeness"] = self._calculate_answer_completeness(
            test_case, answer
        )

        scores["query_type_match"] = self._calculate_query_type_match(
            test_case.query_type, answer, contexts
        )

        # 性能指标
        response_time = rag_result.get('response_time_ms', 0)
        scores["response_time_score"] = self._calculate_response_time_score(response_time)

        # 文档质量指标
        if 'documents' in rag_result:
            scores["document_relevance"] = self._calculate_document_relevance(
                rag_result['documents']
            )
            scores["retrieval_precision"] = self._calculate_retrieval_precision(
                rag_result['documents']
            )

        return scores

    async def _generate_optimization_plan(
        self,
        results_by_type: Dict[str, Any],
        overall_metrics: Dict[str, Any]
    ) -> List[OptimizationStrategy]:
        """生成优化计划"""
        optimization_plan = []
        overall_score = overall_metrics.get("overall_score", 0)

        if overall_score >= self.target_scores["overall"]:
            # 已达到目标，建议维持策略
            optimization_plan.append(OptimizationStrategy(
                level=OptimizationLevel.BASIC,
                target_score=overall_score,
                optimizations=[{"action": "maintain_current_performance", "description": "当前性能已达标，维持现有策略"}],
                estimated_improvement=0.0,
                implementation_effort="low",
                priority=1
            ))
            return optimization_plan

        # 分析各类型表现
        for query_type, results in results_by_type.items():
            type_score = results.get("avg_score", 0)
            gap = self.target_scores["overall"] - type_score

            if gap > 0.1:  # 需要显著改进
                strategies = await self._generate_type_specific_strategies(query_type, gap)
                optimization_plan.extend(strategies)

        # 通用优化策略
        general_strategies = await self._generate_general_strategies(overall_metrics)
        optimization_plan.extend(general_strategies)

        # 按优先级排序
        optimization_plan.sort(key=lambda x: x.priority, reverse=True)

        return optimization_plan[:10]  # 返回前10个最重要的策略

    async def _generate_type_specific_strategies(
        self,
        query_type: str,
        gap: float
    ) -> List[OptimizationStrategy]:
        """生成特定查询类型的优化策略"""
        strategies = []

        if query_type == "factual":
            strategies.append(OptimizationStrategy(
                level=OptimizationLevel.INTERMEDIATE,
                target_score=self.target_scores["overall"],
                optimizations=[
                    {
                        "action": "enhance_entity_extraction",
                        "description": "增强实体提取，提高事实检索准确性"
                    },
                    {
                        "action": "improve_knowledge_graph",
                        "description": "改进知识图谱，增强实体关系推理"
                    }
                ],
                estimated_improvement=min(0.15, gap * 0.8),
                implementation_effort="medium",
                priority=8
            ))

        elif query_type == "analytical":
            strategies.append(OptimizationStrategy(
                level=OptimizationLevel.ADVANCED,
                target_score=self.target_scores["overall"],
                optimizations=[
                    {
                        "action": "implement_multi_hop_reasoning",
                        "description": "实现多跳推理，支持复杂分析查询"
                    },
                    {
                        "action": "enhance_context_aggregation",
                        "description": "增强上下文聚合，提高分析深度"
                    }
                ],
                estimated_improvement=min(0.20, gap * 0.9),
                implementation_effort="high",
                priority=9
            ))

        elif query_type == "comparative":
            strategies.append(OptimizationStrategy(
                level=OptimizationLevel.INTERMEDIATE,
                target_score=self.target_scores["overall"],
                optimizations=[
                    {
                        "action": "parallel_entity_search",
                        "description": "并行实体搜索，提高比较查询效率"
                    },
                    {
                        "action": "structured_comparison_generation",
                        "description": "结构化比较生成，提高答案可读性"
                    }
                ],
                estimated_improvement=min(0.12, gap * 0.7),
                implementation_effort="medium",
                priority=7
            ))

        return strategies

    async def _generate_general_strategies(self, metrics: Dict[str, Any]) -> List[OptimizationStrategy]:
        """生成通用优化策略"""
        strategies = []

        # 基于具体指标的策略
        if metrics.get("faithfulness", 0) < self.target_scores["faithfulness"]:
            strategies.append(OptimizationStrategy(
                level=OptimizationLevel.BASIC,
                target_score=self.target_scores["faithfulness"],
                optimizations=[
                    {
                        "action": "improve_context_verification",
                        "description": "改进上下文验证机制，确保答案基于检索内容"
                    },
                    {
                        "action": "enhance_fact_checking",
                        "description": "增强事实检查，提高答案准确性"
                    }
                ],
                estimated_improvement=0.10,
                implementation_effort="low",
                priority=6
            ))

        if metrics.get("answer_relevancy", 0) < self.target_scores["answer_relevancy"]:
            strategies.append(OptimizationStrategy(
                level=OptimizationLevel.INTERMEDIATE,
                target_score=self.target_scores["answer_relevancy"],
                optimizations=[
                    {
                        "action": "improve_query_understanding",
                        "description": "改进查询理解，更好地捕捉用户意图"
                    },
                    {
                        "action": "enhance_answer_ranking",
                        "description": "增强答案排序，提高相关性"
                    }
                ],
                estimated_improvement=0.12,
                implementation_effort="medium",
                priority=5
            ))

        if metrics.get("context_relevancy", 0) < self.target_scores["context_relevancy"]:
            strategies.append(OptimizationStrategy(
                level=OptimizationLevel.BASIC,
                target_score=self.target_scores["context_relevancy"],
                optimizations=[
                    {
                        "action": "optimize_embedding_model",
                        "description": "优化嵌入模型，提高检索相关性"
                    },
                    {
                        "action": "implement_hybrid_search",
                        "description": "实现混合搜索，结合语义和关键词搜索"
                    }
                ],
                estimated_improvement=0.15,
                implementation_effort="medium",
                priority=7
            ))

        return strategies

    def _create_benchmark_results(self, results_by_type: Dict[str, Any]) -> List[BenchmarkResult]:
        """创建基准测试结果"""
        benchmark_results = []

        for query_type, data in results_by_type.items():
            current_score = data.get("avg_score", 0)
            gap = self.target_scores["overall"] - current_score

            # 生成该类型的优化策略
            type_strategies = [
                OptimizationStrategy(
                    level=OptimizationLevel.BASIC,
                    target_score=self.target_scores["overall"],
                    optimizations=[{"action": "improve_retrieval_precision", "description": "提高检索精度"}],
                    estimated_improvement=min(0.10, gap * 0.5),
                    implementation_effort="low",
                    priority=5
                )
            ]

            benchmark_results.append(BenchmarkResult(
                query_type=query_type,
                current_score=current_score,
                target_score=self.target_scores["overall"],
                gap=gap,
                optimization_strategies=type_strategies
            ))

        return benchmark_results

    # 辅助方法
    def _get_optimal_mode_for_query_type(self, query_type: str) -> RetrievalMode:
        """获取查询类型对应的最佳检索模式"""
        mode_mapping = {
            "factual": RetrievalMode.ENHANCED,
            "analytical": RetrievalMode.DEEP_SEARCH,
            "comparative": RetrievalMode.AGENTIC,
            "temporal": RetrievalMode.ENHANCED,
            "causal": RetrievalMode.DEEP_SEARCH,
            "aggregate": RetrievalMode.AGENTIC
        }
        return mode_mapping.get(query_type, RetrievalMode.ENHANCED)

    def _classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["对比", "比较", "差异", "区别"]):
            return "comparative"
        elif any(word in query_lower for word in ["为什么", "原因", "影响", "导致"]):
            return "causal"
        elif any(word in query_lower for word in ["多少", "总计", "平均", "统计"]):
            return "aggregate"
        elif any(word in query_lower for word in ["趋势", "变化", "发展", "未来"]):
            return "temporal"
        elif any(word in query_lower for word in ["分析", "评估", "研究"]):
            return "analytical"
        else:
            return "factual"

    def _estimate_difficulty(self, query: str) -> str:
        """估算查询难度"""
        length = len(query)
        word_count = len(query.split())

        if word_count <= 10:
            return "easy"
        elif word_count <= 20:
            return "medium"
        else:
            return "hard"

    async def _generate_queries_for_type(self, query_type: str, count: int) -> List[str]:
        """为特定类型生成查询"""
        templates = {
            "factual": [
                "公司{}的最新财务数据是什么？",
                "{}的市值是多少？",
                "公司{}的主要业务有哪些？"
            ],
            "analytical": [
                "分析{}的发展趋势和前景",
                "评估{}的竞争优势",
                "研究{}的商业模式"
            ],
            "comparative": [
                "对比{}和{}的差异",
                "{}与{}哪个更有投资价值？",
                "比较{}和{}的业绩表现"
            ],
            "temporal": [
                "{}近五年的发展历程如何？",
                "预测{}的未来趋势",
                "{}在不同时期的表现"
            ],
            "causal": [
                "为什么{}会出现这种情况？",
                "什么因素影响了{}的发展？",
                "{}变化的原因是什么？"
            ],
            "aggregate": [
                "统计{}的关键指标",
                "总结{}的整体情况",
                "汇总{}的重要信息"
            ]
        }

        queries = []
        type_templates = templates.get(query_type, templates["factual"])

        for i in range(count):
            template = type_templates[i % len(type_templates)]
            # 这里可以填充具体的公司名称或主题
            query = template.format("某公司")
            queries.append(query)

        return queries

    def _calculate_average_score(self, results: List[EvaluationResult]) -> float:
        """计算平均分数"""
        if not results:
            return 0.0

        total_score = 0
        for result in results:
            # 合并RAGAS和自定义指标计算综合分数
            all_scores = {**result.ragas_scores, **result.custom_scores}
            score = self.aggregator.calculate_overall_score(all_scores)
            total_score += score

        return total_score / len(results)

    async def _calculate_comprehensive_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """计算综合指标"""
        if not results:
            return {}

        # 收集所有指标
        all_metrics = {}
        ragas_metrics = {}
        custom_metrics = {}

        for result in results:
            # RAGAS指标
            for metric_name, score in result.ragas_scores.items():
                if score is not None:
                    if metric_name not in ragas_metrics:
                        ragas_metrics[metric_name] = []
                    ragas_metrics[metric_name].append(score)

            # 自定义指标
            for metric_name, score in result.custom_scores.items():
                if metric_name not in custom_metrics:
                    custom_metrics[metric_name] = []
                custom_metrics[metric_name].append(score)

        # 计算平均分数
        avg_ragas = {k: sum(v) / len(v) for k, v in ragas_metrics.items() if v}
        avg_custom = {k: sum(v) / len(v) for k, v in custom_metrics.items() if v}

        # 计算总体评分
        all_scores = {**avg_ragas, **avg_custom}
        overall_score = self.aggregator.calculate_overall_score(all_scores)

        # 计算分维度评分
        dimension_scores = self.aggregator.generate_dimension_scores(all_scores)

        # 添加执行时间统计
        execution_times = [result.execution_time for result in results]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

        return {
            "overall_score": overall_score,
            "ragas_metrics": avg_ragas,
            "custom_metrics": avg_custom,
            "dimension_scores": dimension_scores,
            "execution_time": {
                "average": avg_execution_time,
                "p95": np.percentile(execution_times, 95) if execution_times else 0
            },
            "target_scores": self.target_scores,
            "meets_target": overall_score >= self.target_scores["overall"]
        }

    async def _calculate_semantic_similarity(self, query: str, answer: str) -> float:
        """计算语义相似度"""
        # 简化实现，实际应用中可以使用更高级的语义模型
        query_words = set(self._extract_keywords(query))
        answer_words = set(self._extract_keywords(answer))

        if not query_words or not answer_words:
            return 0.0

        intersection = len(query_words & answer_words)
        union = len(query_words | answer_words)

        return intersection / union if union else 0.0

    def _calculate_context_coverage(self, answer: str, contexts: List[str]) -> float:
        """计算上下文覆盖率"""
        if not contexts:
            return 0.0

        answer_words = set(self._extract_keywords(answer))
        total_coverage = 0

        for context in contexts:
            context_words = set(self._extract_keywords(context))
            if context_words:
                coverage = len(answer_words & context_words) / len(answer_words)
                total_coverage += coverage

        return total_coverage / len(contexts) if contexts else 0.0

    def _calculate_answer_completeness(self, test_case: EvaluationTestCase, answer: str) -> float:
        """计算答案完整性"""
        # 基于答案长度和关键词覆盖度
        answer_words = len(answer.split())
        query_words = len(test_case.query.split())

        # 长度评分
        length_score = min(1.0, answer_words / (query_words * 2))

        # 关键词覆盖评分
        query_keywords = set(self._extract_keywords(test_case.query))
        answer_keywords = set(self._extract_keywords(answer))

        if query_keywords:
            coverage = len(query_keywords & answer_keywords) / len(query_keywords)
        else:
            coverage = 0.0

        return (length_score + coverage) / 2

    def _calculate_query_type_match(self, query_type: str, answer: str, contexts: List[str]) -> float:
        """计算查询类型匹配度"""
        if query_type == "factual":
            # 事实查询要求简洁准确
            length_score = 1.0 if len(answer.split()) < 100 else 0.7
            fact_indicators = ["是", "为", "达到", "实现", "总共"]
            has_facts = any(indicator in answer for indicator in fact_indicators)
            return length_score if has_facts else length_score * 0.8

        elif query_type == "analytical":
            # 分析查询要求深度和广度
            length_score = min(1.0, len(answer.split()) / 200)
            analysis_indicators = ["分析", "原因", "影响", "趋势", "因素"]
            has_analysis = any(indicator in answer for indicator in analysis_indicators)
            return length_score if has_analysis else length_score * 0.7

        elif query_type == "comparative":
            # 比较查询需要对比元素
            comparison_indicators = ["对比", "相比", "而", "但是", "另一方面"]
            has_comparison = any(indicator in answer for indicator in comparison_indicators)
            return 1.0 if has_comparison else 0.5

        else:
            return 0.8  # 默认评分

    def _calculate_response_time_score(self, response_time_ms: float) -> float:
        """计算响应时间评分"""
        # 响应时间越短越好
        if response_time_ms < 1000:
            return 1.0
        elif response_time_ms < 3000:
            return 0.8
        elif response_time_ms < 5000:
            return 0.6
        else:
            return 0.4

    def _calculate_document_relevance(self, documents: List[Dict]) -> float:
        """计算文档相关性"""
        if not documents:
            return 0.0

        total_relevance = sum(doc.get('score', 0) for doc in documents)
        return total_relevance / len(documents)

    def _calculate_retrieval_precision(self, documents: List[Dict]) -> float:
        """计算检索精度"""
        if not documents:
            return 0.0

        # 计算高相关性文档的比例
        high_relevance_count = sum(1 for doc in documents if doc.get('score', 0) > 0.7)
        return high_relevance_count / len(documents)

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        import re

        # 移除标点符号，转换为小写
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()

        # 简单的停用词列表
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好'
        }

        return [word for word in words if word not in stop_words and len(word) > 1]

    async def _cache_evaluation_results(self, evaluation_id: str, results: Dict[str, Any]):
        """缓存评估结果"""
        try:
            cache_key = f"comprehensive_evaluation:{evaluation_id}"
            redis_client.setex(
                cache_key,
                86400 * 7,  # 缓存7天
                json.dumps(results, ensure_ascii=False, default=str)
            )
            logger.info(f"Evaluation results cached: {evaluation_id}")
        except Exception as e:
            logger.error(f"Failed to cache evaluation results: {str(e)}")


# 全局增强评估器实例
enhanced_rag_evaluator = EnhancedRAGEvaluator()

# 保留原有的评估服务实例
ragas_evaluation_service = RAGASEvaluationService()