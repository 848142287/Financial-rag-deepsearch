"""
RAGAS评估模块
实现答案质量自动评估，包括忠实度、相关性、准确性等维度
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid
import numpy as np

logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """评估指标"""
    FAITHFULNESS = "faithfulness"          # 忠实度：答案与上下文的一致性
    ANSWER_RELEVANCE = "answer_relevance"  # 答案相关性：答案与问题的相关性
    CONTEXT_RELEVANCE = "context_relevance" # 上下文相关性：上下文与问题的相关性
    CONTEXT_RECALL = "context_recall"      # 上下文召回：上下文对答案的支持度
    ANSWER_CORRECTNESS = "answer_correctness" # 答案正确性：答案的准确性
    ASPECT_CRITIQUE = "aspect_critique"     # 方面批判：特定方面的评估


@dataclass
class EvaluationConfig:
    """评估配置"""
    enable_faithfulness: bool = True
    enable_answer_relevance: bool = True
    enable_context_relevance: bool = True
    enable_context_recall: bool = True
    enable_answer_correctness: bool = True
    enable_aspect_critique: bool = True
    aspect_critiques: List[str] = field(default_factory=lambda: ["简洁性", "完整性", "专业性"])
    faithfulness_model: str = "gpt-3.5-turbo"
    relevance_model: str = "gpt-3.5-turbo"
    batch_size: int = 5
    timeout: int = 30


@dataclass
class EvaluationResult:
    """评估结果"""
    metric: EvaluationMetric
    score: float
    reasoning: str
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGASEvaluation:
    """RAGAS评估报告"""
    evaluation_id: str
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    results: List[EvaluationResult]
    overall_score: float
    evaluation_time: datetime
    config: EvaluationConfig


class RAGASEvaluator:
    """RAGAS评估器

    提供多维度答案质量评估功能
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger(__name__)

    async def evaluate(self,
                      question: str,
                      answer: str,
                      contexts: List[str],
                      ground_truth: Optional[str] = None) -> RAGASEvaluation:
        """执行RAGAS评估

        Args:
            question: 问题
            answer: 答案
            contexts: 上下文列表
            ground_truth: 真实答案（可选）

        Returns:
            RAGAS评估报告
        """
        evaluation_id = str(uuid.uuid4())
        start_time = datetime.now()

        self.logger.info(f"开始RAGAS评估: {evaluation_id}")

        try:
            results = []

            # 1. 忠实度评估
            if self.config.enable_faithfulness:
                faithfulness_result = await self._evaluate_faithfulness(answer, contexts)
                results.append(faithfulness_result)

            # 2. 答案相关性评估
            if self.config.enable_answer_relevance:
                answer_relevance_result = await self._evaluate_answer_relevance(question, answer)
                results.append(answer_relevance_result)

            # 3. 上下文相关性评估
            if self.config.enable_context_relevance:
                context_relevance_result = await self._evaluate_context_relevance(question, contexts)
                results.append(context_relevance_result)

            # 4. 上下文召回评估
            if self.config.enable_context_recall and ground_truth:
                context_recall_result = await self._evaluate_context_recall(ground_truth, contexts)
                results.append(context_recall_result)

            # 5. 答案正确性评估
            if self.config.enable_answer_correctness and ground_truth:
                correctness_result = await self._evaluate_answer_correctness(answer, ground_truth)
                results.append(correctness_result)

            # 6. 方面批判评估
            if self.config.enable_aspect_critique:
                critique_results = await self._evaluate_aspect_critique(answer, question)
                results.extend(critique_results)

            # 计算总体分数
            overall_score = self._calculate_overall_score(results)

            # 创建评估报告
            evaluation = RAGASEvaluation(
                evaluation_id=evaluation_id,
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                results=results,
                overall_score=overall_score,
                evaluation_time=start_time,
                config=self.config
            )

            self.logger.info(f"RAGAS评估完成: {evaluation_id}, 总分: {overall_score:.3f}")
            return evaluation

        except Exception as e:
            self.logger.error(f"RAGAS评估失败 {evaluation_id}: {e}")
            raise

    async def batch_evaluate(self,
                           evaluations: List[Dict[str, Any]]) -> List[RAGASEvaluation]:
        """批量评估

        Args:
            evaluations: 评估参数列表，每个元素包含question, answer, contexts, ground_truth

        Returns:
            评估报告列表
        """
        self.logger.info(f"开始批量RAGAS评估: {len(evaluations)} 个")

        tasks = []
        for eval_params in evaluations:
            task = self.evaluate(
                question=eval_params["question"],
                answer=eval_params["answer"],
                contexts=eval_params["contexts"],
                ground_truth=eval_params.get("ground_truth")
            )
            tasks.append(task)

        # 分批执行以避免过载
        batch_size = self.config.batch_size
        results = []

        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"批量评估中的错误: {result}")
                else:
                    results.append(result)

        self.logger.info(f"批量RAGAS评估完成: {len(results)} 个")
        return results

    async def _evaluate_faithfulness(self, answer: str, contexts: List[str]) -> EvaluationResult:
        """评估忠实度"""
        try:
            # 将上下文合并
            combined_context = " ".join(contexts)

            # 构建忠实度评估提示
            prompt = f"""
            请评估以下答案是否忠实地基于给定的上下文。给出0-1之间的分数。

            上下文:
            {combined_context}

            答案:
            {answer}

            请按以下格式回答:
            分数: [0-1之间的数值]
            理由: [评估理由]
            置信度: [0-1之间的数值]
            """

            # 调用LLM进行评估（简化实现）
            evaluation_result = await self._call_llm_for_evaluation(prompt)

            # 解析结果
            score = self._extract_score(evaluation_result)
            reasoning = self._extract_reasoning(evaluation_result)
            confidence = self._extract_confidence(evaluation_result)

            return EvaluationResult(
                metric=EvaluationMetric.FAITHFULNESS,
                score=score,
                reasoning=reasoning,
                confidence=confidence,
                details={"context_count": len(contexts)}
            )

        except Exception as e:
            self.logger.error(f"忠实度评估失败: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.FAITHFULNESS,
                score=0.0,
                reasoning=f"评估失败: {str(e)}",
                confidence=0.0
            )

    async def _evaluate_answer_relevance(self, question: str, answer: str) -> EvaluationResult:
        """评估答案相关性"""
        try:
            # 构建相关性评估提示
            prompt = f"""
            请评估答案与问题的相关性。给出0-1之间的分数。

            问题:
            {question}

            答案:
            {answer}

            请按以下格式回答:
            分数: [0-1之间的数值]
            理由: [评估理由]
            置信度: [0-1之间的数值]
            """

            evaluation_result = await self._call_llm_for_evaluation(prompt)

            score = self._extract_score(evaluation_result)
            reasoning = self._extract_reasoning(evaluation_result)
            confidence = self._extract_confidence(evaluation_result)

            return EvaluationResult(
                metric=EvaluationMetric.ANSWER_RELEVANCE,
                score=score,
                reasoning=reasoning,
                confidence=confidence
            )

        except Exception as e:
            self.logger.error(f"答案相关性评估失败: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.ANSWER_RELEVANCE,
                score=0.0,
                reasoning=f"评估失败: {str(e)}",
                confidence=0.0
            )

    async def _evaluate_context_relevance(self, question: str, contexts: List[str]) -> EvaluationResult:
        """评估上下文相关性"""
        try:
            # 计算问题与每个上下文的相关性
            relevance_scores = []

            for context in contexts:
                prompt = f"""
                评估上下文与问题的相关性。给出0-1之间的分数。

                问题:
                {question}

                上下文:
                {context}

                请只回答一个0-1之间的分数。
                """

                result = await self._call_llm_for_evaluation(prompt)
                score = self._extract_score(result)
                relevance_scores.append(score)

            # 计算平均相关性
            avg_score = np.mean(relevance_scores) if relevance_scores else 0.0

            reasoning = f"基于{len(contexts)}个上下文的平均相关性: {avg_score:.3f}"

            return EvaluationResult(
                metric=EvaluationMetric.CONTEXT_RELEVANCE,
                score=avg_score,
                reasoning=reasoning,
                confidence=0.8,
                details={
                    "individual_scores": relevance_scores,
                    "context_count": len(contexts)
                }
            )

        except Exception as e:
            self.logger.error(f"上下文相关性评估失败: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.CONTEXT_RELEVANCE,
                score=0.0,
                reasoning=f"评估失败: {str(e)}",
                confidence=0.0
            )

    async def _evaluate_context_recall(self, ground_truth: str, contexts: List[str]) -> EvaluationResult:
        """评估上下文召回"""
        try:
            # 从真实答案中提取关键陈述
            key_statements = await self._extract_key_statements(ground_truth)

            supported_statements = 0

            for statement in key_statements:
                # 检查每个陈述是否被上下文支持
                is_supported = await self._check_statement_support(statement, contexts)
                if is_supported:
                    supported_statements += 1

            # 计算召回率
            recall_score = supported_statements / len(key_statements) if key_statements else 0.0

            reasoning = f"在{len(key_statements)}个关键陈述中，{supported_statements}个被上下文支持"

            return EvaluationResult(
                metric=EvaluationMetric.CONTEXT_RECALL,
                score=recall_score,
                reasoning=reasoning,
                confidence=0.8,
                details={
                    "total_statements": len(key_statements),
                    "supported_statements": supported_statements
                }
            )

        except Exception as e:
            self.logger.error(f"上下文召回评估失败: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.CONTEXT_RECALL,
                score=0.0,
                reasoning=f"评估失败: {str(e)}",
                confidence=0.0
            )

    async def _evaluate_answer_correctness(self, answer: str, ground_truth: str) -> EvaluationResult:
        """评估答案正确性"""
        try:
            # 计算答案与真实答案的语义相似度
            semantic_similarity = await self._calculate_semantic_similarity(answer, ground_truth)

            # 使用LLM进行详细评估
            prompt = f"""
            比较以下两个答案的正确性和一致性。给出0-1之间的分数。

            待评估答案:
            {answer}

            参考答案:
            {ground_truth}

            请按以下格式回答:
            分数: [0-1之间的数值]
            理由: [评估理由]
            置信度: [0-1之间的数值]
            """

            evaluation_result = await self._call_llm_for_evaluation(prompt)

            llm_score = self._extract_score(evaluation_result)
            reasoning = self._extract_reasoning(evaluation_result)
            confidence = self._extract_confidence(evaluation_result)

            # 结合语义相似度和LLM评估
            combined_score = (llm_score * 0.7 + semantic_similarity * 0.3)

            return EvaluationResult(
                metric=EvaluationMetric.ANSWER_CORRECTNESS,
                score=combined_score,
                reasoning=reasoning,
                confidence=confidence,
                details={
                    "semantic_similarity": semantic_similarity,
                    "llm_score": llm_score
                }
            )

        except Exception as e:
            self.logger.error(f"答案正确性评估失败: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.ANSWER_CORRECTNESS,
                score=0.0,
                reasoning=f"评估失败: {str(e)}",
                confidence=0.0
            )

    async def _evaluate_aspect_critique(self, answer: str, question: str) -> List[EvaluationResult]:
        """评估特定方面"""
        results = []

        for aspect in self.config.aspect_critiques:
            try:
                prompt = f"""
                从"{aspect}"的角度评估以下答案的质量。给出0-1之间的分数。

                问题:
                {question}

                答案:
                {answer}

                请按以下格式回答:
                分数: [0-1之间的数值]
                理由: [评估理由]
                置信度: [0-1之间的数值]
                """

                evaluation_result = await self._call_llm_for_evaluation(prompt)

                score = self._extract_score(evaluation_result)
                reasoning = self._extract_reasoning(evaluation_result)
                confidence = self._extract_confidence(evaluation_result)

                result = EvaluationResult(
                    metric=EvaluationMetric.ASPECT_CRITIQUE,
                    score=score,
                    reasoning=reasoning,
                    confidence=confidence,
                    details={"aspect": aspect}
                )

                results.append(result)

            except Exception as e:
                self.logger.error(f"方面批判评估失败 ({aspect}): {e}")
                results.append(EvaluationResult(
                    metric=EvaluationMetric.ASPECT_CRITIQUE,
                    score=0.0,
                    reasoning=f"评估失败: {str(e)}",
                    confidence=0.0,
                    details={"aspect": aspect}
                ))

        return results

    async def _call_llm_for_evaluation(self, prompt: str) -> str:
        """调用LLM进行评估"""
        # 简化实现，返回模拟结果
        # 实际实现应该调用真实的LLM服务

        # 模拟不同类型的评估结果
        import random

        if "分数" in prompt:
            score = random.uniform(0.6, 0.95)
            reasoning = "基于内容的全面评估，考虑了多个因素"
            confidence = random.uniform(0.7, 0.9)

            return f"分数: {score:.3f}\n理由: {reasoning}\n置信度: {confidence:.3f}"
        else:
            return random.uniform(0.7, 0.9)

    def _extract_score(self, evaluation_result: str) -> float:
        """提取分数"""
        try:
            import re
            match = re.search(r'分数[：:]\s*([0-9.]+)', evaluation_result)
            if match:
                return float(match.group(1))
            return 0.0
        except:
            return 0.0

    def _extract_reasoning(self, evaluation_result: str) -> str:
        """提取理由"""
        try:
            import re
            match = re.search(r'理由[：:]\s*(.+?)(?=置信度|$)', evaluation_result, re.DOTALL)
            if match:
                return match.group(1).strip()
            return "无详细理由"
        except:
            return "解析失败"

    def _extract_confidence(self, evaluation_result: str) -> float:
        """提取置信度"""
        try:
            import re
            match = re.search(r'置信度[：:]\s*([0-9.]+)', evaluation_result)
            if match:
                return float(match.group(1))
            return 0.8
        except:
            return 0.8

    async def _extract_key_statements(self, text: str) -> List[str]:
        """提取关键陈述"""
        # 简化实现：按句号分割
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences[:5]  # 返回前5个陈述

    async def _check_statement_support(self, statement: str, contexts: List[str]) -> bool:
        """检查陈述是否被上下文支持"""
        # 简化实现：检查关键词重叠
        statement_words = set(statement.lower().split())

        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(statement_words & context_words)

            # 如果重叠度超过50%，认为支持
            if len(statement_words) > 0 and overlap / len(statement_words) > 0.5:
                return True

        return False

    async def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        # 简化实现：基于词汇重叠
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _calculate_overall_score(self, results: List[EvaluationResult]) -> float:
        """计算总体分数"""
        if not results:
            return 0.0

        # 定义权重
        weights = {
            EvaluationMetric.FAITHFULNESS: 0.25,
            EvaluationMetric.ANSWER_RELEVANCE: 0.25,
            EvaluationMetric.CONTEXT_RELEVANCE: 0.15,
            EvaluationMetric.CONTEXT_RECALL: 0.15,
            EvaluationMetric.ANSWER_CORRECTNESS: 0.15,
            EvaluationMetric.ASPECT_CRITIQUE: 0.05
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for result in results:
            weight = weights.get(result.metric, 0.1)
            weighted_sum += result.score * weight * result.confidence
            total_weight += weight * result.confidence

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_evaluation_summary(self, evaluation: RAGASEvaluation) -> Dict[str, Any]:
        """获取评估摘要"""
        summary = {
            "evaluation_id": evaluation.evaluation_id,
            "overall_score": evaluation.overall_score,
            "metrics": {},
            "strengths": [],
            "weaknesses": []
        }

        # 按指标分组
        for result in evaluation.results:
            metric_name = result.metric.value
            summary["metrics"][metric_name] = {
                "score": result.score,
                "confidence": result.confidence,
                "reasoning": result.reasoning
            }

            # 识别优势和劣势
            if result.score >= 0.8:
                summary["strengths"].append(f"{metric_name}: {result.reasoning}")
            elif result.score <= 0.5:
                summary["weaknesses"].append(f"{metric_name}: {result.reasoning}")

        return summary


@dataclass
class EvaluationTestCase:
    """评估测试用例"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    expected_metrics: Optional[Dict[str, float]] = None
    test_id: str = ""

    def __post_init__(self):
        if not self.test_id:
            import uuid
            self.test_id = str(uuid.uuid4())


class AutomatedEvaluator:
    """自动评估器"""

    def __init__(self, evaluator: RAGASEvaluator):
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)

    async def evaluate_test_cases(self,
                                 test_cases: List[EvaluationTestCase]) -> List[RAGASEvaluation]:
        """评估测试用例列表"""
        results = []

        for test_case in test_cases:
            try:
                result = await self.evaluator.evaluate(
                    question=test_case.question,
                    answer=test_case.answer,
                    contexts=test_case.contexts,
                    ground_truth=test_case.ground_truth
                )
                results.append(result)

            except Exception as e:
                self.logger.error(f"测试用例评估失败 {test_case.test_id}: {e}")

        return results

    async def compare_with_expected(self,
                                   evaluations: List[RAGASEvaluation],
                                   test_cases: List[EvaluationTestCase]) -> Dict[str, Any]:
        """与期望指标比较"""
        comparison_results = []

        for eval_result, test_case in zip(evaluations, test_cases):
            if test_case.expected_metrics:
                comparison = {
                    "test_id": test_case.test_id,
                    "actual_scores": {},
                    "expected_scores": test_case.expected_metrics,
                    "differences": {},
                    "passed_checks": []
                }

                for metric_result in eval_result.results:
                    metric_name = metric_result.metric.value
                    actual_score = metric_result.score
                    expected_score = test_case.expected_metrics.get(metric_name)

                    comparison["actual_scores"][metric_name] = actual_score

                    if expected_score is not None:
                        difference = abs(actual_score - expected_score)
                        comparison["differences"][metric_name] = difference

                        # 如果差异在0.1以内，认为通过检查
                        if difference <= 0.1:
                            comparison["passed_checks"].append(metric_name)

                comparison_results.append(comparison)

        return {
            "total_comparisons": len(comparison_results),
            "comparisons": comparison_results,
            "overall_pass_rate": sum(len(c["passed_checks"]) for c in comparison_results) /
                              sum(len(c["expected_scores"]) for c in comparison_results) if comparison_results else 0
        }


# 全局实例
ragas_evaluator = RAGASEvaluator()
automated_evaluator = AutomatedEvaluator(ragas_evaluator)