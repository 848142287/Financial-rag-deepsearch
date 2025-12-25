"""
AgentRAG核心执行引擎
实现查询理解、计划生成、迭代检索、验证评估的完整流程
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from enum import Enum

from ..llm_service import llm_service
from ..knowledge_base.retrieval_engine import retrieval_engine
from ..knowledge_base.entity_extractor import financial_entity_extractor
from .query_understanding import QueryUnderstanding
from .planning import TaskPlanner
from .iteration_engine import IterationEngine
from .verification import VerificationEngine
from .ragas_evaluator import RAGASEvaluator

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型"""
    FACTUAL = "factual"          # 事实查询
    ANALYTICAL = "analytical"    # 分析查询
    COMPARATIVE = "comparative"  # 比较查询
    TEMPORAL = "temporal"        # 时间序列查询
    CAUSAL = "causal"           # 因果查询
    AGGREGATE = "aggregate"     # 聚合查询


@dataclass
class QueryPlan:
    """查询计划"""
    query_id: str
    original_query: str
    query_type: QueryType
    sub_queries: List[str]
    retrieval_strategies: List[str]
    entities: List[Dict[str, Any]]
    time_constraints: Optional[Dict[str, Any]] = None
    max_iterations: int = 3
    confidence_threshold: float = 0.7
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RetrievalStep:
    """检索步骤"""
    step_id: int
    query: str
    strategy: str
    results: List[Dict[str, Any]]
    confidence: float
    execution_time: float
    feedback: Optional[str] = None


@dataclass
class IterationResult:
    """迭代结果"""
    iteration: int
    query: str
    retrieved_docs: List[Dict[str, Any]]
    synthesized_answer: str
    confidence_score: float
    improvement_suggestions: List[str]
    verification_results: Dict[str, Any]


@dataclass
class AgentRAGResult:
    """AgentRAG执行结果"""
    query_plan: QueryPlan
    iterations: List[IterationResult]
    final_answer: str
    confidence_score: float
    ragas_metrics: Dict[str, float]
    execution_trace: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class AgentRAGEngine:
    """AgentRAG执行引擎"""

    def __init__(self):
        # 初始化各个模块
        self.query_understanding = QueryUnderstanding()
        self.task_planner = TaskPlanner()
        self.iteration_engine = IterationEngine()
        self.verification_engine = VerificationEngine()
        self.ragas_evaluator = RAGASEvaluator()

        # 配置参数
        self.max_iterations = 3
        self.confidence_threshold = 0.7
        self.enable_ragas_evaluation = True

    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AgentRAGResult:
        """
        执行AgentRAG流程

        Args:
            query: 用户查询
            context: 查询上下文
            options: 执行选项

        Returns:
            AgentRAG执行结果
        """
        start_time = datetime.utcnow()
        query_id = f"query_{int(start_time.timestamp())}"

        logger.info(f"开始执行AgentRAG: {query_id} - {query[:50]}...")

        try:
            # 第一步：查询理解
            logger.info(f"第一步：查询理解 - {query_id}")
            query_understanding = await self.query_understanding.understand(query, context)

            # 第二步：计划生成
            logger.info(f"第二步：计划生成 - {query_id}")
            query_plan = await self.task_planner.create_plan(
                query_id, query, query_understanding
            )

            # 第三步：迭代检索
            logger.info(f"第三步：迭代检索 - {query_id}")
            iterations = await self._execute_iterations(query_plan)

            # 第四步：验证评估
            logger.info(f"第四步：验证评估 - {query_id}")
            final_answer, verification_results = await self._verify_and_evaluate(
                query_plan, iterations
            )

            # RAGAS评估
            ragas_metrics = {}
            if self.enable_ragas_evaluation:
                logger.info(f"RAGAS评估 - {query_id}")
                ragas_metrics = await self.ragas_evaluator.evaluate(
                    query, final_answer, iterations[-1].retrieved_docs if iterations else []
                )

            # 构建执行结果
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            result = AgentRAGResult(
                query_plan=query_plan,
                iterations=iterations,
                final_answer=final_answer,
                confidence_score=iterations[-1].confidence_score if iterations else 0.0,
                ragas_metrics=ragas_metrics,
                execution_trace=self._build_execution_trace(query_plan, iterations),
                metadata={
                    'execution_time': execution_time,
                    'total_iterations': len(iterations),
                    'verification_results': verification_results,
                    'context': context,
                    'options': options
                }
            )

            logger.info(f"AgentRAG执行完成: {query_id}, 耗时: {execution_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"AgentRAG执行失败: {query_id}, 错误: {e}")
            raise

    async def _execute_iterations(
        self,
        query_plan: QueryPlan
    ) -> List[IterationResult]:
        """执行迭代检索"""
        iterations = []
        current_query = query_plan.original_query
        accumulated_context = []

        for iteration in range(1, query_plan.max_iterations + 1):
            logger.info(f"开始第 {iteration} 次迭代")

            # 执行检索
            retrieval_results = await self.iteration_engine.retrieve(
                current_query,
                query_plan.retrieval_strategies,
                query_plan.entities,
                accumulated_context
            )

            # 合并检索结果
            retrieved_docs = []
            for results in retrieval_results.values():
                retrieved_docs.extend(results)

            # 去重和排序
            retrieved_docs = self._deduplicate_and_rank(retrieved_docs)

            # 生成答案
            synthesized_answer = await self._synthesize_answer(
                query_plan.original_query,
                retrieved_docs,
                accumulated_context,
                iteration
            )

            # 计算置信度
            confidence_score = await self._calculate_confidence(
                query_plan.original_query,
                synthesized_answer,
                retrieved_docs
            )

            # 获取改进建议
            improvement_suggestions = []
            if iteration < query_plan.max_iterations and confidence_score < query_plan.confidence_threshold:
                improvement_suggestions = await self._get_improvement_suggestions(
                    query_plan.original_query,
                    synthesized_answer,
                    retrieved_docs,
                    iteration
                )
                current_query = self._refine_query(
                    query_plan.original_query,
                    improvement_suggestions
                )

            # 更新累积上下文
            accumulated_context.extend(retrieved_docs[:5])  # 保留最相关的5个文档

            # 创建迭代结果
            iteration_result = IterationResult(
                iteration=iteration,
                query=current_query,
                retrieved_docs=retrieved_docs,
                synthesized_answer=synthesized_answer,
                confidence_score=confidence_score,
                improvement_suggestions=improvement_suggestions,
                verification_results={}
            )

            iterations.append(iteration_result)

            # 检查是否继续迭代
            if confidence_score >= query_plan.confidence_threshold:
                logger.info(f"达到置信度阈值，停止迭代: {confidence_score}")
                break

        return iterations

    async def _verify_and_evaluate(
        self,
        query_plan: QueryPlan,
        iterations: List[IterationResult]
    ) -> Tuple[str, Dict[str, Any]]:
        """验证和评估最终结果"""
        if not iterations:
            return "无法找到相关信息，请尝试其他查询方式。", {}

        final_iteration = iterations[-1]
        final_answer = final_iteration.synthesized_answer

        # 使用验证引擎
        verification_results = await self.verification_engine.verify(
            query_plan.original_query,
            final_answer,
            final_iteration.retrieved_docs
        )

        # 如果验证失败，尝试修正答案
        if not verification_results.get('is_valid', False):
            logger.info("答案验证失败，尝试修正")
            corrected_answer = await self._correct_answer(
                query_plan.original_query,
                final_answer,
                verification_results
            )
            final_answer = corrected_answer

        return final_answer, verification_results

    async def _synthesize_answer(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        context: List[Dict[str, Any]],
        iteration: int
    ) -> str:
        """生成答案"""
        # 构建上下文
        context_text = ""
        for i, doc in enumerate(retrieved_docs[:10]):  # 限制文档数量
            context_text += f"\n[文档{i+1}]:\n{doc.get('content', '')[:500]}...\n"

        # 添加历史上下文
        if context:
            context_text += "\n历史相关文档:\n"
            for i, doc in enumerate(context[:5]):
                context_text += f"[历史{i+1}]: {doc.get('content', '')[:200]}...\n"

        # 构建prompt
        prompt = f"""
请基于以下文档内容，回答用户查询。

用户查询: {query}
当前迭代: {iteration}

相关文档:
{context_text}

请提供准确、全面的答案，并引用相关文档。
"""

        try:
            response = await llm_service.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return "抱歉，无法生成答案，请稍后再试。"

    async def _calculate_confidence(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> float:
        """计算答案置信度"""
        if not retrieved_docs or not answer:
            return 0.0

        # 基础置信度（基于检索文档质量）
        doc_confidence = sum(doc.get('score', 0) for doc in retrieved_docs[:5]) / min(5, len(retrieved_docs))

        # 答案完整性评估
        answer_completeness = min(len(answer.split('.')) / 3, 1.0)  # 假设3句话为完整答案

        # 文档相关性评估
        relevance_keywords = set(query.lower().split())
        relevant_docs = sum(
            1 for doc in retrieved_docs
            if any(keyword in doc.get('content', '').lower() for keyword in relevance_keywords)
        ) / max(len(retrieved_docs), 1)

        # 综合置信度
        confidence = 0.4 * doc_confidence + 0.3 * answer_completeness + 0.3 * relevance_docs

        return min(confidence, 1.0)

    async def _get_improvement_suggestions(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        iteration: int
    ) -> List[str]:
        """获取改进建议"""
        prompt = f"""
当前答案不够理想，请提供改进建议。

原始查询: {query}
当前答案: {answer}
已检索文档数量: {len(retrieved_docs)}
当前迭代: {iteration}

请提供3-5条具体的改进建议，包括：
1. 需要补充哪些信息
2. 如何调整检索策略
3. 如何改进答案结构
"""

        try:
            response = await llm_service.generate_response(prompt)
            # 解析建议
            suggestions = []
            for line in response.split('\n'):
                if line.strip() and any(char in line for char in ['1', '2', '3', '4', '5']):
                    suggestions.append(line.strip())
            return suggestions[:5]  # 最多5条建议
        except Exception as e:
            logger.error(f"获取改进建议失败: {e}")
            return ["尝试不同的检索策略", "扩大检索范围", "关注更具体的信息"]

    def _refine_query(self, original_query: str, suggestions: List[str]) -> str:
        """根据建议优化查询"""
        # 简单实现：添加关键词
        refined_query = original_query

        # 从建议中提取关键词
        keywords = []
        for suggestion in suggestions:
            if '更多' in suggestion or '补充' in suggestion:
                # 提取相关关键词
                words = suggestion.split()
                keywords.extend([w for w in words if len(w) > 2])

        if keywords:
            refined_query += f" {' '.join(keywords[:3])}"  # 添加前3个关键词

        return refined_query

    async def _correct_answer(
        self,
        query: str,
        answer: str,
        verification_results: Dict[str, Any]
    ) -> str:
        """修正答案"""
        errors = verification_results.get('errors', [])
        if not errors:
            return answer

        prompt = f"""
答案验证发现以下问题，请修正答案。

原始查询: {query}
当前答案: {answer}
验证错误:
{chr(10).join(f"- {error}" for error in errors)}

请基于验证结果修正答案，确保准确性和完整性。
"""

        try:
            corrected_answer = await llm_service.generate_response(prompt)
            return corrected_answer
        except Exception as e:
            logger.error(f"修正答案失败: {e}")
            return answer

    def _deduplicate_and_rank(
        self,
        docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """去重和排序文档"""
        if not docs:
            return []

        # 去重（基于文档内容相似度）
        unique_docs = []
        seen_contents = set()

        for doc in docs:
            content = doc.get('content', '')
            # 简单去重：基于前100个字符
            content_hash = content[:100].strip()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)

        # 按分数排序
        unique_docs.sort(key=lambda x: x.get('score', 0), reverse=True)

        return unique_docs

    def _build_execution_trace(
        self,
        query_plan: QueryPlan,
        iterations: List[IterationResult]
    ) -> List[Dict[str, Any]]:
        """构建执行轨迹"""
        trace = []

        # 添加查询理解阶段
        trace.append({
            'phase': 'query_understanding',
            'timestamp': query_plan.created_at.isoformat(),
            'query_type': query_plan.query_type.value,
            'entities': query_plan.entities,
            'sub_queries': query_plan.sub_queries
        })

        # 添加计划生成阶段
        trace.append({
            'phase': 'planning',
            'timestamp': query_plan.created_at.isoformat(),
            'retrieval_strategies': query_plan.retrieval_strategies,
            'max_iterations': query_plan.max_iterations
        })

        # 添加迭代检索阶段
        for iteration in iterations:
            trace.append({
                'phase': 'iteration',
                'iteration': iteration.iteration,
                'query': iteration.query,
                'retrieved_count': len(iteration.retrieved_docs),
                'confidence': iteration.confidence_score,
                'improvements': iteration.improvement_suggestions
            })

        return trace


# 全局AgentRAG引擎实例
agent_rag_engine = AgentRAGEngine()