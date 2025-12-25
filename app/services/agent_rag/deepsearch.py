"""
DeepSearch优化模块
实现深度搜索优化，包括多轮检索、结果重排、答案增强等
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging
from datetime import datetime

from ..knowledge_base.retrieval_engine import retrieval_engine, RetrievalResult
from ..llm_service import llm_service
from ..knowledge_base.entity_extractor import financial_entity_extractor
from .ragas_evaluator import ragas_evaluator

logger = logging.getLogger(__name__)


@dataclass
class SearchContext:
    """搜索上下文"""
    original_query: str
    current_query: str
    iteration: int
    accumulated_results: List[RetrievalResult]
    retrieved_entities: List[Dict[str, Any]]
    query_refinements: List[str]
    confidence_history: List[float]


@dataclass
class DeepSearchResult:
    """深度搜索结果"""
    query: str
    final_answer: str
    all_iterations: List[List[RetrievalResult]]
    search_trace: List[Dict[str, Any]]
    confidence_score: float
    optimization_actions: List[str]
    ragas_metrics: Dict[str, float]


class DeepSearchOptimizer:
    """DeepSearch优化器"""

    def __init__(self):
        self.max_iterations = 3
        self.confidence_threshold = 0.8
        self.diversity_threshold = 0.3
        self.entity_expansion_factor = 2.0
        self.query_refinement_strategies = [
            'entity_expansion',
            'synonym_replacement',
            'temporal_extension',
            'aspect_enrichment'
        ]

    async def deep_search(
        self,
        query: str,
        initial_results: Optional[List[RetrievalResult]] = None,
        max_iterations: Optional[int] = None
    ) -> DeepSearchResult:
        """
        执行深度搜索优化

        Args:
            query: 原始查询
            initial_results: 初始检索结果
            max_iterations: 最大迭代次数

        Returns:
            深度搜索结果
        """
        logger.info(f"开始DeepSearch优化: {query}")

        max_iterations = max_iterations or self.max_iterations
        context = SearchContext(
            original_query=query,
            current_query=query,
            iteration=0,
            accumulated_results=initial_results or [],
            retrieved_entities=[],
            query_refinements=[],
            confidence_history=[]
        )

        all_iterations = []
        search_trace = []
        optimization_actions = []

        try:
            # 迭代优化
            for iteration in range(max_iterations):
                logger.info(f"DeepSearch迭代 {iteration + 1}/{max_iterations}")

                context.iteration = iteration + 1

                # 执行检索
                current_results = await self._execute_retrieval(context)
                all_iterations.append(current_results)

                # 分析和优化
                optimization_result = await self._analyze_and_optimize(
                    context, current_results
                )

                # 更新上下文
                context.accumulated_results.extend(current_results)
                context.retrieved_entities.extend(optimization_result['entities'])
                context.query_refinements.extend(optimization_result['query_refinements'])
                context.confidence_history.append(optimization_result['confidence'])

                # 记录轨迹
                trace_entry = {
                    'iteration': iteration + 1,
                    'query': context.current_query,
                    'results_count': len(current_results),
                    'confidence': optimization_result['confidence'],
                    'optimizations': optimization_result['actions']
                }
                search_trace.append(trace_entry)

                # 更新优化动作
                optimization_actions.extend(optimization_result['actions'])

                # 检查是否继续
                if optimization_result['confidence'] >= self.confidence_threshold:
                    logger.info(f"达到置信度阈值: {optimization_result['confidence']}")
                    break

                # 优化查询
                context.current_query = await self._optimize_query(
                    context, optimization_result
                )

                # 去重累积结果
                context.accumulated_results = self._deduplicate_results(
                    context.accumulated_results
                )

            # 生成最终答案
            final_answer = await self._generate_final_answer(context)

            # RAGAS评估
            ragas_metrics = await ragas_evaluator.evaluate(
                query, final_answer, [
                    r.__dict__ for r in context.accumulated_results
                ]
            )

            # 计算最终置信度
            final_confidence = np.mean(context.confidence_history) if context.confidence_history else 0.5

            result = DeepSearchResult(
                query=query,
                final_answer=final_answer,
                all_iterations=all_iterations,
                search_trace=search_trace,
                confidence_score=final_confidence,
                optimization_actions=list(set(optimization_actions)),
                ragas_metrics=ragas_metrics
            )

            logger.info(f"DeepSearch完成: 置信度 {final_confidence:.3f}")
            return result

        except Exception as e:
            logger.error(f"DeepSearch失败: {e}")
            # 返回基础结果
            return await self._fallback_result(query, initial_results or [])

    async def _execute_retrieval(
        self,
        context: SearchContext
    ) -> List[RetrievalResult]:
        """执行检索"""
        try:
            # 使用多策略检索
            results = await retrieval_engine.retrieve(
                query=context.current_query,
                strategies=['vector', 'graph', 'keyword'],
                top_k=20,
                rerank=True
            )

            # 过滤和优化结果
            optimized_results = self._optimize_retrieval_results(results, context)

            return optimized_results

        except Exception as e:
            logger.error(f"检索执行失败: {e}")
            return []

    def _optimize_retrieval_results(
        self,
        results: List[RetrievalResult],
        context: SearchContext
    ) -> List[RetrievalResult]:
        """优化检索结果"""
        if not results:
            return results

        # 多样性过滤
        diverse_results = self._ensure_diversity(results)

        # 实体相关性加权
        entity_weighted_results = self._apply_entity_weighting(
            diverse_results, context.retrieved_entities
        )

        # 时间相关性调整
        time_adjusted_results = self._apply_temporal_weighting(
            entity_weighted_results, context.original_query
        )

        return time_adjusted_results

    def _ensure_diversity(
        self,
        results: List[RetrievalResult],
        diversity_threshold: float = 0.3
    ) -> List[RetrievalResult]:
        """确保结果多样性"""
        if not results:
            return results

        diverse_results = [results[0]]  # 保留第一个结果

        for result in results[1:]:
            # 计算与已选结果的相似度
            is_diverse = True
            for selected in diverse_results:
                similarity = self._calculate_content_similarity(
                    result.content, selected.content
                )
                if similarity > (1 - diversity_threshold):
                    is_diverse = False
                    break

            if is_diverse:
                diverse_results.append(result)

        return diverse_results

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度"""
        # 简单的词重叠度计算
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _apply_entity_weighting(
        self,
        results: List[RetrievalResult],
        entities: List[Dict[str, Any]]
    ) -> List[RetrievalResult]:
        """应用实体相关性加权"""
        if not entities:
            return results

        entity_texts = {e['text'] for e in entities}

        for result in results:
            entity_count = sum(1 for entity in entity_texts if entity in result.content)
            entity_boost = min(entity_count * 0.1, 0.3)  # 最多30%的加权
            result.score = min(result.score + entity_boost, 1.0)

        return results

    def _apply_temporal_weighting(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """应用时间相关性加权"""
        # 检查查询中的时间关键词
        temporal_keywords = ['最近', '2024', '2023', '今年', '去年', '本季度']
        has_temporal_query = any(keyword in query for keyword in temporal_keywords)

        if not has_temporal_query:
            return results

        # 假设元数据中有时间信息
        for result in results:
            metadata = result.metadata or {}
            doc_date = metadata.get('document_date')

            if doc_date:
                # 简单的时间相关性计算
                if '2024' in doc_date:
                    result.score *= 1.2
                elif '2023' in doc_date:
                    result.score *= 1.1

        return results

    async def _analyze_and_optimize(
        self,
        context: SearchContext,
        results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """分析和优化检索结果"""
        analysis = {
            'entities': [],
            'query_refinements': [],
            'confidence': 0.0,
            'actions': []
        }

        # 提取实体
        if context.current_query:
            entities = await financial_entity_extractor.extract_entities(context.current_query)
            analysis['entities'] = [e.__dict__ for e in entities]
            if entities:
                analysis['actions'].append('entity_extraction')

        # 计算置信度
        if results:
            avg_score = np.mean([r.score for r in results])
            diversity = self._calculate_diversity_score(results)
            coverage = self._calculate_coverage_score(context.original_query, results)

            analysis['confidence'] = 0.5 * avg_score + 0.3 * diversity + 0.2 * coverage

            # 根据分数决定优化动作
            if avg_score < 0.6:
                analysis['actions'].append('improve_relevance')
            if diversity < 0.5:
                analysis['actions'].append('increase_diversity')
            if coverage < 0.5:
                analysis['actions'].append('expand_coverage')

        # 生成查询优化建议
        suggestions = await self._generate_query_suggestions(context, results)
        analysis['query_refinements'] = suggestions
        if suggestions:
            analysis['actions'].append('query_refinement')

        return analysis

    def _calculate_diversity_score(self, results: List[RetrievalResult]) -> float:
        """计算多样性分数"""
        if len(results) <= 1:
            return 1.0

        similarities = []
        for i, result1 in enumerate(results):
            for result2 in results[i+1:]:
                similarity = self._calculate_content_similarity(
                    result1.content, result2.content
                )
                similarities.append(similarity)

        # 多样性 = 1 - 平均相似度
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity

    def _calculate_coverage_score(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> float:
        """计算查询覆盖度"""
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0

        covered_words = set()
        all_content = ' '.join([r.content for r in results]).lower()

        for word in query_words:
            if word in all_content:
                covered_words.add(word)

        return len(covered_words) / len(query_words)

    async def _generate_query_suggestions(
        self,
        context: SearchContext,
        results: List[RetrievalResult]
    ) -> List[str]:
        """生成查询优化建议"""
        suggestions = []

        # 基于实体扩展
        if context.retrieved_entities:
            entity_texts = [e['text'] for e in context.retrieved_entities[:3]]
            entity_query = f"{context.original_query} {' '.join(entity_texts)}"
            suggestions.append(entity_query)

        # 基于结果内容的关键词
        if results:
            # 提取高频关键词
            all_words = []
            for result in results[:5]:
                all_words.extend(result.content.lower().split())

            word_counts = {}
            for word in all_words:
                if len(word) > 2:  # 过滤短词
                    word_counts[word] = word_counts.get(word, 0) + 1

            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_words:
                keyword_query = f"{context.original_query} {top_words[0][0]}"
                suggestions.append(keyword_query)

        # 时间扩展
        temporal_suggestions = [
            f"{context.original_query} 2024年",
            f"{context.original_query} 最新",
            f"{context.original_query} 趋势分析"
        ]
        suggestions.extend(temporal_suggestions[:2])

        return suggestions[:3]  # 最多3个建议

    async def _optimize_query(
        self,
        context: SearchContext,
        optimization_result: Dict[str, Any]
    ) -> str:
        """优化查询"""
        # 选择最佳优化建议
        refinements = optimization_result['query_refinements']

        if not refinements:
            return context.original_query

        # 简单策略：选择第一个建议
        # 实际可以基于历史表现选择
        return refinements[0]

    def _deduplicate_results(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """去重结果"""
        if not results:
            return results

        unique_results = []
        seen_content = set()

        for result in results:
            # 使用内容哈希去重
            content_hash = hash(result.content[:100])  # 使用前100字符
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)

        # 按分数排序
        unique_results.sort(key=lambda x: x.score, reverse=True)

        return unique_results

    async def _generate_final_answer(
        self,
        context: SearchContext
    ) -> str:
        """生成最终答案"""
        # 准备上下文
        context_text = ""
        for i, result in enumerate(context.accumulated_results[:10]):  # 限制文档数量
            context_text += f"\n[文档{i+1}]:\n{result.content[:500]}...\n"

        # 构建prompt
        prompt = f"""
基于以下检索到的文档，全面回答用户查询。

原始查询: {context.original_query}
优化后的查询: {context.current_query}
搜索迭代次数: {context.iteration}

相关文档:
{context_text}

请提供准确、全面的答案，包含：
1. 直接回答
2. 相关细节和背景
3. 数据来源说明（如适用）
"""

        try:
            response = await llm_service.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"生成最终答案失败: {e}")
            return "抱歉，无法生成完整的答案，请尝试重新查询。"

    async def _fallback_result(
        self,
        query: str,
        initial_results: List[RetrievalResult]
    ) -> DeepSearchResult:
        """生成回退结果"""
        logger.warning("使用回退结果生成")

        simple_answer = "基于检索到的信息，我无法提供完整的答案。请尝试更具体的查询。"

        return DeepSearchResult(
            query=query,
            final_answer=simple_answer,
            all_iterations=[initial_results] if initial_results else [],
            search_trace=[{'fallback': True}],
            confidence_score=0.3,
            optimization_actions=['fallback_mode'],
            ragas_metrics={'faithfulness': 0.3, 'overall_score': 0.3}
        )


# 全局DeepSearch优化器实例
deepsearch_optimizer = DeepSearchOptimizer()