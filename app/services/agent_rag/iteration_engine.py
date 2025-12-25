"""
迭代检索引擎
实现多轮检索和结果优化
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import logging

from ..knowledge_base.retrieval_engine import retrieval_engine, RetrievalResult

logger = logging.getLogger(__name__)


class IterationEngine:
    """迭代检索引擎"""

    def __init__(self):
        self.max_iterations = 3
        self.improvement_threshold = 0.1
        self.diversity_factor = 0.3

    async def retrieve(
        self,
        query: str,
        strategies: List[str],
        entities: List[Dict[str, Any]],
        accumulated_context: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        执行迭代检索

        Args:
            query: 当前查询
            strategies: 检索策略
            entities: 已识别实体
            accumulated_context: 累积上下文

        Returns:
            各策略的检索结果
        """
        results = {}

        # 并行执行各策略
        tasks = []
        for strategy in strategies:
            task = self._execute_strategy(strategy, query, entities, accumulated_context)
            tasks.append((strategy, task))

        # 等待所有策略完成
        for strategy, task in tasks:
            try:
                strategy_results = await task
                results[strategy] = strategy_results
                logger.info(f"策略 {strategy} 检索到 {len(strategy_results)} 个结果")
            except Exception as e:
                logger.error(f"策略 {strategy} 执行失败: {e}")
                results[strategy] = []

        return results

    async def _execute_strategy(
        self,
        strategy: str,
        query: str,
        entities: List[Dict[str, Any]],
        accumulated_context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """执行单个检索策略"""
        if strategy == 'vector':
            return await self._vector_retrieval(query, entities)
        elif strategy == 'graph':
            return await self._graph_retrieval(query, entities)
        elif strategy == 'keyword':
            return await self._keyword_retrieval(query, entities)
        else:
            logger.warning(f"未知检索策略: {strategy}")
            return []

    async def _vector_retrieval(
        self,
        query: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """向量检索"""
        try:
            # 基础向量检索
            results = await retrieval_engine._vector_retrieve(query, {}, entities, 15)

            # 基于实体增强查询
            if entities:
                entity_names = [e['text'] for e in entities if e.get('type') == 'COMPANY']
                if entity_names:
                    enhanced_query = f"{query} {' '.join(entity_names[:3])}"
                    enhanced_results = await retrieval_engine._vector_retrieve(
                        enhanced_query, {}, entities, 10
                    )
                    results.extend(enhanced_results)

            # 去重和排序
            unique_results = self._deduplicate_vector_results(results)
            return [r.__dict__ for r in unique_results]

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    async def _graph_retrieval(
        self,
        query: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """图谱检索"""
        try:
            # 提取关键实体
            key_entities = [e for e in entities if e.get('type') in ['COMPANY', 'PERSON', 'STOCK']]

            if not key_entities:
                return []

            # 基于实体检索
            results = await retrieval_engine._graph_retrieve(query, {}, key_entities, 10)

            return [r.__dict__ for r in results]

        except Exception as e:
            logger.error(f"图谱检索失败: {e}")
            return []

    async def _keyword_retrieval(
        self,
        query: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """关键词检索"""
        try:
            # 提取关键词
            keywords = self._extract_keywords(query, entities)

            # 执行关键词检索（这里使用简化的实现）
            # 实际应该使用Elasticsearch或类似引擎
            results = await retrieval_engine._keyword_retrieve(query, {}, entities, 20)

            return [r.__dict__ for r in results]

        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            return []

    def _extract_keywords(self, query: str, entities: List[Dict[str, Any]]) -> List[str]:
        """提取关键词"""
        keywords = query.split()

        # 添加实体关键词
        for entity in entities:
            if entity.get('text'):
                keywords.append(entity['text'])

        # 去重和过滤
        unique_keywords = list(set(keywords))
        return [kw for kw in unique_keywords if len(kw) > 1]

    def _deduplicate_vector_results(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """去重向量检索结果"""
        if not results:
            return results

        unique_results = []
        seen_ids = set()

        for result in results:
            result_id = result.document_id + '_' + str(result.chunk_id)
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)

        # 按分数排序
        unique_results.sort(key=lambda x: x.score, reverse=True)

        return unique_results