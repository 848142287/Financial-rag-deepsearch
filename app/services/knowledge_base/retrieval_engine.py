"""
智能检索引擎
实现多策略检索：向量检索、图谱检索、关键词检索
支持结果融合和DeepSearch优化
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging

from ..embedding_service import embedding_service
from ..milvus_service import milvus_service
from ..neo4j_service import neo4j_service
from ..llm_service import llm_service
from .entity_extractor import financial_entity_extractor
from ...core.redis_client import redis_client

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    document_id: str
    chunk_id: str
    score: float
    source: str  # vector, graph, keyword
    metadata: Dict[str, Any]
    explanation: Optional[str] = None


class MultiStrategyRetrievalEngine:
    """多策略检索引擎"""

    def __init__(self):
        self.retrieval_strategies = {
            'vector': self._vector_retrieve,
            'graph': self._graph_retrieve,
            'keyword': self._keyword_retrieve
        }

        # 融合权重
        self.fusion_weights = {
            'vector': 0.5,
            'graph': 0.3,
            'keyword': 0.2
        }

        # 缓存配置
        self.cache_ttl = 3600  # 1小时

    async def retrieve(
        self,
        query: str,
        strategies: List[str] = None,
        top_k: int = 10,
        rerank: bool = True,
        use_cache: bool = True
    ) -> List[RetrievalResult]:
        """
        多策略检索

        Args:
            query: 查询语句
            strategies: 使用的检索策略
            top_k: 返回结果数量
            rerank: 是否重排序
            use_cache: 是否使用缓存

        Returns:
            检索结果列表
        """
        if not strategies:
            strategies = ['vector', 'graph', 'keyword']

        # 生成缓存key
        cache_key = f"retrieval:{hash(query)}:{','.join(sorted(strategies))}:{top_k}"

        # 尝试从缓存获取
        if use_cache:
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"检索结果命中缓存: {query[:50]}...")
                return cached_result

        logger.info(f"开始多策略检索: {query[:50]}...")

        # 意图识别
        intent = await self._identify_intent(query)
        logger.info(f"查询意图: {intent}")

        # 实体抽取
        entities = await financial_entity_extractor.extract_entities(query)

        # 并行执行多策略检索
        retrieval_tasks = []
        for strategy in strategies:
            if strategy in self.retrieval_strategies:
                task = self.retrieval_strategies[strategy](query, intent, entities, top_k)
                retrieval_tasks.append(task)

        strategy_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

        # 合并结果
        all_results = []
        for i, result in enumerate(strategy_results):
            if isinstance(result, Exception):
                logger.error(f"检索策略 {strategies[i]} 失败: {result}")
                continue
            all_results.extend(result)

        # 结果融合
        if len(strategies) > 1:
            fused_results = self._fuse_results(all_results, strategies)
        else:
            fused_results = all_results

        # 重排序
        if rerank:
            fused_results = await self._rerank_results(query, fused_results, intent)

        # DeepSearch优化
        optimized_results = await self._deep_search_optimize(query, fused_results)

        # 截取top_k结果
        final_results = optimized_results[:top_k]

        # 缓存结果
        if use_cache:
            await self._save_to_cache(cache_key, final_results)

        return final_results

    async def _identify_intent(self, query: str) -> Dict[str, Any]:
        """识别查询意图"""
        # TODO: 实现更精细的意图识别
        intent = {
            'type': 'general',
            'category': 'fact retrieval',  # fact retrieval, analysis, comparison
            'entities': [],
            'time_range': None,
            'confidence': 0.8
        }

        # 简单的意图分类
        if any(keyword in query for keyword in ['比较', '对比', '区别']):
            intent['category'] = 'comparison'
        elif any(keyword in query for keyword in ['分析', '评价', '如何', '为什么']):
            intent['category'] = 'analysis'
        elif any(keyword in query for keyword in ['最新', '最近', '当前']):
            intent['time_range'] = 'recent'

        return intent

    async def _vector_retrieve(
        self,
        query: str,
        intent: Dict[str, Any],
        entities: List,
        top_k: int
    ) -> List[RetrievalResult]:
        """向量检索"""
        try:
            # 生成查询向量
            query_embedding = await embedding_service.generate_embeddings([query])

            # Milvus检索
            search_results = await milvus_service.search(
                query_vectors=query_embedding,
                limit=top_k,
                expr=None  # 可以添加过滤条件
            )

            results = []
            for result in search_results:
                results.append(RetrievalResult(
                    content=result['content'],
                    document_id=result['document_id'],
                    chunk_id=result.get('chunk_id'),
                    score=result['score'],
                    source='vector',
                    metadata=result.get('metadata', {}),
                    explanation=f"语义相似度: {result['score']:.3f}"
                ))

            return results

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    async def _graph_retrieve(
        self,
        query: str,
        intent: Dict[str, Any],
        entities: List,
        top_k: int
    ) -> List[RetrievalResult]:
        """图谱检索"""
        try:
            # 使用实体进行图谱检索
            if not entities:
                return []

            # 获取实体相关的文档
            entity_names = [e.text for e in entities if e.type in ['COMPANY', 'PERSON', 'STOCK']]

            if not entity_names:
                return []

            # Neo4j检索
            graph_results = await neo4j_service.get_related_documents(
                entity_names,
                limit=top_k
            )

            results = []
            for result in graph_results:
                results.append(RetrievalResult(
                    content=result.get('content', ''),
                    document_id=result.get('document_id'),
                    chunk_id=result.get('chunk_id'),
                    score=result.get('score', 0.5),
                    source='graph',
                    metadata={
                        'entities': entity_names,
                        'relations': result.get('relations', [])
                    },
                    explanation=f"通过实体关联: {', '.join(entity_names[:3])}"
                ))

            return results

        except Exception as e:
            logger.error(f"图谱检索失败: {e}")
            return []

    async def _keyword_retrieve(
        self,
        query: str,
        intent: Dict[str, Any],
        entities: List,
        top_k: int
    ) -> List[RetrievalResult]:
        """关键词检索"""
        try:
            # TODO: 实现关键词检索
            # 可以使用Elasticsearch或数据库全文检索

            # 简单实现：从向量结果中按关键词匹配
            # 实际应该使用专门的全文搜索引擎

            results = []
            # 这里返回空列表，表示关键词检索暂未实现
            return results

        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            return []

    def _fuse_results(
        self,
        results: List[RetrievalResult],
        strategies: List[str]
    ) -> List[RetrievalResult]:
        """融合多策略结果"""
        if not results:
            return []

        # 按document_id和chunk_id分组
        grouped_results = {}
        for result in results:
            key = (result.document_id, result.chunk_id)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)

        # 融合分数
        fused_results = []
        for (doc_id, chunk_id), group in grouped_results.items():
            # 加权平均分数
            weighted_score = 0.0
            source_contributions = {}

            for result in group:
                weight = self.fusion_weights.get(result.source, 0.1)
                weighted_score += result.score * weight
                source_contributions[result.source] = result.score

            # 取最高分数的内容作为代表
            representative = max(group, key=lambda x: x.score)

            # 创建融合结果
            fused_result = RetrievalResult(
                content=representative.content,
                document_id=doc_id,
                chunk_id=chunk_id,
                score=weighted_score,
                source='fusion',
                metadata={
                    **representative.metadata,
                    'fusion_sources': source_contributions,
                    'strategies': strategies
                },
                explanation=f"多策略融合分数: {weighted_score:.3f}"
            )

            fused_results.append(fused_result)

        # 按分数排序
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results

    async def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult],
        intent: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """重排序结果"""
        if not results:
            return results

        try:
            # 使用LLM进行重排序
            # 构建重排序prompt
            prompt = f"""
请根据以下查询，对检索结果进行重新排序。

查询：{query}
查询意图：{intent['category']}

检索结果：
{chr(10).join([f"{i+1}. {r.content[:200]}..." for i, r in enumerate(results)])}

请返回重新排序后的结果索引（从0开始），格式为：[index1, index2, index3, ...]
"""

            response = await llm_service.generate_response(prompt)

            # 解析LLM返回的排序
            try:
                # 提取数字索引
                import re
                indices = re.findall(r'\d+', response)
                indices = [int(i) for i in indices if int(i) < len(results)]

                if indices:
                    # 应用新的排序
                    reranked = [results[i] for i in indices]
                    # 添加未排序的结果
                    for i, r in enumerate(results):
                        if i not in indices:
                            reranked.append(r)
                    return reranked
            except Exception as e:
                logger.warning(f"解析LLM重排序结果失败: {e}")

            # 如果重排序失败，保持原排序
            return results

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return results

    async def _deep_search_optimize(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """DeepSearch优化"""
        if not results:
            return results

        try:
            # 评估每个结果的相关性
            optimized_results = []

            for result in results:
                # 计算查询-内容相关性
                relevance_score = await self._calculate_relevance(query, result.content)

                # 更新分数（原分数和相关性分数的加权平均）
                result.score = 0.7 * result.score + 0.3 * relevance_score
                result.metadata['deep_search_score'] = relevance_score

                optimized_results.append(result)

            # 重新排序
            optimized_results.sort(key=lambda x: x.score, reverse=True)
            return optimized_results

        except Exception as e:
            logger.error(f"DeepSearch优化失败: {e}")
            return results

    async def _calculate_relevance(self, query: str, content: str) -> float:
        """计算查询和内容的相关性"""
        try:
            # 使用向量相似度计算相关性
            query_embedding = await embedding_service.generate_embeddings([query])
            content_embedding = await embedding_service.generate_embeddings([content])

            # 计算余弦相似度
            query_vec = np.array(query_embedding[0])
            content_vec = np.array(content_embedding[0])

            dot_product = np.dot(query_vec, content_vec)
            norm_query = np.linalg.norm(query_vec)
            norm_content = np.linalg.norm(content_vec)

            if norm_query == 0 or norm_content == 0:
                return 0.0

            similarity = dot_product / (norm_query * norm_content)
            return float(similarity)

        except Exception as e:
            logger.error(f"计算相关性失败: {e}")
            return 0.5  # 默认中等相关性

    async def _get_from_cache(self, cache_key: str) -> Optional[List[RetrievalResult]]:
        """从缓存获取结果"""
        try:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                # 反序列化RetrievalResult对象
                data = json.loads(cached_data)
                results = []
                for item in data:
                    result = RetrievalResult(
                        content=item['content'],
                        document_id=item['document_id'],
                        chunk_id=item['chunk_id'],
                        score=item['score'],
                        source=item['source'],
                        metadata=item['metadata'],
                        explanation=item.get('explanation')
                    )
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"从缓存获取失败: {e}")
        return None

    async def _save_to_cache(
        self,
        cache_key: str,
        results: List[RetrievalResult]
    ):
        """保存结果到缓存"""
        try:
            # 序列化RetrievalResult对象
            data = []
            for result in results:
                data.append({
                    'content': result.content,
                    'document_id': result.document_id,
                    'chunk_id': result.chunk_id,
                    'score': result.score,
                    'source': result.source,
                    'metadata': result.metadata,
                    'explanation': result.explanation
                })
            serialized_data = json.dumps(data, ensure_ascii=False)
            await redis_client.setex(cache_key, self.cache_ttl, serialized_data)
        except Exception as e:
            logger.error(f"保存到缓存失败: {e}")

    async def retrieve_with_feedback(
        self,
        query: str,
        feedback_history: List[Dict[str, Any]] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """基于反馈的检索"""
        # TODO: 实现基于用户反馈的检索优化
        return await self.retrieve(query, **kwargs)


# 全局检索引擎实例
retrieval_engine = MultiStrategyRetrievalEngine()