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
from ...core.cache.enhanced_cache import enhanced_cache
from ...core.async_optimizer import async_optimizer

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

        # 融合权重 - 优化配置以提高知识图谱权重
        self.fusion_weights = {
            'vector': 0.45,
            'graph': 0.40,  # 提高知识图谱权重，增强实体关系推理能力
            'keyword': 0.15
        }

        # 知识图谱增强配置 - 长期开启
        self.enable_knowledge_graph = True  # 持续启用知识图谱增强
        self.graph_rag_config = {
            'max_depth': 3,  # 增加图谱遍历深度
            'max_entities': 15,  # 增加实体数量
            'max_relations': 30,  # 增加关系统计
            'use_entity_linking': True,
            'use_relation_traversal': True
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
        多策略检索 - 增强缓存版本

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

        # 生成缓存key - 使用增强缓存服务的键生成方法
        cache_key = enhanced_cache._generate_cache_key(
            "retrieval",
            query,
            ",".join(sorted(strategies)),
            top_k
        )

        # 尝试从多层缓存获取
        if use_cache:
            cached_result = await enhanced_cache.get_multi_layer(cache_key)
            if cached_result:
                logger.info(f"检索结果命中多层缓存: {query[:50]}...")
                return [RetrievalResult(**r) for r in cached_result]

        logger.info(f"开始多策略检索: {query[:50]}...")

        # 意图识别 - 使用缓存
        intent = await self._identify_intent_cached(query)
        logger.info(f"查询意图: {intent}")

        # 实体抽取
        entities = await financial_entity_extractor.extract_entities(query)

        # 并行执行多策略检索 - 使用优化的并发控制
        retrieval_tasks = []
        for strategy in strategies:
            if strategy in self.retrieval_strategies:
                task = self.retrieval_strategies[strategy](query, intent, entities, top_k)
                retrieval_tasks.append(task)

        # 使用异步优化器进行并发控制
        strategy_results = await async_optimizer.execute_with_concurrency_control(
            tasks=retrieval_tasks,
            max_concurrency=3,  # 最多3个策略并行执行
            task_type="retrieval_strategies"
        )

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

        # 缓存结果到L1和L2层
        if use_cache:
            # 序列化结果
            serialized_results = [
                {
                    "content": r.content,
                    "document_id": r.document_id,
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "source": r.source,
                    "metadata": r.metadata,
                    "explanation": r.explanation
                }
                for r in final_results
            ]
            # 存入L1缓存（5分钟）
            await enhanced_cache.set(cache_key, serialized_results, layer="L1")
            # 存入L2缓存（30分钟）
            await enhanced_cache.set(cache_key, serialized_results, layer="L2")

        return final_results

    async def _identify_intent_cached(self, query: str) -> Dict[str, Any]:
        """带缓存的意图识别"""
        cache_key = enhanced_cache._generate_cache_key("intent", query)

        cached_intent = await enhanced_cache.get(cache_key, layer="L1")
        if cached_intent:
            return cached_intent

        intent = await self._identify_intent(query)
        await enhanced_cache.set(cache_key, intent, ttl=600, layer="L1")  # 10分钟

        return intent

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
            query_embeddings = await embedding_service.generate_embeddings([query])
            query_embedding = query_embeddings[0] if query_embeddings else []

            # Milvus检索
            search_results = await milvus_service.search(
                query_embedding=query_embedding,
                limit=top_k
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
        """图谱检索 - 增强版本"""
        try:
            # 知识图谱增强开关检查
            if not self.enable_knowledge_graph:
                logger.info("知识图谱增强功能已禁用")
                return []

            # 使用实体进行图谱检索
            if not entities:
                # 即使没有实体，也尝试从查询中提取关键词进行图谱检索
                logger.info("未检测到实体，尝试基于关键词的图谱检索")
                entity_names = self._extract_key_entities_from_query(query)
            else:
                # 获取实体相关的文档
                entity_names = [e.text for e in entities if e.type in ['COMPANY', 'PERSON', 'STOCK', 'ORGANIZATION', 'LOCATION', 'EVENT']]

            if not entity_names:
                logger.info("未找到相关实体")
                return []

            logger.info(f"使用实体进行知识图谱检索: {entity_names[:5]}")

            # Neo4j检索 - 使用增强配置
            try:
                graph_results = await neo4j_service.get_related_documents(
                    entity_names,
                    limit=top_k,
                    max_depth=self.graph_rag_config['max_depth'],
                    max_relations=self.graph_rag_config['max_relations']
                )
            except Exception as e:
                logger.warning(f"Neo4j图谱检索失败，将使用仅向量检索: {e}")
                return []

            results = []
            for result in graph_results:
                # 计算图谱检索分数 - 考虑关系数量和实体关联度
                base_score = result.get('score', 0.5)
                relation_count = len(result.get('relations', []))
                entity_match_count = len(entity_names)

                # 增强分数计算
                enhanced_score = base_score * (1 + 0.1 * relation_count) * (1 + 0.05 * entity_match_count)
                enhanced_score = min(enhanced_score, 1.0)  # 限制最大值为1.0

                results.append(RetrievalResult(
                    content=result.get('content', ''),
                    document_id=result.get('document_id'),
                    chunk_id=result.get('chunk_id'),
                    score=enhanced_score,
                    source='graph',
                    metadata={
                        'entities': entity_names,
                        'relations': result.get('relations', []),
                        'relation_count': relation_count,
                        'graph_enhancement': True
                    },
                    explanation=f"知识图谱增强: 通过{entity_match_count}个实体和{relation_count}个关系关联"
                ))

            logger.info(f"知识图谱检索完成，返回{len(results)}个结果")
            return results

        except Exception as e:
            logger.error(f"图谱检索失败: {e}", exc_info=True)
            return []

    def _extract_key_entities_from_query(self, query: str) -> List[str]:
        """从查询中提取关键实体"""
        import re
        entities = []

        # 提取大写字母开头的词
        words = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(words)

        # 提取数字+单位
        numbers = re.findall(r'\b\d+[%年月日元]\b', query)
        entities.extend(numbers)

        # 提取中文关键词（简单实现）
        chinese_keywords = re.findall(r'[\u4e00-\u9fa5]{2,4}', query)
        entities.extend(chinese_keywords[:5])

        return list(set(entities))  # 去重

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
        """重排序结果 - 优化版提示词"""
        if not results:
            return results

        try:
            # 使用LLM进行智能重排序
            # 构建优化的重排序提示词
            system_prompt = """你是一个信息检索质量评估专家。你的任务是评估文档片段与用户查询的相关性，并进行排序。

## 评估标准
1. **直接相关性**: 文档是否直接回答了用户的问题
2. **信息完整性**: 是否包含问题所需的完整信息
3. **实体匹配**: 是否包含问题中的关键实体
4. **数据支撑**: 是否提供具体数据支持
5. **时效性**: 信息是否是最新的（如适用）

## 输出要求
请返回排序后的文档索引列表（从0开始），格式为：[0, 3, 1, 2, ...]
- 最相关的排在前面
- 索引必须全部包含，不能遗漏
- 只返回数字列表，不要其他内容"""

            # 构建检索结果描述
            results_desc = []
            for i, r in enumerate(results):
                source_label = {
                    'vector': '向量检索',
                    'graph': '知识图谱',
                    'fusion': '融合检索',
                    'keyword': '关键词检索'
                }.get(r.source, r.source)

                desc = f"""
文档{i}:
- 来源: {source_label}
- 相似度分数: {r.score:.3f}
- 说明: {r.explanation or '无'}
- 内容: {r.content[:300]}...
"""
                results_desc.append(desc)

            user_prompt = f"""## 用户查询
问题: {query}
查询意图: {intent.get('category', 'general')}

## 待排序的检索结果
共{len(results)}个文档片段：
{''.join(results_desc)}

## 任务
请根据与查询的相关性，对上述文档进行重新排序。
返回排序后的索引列表（最相关的在前）。"""

            response = await llm_service.simple_chat(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2  # 使用较低温度以获得稳定结果
            )

            # 解析LLM返回的排序
            try:
                import re
                # 提取数字索引
                indices = re.findall(r'\d+', response)
                indices = [int(i) for i in indices if 0 <= int(i) < len(results)]

                # 验证是否包含所有索引
                if len(indices) == len(results) and set(indices) == set(range(len(results))):
                    # 应用新的排序
                    reranked = [results[i] for i in indices]
                    logger.info(f"LLM重排序成功，新排序: {indices[:5]}...")
                    return reranked
                else:
                    logger.warning(f"LLM重排序索引不完整，使用原排序")

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