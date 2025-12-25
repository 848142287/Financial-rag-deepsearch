"""
Agentic RAG执行阶段
并行执行多路检索，融合结果
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime
import time

from app.services.milvus_service import MilvusService
from app.services.neo4j_service import Neo4jService
from app.services.embedding_service import RerankService
from app.services.progress_tracker import progress_tracker, TaskStatus, TaskStep

logger = logging.getLogger(__name__)


class RetrievalType(Enum):
    """检索类型"""
    VECTOR = "vector"
    GRAPH = "graph"
    KEYWORD = "keyword"


@dataclass
class RetrievalResult:
    """检索结果"""
    retrieval_type: RetrievalType
    query_used: str
    results: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any]
    execution_time_ms: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['retrieval_type'] = self.retrieval_type.value
        return data


@dataclass
class FusedResult:
    """融合结果"""
    task_id: str
    plan_id: str
    final_results: List[Dict[str, Any]]
    retrieval_summary: Dict[str, Any]
    fusion_stats: Dict[str, Any]
    execution_time_ms: float
    quality_score: float
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


class ExecutePhase:
    """执行阶段处理器"""

    def __init__(self):
        self.milvus_service = MilvusService()
        self.neo4j_service = Neo4jService()
        self.rerank_service = RerankService()

        # 缓存热门查询结果
        self.result_cache = {}
        self.cache_ttl = 300  # 5分钟

    async def execute_retrieval(
        self,
        plan,
        document_ids: Optional[List[int]] = None,
        task_id: Optional[str] = None
    ) -> FusedResult:
        """
        执行检索计划

        Args:
            plan: 检索计划
            document_ids: 限定文档ID列表
            task_id: 任务ID（用于进度跟踪）

        Returns:
            融合后的检索结果
        """
        start_time = time.time()

        try:
            logger.info(f"Executing retrieval plan: {plan.task_id}")

            # 更新进度
            if task_id:
                await progress_tracker.update_progress(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,
                    current_step=TaskStep.RETRIEVAL,
                    progress_percentage=10.0,
                    message="开始执行检索计划"
                )

            # 1. 并行执行多路检索
            retrieval_results = await self._execute_parallel_retrieval(
                plan, document_ids, task_id
            )

            # 2. 结果收集与融合
            fused_result = await self._fuse_results(
                plan, retrieval_results, task_id
            )

            # 3. 计算执行时间
            execution_time = (time.time() - start_time) * 1000
            fused_result.execution_time_ms = execution_time

            # 更新进度
            if task_id:
                await progress_tracker.update_progress(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,
                    current_step=TaskStep.SYNTHESIS,
                    progress_percentage=70.0,
                    message=f"检索完成，获得 {len(fused_result.final_results)} 个结果",
                    details={
                        "result_count": len(fused_result.final_results),
                        "quality_score": fused_result.quality_score
                    }
                )

            logger.info(f"Retrieval execution completed: {execution_time:.2f}ms")
            return fused_result

        except Exception as e:
            logger.error(f"Error in Execute Phase: {str(e)}")
            if task_id:
                await progress_tracker.fail_task(task_id, str(e))
            raise

    async def _execute_parallel_retrieval(
        self,
        plan,
        document_ids: Optional[List[int]],
        task_id: Optional[str]
    ) -> List[RetrievalResult]:
        """并行执行多路检索"""
        tasks = []

        # 向量检索任务
        if plan.retrieval_strategy in ['vector_primary', 'hybrid', 'keyword_enhanced']:
            tasks.append(
                self._execute_vector_retrieval(plan, document_ids, task_id)
            )

        # 图谱检索任务
        if plan.retrieval_strategy in ['graph_primary', 'hybrid', 'temporal_focused']:
            tasks.append(
                self._execute_graph_retrieval(plan, document_ids, task_id)
            )

        # 关键词检索任务
        if plan.retrieval_strategy in ['keyword_enhanced', 'hybrid']:
            tasks.append(
                self._execute_keyword_retrieval(plan, document_ids, task_id)
            )

        # 并行执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        retrieval_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Retrieval task failed: {str(result)}")
                retrieval_results.append(
                    RetrievalResult(
                        retrieval_type=RetrievalType.VECTOR,  # 默认类型
                        query_used="",
                        results=[],
                        scores=[],
                        metadata={},
                        execution_time_ms=0,
                        success=False,
                        error=str(result)
                    )
                )
            else:
                retrieval_results.append(result)

        return retrieval_results

    async def _execute_vector_retrieval(
        self,
        plan,
        document_ids: Optional[List[int]],
        task_id: Optional[str]
    ) -> RetrievalResult:
        """执行向量检索"""
        start_time = time.time()
        query = plan.main_queries[0]  # 使用第一个主查询

        try:
            # 检查缓存
            cache_key = f"vector:{hash(query)}:{hash(str(document_ids))}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Using cached vector result for query: {query[:50]}...")
                return cached_result

            # 生成查询向量
            from app.services.embedding_service import embedding_service
            query_vector = await embedding_service.get_embedding(query)

            # 执行向量检索
            search_results = await self.milvus_service.search_vectors(
                collection_name="documents",
                query_vectors=[query_vector],
                top_k=plan.retrieval_params['vector_top_k'],
                expr=self._build_filter_expr(document_ids)
            )

            # 处理结果
            results = []
            scores = []
            for hit in search_results[0]:  # 第一个查询的结果
                results.append({
                    'id': hit.get('id'),
                    'content': hit.get('content', ''),
                    'metadata': hit.get('metadata', {}),
                    'source': 'vector'
                })
                scores.append(hit.get('score', 0.0))

            execution_time = (time.time() - start_time) * 1000

            retrieval_result = RetrievalResult(
                retrieval_type=RetrievalType.VECTOR,
                query_used=query,
                results=results,
                scores=scores,
                metadata={
                    'total_results': len(results),
                    'collection': 'documents',
                    'top_k': plan.retrieval_params['vector_top_k']
                },
                execution_time_ms=execution_time,
                success=True
            )

            # 缓存结果
            await self._cache_result(cache_key, retrieval_result)

            return retrieval_result

        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}")
            return RetrievalResult(
                retrieval_type=RetrievalType.VECTOR,
                query_used=query,
                results=[],
                scores=[],
                metadata={},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )

    async def _execute_graph_retrieval(
        self,
        plan,
        document_ids: Optional[List[int]],
        task_id: Optional[str]
    ) -> RetrievalResult:
        """执行图谱检索"""
        start_time = time.time()
        query = plan.main_queries[0]

        try:
            # 检查缓存
            cache_key = f"graph:{hash(query)}:{hash(str(document_ids))}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Using cached graph result for query: {query[:50]}...")
                return cached_result

            # 提取实体
            from app.services.knowledge_base.entity_extractor import EntityExtractor
            entity_extractor = EntityExtractor()
            entities = await entity_extractor.extract_entities(query)

            # 执行图谱查询
            graph_results = []
            for entity in entities[:5]:  # 最多使用5个实体
                entity_nodes = await self.neo4j_service.search_entity(
                    entity_name=entity.text,
                    entity_type=entity.type,
                    limit=10
                )
                graph_results.extend(entity_nodes)

            # 获取相关文档
            document_ids_set = set()
            for node in graph_results:
                if 'document_id' in node:
                    document_ids_set.add(node['document_id'])

            # 获取文档内容
            if document_ids_set:
                from app.models.document import DocumentChunk
                documents = await self._get_documents_by_ids(list(document_ids_set))
                graph_results.extend(documents)

            # 计算相关性分数
            results = []
            scores = []
            for result in graph_results:
                # 简单的评分策略
                score = 0.5
                if 'content' in result and query in result['content']:
                    score += 0.3
                if 'relevance' in result:
                    score = result['relevance']

                results.append({
                    'id': result.get('id'),
                    'content': result.get('content', ''),
                    'metadata': result.get('metadata', {}),
                    'source': 'graph',
                    'entities': [e.text for e in entities]
                })
                scores.append(score)

            execution_time = (time.time() - start_time) * 1000

            retrieval_result = RetrievalResult(
                retrieval_type=RetrievalType.GRAPH,
                query_used=query,
                results=results,
                scores=scores,
                metadata={
                    'total_results': len(results),
                    'entities_used': [e.text for e in entities],
                    'entity_types': [e.type for e in entities]
                },
                execution_time_ms=execution_time,
                success=True
            )

            # 缓存结果
            await self._cache_result(cache_key, retrieval_result)

            return retrieval_result

        except Exception as e:
            logger.error(f"Graph retrieval failed: {str(e)}")
            return RetrievalResult(
                retrieval_type=RetrievalType.GRAPH,
                query_used=query,
                results=[],
                scores=[],
                metadata={},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )

    async def _execute_keyword_retrieval(
        self,
        plan,
        document_ids: Optional[List[int]],
        task_id: Optional[str]
    ) -> RetrievalResult:
        """执行关键词检索"""
        start_time = time.time()
        query = plan.main_queries[0]

        try:
            # 检查缓存
            cache_key = f"keyword:{hash(query)}:{hash(str(document_ids))}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Using cached keyword result for query: {query[:50]}...")
                return cached_result

            # 提取关键词
            keywords = self._extract_keywords(query)

            # 执行关键词搜索（这里简化为基于文本匹配）
            from app.models.document import DocumentChunk
            # 这里应该调用实际的搜索引擎
            # 暂时返回空结果
            results = []
            scores = []

            execution_time = (time.time() - start_time) * 1000

            retrieval_result = RetrievalResult(
                retrieval_type=RetrievalType.KEYWORD,
                query_used=query,
                results=results,
                scores=scores,
                metadata={
                    'keywords': keywords,
                    'total_results': len(results)
                },
                execution_time_ms=execution_time,
                success=True
            )

            # 缓存结果
            await self._cache_result(cache_key, retrieval_result)

            return retrieval_result

        except Exception as e:
            logger.error(f"Keyword retrieval failed: {str(e)}")
            return RetrievalResult(
                retrieval_type=RetrievalType.KEYWORD,
                query_used=query,
                results=[],
                scores=[],
                metadata={},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )

    async def _fuse_results(
        self,
        plan,
        retrieval_results: List[RetrievalResult],
        task_id: Optional[str]
    ) -> FusedResult:
        """融合检索结果"""
        # 更新进度
        if task_id:
            await progress_tracker.update_progress(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                current_step=TaskStep.EVIDENCE_COLLECTION,
                progress_percentage=40.0,
                message="正在融合检索结果"
            )

        # 1. 去重
        deduplicated_results = self._deduplicate_results(retrieval_results)

        # 2. 加权融合
        fused_scores = self._weighted_fusion(
            deduplicated_results,
            plan.retrieval_params['fusion_weights']
        )

        # 3. 重排序
        if plan.retrieval_params.get('rerank', False):
            final_results = await self._rerank_results(
                plan.processed_query,
                deduplicated_results,
                fused_scores
            )
        else:
            final_results = sorted(
                zip(deduplicated_results, fused_scores),
                key=lambda x: x[1],
                reverse=True
            )[:plan.estimated_results]

        # 4. 多样化（如果启用）
        if plan.retrieval_params.get('diversify', False):
            final_results = self._diversify_results(final_results)

        # 5. 计算质量分数
        quality_score = self._calculate_quality_score(final_results, plan.quality_threshold)

        # 创建融合结果
        fused_result = FusedResult(
            task_id=task_id or "unknown",
            plan_id=plan.task_id,
            final_results=[{
                'content': result[0]['content'],
                'score': result[1],
                'metadata': result[0].get('metadata', {}),
                'sources': result[0].get('sources', ['vector'])
            } for result in final_results],
            retrieval_summary={
                'vector_results': sum(1 for r in retrieval_results if r.retrieval_type == RetrievalType.VECTOR),
                'graph_results': sum(1 for r in retrieval_results if r.retrieval_type == RetrievalType.GRAPH),
                'keyword_results': sum(1 for r in retrieval_results if r.retrieval_type == RetrievalType.KEYWORD),
                'total_before_fusion': sum(len(r.results) for r in retrieval_results),
                'total_after_fusion': len(final_results)
            },
            fusion_stats={
                'fusion_weights': plan.retrieval_params['fusion_weights'],
                'deduplication_ratio': 1 - len(final_results) / max(sum(len(r.results) for r in retrieval_results), 1),
                'avg_score': sum(r[1] for r in final_results) / len(final_results) if final_results else 0
            },
            execution_time_ms=0,  # 将在调用处设置
            quality_score=quality_score,
            created_at=datetime.utcnow()
        )

        return fused_result

    def _deduplicate_results(
        self,
        retrieval_results: List[RetrievalResult]
    ) -> List[Dict[str, Any]]:
        """去重结果"""
        seen = set()
        deduplicated = []

        for result in retrieval_results:
            for item in result.results:
                # 使用内容哈希去重
                content_hash = hash(item.get('content', ''))
                if content_hash not in seen:
                    seen.add(content_hash)
                    # 合并来源信息
                    if 'sources' not in item:
                        item['sources'] = []
                    item['sources'].append(result.retrieval_type.value)
                    deduplicated.append(item)

        return deduplicated

    def _weighted_fusion(
        self,
        results: List[Dict[str, Any]],
        weights: Dict[str, float]
    ) -> List[float]:
        """加权融合分数"""
        fused_scores = []

        for result in results:
            score = 0.0
            total_weight = 0.0

            sources = result.get('sources', [])
            for source in sources:
                if source in weights:
                    score += weights[source]
                    total_weight += weights[source]

            # 归一化分数
            if total_weight > 0:
                fused_scores.append(score / total_weight)
            else:
                fused_scores.append(0.5)  # 默认分数

        return fused_scores

    async def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        scores: List[float]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """重排序结果"""
        try:
            # 使用重排序服务
            reranked = await self.rerank_service.rerank(
                query=query,
                documents=[r['content'] for r in results],
                initial_scores=scores
            )
            return list(zip(results, reranked))
        except Exception as e:
            logger.error(f"Rerank failed: {str(e)}")
            # 返回原始排序
            return sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

    def _diversify_results(
        self,
        results: List[Tuple[Dict[str, Any], float]],
        diversity_threshold: float = 0.8
    ) -> List[Tuple[Dict[str, Any], float]]:
        """多样化结果"""
        diversified = []
        seen_content = set()

        for result, score in results:
            content = result.get('content', '')
            # 简单的内容相似度检查
            is_similar = False
            for seen in seen_content:
                # 使用简单的词汇重叠作为相似度度量
                content_words = set(content.lower().split())
                seen_words = set(seen.lower().split())
                overlap = len(content_words & seen_words)
                union = len(content_words | seen_words)
                similarity = overlap / union if union > 0 else 0

                if similarity > diversity_threshold:
                    is_similar = True
                    break

            if not is_similar:
                diversified.append((result, score))
                seen_content.add(content)

        return diversified

    def _calculate_quality_score(
        self,
        results: List[Tuple[Dict[str, Any], float]],
        threshold: float
    ) -> float:
        """计算质量分数"""
        if not results:
            return 0.0

        # 平均分数
        avg_score = sum(r[1] for r in results) / len(results)

        # 达到阈值的比例
        above_threshold = sum(1 for r in results if r[1] >= threshold)
        threshold_ratio = above_threshold / len(results)

        # 综合质量分数
        quality_score = (avg_score * 0.7 + threshold_ratio * 0.3)

        return min(1.0, quality_score)

    def _build_filter_expr(self, document_ids: Optional[List[int]]) -> str:
        """构建过滤表达式"""
        if not document_ids:
            return ""

        # 构建文档ID过滤
        id_filters = [f"document_id == {doc_id}" for doc_id in document_ids]
        return " or ".join(id_filters)

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        import re
        # 提取中文和英文单词
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', query)
        # 过滤停用词
        stop_words = {'的', '是', '在', '了', '和', '与', 'the', 'is', 'at', 'which', 'on'}
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        return keywords[:10]  # 最多10个关键词

    async def _get_cached_result(self, cache_key: str) -> Optional[RetrievalResult]:
        """获取缓存结果"""
        # 这里应该使用Redis或其他缓存系统
        # 暂时使用内存缓存
        cached = self.result_cache.get(cache_key)
        if cached and (time.time() - cached['timestamp']) < self.cache_ttl:
            return cached['result']
        return None

    async def _cache_result(self, cache_key: str, result: RetrievalResult):
        """缓存结果"""
        self.result_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }

    async def _get_documents_by_ids(self, document_ids: List[int]) -> List[Dict]:
        """根据ID获取文档"""
        # 这里应该查询数据库
        # 暂时返回空列表
        return []