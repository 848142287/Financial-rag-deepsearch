"""
Agentic RAG 执行阶段
并行执行多路检索，融合结果
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
import uuid

from .planner import RetrievalPlan, RetrievalStrategy
from ..document_processing.content_filter import DocumentContentFilter

logger = logging.getLogger(__name__)


class RetrievalMethod(Enum):
    """检索方法"""
    VECTOR_SEARCH = "vector_search"       # 向量检索
    GRAPH_SEARCH = "graph_search"         # 图谱检索
    KEYWORD_SEARCH = "keyword_search"     # 关键词检索
    TEMPORAL_SEARCH = "temporal_search"   # 时序检索
    HYBRID_SEARCH = "hybrid_search"       # 混合检索
    FUZZY_SEARCH = "fuzzy_search"         # 模糊检索
    SEMANTIC_SEARCH = "semantic_search"   # 语义检索
    MULTI_MODAL_SEARCH = "multi_modal_search"  # 多模态检索


@dataclass
class RetrievalResult:
    """检索结果"""
    method: RetrievalMethod
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    query_used: str = ""
    retrieval_time: float = 0.0


@dataclass
class FusedResult:
    """融合结果"""
    content: str
    sources: List[str]
    overall_score: float
    method_contributions: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """执行结果"""
    plan_id: str
    original_query: str
    retrieval_results: List[RetrievalResult]
    fused_results: List[FusedResult]
    execution_stats: Dict[str, Any]
    total_time: float
    success: bool
    error_message: Optional[str] = None


class AgenticRAGExecutor:
    """Agentic RAG 执行器"""

    def __init__(self):
        # 初始化检索器
        self.retrievers = self._initialize_retrievers()

        # 内容过滤器
        self.content_filter = DocumentContentFilter()

        # 结果融合器
        self.fusion_weights = {
            RetrievalMethod.VECTOR_SEARCH: 0.3,
            RetrievalMethod.GRAPH_SEARCH: 0.35,
            RetrievalMethod.KEYWORD_SEARCH: 0.2,
            RetrievalMethod.TEMPORAL_SEARCH: 0.15
        }

    def _initialize_retrievers(self) -> Dict[RetrievalMethod, Any]:
        """初始化检索器"""
        retrievers = {}

        try:
            # 向量检索器
            from app.services.knowledge_base.vector_store import VectorStoreManager
            retrievers[RetrievalMethod.VECTOR_SEARCH] = VectorStoreManager()
            logger.info("向量检索器初始化成功")
        except ImportError:
            logger.warning("向量检索器初始化失败，使用模拟实现")
            retrievers[RetrievalMethod.VECTOR_SEARCH] = None

        try:
            # 图谱检索器
            from app.services.knowledge_base.knowledge_graph import KnowledgeGraphManager
            retrievers[RetrievalMethod.GRAPH_SEARCH] = KnowledgeGraphManager()
            logger.info("图谱检索器初始化成功")
        except ImportError:
            logger.warning("图谱检索器初始化失败，使用模拟实现")
            retrievers[RetrievalMethod.GRAPH_SEARCH] = None

        try:
            # 关键词检索器
            from app.services.search.keyword_searcher import KeywordSearcher
            retrievers[RetrievalMethod.KEYWORD_SEARCH] = KeywordSearcher()
            logger.info("关键词检索器初始化成功")
        except ImportError:
            logger.warning("关键词检索器初始化失败，使用模拟实现")
            retrievers[RetrievalMethod.KEYWORD_SEARCH] = None

        return retrievers

    async def execute_plan(self, plan: RetrievalPlan) -> ExecutionResult:
        """
        执行检索计划

        Args:
            plan: 检索计划

        Returns:
            ExecutionResult: 执行结果
        """
        start_time = time.time()
        execution_id = str(uuid.uuid4())

        try:
            logger.info(f"开始执行检索计划: {plan.plan_id}")

            # 1. 执行检索任务
            all_results = await self._execute_retrieval_tasks(plan)

            # 2. 内容过滤
            filtered_results = await self._filter_results(all_results)

            # 3. 结果融合
            fused_results = await self._fuse_results(filtered_results, plan)

            # 4. 统计信息
            execution_stats = self._calculate_execution_stats(all_results, fused_results)

            total_time = time.time() - start_time

            result = ExecutionResult(
                plan_id=plan.plan_id,
                original_query=plan.query_analysis.original_query,
                retrieval_results=filtered_results,
                fused_results=fused_results,
                execution_stats=execution_stats,
                total_time=total_time,
                success=True
            )

            logger.info(f"检索计划执行完成: {plan.plan_id}, 耗时: {total_time:.2f}秒")
            return result

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"检索计划执行失败 {plan.plan_id}: {str(e)}")

            return ExecutionResult(
                plan_id=plan.plan_id,
                original_query=plan.query_analysis.original_query,
                retrieval_results=[],
                fused_results=[],
                execution_stats={"error": str(e)},
                total_time=total_time,
                success=False,
                error_message=str(e)
            )

    async def _execute_retrieval_tasks(self, plan: RetrievalPlan) -> List[RetrievalResult]:
        """执行检索任务"""
        all_results = []

        # 并行执行主要策略
        primary_strategy = plan.strategies[0]
        primary_results = await self._execute_strategy(primary_strategy, plan.main_query)
        all_results.extend(primary_results)

        # 执行备选查询
        for alt_query in plan.alternative_queries:
            alt_results = await self._execute_strategy(primary_strategy, alt_query)
            all_results.extend(alt_results)

        # 执行备用策略（如果存在）
        for strategy in plan.strategies[1:]:
            strategy_results = await self._execute_strategy(strategy, plan.main_query)
            all_results.extend(strategy_results)

        return all_results

    async def _execute_strategy(self, strategy: RetrievalStrategy, query: str) -> List[RetrievalResult]:
        """执行单个策略"""
        results = []
        method = RetrievalMethod(strategy.primary_method)

        # 执行主要检索方法
        primary_results = await self._execute_single_retrieval(method, query, strategy.parameters)
        results.extend(primary_results)

        # 执行次要检索方法
        for secondary_method_name in strategy.secondary_methods:
            secondary_method = RetrievalMethod(secondary_method_name)
            secondary_results = await self._execute_single_retrieval(secondary_method, query, strategy.parameters)
            results.extend(secondary_results)

        return results

    async def _execute_single_retrieval(self, method: RetrievalMethod, query: str, parameters: Dict[str, Any]) -> List[RetrievalResult]:
        """执行单个检索方法"""
        start_time = time.time()

        try:
            retriever = self.retrievers.get(method)

            if retriever is None:
                # 使用模拟实现
                return await self._simulate_retrieval(method, query, parameters)

            # 根据方法类型执行检索
            if method == RetrievalMethod.VECTOR_SEARCH:
                results = await self._vector_search(retriever, query, parameters)
            elif method == RetrievalMethod.GRAPH_SEARCH:
                results = await self._graph_search(retriever, query, parameters)
            elif method == RetrievalMethod.KEYWORD_SEARCH:
                results = await self._keyword_search(retriever, query, parameters)
            elif method == RetrievalMethod.TEMPORAL_SEARCH:
                results = await self._temporal_search(retriever, query, parameters)
            else:
                results = await self._simulate_retrieval(method, query, parameters)

            # 添加执行时间
            retrieval_time = time.time() - start_time
            for result in results:
                result.retrieval_time = retrieval_time

            return results

        except Exception as e:
            logger.error(f"检索方法 {method.value} 执行失败: {str(e)}")
            return []

    async def _vector_search(self, retriever, query: str, parameters: Dict[str, Any]) -> List[RetrievalResult]:
        """执行向量检索"""
        try:
            top_k = parameters.get("top_k", 10)
            similarity_threshold = parameters.get("similarity_threshold", 0.7)

            search_results = await retriever.search(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )

            results = []
            for item in search_results:
                result = RetrievalResult(
                    method=RetrievalMethod.VECTOR_SEARCH,
                    content=item.get("content", ""),
                    source=item.get("metadata", {}).get("source", ""),
                    score=item.get("score", 0.0),
                    metadata=item.get("metadata", {}),
                    query_used=query
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
            return []

    async def _graph_search(self, retriever, query: str, parameters: Dict[str, Any]) -> List[RetrievalResult]:
        """执行图谱检索"""
        try:
            max_depth = parameters.get("max_depth", 2)
            entity_expansion = parameters.get("entity_expansion", True)

            search_results = await retriever.search_related_entities(
                entities=[query],  # 简化处理
                max_depth=max_depth,
                expand_entities=entity_expansion
            )

            results = []
            for item in search_results:
                result = RetrievalResult(
                    method=RetrievalMethod.GRAPH_SEARCH,
                    content=item.get("description", ""),
                    source="knowledge_graph",
                    score=item.get("weight", 0.0),
                    metadata=item,
                    query_used=query
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"图谱检索失败: {str(e)}")
            return []

    async def _keyword_search(self, retriever, query: str, parameters: Dict[str, Any]) -> List[RetrievalResult]:
        """执行关键词检索"""
        try:
            # 简化的关键词搜索实现
            search_results = await retriever.search_keywords(
                keywords=query.split(),
                max_results=parameters.get("top_k", 10)
            )

            results = []
            for item in search_results:
                result = RetrievalResult(
                    method=RetrievalMethod.KEYWORD_SEARCH,
                    content=item.get("content", ""),
                    source=item.get("source", ""),
                    score=item.get("relevance", 0.0),
                    metadata=item.get("metadata", {}),
                    query_used=query
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"关键词检索失败: {str(e)}")
            return []

    async def _temporal_search(self, retriever, query: str, parameters: Dict[str, Any]) -> List[RetrievalResult]:
        """执行时序检索"""
        try:
            time_range = parameters.get("time_range", "recent_1_year")
            top_k = parameters.get("top_k", 15)

            # 简化的时序搜索实现
            temporal_query = f"{query} {time_range}"
            search_results = await retriever.search_temporal(
                query=temporal_query,
                time_range=time_range,
                max_results=top_k
            )

            results = []
            for item in search_results:
                result = RetrievalResult(
                    method=RetrievalMethod.TEMPORAL_SEARCH,
                    content=item.get("content", ""),
                    source=item.get("source", ""),
                    score=item.get("relevance", 0.0),
                    metadata={
                        **item.get("metadata", {}),
                        "time_range": time_range
                    },
                    query_used=temporal_query
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"时序检索失败: {str(e)}")
            return []

    async def _simulate_retrieval(self, method: RetrievalMethod, query: str, parameters: Dict[str, Any]) -> List[RetrievalResult]:
        """模拟检索实现"""
        import random

        # 生成模拟结果
        result_count = parameters.get("top_k", 10)
        results = []

        for i in range(min(result_count, 5)):  # 最多5个模拟结果
            content = f"模拟{method.value}结果 {i+1}：{query}的相关信息..."
            score = random.uniform(0.6, 0.9)

            result = RetrievalResult(
                method=method,
                content=content,
                source=f"模拟来源_{i+1}",
                score=score,
                metadata={"simulated": True, "index": i+1},
                query_used=query
            )
            results.append(result)

        return results

    async def _filter_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """过滤检索结果"""
        filtered_results = []

        for result in results:
            # 应用内容过滤器
            filter_result = self.content_filter.filter_content(
                result.content,
                content_type="financial"
            )

            # 只保留质量分数较高的结果
            if filter_result.quality_score >= 0.5:
                # 更新内容
                result.content = filter_result.filtered_content
                # 调整分数
                result.score *= filter_result.quality_score
                result.metadata["filter_quality"] = filter_result.quality_score

                filtered_results.append(result)

        return filtered_results

    async def _fuse_results(self, results: List[RetrievalResult], plan: RetrievalPlan) -> List[FusedResult]:
        """融合检索结果"""
        if not results:
            return []

        # 按方法分组
        method_groups = {}
        for result in results:
            method = result.method.value
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(result)

        # 计算融合权重
        used_methods = set(result.method for result in results)
        active_weights = {method.value: self.fusion_weights.get(method, 0.1) for method in used_methods}

        # 归一化权重
        total_weight = sum(active_weights.values())
        if total_weight > 0:
            active_weights = {k: v/total_weight for k, v in active_weights.items()}

        # 去重和排序
        unique_results = self._deduplicate_and_sort(results, active_weights)

        # 创建融合结果
        fused_results = []
        for result in unique_results[:10]:  # 取前10个结果
            fused_result = FusedResult(
                content=result.content,
                sources=[result.source],
                overall_score=result.score,
                method_contributions={result.method.value: active_weights.get(result.method.value, 0.1)},
                metadata={
                    **result.metadata,
                    "method": result.method.value,
                    "query_used": result.query_used
                }
            )
            fused_results.append(fused_result)

        return fused_results

    def _deduplicate_and_sort(self, results: List[RetrievalResult], weights: Dict[str, float]) -> List[RetrievalResult]:
        """去重和排序"""
        # 内容相似度去重（简化实现）
        unique_results = []
        seen_contents = set()

        for result in results:
            # 生成内容哈希进行去重
            content_hash = hash(result.content[:200])  # 使用前200字符

            if content_hash not in seen_contents:
                # 应用方法权重
                method_weight = weights.get(result.method.value, 0.1)
                result.score *= (1 + method_weight)

                unique_results.append(result)
                seen_contents.add(content_hash)

        # 按分数排序
        unique_results.sort(key=lambda x: x.score, reverse=True)

        return unique_results

    def _calculate_execution_stats(self, original_results: List[RetrievalResult], fused_results: List[FusedResult]) -> Dict[str, Any]:
        """计算执行统计信息"""
        stats = {
            "total_retrieval_results": len(original_results),
            "fused_results": len(fused_results),
            "method_distribution": {},
            "average_score": 0.0,
            "score_distribution": {"high": 0, "medium": 0, "low": 0}
        }

        # 方法分布统计
        for result in original_results:
            method = result.method.value
            stats["method_distribution"][method] = stats["method_distribution"].get(method, 0) + 1

        # 分数统计
        if fused_results:
            scores = [r.overall_score for r in fused_results]
            stats["average_score"] = sum(scores) / len(scores)

            # 分数分布
            for score in scores:
                if score >= 0.8:
                    stats["score_distribution"]["high"] += 1
                elif score >= 0.6:
                    stats["score_distribution"]["medium"] += 1
                else:
                    stats["score_distribution"]["low"] += 1

        return stats


# 全局执行器实例
rag_executor = AgenticRAGExecutor()