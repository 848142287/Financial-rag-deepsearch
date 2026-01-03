"""
统一 RAG 服务入口
整合所有 RAG 功能到单一入口点
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
import asyncio

from app.core.logging_utils import get_contextual_logger
from app.core.cache.unified_cache import get_rag_cache

logger = get_contextual_logger(__name__)

class RAGMode(Enum):
    """RAG 模式"""
    # 自动模式：根据查询自动选择最佳策略
    AUTO = "auto"
    # Agentic 模式：多阶段智能检索
    AGENTIC = "agentic"
    # 简单检索：直接向量检索
    SIMPLE = "simple"
    # 混合检索：向量 + 关键词
    HYBRID = "hybrid"
    # 图谱检索：基于知识图谱
    GRAPH = "graph"
    # 深度检索：多轮迭代检索
    DEEP = "deep"

class QueryType(Enum):
    """查询类型"""
    FACTUAL = "factual"        # 事实查询
    ANALYTICAL = "analytical"  # 分析查询
    COMPARISON = "comparison"  # 比较查询
    SUMMARIZATION = "summary"  # 摘要查询
    RECOMMENDATION = "recommend"  # 推荐查询

@dataclass
class RAGQuery:
    """RAG 查询"""
    text: str
    mode: RAGMode = RAGMode.AUTO
    query_type: Optional[QueryType] = None
    retrieval_level: int = 3
    top_k: int = 5
    filters: Dict[str, Any] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    use_cache: bool = True

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}

@dataclass
class RAGResult:
    """RAG 结果"""
    answer: str
    sources: List[Dict[str, Any]]
    query_type: QueryType
    mode_used: RAGMode
    retrieval_details: Dict[str, Any]
    confidence: float
    latency_ms: int
    cached: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'answer': self.answer,
            'sources': self.sources,
            'query_type': self.query_type.value,
            'mode_used': self.mode_used.value,
            'retrieval_details': self.retrieval_details,
            'confidence': self.confidence,
            'latency_ms': self.latency_ms,
            'cached': self.cached,
            'metadata': self.metadata
        }

class UnifiedRAGService:
    """
    统一 RAG 服务
    作为所有 RAG 功能的单一入口点

    整合：
    - consolidated_rag_service.py
    - rag_unified/rag_service.py
    - retrieval/unified_rag_service.py
    - agentic_rag/agentic_rag_system.py
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__)
        self.cache = get_rag_cache()

        # 延迟加载各子服务
        self._agentic_rag = None
        self._simple_retriever = None
        self._hybrid_retriever = None
        self._graph_retriever = None
        self._deep_retriever = None

    async def query(
        self,
        query: Union[str, RAGQuery],
        **kwargs
    ) -> RAGResult:
        """
        执行 RAG 查询

        Args:
            query: 查询文本或 RAGQuery 对象
            **kwargs: 额外参数

        Returns:
            RAGResult 对象
        """
        import time

        # 标准化查询对象
        if isinstance(query, str):
            rag_query = RAGQuery(text=query, **kwargs)
        else:
            rag_query = query

        start_time = time.time()

        try:
            # 检查缓存
            if rag_query.use_cache:
                cached_result = await self._get_from_cache(rag_query)
                if cached_result:
                    cached_result.cached = True
                    self.logger.info(f"缓存命中: {rag_query.text[:50]}...")
                    return cached_result

            # 确定查询模式
            mode = await self._determine_mode(rag_query)

            # 执行查询
            result = await self._execute_query(rag_query, mode)

            # 计算延迟
            result.latency_ms = int((time.time() - start_time) * 1000)

            # 缓存结果
            if rag_query.use_cache:
                await self._store_in_cache(rag_query, result)

            return result

        except Exception as e:
            self.logger.error(f"RAG 查询失败: {str(e)}")
            raise RetrievalError(
                message=f"RAG 查询失败: {str(e)}",
                query=rag_query.text[:100],
                details={'mode': mode.value if 'mode' in locals() else 'unknown'}
            )

    async def _determine_mode(self, query: RAGQuery) -> RAGMode:
        """确定最佳 RAG 模式"""
        if query.mode != RAGMode.AUTO:
            return query.mode

        # 简单的启发式规则
        # 可以用更复杂的 ML 模型替代
        text = query.text.lower()

        # 检测查询类型
        if any(word in text for word in ['比较', '对比', '区别', '差异']):
            query.query_type = QueryType.COMPARISON
            return RAGMode.AGENTIC  # 比较查询需要多源信息

        elif any(word in text for word in ['总结', '摘要', '概括']):
            query.query_type = QueryType.SUMMARIZATION
            return RAGMode.DEEP  # 摘要需要深度检索

        elif any(word in text for word in ['推荐', '建议', '如何']):
            query.query_type = QueryType.RECOMMENDATION
            return RAGMode.HYBRID  # 推荐需要混合检索

        elif len(text.split()) < 5:  # 短查询
            query.query_type = QueryType.FACTUAL
            return RAGMode.SIMPLE

        else:
            query.query_type = QueryType.ANALYTICAL
            return RAGMode.AGENTIC  # 默认使用 Agentic 模式

    async def _execute_query(
        self,
        query: RAGQuery,
        mode: RAGMode
    ) -> RAGResult:
        """执行指定模式的查询"""
        if mode == RAGMode.AGENTIC:
            return await self._agentic_query(query)
        elif mode == RAGMode.SIMPLE:
            return await self._simple_query(query)
        elif mode == RAGMode.HYBRID:
            return await self._hybrid_query(query)
        elif mode == RAGMode.GRAPH:
            return await self._graph_query(query)
        elif mode == RAGMode.DEEP:
            return await self._deep_query(query)
        else:
            # 默认使用 Agentic
            return await self._agentic_query(query)

    async def _agentic_query(self, query: RAGQuery) -> RAGResult:
        """Agentic 模式查询"""
        try:
            from app.services.agentic_rag.agentic_rag_system import AgenticRAGSystem

            if self._agentic_rag is None:
                self._agentic_rag = AgenticRAGSystem()

            result = await self._agentic_rag.query(
                query.text,
                retrieval_level=query.retrieval_level,
                user_id=query.user_id
            )

            return RAGResult(
                answer=result.get('answer', ''),
                sources=result.get('sources', []),
                query_type=query.query_type or QueryType.ANALYTICAL,
                mode_used=RAGMode.AGENTIC,
                retrieval_details=result.get('retrieval_details', {}),
                confidence=result.get('confidence', 0.0),
                latency_ms=0
            )

        except Exception as e:
            self.logger.error(f"Agentic 查询失败: {e}")
            raise

    async def _simple_query(self, query: RAGQuery) -> RAGResult:
        """简单查询（向量检索）"""
        try:
            from app.services.retrieval.unified_retrieval_service import (
                UnifiedRetrievalService,
                RetrievalConfig,
                RetrievalMode
            )

            if self._simple_retriever is None:
                # 配置检索服务
                config = RetrievalConfig(
                    mode=RetrievalMode.FAST,  # 简单模式使用快速检索
                    final_top_k=query.top_k or 10
                )
                self._simple_retriever = UnifiedRetrievalService(config=config)

            # 调用检索服务
            retrieval_result = await self._simple_retriever.retrieve(
                query=query.text,
                mode=RetrievalMode.FAST
            )

            # 提取结果列表
            results = retrieval_result.get("results", [])

            # 生成答案
            answer = await self._generate_answer(query.text, results)

            return RAGResult(
                answer=answer,
                sources=results,
                query_type=QueryType.FACTUAL,
                mode_used=RAGMode.SIMPLE,
                retrieval_details={
                    'count': len(results),
                    'stats': retrieval_result.get('stats', {})
                },
                confidence=0.8,
                latency_ms=int(retrieval_result.get('stats', {}).get('latency', 0) * 1000)
            )

        except Exception as e:
            self.logger.error(f"简单查询失败: {e}")
            raise

    async def _hybrid_query(self, query: RAGQuery) -> RAGResult:
        """混合查询（向量 + 关键词）"""
        # 实现混合检索逻辑
        # 这里简化为调用简单检索
        return await self._simple_query(query)

    async def _graph_query(self, query: RAGQuery) -> RAGResult:
        """图谱查询"""
        try:
            from app.services.unified_knowledge_graph import UnifiedKnowledgeGraph

            if self._graph_retriever is None:
                self._graph_retriever = UnifiedKnowledgeGraph()

            result = await self._graph_retriever.query(query.text)

            return RAGResult(
                answer=result.get('answer', ''),
                sources=result.get('entities', []),
                query_type=QueryType.ANALYTICAL,
                mode_used=RAGMode.GRAPH,
                retrieval_details={},
                confidence=0.75,
                latency_ms=0
            )

        except Exception as e:
            self.logger.error(f"图谱查询失败: {e}")
            raise

    async def _deep_query(self, query: RAGQuery) -> RAGResult:
        """深度查询（多轮迭代）"""
        # 实现深度检索逻辑
        # 这里简化为调用 Agentic 检索
        return await self._agentic_query(query)

    async def _generate_answer(self, query: str, sources: List[Dict]) -> str:
        """生成答案"""
        try:
            from app.services.llm.unified_llm_service import LLMService

            llm = LLMService()
            prompt = f"根据以下内容回答问题：\n\n问题：{query}\n\n"
            for i, source in enumerate(sources[:3], 1):
                prompt += f"来源{i}：{source.get('content', '')[:200]}...\n"

            answer = await llm.generate(prompt)
            return answer

        except Exception as e:
            self.logger.error(f"答案生成失败: {e}")
            return "无法生成答案"

    async def _get_from_cache(self, query: RAGQuery) -> Optional[RAGResult]:
        """从缓存获取结果"""
        try:
            cache_key = self.cache._generate_key('query', query.text, query.mode.value)
            cached = await self.cache.get(cache_key)

            if cached:
                return RAGResult(**cached)

        except Exception as e:
            self.logger.warning(f"缓存获取失败: {e}")

        return None

    async def _store_in_cache(self, query: RAGQuery, result: RAGResult):
        """存储结果到缓存"""
        try:
            cache_key = self.cache._generate_key('query', query.text, query.mode.value)
            await self.cache.set(cache_key, result.to_dict(), ttl=1800)

        except Exception as e:
            self.logger.warning(f"缓存存储失败: {e}")

    async def batch_query(
        self,
        queries: List[Union[str, RAGQuery]],
        **kwargs
    ) -> List[RAGResult]:
        """
        批量查询

        Args:
            queries: 查询列表
            **kwargs: 共享参数

        Returns:
            RAGResult 列表
        """
        tasks = [self.query(q, **kwargs) for q in queries]
        return await asyncio.gather(*tasks)

    async def get_feedback(self, query_id: str) -> Dict[str, Any]:
        """获取查询反馈"""
        # 实现反馈获取逻辑
        return {'query_id': query_id, 'feedback': []}

    async def submit_feedback(
        self,
        query_id: str,
        rating: int,
        comment: Optional[str] = None
    ) -> bool:
        """提交反馈"""
        # 实现反馈提交逻辑
        return True

# 全局单例
_unified_rag_service: Optional[UnifiedRAGService] = None

def get_rag_service() -> UnifiedRAGService:
    """获取统一 RAG 服务实例"""
    global _unified_rag_service
    if _unified_rag_service is None:
        _unified_rag_service = UnifiedRAGService()
    return _unified_rag_service

# 便捷函数
async def rag_query(
    query: str,
    mode: RAGMode = RAGMode.AUTO,
    **kwargs
) -> RAGResult:
    """
    RAG 查询的便捷函数

    Example:
        result = await rag_query("什么是机器学习？")
        print(result.answer)

        # 使用特定模式
        result = await rag_query(
            "比较深度学习和机器学习",
            mode=RAGMode.AGENTIC
        )
    """
    service = get_rag_service()
    rag_query = RAGQuery(text=query, mode=mode, **kwargs)
    return await service.query(rag_query)

async def simple_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    简单搜索的便捷函数

    Example:
        results = await simple_search("神经网络", top_k=10)
    """
    result = await rag_query(query, mode=RAGMode.SIMPLE, top_k=top_k)
    return result.sources
