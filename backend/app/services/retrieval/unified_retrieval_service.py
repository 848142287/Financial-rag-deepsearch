"""
统一检索服务入口
整合所有检索功能，提供统一接口

保留的服务：
1. OptimizedRetrievalServiceV2 - 优化的混合检索（主要）
2. HybridRetrievalService - RAG检索（辅助）
"""

from typing import List, Dict, Any, Optional
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

class UnifiedRetrievalService:
    """
    统一检索服务

    整合所有检索功能：
    - 向量检索（Milvus）
    - 知识图谱检索（Neo4j）
    - 混合检索
    - 语义检索
    """

    def __init__(self):
        """初始化统一检索服务"""
        self.optimized_retrieval = None
        self.hybrid_retrieval = None
        self._initialized = False

    async def initialize(self):
        """初始化服务"""
        if self._initialized:
            return

        # 初始化优化的检索服务V2

        self.optimized_retrieval = get_optimized_retrieval_v2()
        await self.optimized_retrieval.initialize()

        # 初始化混合检索服务（用于RAG）
        from app.services.rag.retrieval.hybrid_retrieval_service import HybridRetrievalService
        self.hybrid_retrieval = HybridRetrievalService()
        await self.hybrid_retrieval.initialize()

        self._initialized = True
        logger.info("✅ 统一检索服务初始化完成")

    async def search(
        self,
        query: str,
        top_k: int = 10,
        retrieval_mode: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        统一检索入口

        Args:
            query: 查询文本
            top_k: 返回结果数量
            retrieval_mode: 检索模式
                - "hybrid": 混合检索（向量+图谱）- 默认
                - "vector": 仅向量检索
                - "graph": 仅知识图谱检索
                - "semantic": 语义检索
            filters: 过滤条件

        Returns:
            检索结果
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"🔍 统一检索: query='{query}', mode={retrieval_mode}, top_k={top_k}")

        # 根据模式选择检索方法
        if retrieval_mode == "hybrid":
            # 混合检索（使用优化的V2）
            return await self.optimized_retrieval.search(query, top_k, filters)

        elif retrieval_mode == "vector":
            # 仅向量检索
            return await self._search_vector_only(query, top_k, filters)

        elif retrieval_mode == "graph":
            # 仅知识图谱检索
            return await self._search_graph_only(query, top_k)

        elif retrieval_mode == "semantic":
            # 语义检索（使用hybrid_retrieval）
            return await self._search_semantic(query, top_k)

        else:
            logger.warning(f"⚠️ 未知检索模式: {retrieval_mode}，使用混合检索")
            return await self.optimized_retrieval.search(query, top_k, filters)

    async def _search_vector_only(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """仅向量检索"""
        # 使用optimized_retrieval的Milvus检索
        results = await self.optimized_retrieval._search_milvus([query], top_k)

        return {
            'query': query,
            'mode': 'vector',
            'results': results,
            'total_found': len(results)
        }

    async def _search_graph_only(
        self,
        query: str,
        top_k: int
    ) -> Dict[str, Any]:
        """仅知识图谱检索"""
        results = await self.optimized_retrieval._search_neo4j(query, top_k)

        return {
            'query': query,
            'mode': 'graph',
            'results': results,
            'total_found': len(results)
        }

    async def _search_semantic(
        self,
        query: str,
        top_k: int
    ) -> Dict[str, Any]:
        """语义检索"""
        # 使用hybrid_retrieval的语义检索
        results = await self.hybrid_retrieval.semantic_search(query, top_k)

        return {
            'query': query,
            'mode': 'semantic',
            'results': results,
            'total_found': len(results)
        }

    async def get_document_context(
        self,
        document_ids: List[int],
        max_chunks: int = 10
    ) -> Dict[str, Any]:
        """
        获取文档上下文

        Args:
            document_ids: 文档ID列表
            max_chunks: 每个文档最大chunk数

        Returns:
            文档上下文
        """
        # 使用hybrid_retrieval获取上下文
        contexts = await self.hybrid_retrieval.get_document_context(
            document_ids,
            max_chunks
        )

        return {
            'contexts': contexts,
            'total_documents': len(document_ids)
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        if not self._initialized:
            return {
                'status': 'uninitialized',
                'timestamp': None
            }

        # 检查各个服务
        optimized_health = await self.optimized_retrieval.search(
            "健康检查",
            top_k=1
        )

        return {
            'status': 'healthy' if optimized_health else 'degraded',
            'services': {
                'optimized_retrieval': bool(optimized_health),
                'hybrid_retrieval': bool(self.hybrid_retrieval)
            }
        }

# 全局实例
_unified_retrieval_instance: Optional[UnifiedRetrievalService] = None

def get_unified_retrieval_service() -> UnifiedRetrievalService:
    """
    获取统一检索服务实例

    Returns:
        统一检索服务实例
    """
    global _unified_retrieval_instance

    if _unified_retrieval_instance is None:
        _unified_retrieval_instance = UnifiedRetrievalService()
        logger.info("✅ 初始化统一检索服务")

    return _unified_retrieval_instance

# 向后兼容的导出
__all__ = [
    'UnifiedRetrievalService',
    'get_unified_retrieval_service'
]
