"""
统一向量存储服务入口
整合所有向量存储功能

保留的实现：
1. MilvusVectorStore - 主要向量存储
2. EnhancedMilvusService - 兼容层
"""

from typing import List, Dict, Any, Optional
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class UnifiedVectorStoreService:
    """
    统一向量存储服务

    功能：
    - 向量存储（Milvus）
    - 向量检索
    - 向量删除
    - 集合管理
    """

    def __init__(self):
        """初始化服务"""
        self.milvus_store = None
        self._initialized = False

    async def initialize(self):
        """初始化服务"""
        if self._initialized:
            return

        # 使用Milvus向量存储（重构版）
        from app.services.vectorstore.milvus_vector_store import MilvusVectorStore

        self.milvus_store = MilvusVectorStore()
        await self.milvus_store.initialize()

        self._initialized = True
        logger.info("✅ 统一向量存储服务初始化完成")

    async def insert_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        插入文档向量

        Args:
            collection_name: 集合名称
            documents: 文档列表，每个文档包含:
                - document_id: 文档ID
                - chunk_index: 块索引
                - text: 文本内容
                - vector: 向量数据
                - metadata: 元数据

        Returns:
            插入结果
        """
        if not self._initialized:
            await self.initialize()

        return await self.milvus_store.insert_documents(
            collection_name=collection_name,
            documents=documents
        )

    async def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 10,
        expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        向量检索

        Args:
            collection_name: 集合名称
            query_vectors: 查询向量列表
            top_k: 返回结果数量
            expr: 过滤表达式

        Returns:
            检索结果列表
        """
        if not self._initialized:
            await self.initialize()

        return await self.milvus_store.search(
            collection_name=collection_name,
            query_vectors=query_vectors,
            top_k=top_k,
            expr=expr
        )

    async def delete_documents(
        self,
        collection_name: str,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """
        删除文档向量

        Args:
            collection_name: 集合名称
            document_ids: 文档ID列表

        Returns:
            删除结果
        """
        if not self._initialized:
            await self.initialize()

        return await self.milvus_store.delete_documents(
            collection_name=collection_name,
            document_ids=document_ids
        )

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        description: str = ""
    ) -> bool:
        """
        创建集合

        Args:
            collection_name: 集合名称
            dimension: 向量维度
            description: 描述

        Returns:
            是否成功
        """
        if not self._initialized:
            await self.initialize()

        return await self.milvus_store.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            description=description
        )

    async def drop_collection(
        self,
        collection_name: str
    ) -> bool:
        """
        删除集合

        Args:
            collection_name: 集合名称

        Returns:
            是否成功
        """
        if not self._initialized:
            await self.initialize()

        return await self.milvus_store.drop_collection(
            collection_name=collection_name
        )

    async def get_collection_stats(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """
        获取集合统计信息

        Args:
            collection_name: 集合名称

        Returns:
            统计信息
        """
        if not self._initialized:
            await self.initialize()

        return await self.milvus_store.get_collection_stats(
            collection_name=collection_name
        )

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            if not self._initialized:
                await self.initialize()

            # 检查Milvus连接
            stats = await self.milvus_store.get_collection_stats("financial_documents")

            return {
                'status': 'healthy',
                'milvus_connected': True,
                'collection_stats': stats
            }
        except Exception as e:
            logger.error(f"❌ 健康检查失败: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


# 全局实例
_unified_vector_store_instance: Optional[UnifiedVectorStoreService] = None


def get_unified_vector_store() -> UnifiedVectorStoreService:
    """
    获取统一向量存储服务实例

    Returns:
        统一向量存储服务实例
    """
    global _unified_vector_store_instance

    if _unified_vector_store_instance is None:
        _unified_vector_store_instance = UnifiedVectorStoreService()
        logger.info("✅ 初始化统一向量存储服务")

    return _unified_vector_store_instance


# 向后兼容的导出
__all__ = [
    'UnifiedVectorStoreService',
    'get_unified_vector_store'
]
