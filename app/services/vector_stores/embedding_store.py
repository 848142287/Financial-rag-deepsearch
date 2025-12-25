"""
嵌入向量存储模块
提供嵌入向量的存储和管理功能
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import uuid

from .base_vector_store import BaseVectorStore, VectorData, VectorConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMetadata:
    """嵌入向量元数据"""
    model_name: str
    embedding_dim: int
    created_at: datetime
    source_type: str  # text, image, table, etc.
    chunk_index: Optional[int] = None
    document_id: Optional[str] = None


class EmbeddingStore:
    """嵌入向量存储器

    专门用于存储和管理文档嵌入向量
    """

    def __init__(self, vector_store: BaseVectorStore, config: Optional[VectorConfig] = None):
        """初始化嵌入存储器

        Args:
            vector_store: 向量存储后端
            config: 向量存储配置
        """
        self.vector_store = vector_store
        self.config = config or VectorConfig()
        self.logger = logging.getLogger(__name__)

    async def store_embedding(self,
                            content: str,
                            embedding: List[float],
                            metadata: Optional[Dict[str, Any]] = None,
                            embedding_id: Optional[str] = None) -> str:
        """存储单个嵌入向量

        Args:
            content: 原始内容
            embedding: 嵌入向量
            metadata: 元数据
            embedding_id: 向量ID，如果不提供则自动生成

        Returns:
            嵌入向量ID
        """
        if not embedding_id:
            embedding_id = str(uuid.uuid4())

        vector_data = VectorData(
            id=embedding_id,
            vector=embedding,
            content=content,
            metadata=metadata or {}
        )

        await self.vector_store.insert(vector_data)
        self.logger.info(f"存储嵌入向量: {embedding_id}")
        return embedding_id

    async def store_batch_embeddings(self,
                                    embeddings: List[Tuple[str, List[float], Dict[str, Any]]]) -> List[str]:
        """批量存储嵌入向量

        Args:
            embeddings: (content, embedding, metadata) 元组列表

        Returns:
            嵌入向量ID列表
        """
        vector_data_list = []
        embedding_ids = []

        for content, embedding, metadata in embeddings:
            embedding_id = str(uuid.uuid4())
            vector_data = VectorData(
                id=embedding_id,
                vector=embedding,
                content=content,
                metadata=metadata
            )
            vector_data_list.append(vector_data)
            embedding_ids.append(embedding_id)

        await self.vector_store.batch_insert(vector_data_list)
        self.logger.info(f"批量存储嵌入向量: {len(embedding_ids)} 个")
        return embedding_ids

    async def search_similar(self,
                           query_embedding: List[float],
                           top_k: int = 10,
                           filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索相似嵌入向量

        Args:
            query_embedding: 查询嵌入向量
            top_k: 返回数量
            filter_expr: 过滤表达式

        Returns:
            相似向量列表
        """
        results = await self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
            filter_expr=filter_expr
        )
        return results

    async def get_embedding(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """获取单个嵌入向量

        Args:
            embedding_id: 嵌入向量ID

        Returns:
            嵌入向量数据
        """
        return await self.vector_store.get(embedding_id)

    async def update_embedding(self,
                             embedding_id: str,
                             content: Optional[str] = None,
                             embedding: Optional[List[float]] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """更新嵌入向量

        Args:
            embedding_id: 嵌入向量ID
            content: 新内容
            embedding: 新嵌入向量
            metadata: 新元数据

        Returns:
            是否更新成功
        """
        try:
            update_data = {}
            if content is not None:
                update_data['content'] = content
            if embedding is not None:
                update_data['vector'] = embedding
            if metadata is not None:
                update_data['metadata'] = metadata

            await self.vector_store.update(embedding_id, update_data)
            self.logger.info(f"更新嵌入向量: {embedding_id}")
            return True
        except Exception as e:
            self.logger.error(f"更新嵌入向量失败 {embedding_id}: {e}")
            return False

    async def delete_embedding(self, embedding_id: str) -> bool:
        """删除嵌入向量

        Args:
            embedding_id: 嵌入向量ID

        Returns:
            是否删除成功
        """
        try:
            await self.vector_store.delete(embedding_id)
            self.logger.info(f"删除嵌入向量: {embedding_id}")
            return True
        except Exception as e:
            self.logger.error(f"删除嵌入向量失败 {embedding_id}: {e}")
            return False

    async def get_document_embeddings(self, document_id: str) -> List[Dict[str, Any]]:
        """获取文档的所有嵌入向量

        Args:
            document_id: 文档ID

        Returns:
            嵌入向量列表
        """
        filter_expr = f"metadata.document_id == '{document_id}'"
        return await self.vector_store.search(
            query_vector=[],
            top_k=1000,  # 大数量获取所有向量
            filter_expr=filter_expr
        )

    async def delete_document_embeddings(self, document_id: str) -> int:
        """删除文档的所有嵌入向量

        Args:
            document_id: 文档ID

        Returns:
            删除的向量数量
        """
        try:
            filter_expr = f"metadata.document_id == '{document_id}'"
            deleted_count = await self.vector_store.delete_by_filter(filter_expr)
            self.logger.info(f"删除文档 {document_id} 的嵌入向量: {deleted_count} 个")
            return deleted_count
        except Exception as e:
            self.logger.error(f"删除文档嵌入向量失败 {document_id}: {e}")
            return 0

    async def get_statistics(self) -> Dict[str, Any]:
        """获取嵌入存储统计信息

        Returns:
            统计信息
        """
        try:
            total_count = await self.vector_store.count()
            collection_info = await self.vector_store.get_collection_info()

            return {
                "total_embeddings": total_count,
                "collection_info": collection_info,
                "store_config": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "index_type": self.config.index_type,
                    "metric_type": self.config.metric_type
                }
            }
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}