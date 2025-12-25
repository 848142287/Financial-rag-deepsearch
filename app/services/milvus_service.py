"""
Milvus向量数据库服务
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
import logging
import json
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class MilvusService:
    """Milvus向量数据库服务"""

    # 向量维度配置 - 支持不同模型的向量维度
    EMBEDDING_DIMENSIONS = {
        "bge-large-zh-v1.5": 1024,
        "bge-base-zh-v1.5": 768,
        "text-embedding-v4": 1536,
        "text-embedding-v3": 1024,
        "qwen2.5-vl-embedding": 1536,
        "default": 1536  # 默认使用1536维
    }

    def __init__(self, embedding_model: str = "text-embedding-v4"):
        self.collection_name = settings.milvus_collection_name
        self.host = settings.milvus_host
        self.port = settings.milvus_port
        self.collection = None
        self.embedding_model = embedding_model
        # 根据模型名称获取向量维度，默认1536
        self.embedding_dim = self.EMBEDDING_DIMENSIONS.get(
            embedding_model,
            self.EMBEDDING_DIMENSIONS["default"]
        )

    async def connect(self):
        """连接到Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"成功连接到Milvus: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise

    async def disconnect(self):
        """断开Milvus连接"""
        try:
            connections.disconnect("default")
            logger.info("已断开Milvus连接")
        except Exception as e:
            logger.error(f"断开Milvus连接失败: {e}")

    async def init_collections(self):
        """初始化集合"""
        try:
            await self.connect()

            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                # 检查集合schema是否匹配
                collection = Collection(self.collection_name)
                schema = collection.schema

                # 检查embedding字段维度
                embedding_dim = None
                for field in schema.fields:
                    if field.name == "embedding":
                        embedding_dim = field.params.get("dim")
                        break

                # 如果维度不匹配，重新创建集合
                if embedding_dim and embedding_dim != self.embedding_dim:
                    logger.warning(f"集合维度不匹配(当前:{embedding_dim}, 期望:{self.embedding_dim})，重新创建集合")
                    utility.drop_collection(self.collection_name)
                else:
                    self.collection = collection
                    # 创建索引（如果不存在）
                    if not collection.indexes:
                        await self._create_index()
                    collection.load()
                    logger.info(f"集合 {self.collection_name} 已存在并加载")
                    return

            # 创建新集合
            if not utility.has_collection(self.collection_name):
                await self._create_collection()

            # 创建索引
            await self._create_index()

            # 加载集合
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"集合 {self.collection_name} 初始化完成")

        except Exception as e:
            logger.error(f"初始化Milvus集合失败: {e}")
            raise

    async def _create_collection(self):
        """创建集合"""
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.INT64),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.INT64)
        ]

        # 创建集合schema
        schema = CollectionSchema(
            fields=fields,
            description=f"文档嵌入向量集合 (维度:{self.embedding_dim}, 模型:{self.embedding_model})"
        )

        # 创建集合
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        logger.info(f"创建集合 {self.collection_name} 成功 (向量维度:{self.embedding_dim})")

    async def _create_index(self):
        """创建索引"""
        # 为向量字段创建IVF_FLAT索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        logger.info("向量索引创建成功")

    async def insert_embeddings(
        self,
        document_id: int,
        chunks: List[Dict[str, Any]]
    ) -> List[int]:
        """
        插入嵌入向量

        Args:
            document_id: 文档ID
            chunks: 文档块列表

        Returns:
            插入的记录ID列表
        """
        try:
            entities = []

            for i, chunk in enumerate(chunks):
                entities.append({
                    "document_id": document_id,
                    "chunk_id": chunk.get("chunk_index", i),
                    "content": chunk["content"][:65535],  # 限制长度
                    "embedding": chunk["embedding"],
                    "metadata": {
                        "chunk_index": chunk.get("chunk_index", i),
                        "start_char": chunk.get("metadata", {}).get("start_char", 0),
                        "end_char": chunk.get("metadata", {}).get("end_char", 0)
                    },
                    "created_at": int(datetime.now().timestamp() * 1000)
                })

            # 批量插入
            result = self.collection.insert(entities)

            # 刷新使数据可见
            self.collection.flush()

            logger.info(f"成功插入 {len(entities)} 个嵌入向量，文档ID: {document_id}")
            return result.primary_keys

        except Exception as e:
            logger.error(f"插入嵌入向量失败: {e}")
            raise

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        document_ids: Optional[List[int]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        向量搜索

        Args:
            query_embedding: 查询向量
            limit: 返回结果数量
            document_ids: 限定文档ID范围
            score_threshold: 相似度阈值

        Returns:
            搜索结果列表
        """
        try:
            # 确保集合已初始化
            if self.collection is None:
                await self.init_collections()

            # 构建搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }

            # 构建表达式（过滤条件）
            expr = None
            if document_ids:
                doc_ids_str = ", ".join(map(str, document_ids))
                expr = f"document_id in [{doc_ids_str}]"

            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=[
                    "document_id", "chunk_id", "content", "metadata", "created_at"
                ]
            )

            # 处理搜索结果
            search_results = []
            for hit in results[0]:
                if hit.score >= score_threshold:
                    search_results.append({
                        "document_id": hit.entity.get("document_id"),
                        "chunk_id": hit.entity.get("chunk_id"),
                        "content": hit.entity.get("content"),
                        "score": float(hit.score),
                        "metadata": hit.entity.get("metadata", {}),
                        "created_at": hit.entity.get("created_at")
                    })

            logger.info(f"向量搜索完成，返回 {len(search_results)} 个结果")
            return search_results

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            raise

    async def delete_document(self, document_id: int):
        """删除文档的所有向量"""
        try:
            # 构建删除表达式
            expr = f"document_id == {document_id}"

            # 执行删除
            self.collection.delete(expr)

            # 刷新
            self.collection.flush()

            logger.info(f"删除文档 {document_id} 的向量数据成功")

        except Exception as e:
            logger.error(f"删除文档向量失败: {e}")
            raise

    async def update_document(
        self,
        document_id: int,
        chunks: List[Dict[str, Any]]
    ) -> List[int]:
        """更新文档向量"""
        try:
            # 先删除旧的向量
            await self.delete_document(document_id)

            # 插入新的向量
            return await self.insert_embeddings(document_id, chunks)

        except Exception as e:
            logger.error(f"更新文档向量失败: {e}")
            raise

    async def get_document_stats(self, document_id: int) -> Dict[str, Any]:
        """获取文档统计信息"""
        try:
            # 构建查询表达式
            expr = f"document_id == {document_id}"

            # 执行查询
            results = self.collection.query(
                expr=expr,
                output_fields=["chunk_id", "created_at"],
                limit=10000  # 设置足够大的限制
            )

            stats = {
                "document_id": document_id,
                "total_chunks": len(results),
                "first_chunk_time": None,
                "last_chunk_time": None
            }

            if results:
                timestamps = [r.get("created_at") for r in results]
                stats["first_chunk_time"] = min(timestamps)
                stats["last_chunk_time"] = max(timestamps)

            return stats

        except Exception as e:
            logger.error(f"获取文档统计信息失败: {e}")
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            stats = {
                "collection_name": self.collection_name,
                "total_entities": self.collection.num_entities,
                "index_status": "已创建"
            }

            return stats

        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            raise

    async def search_vectors(
        self,
        query_embedding: List[float],
        limit: int = 10,
        document_ids: Optional[List[int]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        向量搜索（search_vectors方法别名）

        Args:
            query_embedding: 查询向量
            limit: 返回结果数量
            document_ids: 限定文档ID范围
            score_threshold: 相似度阈值

        Returns:
            搜索结果列表
        """
        return await self.search(
            query_embedding=query_embedding,
            limit=limit,
            document_ids=document_ids,
            score_threshold=score_threshold
        )

    async def hybrid_search(
        self,
        query_embedding: List[float],
        keywords: List[str],
        limit: int = 10,
        document_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        混合搜索（向量+关键词）

        Args:
            query_embedding: 查询向量
            keywords: 关键词列表
            limit: 返回结果数量
            document_ids: 限定文档ID范围

        Returns:
            搜索结果列表
        """
        try:
            # 向量搜索
            vector_results = await self.search(
                query_embedding=query_embedding,
                limit=limit * 2,  # 获取更多结果用于重排序
                document_ids=document_ids
            )

            # 关键词过滤
            if keywords:
                filtered_results = []
                for result in vector_results:
                    content = result.get("content", "").lower()
                    if any(keyword.lower() in content for keyword in keywords):
                        filtered_results.append(result)
                vector_results = filtered_results

            # 限制结果数量
            return vector_results[:limit]

        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            raise


# 全局Milvus服务实例
milvus_service = MilvusService()