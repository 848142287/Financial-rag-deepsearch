"""
向量管理器
统一管理所有向量存储和检索操作
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
import numpy as np
from datetime import datetime

from .base_vector_store import VectorConfig, VectorData, SearchResult
from .milvus_store import MilvusVectorStore

logger = logging.getLogger(__name__)


class VectorManager:
    """向量管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.vector_stores = {}
        self.default_store = None

        # 集合配置
        self.collection_configs = {
            "text_chunks": {"dimension": 768, "description": "文本块向量"},
            "entities": {"dimension": 768, "description": "实体向量"},
            "relations": {"dimension": 768, "description": "关系向量"},
            "documents": {"dimension": 768, "description": "文档向量"}
        }

        # 初始化默认向量存储
        self._initialize_default_store()

    def _initialize_default_store(self):
        """初始化默认向量存储"""
        try:
            vector_config = VectorConfig(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 19530),
                db_name=self.config.get("db_name", "knowledge_graph")
            )

            # 使用Milvus作为默认存储
            self.default_store = MilvusVectorStore(vector_config)
            self.vector_stores["default"] = self.default_store

            logger.info("默认向量存储初始化成功")

        except Exception as e:
            logger.error(f"初始化默认向量存储失败: {str(e)}")

    async def initialize(self) -> bool:
        """初始化向量管理器"""
        try:
            # 连接到所有向量存储
            for name, store in self.vector_stores.items():
                success = await store.connect()
                if success:
                    logger.info(f"向量存储 {name} 连接成功")
                else:
                    logger.error(f"向量存储 {name} 连接失败")

            # 创建默认集合
            await self._create_default_collections()

            return True

        except Exception as e:
            logger.error(f"向量管理器初始化失败: {str(e)}")
            return False

    async def _create_default_collections(self):
        """创建默认集合"""
        if not self.default_store:
            return

        for collection_name, config in self.collection_configs.items():
            try:
                await self.default_store.create_collection(
                    collection_name=collection_name,
                    dimension=config["dimension"],
                    collection_type=collection_name.rstrip("s")
                )
                await self.default_store.create_index(
                    collection_name=collection_name,
                    index_type="HNSW"
                )
                logger.info(f"创建集合 {collection_name} 成功")
            except Exception as e:
                logger.warning(f"创建集合 {collection_name} 失败: {str(e)}")

    async def store_text_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """存储文本块向量"""
        if not self.default_store:
            raise RuntimeError("默认向量存储未初始化")

        vectors = []
        for chunk in chunks:
            vector_data = VectorData(
                id=chunk.get("chunk_id", str(len(vectors))),
                vector=chunk.get("embedding", []),
                content=chunk.get("text", ""),
                metadata={
                    "source": chunk.get("source", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "content_type": chunk.get("content_type", "text"),
                    "entities": chunk.get("entities", []),
                    "relations": chunk.get("relations", [])
                },
                confidence=chunk.get("confidence", 1.0)
            )
            vectors.append(vector_data)

        return await self.default_store.insert_vectors("text_chunks", vectors)

    async def store_entities(self, entities: List[Dict[str, Any]]) -> List[str]:
        """存储实体向量"""
        if not self.default_store:
            raise RuntimeError("默认向量存储未初始化")

        vectors = []
        for entity in entities:
            vector_data = VectorData(
                id=entity.get("id", str(len(vectors))),
                vector=entity.get("embedding", []),
                content=entity.get("text", ""),
                metadata={
                    "type": entity.get("type", ""),
                    "properties": entity.get("properties", {}),
                    "source_documents": entity.get("source_documents", []),
                    "confidence": entity.get("confidence", 1.0)
                },
                confidence=entity.get("confidence", 1.0)
            )
            vectors.append(vector_data)

        return await self.default_store.insert_vectors("entities", vectors)

    async def store_relations(self, relations: List[Dict[str, Any]]) -> List[str]:
        """存储关系向量"""
        if not self.default_store:
            raise RuntimeError("默认向量存储未初始化")

        vectors = []
        for relation in relations:
            # 构建关系文本
            relation_text = f"{relation.get('subject', '')} {relation.get('predicate', '')} {relation.get('object', '')}"

            vector_data = VectorData(
                id=relation.get("id", str(len(vectors))),
                vector=relation.get("embedding", []),
                content=relation_text,
                metadata={
                    "subject": relation.get("subject", ""),
                    "predicate": relation.get("predicate", ""),
                    "object": relation.get("object", ""),
                    "relation_type": relation.get("type", ""),
                    "source_documents": relation.get("source_documents", []),
                    "confidence": relation.get("confidence", 1.0)
                },
                confidence=relation.get("confidence", 1.0)
            )
            vectors.append(vector_data)

        return await self.default_store.insert_vectors("relations", vectors)

    async def search_similar_texts(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None
    ) -> List[SearchResult]:
        """搜索相似文本"""
        if not self.default_store:
            raise RuntimeError("默认向量存储未初始化")

        results = await self.default_store.search_vectors(
            collection_name="text_chunks",
            query_vectors=[query_vector],
            top_k=top_k,
            filter_expr=filter_expr
        )

        return results[0] if results else []

    async def search_similar_entities(
        self,
        query_vector: List[float],
        entity_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """搜索相似实体"""
        if not self.default_store:
            raise RuntimeError("默认向量存储未初始化")

        # 构建过滤表达式
        filter_expr = f"entity_type == '{entity_type}'" if entity_type else None

        results = await self.default_store.search_vectors(
            collection_name="entities",
            query_vectors=[query_vector],
            top_k=top_k,
            filter_expr=filter_expr
        )

        return results[0] if results else []

    async def search_similar_relations(
        self,
        query_vector: List[float],
        relation_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """搜索相似关系"""
        if not self.default_store:
            raise RuntimeError("默认向量存储未初始化")

        # 构建过滤表达式
        filter_expr = f"predicate == '{relation_type}'" if relation_type else None

        results = await self.default_store.search_vectors(
            collection_name="relations",
            query_vectors=[query_vector],
            top_k=top_k,
            filter_expr=filter_expr
        )

        return results[0] if results else []

    async def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        search_types: List[str] = None,
        top_k: int = 10
    ) -> Dict[str, List[SearchResult]]:
        """混合搜索"""
        if search_types is None:
            search_types = ["text_chunks", "entities", "relations"]

        results = {}

        for search_type in search_types:
            try:
                search_results = await self.default_store.hybrid_search(
                    collection_name=search_type,
                    query_vector=query_vector,
                    text_query=query,
                    top_k=top_k
                )
                results[search_type] = search_results
            except Exception as e:
                logger.error(f"搜索 {search_type} 失败: {str(e)}")
                results[search_type] = []

        return results

    async def get_recommendations(
        self,
        entity_id: str,
        recommendation_type: str = "similar_entities",
        top_k: int = 5
    ) -> List[SearchResult]:
        """获取推荐"""
        try:
            # 获取实体的向量
            entity_results = await self.default_store.search_vectors(
                collection_name="entities",
                query_vectors=[],
                top_k=1,
                filter_expr=f"id == '{entity_id}'"
            )

            if not entity_results or not entity_results[0]:
                return []

            entity_vector = entity_results[0][0].metadata.get("vector", [])
            if not entity_vector:
                return []

            # 基于推荐类型搜索
            if recommendation_type == "similar_entities":
                return await self.search_similar_entities(entity_vector, top_k=top_k)
            elif recommendation_type == "related_texts":
                return await self.search_similar_texts(entity_vector, top_k=top_k)
            elif recommendation_type == "related_relations":
                return await self.search_similar_relations(entity_vector, top_k=top_k)
            else:
                return []

        except Exception as e:
            logger.error(f"获取推荐失败: {str(e)}")
            return []

    async def update_vector(
        self,
        collection_name: str,
        vector_id: str,
        new_vector: List[float],
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """更新向量"""
        if not self.default_store:
            raise RuntimeError("默认向量存储未初始化")

        # 先获取现有数据
        search_results = await self.default_store.search_vectors(
            collection_name=collection_name,
            query_vectors=[],
            top_k=1,
            filter_expr=f"id == '{vector_id}'"
        )

        if not search_results or not search_results[0]:
            logger.warning(f"未找到向量 {vector_id}")
            return False

        existing_result = search_results[0][0]

        # 创建更新后的向量数据
        vector_data = VectorData(
            id=vector_id,
            vector=new_vector,
            content=existing_result.content,
            metadata=new_metadata or existing_result.metadata,
            confidence=existing_result.confidence
        )

        return await self.default_store.update_vector(collection_name, vector_data)

    async def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> bool:
        """删除向量"""
        if not self.default_store:
            raise RuntimeError("默认向量存储未初始化")

        return await self.default_store.delete_vectors(collection_name, vector_ids)

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.default_store:
            return {}

        stats = {
            "collections": {},
            "total_vectors": 0
        }

        for collection_name in self.collection_configs.keys():
            try:
                collection_stats = await self.default_store.get_collection_stats(collection_name)
                stats["collections"][collection_name] = collection_stats
                stats["total_vectors"] += collection_stats.get("num_entities", 0)
            except Exception as e:
                logger.error(f"获取集合 {collection_name} 统计失败: {str(e)}")
                stats["collections"][collection_name] = {"error": str(e)}

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "status": "healthy",
            "stores": {},
            "total_collections": 0,
            "total_vectors": 0
        }

        for name, store in self.vector_stores.items():
            store_health = await store.health_check()
            health_status["stores"][name] = store_health

            if store_health["status"] != "healthy":
                health_status["status"] = "degraded"

        # 获取统计信息
        stats = await self.get_statistics()
        health_status["total_collections"] = len([c for c in stats.get("collections", {}).values() if "error" not in c])
        health_status["total_vectors"] = stats.get("total_vectors", 0)

        return health_status

    async def cleanup(self):
        """清理资源"""
        try:
            for store in self.vector_stores.values():
                await store.disconnect()
            logger.info("向量管理器清理完成")
        except Exception as e:
            logger.error(f"清理失败: {str(e)}")

    async def batch_insert(
        self,
        collection_name: str,
        vectors: List[VectorData],
        batch_size: int = 1000
    ) -> List[str]:
        """批量插入向量"""
        if not self.default_store:
            raise RuntimeError("默认向量存储未初始化")

        all_ids = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            batch_ids = await self.default_store.insert_vectors(collection_name, batch)
            all_ids.extend(batch_ids)

            logger.info(f"批量插入进度: {min(i + batch_size, len(vectors))}/{len(vectors)}")

        return all_ids