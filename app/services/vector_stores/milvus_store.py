"""
Milvus向量数据库实现
用于存储和检索文本、实体、关系的向量表示
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
from datetime import datetime

try:
    from pymilvus import (
        connections, Collection, CollectionSchema, FieldSchema, DataType,
        utility, MilvusException
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("PyMilvus未安装，请运行: pip install pymilvus")

from .base_vector_store import BaseVectorStore, VectorConfig, SearchResult, VectorData

logger = logging.getLogger(__name__)


class CollectionType(Enum):
    """集合类型"""
    TEXT = "text"
    ENTITY = "entity"
    RELATION = "relation"
    MULTIMODAL = "multimodal"


class MilvusVectorStore(BaseVectorStore):
    """Milvus向量存储实现"""

    def __init__(self, config: VectorConfig):
        if not MILVUS_AVAILABLE:
            raise ImportError("需要安装PyMilvus: pip install pymilvus")

        super().__init__(config)
        self.host = config.host
        self.port = config.port
        self.db_name = config.db_name
        self.collections = {}
        self.index_params = {
            "metric_type": config.metric_type or "IP",
            "index_type": config.index_type or "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        # 连接到Milvus
        self._connect()

    def _connect(self):
        """连接Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                db_name=self.db_name
            )
            logger.info(f"连接Milvus成功: {self.host}:{self.port}/{self.db_name}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {str(e)}")
            raise

    async def connect(self) -> bool:
        """连接到Milvus"""
        try:
            # 连接到Milvus服务器
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                db_name=self.db_name
            )

            logger.info(f"成功连接到Milvus: {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"连接Milvus失败: {str(e)}")
            return False

    async def disconnect(self):
        """断开Milvus连接"""
        try:
            connections.disconnect("default")
            logger.info("已断开Milvus连接")
        except Exception as e:
            logger.error(f"断开连接失败: {str(e)}")

    async def create_collection(self, collection_name: str, dimension: int, collection_type: str = "text") -> bool:
        """创建集合"""
        try:
            # 检查集合是否已存在
            if utility.has_collection(collection_name):
                logger.info(f"集合 {collection_name} 已存在")
                collection = Collection(collection_name)
                self.collections[collection_name] = collection
                return True

            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="confidence", dtype=DataType.FLOAT),
                FieldSchema(name="created_at", dtype=DataType.INT64)
            ]

            # 根据类型添加特定字段
            if collection_type == CollectionType.ENTITY.value:
                fields.append(FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=100))
                fields.append(FieldSchema(name="entity_label", dtype=DataType.VARCHAR, max_length=500))
            elif collection_type == CollectionType.RELATION.value:
                fields.append(FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100))
                fields.append(FieldSchema(name="predicate", dtype=DataType.VARCHAR, max_length=100))
                fields.append(FieldSchema(name="object", dtype=DataType.VARCHAR, max_length=100))

            # 创建集合Schema
            schema = CollectionSchema(
                fields=fields,
                description=f"Vector collection for {collection_type}"
            )

            # 创建集合
            collection = Collection(collection_name, schema)
            self.collections[collection_name] = collection

            logger.info(f"成功创建集合 {collection_name}")
            return True

        except Exception as e:
            logger.error(f"创建集合失败: {str(e)}")
            return False

    async def insert_vectors(self, collection_name: str, vectors: List[VectorData]) -> List[str]:
        """插入向量"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                collection = Collection(collection_name)
                self.collections[collection_name] = collection

            # 准备数据
            data = {
                "id": [v.id for v in vectors],
                "vector": [v.vector.tolist() if isinstance(v.vector, np.ndarray) else v.vector for v in vectors],
                "content": [v.content for v in vectors],
                "metadata": [v.metadata or {} for v in vectors],
                "confidence": [v.confidence for v in vectors],
                "created_at": [int(datetime.now().timestamp() * 1000) for v in vectors]
            }

            # 根据集合类型添加特定字段
            collection_type = self._get_collection_type(collection_name)
            if collection_type == CollectionType.ENTITY.value:
                data["entity_type"] = [v.metadata.get("type", "") for v in vectors]
                data["entity_label"] = [v.content for v in vectors]
            elif collection_type == CollectionType.RELATION.value:
                data["subject"] = [v.metadata.get("subject", "") for v in vectors]
                data["predicate"] = [v.metadata.get("predicate", "") for v in vectors]
                data["object"] = [v.metadata.get("object", "") for v in vectors]

            # 批量插入
            insert_result = collection.insert([data[field] for field in data])

            # 刷新以使数据可见
            collection.flush()

            logger.info(f"成功插入 {len(vectors)} 个向量到集合 {collection_name}")
            return insert_result.primary_keys

        except Exception as e:
            logger.error(f"插入向量失败: {str(e)}")
            return []

    async def search_vectors(
        self,
        collection_name: str,
        query_vectors: Union[List[float], List[List[float]]],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[List[SearchResult]]:
        """搜索向量"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                collection = Collection(collection_name)
                self.collections[collection_name] = collection

            # 加载集合到内存
            collection.load()

            # 准备搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }

            # 默认输出字段
            if output_fields is None:
                output_fields = ["id", "content", "metadata", "confidence"]

            # 执行搜索
            results = collection.search(
                data=query_vectors,
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields
            )

            # 转换结果格式
            formatted_results = []
            for hits in results:
                search_results = []
                for hit in hits:
                    result = SearchResult(
                        id=hit.id,
                        score=hit.score,
                        content=hit.entity.get("content", ""),
                        metadata=hit.entity.get("metadata", {}),
                        confidence=hit.entity.get("confidence", 0.0)
                    )
                    search_results.append(result)
                formatted_results.append(search_results)

            logger.info(f"在集合 {collection_name} 中搜索到 {len(formatted_results)} 组结果")
            return formatted_results

        except Exception as e:
            logger.error(f"搜索向量失败: {str(e)}")
            return []

    async def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """删除向量"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                collection = Collection(collection_name)
                self.collections[collection_name] = collection

            # 构建删除表达式
            id_list = "', '".join(ids)
            expr = f"id in ['{id_list}']"

            # 执行删除
            collection.delete(expr)

            logger.info(f"成功从集合 {collection_name} 删除 {len(ids)} 个向量")
            return True

        except Exception as e:
            logger.error(f"删除向量失败: {str(e)}")
            return False

    async def update_vector(self, collection_name: str, vector_data: VectorData) -> bool:
        """更新向量"""
        try:
            # Milvus不支持直接更新，需要先删除再插入
            await self.delete_vectors(collection_name, [vector_data.id])
            await self.insert_vectors(collection_name, [vector_data])

            logger.info(f"成功更新集合 {collection_name} 中的向量 {vector_data.id}")
            return True

        except Exception as e:
            logger.error(f"更新向量失败: {str(e)}")
            return False

    async def create_index(self, collection_name: str, index_type: str = "IVF_FLAT") -> bool:
        """创建索引"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                collection = Collection(collection_name)
                self.collections[collection_name] = collection

            # 获取索引参数
            index_params = self.index_params.get(index_type)
            if not index_params:
                logger.error(f"不支持的索引类型: {index_type}")
                return False

            # 创建索引
            collection.create_index(
                field_name="vector",
                index_params=index_params
            )

            logger.info(f"成功为集合 {collection_name} 创建 {index_type} 索引")
            return True

        except Exception as e:
            logger.error(f"创建索引失败: {str(e)}")
            return False

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                collection = Collection(collection_name)
                self.collections[collection_name] = collection

            # 获取统计信息
            stats = collection.describe()
            num_entities = collection.num_entities

            # 获取索引信息
            indexes = collection.indexes
            index_info = []
            for index in indexes:
                index_info.append({
                    "name": index.index_name,
                    "field_name": index.field_name,
                    "index_type": index.params.get("index_type", "unknown")
                })

            return {
                "collection_name": collection_name,
                "num_entities": num_entities,
                "fields": stats.get("fields", []),
                "indexes": index_info,
                "created_time": stats.get("created_time"),
                "description": stats.get("description", "")
            }

        except Exception as e:
            logger.error(f"获取集合统计失败: {str(e)}")
            return {}

    def _get_collection_type(self, collection_name: str) -> str:
        """根据集合名称推断类型"""
        if "entity" in collection_name.lower():
            return CollectionType.ENTITY.value
        elif "relation" in collection_name.lower():
            return CollectionType.RELATION.value
        elif "text" in collection_name.lower():
            return CollectionType.TEXT.value
        else:
            return CollectionType.MULTIMODAL.value

    async def drop_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)

                # 从缓存中移除
                if collection_name in self.collections:
                    del self.collections[collection_name]

                logger.info(f"成功删除集合 {collection_name}")
                return True

            logger.warning(f"集合 {collection_name} 不存在")
            return False

        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            return False

    async def list_collections(self) -> List[str]:
        """列出所有集合"""
        try:
            collections = utility.list_collections()
            return collections
        except Exception as e:
            logger.error(f"列出集合失败: {str(e)}")
            return []

    async def hybrid_search(
        self,
        collection_name: str,
        query_vector: List[float],
        text_query: Optional[str] = None,
        top_k: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> List[SearchResult]:
        """混合搜索（向量+文本）"""
        try:
            # 向量搜索
            vector_results = await self.search_vectors(
                collection_name=collection_name,
                query_vectors=[query_vector],
                top_k=top_k * 2  # 获取更多结果用于重排序
            )
            vector_results = vector_results[0] if vector_results else []

            results = vector_results

            # 如果有文本查询，进行文本过滤和重排序
            if text_query:
                # 简单的文本匹配（实际应用中可以使用更复杂的文本搜索）
                text_filtered = []
                for result in vector_results:
                    if text_query.lower() in result.content.lower():
                        # 计算文本匹配分数
                        text_score = self._calculate_text_similarity(text_query, result.content)
                        # 综合分数
                        result.score = vector_weight * result.score + text_weight * text_score
                        text_filtered.append(result)

                if text_filtered:
                    # 按综合分数排序
                    text_filtered.sort(key=lambda x: x.score, reverse=True)
                    results = text_filtered[:top_k]

            return results

        except Exception as e:
            logger.error(f"混合搜索失败: {str(e)}")
            return []

    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """计算文本相似度"""
        # 简单的Jaccard相似度
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())

        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)

        return len(intersection) / len(union) if union else 0.0

    async def batch_search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 10,
        batch_size: int = 100
    ) -> List[List[SearchResult]]:
        """批量搜索"""
        results = []

        # 分批处理
        for i in range(0, len(query_vectors), batch_size):
            batch = query_vectors[i:i + batch_size]
            batch_results = await self.search_vectors(collection_name, batch, top_k)
            results.extend(batch_results)

        return results

    # 新增的增强方法
    async def create_partitioned_collection(
        self,
        collection_name: str,
        dimension: int,
        collection_type: str = "text",
        partition_names: Optional[List[str]] = None
    ) -> bool:
        """创建带分区的集合"""
        try:
            # 创建基础集合
            await self.create_collection(collection_name, dimension, collection_type)

            collection = self.collections.get(collection_name)
            if not collection:
                return False

            # 创建分区
            if partition_names:
                for partition_name in partition_names:
                    if partition_name not in collection.partitions:
                        collection.create_partition(partition_name)
                        logger.info(f"创建分区: {partition_name}")

            # 默认分区（按文档类型）
            default_partitions = ["text", "table", "image", "formula"]
            for partition_name in default_partitions:
                if partition_name not in collection.partitions:
                    collection.create_partition(partition_name)

            return True

        except Exception as e:
            logger.error(f"创建分区集合失败: {str(e)}")
            return False

    async def insert_with_partition(
        self,
        collection_name: str,
        vectors: List[VectorData],
        partition_name: Optional[str] = None
    ) -> List[str]:
        """插入向量到指定分区"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                collection = Collection(collection_name)
                self.collections[collection_name] = collection

            # 如果没有指定分区，根据内容类型自动选择
            if not partition_name:
                partition_name = self._determine_partition(vectors[0] if vectors else None)

            # 确保分区存在
            if partition_name and partition_name not in collection.partitions:
                collection.create_partition(partition_name)

            # 准备数据
            data = {
                "id": [v.id for v in vectors],
                "vector": [v.vector.tolist() if isinstance(v.vector, np.ndarray) else v.vector for v in vectors],
                "content": [v.content for v in vectors],
                "metadata": [v.metadata or {} for v in vectors],
                "confidence": [v.confidence for v in vectors],
                "created_at": [int(datetime.now().timestamp() * 1000) for v in vectors]
            }

            # 添加增强元数据
            enhanced_metadata = []
            for v in vectors:
                meta = v.metadata or {}
                # 添加文档信息
                if "document_id" not in meta:
                    meta["document_id"] = meta.get("doc_id", "")
                # 添加页码信息
                if "page_num" not in meta:
                    meta["page_num"] = meta.get("page", 0)
                # 添加块索引
                if "chunk_index" not in meta:
                    meta["chunk_index"] = meta.get("index", 0)
                enhanced_metadata.append(meta)

            data["metadata"] = enhanced_metadata

            # 插入到指定分区
            if partition_name:
                mr = collection.insert(
                    [data[field] for field in data],
                    partition_name=partition_name
                )
            else:
                mr = collection.insert([data[field] for field in data])

            collection.flush()

            logger.info(f"成功插入 {len(vectors)} 个向量到分区 {partition_name or 'default'}")
            return mr.primary_keys

        except Exception as e:
            logger.error(f"插入分区向量失败: {str(e)}")
            return []

    async def enhanced_hybrid_search(
        self,
        collection_name: str,
        query_vector: List[float],
        query_text: Optional[str] = None,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        partitions: Optional[List[str]] = None,
        rerank: bool = True
    ) -> List[SearchResult]:
        """增强的混合搜索"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                collection = Collection(collection_name)
                self.collections[collection_name] = collection

            # 加载指定分区
            if partitions:
                for partition in partitions:
                    if partition in collection.partitions:
                        collection.load([partition])
            else:
                collection.load()

            # 向量搜索
            search_params = {
                "metric_type": self.index_params["metric_type"],
                "params": {"nprobe": 10}
            }

            # 执行向量搜索
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k * 2 if rerank else top_k,  # 获取更多结果用于重排序
                expr=filter_expr,
                output_fields=["id", "content", "metadata", "confidence", "created_at"]
            )[0]

            # 转换为SearchResult格式
            search_results = []
            for hit in results:
                metadata = hit.entity.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}

                # 计算综合分数
                score = hit.score
                if query_text and rerank:
                    text_sim = self._calculate_text_similarity(query_text, hit.entity.get("content", ""))
                    # 综合向量相似度和文本相似度
                    score = 0.7 * score + 0.3 * text_sim

                search_results.append(SearchResult(
                    id=hit.entity.get("id"),
                    content=hit.entity.get("content", ""),
                    metadata=metadata,
                    score=score,
                    distance=1 - hit.score if self.index_params["metric_type"] == "IP" else hit.distance
                ))

            # 如果启用重排序且提供了文本查询
            if rerank and query_text:
                search_results = self._rerank_results(query_text, search_results)

            # 限制返回数量
            return search_results[:top_k]

        except Exception as e:
            logger.error(f"增强混合搜索失败: {str(e)}")
            return []
        finally:
            try:
                collection.release()
            except:
                pass

    async def delete_by_document_id(
        self,
        collection_name: str,
        document_id: str
    ) -> int:
        """根据文档ID删除向量"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                collection = Collection(collection_name)
                self.collections[collection_name] = collection

            # 构建删除表达式
            expr = f'metadata["document_id"] == "{document_id}"'

            # 执行删除
            result = collection.delete(expr)
            collection.flush()

            logger.info(f"删除文档 {document_id} 的向量，数量: {result.delete_count}")
            return result.delete_count

        except Exception as e:
            logger.error(f"根据文档ID删除向量失败: {str(e)}")
            return 0

    def _determine_partition(self, vector_data: Optional[VectorData]) -> str:
        """根据向量数据确定分区"""
        if not vector_data:
            return "text"

        metadata = vector_data.metadata or {}
        chunk_type = metadata.get("type", metadata.get("chunk_type", "text"))

        # 根据类型选择分区
        if chunk_type == "table":
            return "table"
        elif chunk_type == "image":
            return "image"
        elif chunk_type == "formula":
            return "formula"
        else:
            return "text"