"""
增强的 Milvus 服务 - 支持完整元数据存储
"""
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

from app.core.vector_config import vector_config, get_dimension

logger = logging.getLogger(__name__)


class EnhancedMilvusService:
    """增强的 Milvus 服务 - 支持完整元数据"""

    def __init__(self):
        self.collection_name = vector_config.collection_name
        self.host = vector_config.host
        self.port = vector_config.port
        self.collection = None

    async def connect(self):
        """连接到 Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"成功连接到 Milvus: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            raise

    async def init_collection(self):
        """初始化集合 - 支持完整元数据"""
        try:
            await self.connect()

            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                schema = collection.schema

                # 验证维度
                for field in schema.fields:
                    if field.name == "embedding":
                        existing_dim = field.params.get("dim")
                        expected_dim = get_dimension()
                        if existing_dim != expected_dim:
                            logger.warning(
                                f"集合维度不匹配! 现有:{existing_dim}, 期望:{expected_dim}"
                            )
                        break

                self.collection = collection
                collection.load()
                logger.info(f"集合 {self.collection_name} 已存在并加载")
                return

            # 创建新集合 - 包含完整的元数据字段
            await self._create_enhanced_collection()

            # 创建索引
            await self._create_index()

            # 加载集合
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"集合 {self.collection_name} 初始化完成")

        except Exception as e:
            logger.error(f"初始化 Milvus 集合失败: {e}")
            raise

    async def _create_enhanced_collection(self):
        """创建增强的集合 - 包含完整元数据字段"""
        dimension = get_dimension()

        # 定义字段 - 增强版
        fields = [
            # 基础字段
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),

            # 内容字段
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),

            # 向量字段
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),

            # 基础元数据
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),  # text/table/image

            # LLM 提取的元数据（存储在 JSON 字段中）
            FieldSchema(name="llm_metadata", dtype=DataType.JSON),

            # 创建时间
            FieldSchema(name="created_at", dtype=DataType.INT64)
        ]

        # 创建集合 schema
        schema = CollectionSchema(
            fields=fields,
            description=f"Financial documents with full metadata (dimension: {dimension})"
        )

        # 创建集合
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        logger.info(f"创建增强集合 {self.collection_name} (维度: {dimension})")

    async def _create_index(self):
        """创建索引"""
        index_params = {
            "metric_type": vector_config.metric_type,
            "index_type": vector_config.index_type,
            "params": {"nlist": vector_config.nlist}
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        logger.info("向量索引创建成功")

    async def insert_chunks_with_full_metadata(
        self,
        document_id: str,
        chunks_data: List[Dict[str, Any]]
    ) -> List[int]:
        """
        插入文档块及其完整元数据

        Args:
            document_id: 文档ID
            chunks_data: 文档块数据列表，每个元素包含:
                {
                    "chunk_id": str,
                    "content": str,
                    "embedding": List[float],
                    "chunk_index": int,
                    "page_number": int,
                    "chunk_type": str,
                    "llm_metadata": {  # LLM 提取的完整元数据
                        "summary": str,
                        "keywords": List[str],
                        "entities": List[str],
                        "topic": str,
                        "importance_score": float,
                        "chapter_id": Optional[str],
                        "position": Optional[Tuple[int, int]]
                    }
                }

        Returns:
            插入的记录ID列表
        """
        try:
            entities = []

            for chunk in chunks_data:
                # 准备基础数据
                entity = {
                    "document_id": document_id,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "content": chunk["content"][:65535],  # 限制长度
                    "embedding": chunk["embedding"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "page_number": chunk.get("page_number", 0),
                    "chunk_type": chunk.get("chunk_type", "text"),
                    # LLM 提取的完整元数据
                    "llm_metadata": chunk.get("llm_metadata", {}),
                    "created_at": int(datetime.now().timestamp() * 1000)
                }

                entities.append(entity)

            # 批量插入
            result = self.collection.insert(entities)

            # 刷新使数据可见
            self.collection.flush()

            logger.info(
                f"成功插入 {len(entities)} 个文档块到 {self.collection_name}, "
                f"文档ID: {document_id}"
            )
            return result.primary_keys

        except Exception as e:
            logger.error(f"插入文档块失败: {e}")
            raise

    async def search_with_metadata(
        self,
        query_embedding: List[float],
        limit: int = 10,
        document_ids: Optional[List[str]] = None,
        score_threshold: float = 0.0,
        chunk_types: Optional[List[str]] = None,
        output_full_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        增强的向量搜索 - 返回完整元数据

        Args:
            query_embedding: 查询向量
            limit: 返回结果数量
            document_ids: 限定文档ID范围
            score_threshold: 相似度阈值
            chunk_types: 限定文档块类型
            output_full_metadata: 是否输出完整元数据

        Returns:
            搜索结果列表，包含完整元数据
        """
        try:
            # 确保集合已初始化
            if self.collection is None:
                await self.init_collection()

            # 构建搜索参数
            search_params = {
                "metric_type": vector_config.metric_type,
                "params": {"nprobe": vector_config.nprobe}
            }

            # 构建过滤表达式
            expr_parts = []

            if document_ids:
                doc_ids_str = ", ".join([f'"{did}"' for did in document_ids])
                expr_parts.append(f"document_id in [{doc_ids_str}]")

            if chunk_types:
                types_str = ", ".join([f'"{ct}"' for ct in chunk_types])
                expr_parts.append(f"chunk_type in [{types_str}]")

            expr = " and ".join(expr_parts) if expr_parts else None

            # 输出字段
            output_fields = [
                "document_id", "chunk_id", "content",
                "chunk_index", "page_number", "chunk_type",
                "llm_metadata", "created_at"
            ]

            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=output_fields
            )

            # 处理搜索结果
            search_results = []
            for hit in results[0]:
                if hit.score >= score_threshold:
                    result = {
                        "document_id": hit.entity.get("document_id"),
                        "chunk_id": hit.entity.get("chunk_id"),
                        "content": hit.entity.get("content"),
                        "score": float(hit.score),
                        "metadata": {
                            "chunk_index": hit.entity.get("chunk_index"),
                            "page_number": hit.entity.get("page_number"),
                            "chunk_type": hit.entity.get("chunk_type"),
                        },
                        "created_at": hit.entity.get("created_at")
                    }

                    # 添加完整 LLM 元数据
                    if output_full_metadata:
                        result["llm_metadata"] = hit.entity.get("llm_metadata", {})

                    search_results.append(result)

            logger.info(f"向量搜索完成，返回 {len(search_results)} 个结果")
            return search_results

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            raise

    async def update_chunk_metadata(
        self,
        chunk_id: str,
        llm_metadata: Dict[str, Any]
    ) -> bool:
        """
        更新文档块的 LLM 元数据

        Args:
            chunk_id: 文档块ID
            llm_metadata: LLM 提取的元数据

        Returns:
            是否成功
        """
        try:
            # Milvus 不支持直接更新，需要先删除再插入
            # 查询现有数据
            results = self.collection.query(
                expr=f'chunk_id == "{chunk_id}"',
                output_fields=["*"]
            )

            if not results:
                logger.warning(f"Chunk {chunk_id} not found")
                return False

            # 删除旧数据
            self.collection.delete(expr=f'chunk_id == "{chunk_id}"')

            # 重新插入（更新元数据）
            for old_data in results:
                old_data["llm_metadata"] = llm_metadata
                self.collection.insert([old_data])

            self.collection.flush()
            logger.info(f"成功更新 chunk {chunk_id} 的元数据")
            return True

        except Exception as e:
            logger.error(f"更新元数据失败: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            stats = {
                "collection_name": self.collection_name,
                "total_entities": self.collection.num_entities,
                "embedding_dimension": get_dimension(),
                "index_status": "已创建"
            }

            return stats

        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            raise


# 全局实例
enhanced_milvus_service = EnhancedMilvusService()
