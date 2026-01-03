"""
统一Milvus向量存储服务（重构版）
基于BaseVectorStore抽象类，提供统一的接口

特性：
- 清晰的职责分离
- 统一的错误处理
- 完善的性能监控
- 标准化的数据结构
"""

from typing import List, Dict, Any, Optional
import asyncio
import time

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

from app.core.structured_logging import get_structured_logger
from app.core.config import settings

from .base_vector_store import (
    BaseVectorStore,
    VectorStoreConfig,
    VectorStoreType,
    ChunkType,
    SearchResult,
    CollectionStats
)

logger = get_structured_logger(__name__)


class MilvusVectorStore(BaseVectorStore):
    """
    Milvus向量存储服务（重构版）

    整合了unified_milvus_service和enhanced_milvus_service的功能
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        # 从settings创建配置
        if config is None:
            config = VectorStoreConfig(
                store_type=VectorStoreType.MILVUS,
                collection_name=getattr(settings, 'milvus_collection_name', 'financial_documents'),
                embedding_dimension=getattr(settings, 'bge_embedding_dimension', 1024),
                host=getattr(settings, 'milvus_host', 'localhost'),
                port=getattr(settings, 'milvus_port', 19530),
                timeout=getattr(settings, 'milvus_timeout', 30),
                index_type=getattr(settings, 'milvus_index_type', 'IVF_FLAT'),
                metric_type=getattr(settings, 'milvus_metric_type', 'COSINE'),
                nlist=getattr(settings, 'milvus_nlist', 128),
                nprobe=getattr(settings, 'milvus_nprobe', 10),
                batch_size=100
            )

        super().__init__(config)

        # Milvus特定属性
        self.collection: Optional[Collection] = None
        self._last_health_check = 0
        self.config.health_check_interval = 60

    # ========================================================================
    # 连接管理
    # ========================================================================

    async def connect(self) -> bool:
        """连接到Milvus（带重试机制）"""
        for attempt in range(self.config.max_retries):
            try:
                # 如果已连接，先断开
                if self._is_connected:
                    try:
                        connections.disconnect("default")
                    except Exception:
                        pass

                connections.connect(
                    alias="default",
                    host=self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout
                )
                self._is_connected = True
                self._last_health_check = time.time()
                logger.info(f"✓ 成功连接到Milvus: {self.config.host}:{self.config.port}")
                return True

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"连接Milvus失败(尝试{attempt+1}/{self.config.max_retries}): {e}, {wait_time:.1f}秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"✗ 连接Milvus失败(已重试{self.config.max_retries}次): {e}")
                    self._update_stats("error")
                    raise

    async def disconnect(self):
        """断开Milvus连接"""
        try:
            connections.disconnect("default")
            self._is_connected = False
            logger.info("✓ 已断开Milvus连接")
        except Exception as e:
            logger.error(f"断开Milvus连接失败: {e}")

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            current_time = time.time()
            if current_time - self._last_health_check < self.config.health_check_interval:
                return self._is_connected

            if self.collection and utility.has_collection(self.collection_name):
                # 执行简单查询
                self.collection.query(expr="id < 0", limit=1)

            self._last_health_check = current_time
            return True

        except Exception as e:
            logger.warning(f"Milvus健康检查失败: {e}")
            self._is_connected = False
            return False

    # ========================================================================
    # 集合管理
    # ========================================================================

    async def init_collection(self, force_recreate: bool = False):
        """
        初始化集合

        Args:
            force_recreate: 是否强制重新创建集合
        """
        try:
            await self.ensure_connected()

            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                if force_recreate:
                    logger.warning(f"删除现有集合: {self.collection_name}")
                    utility.drop_collection(self.collection_name)
                else:
                    self.collection = Collection(self.collection_name)
                    self._validate_collection_schema()
                    self.collection.load()
                    logger.info(f"✓ 集合 {self.collection_name} 已加载")
                    return

            # 创建新集合
            await self._create_collection()
            await self._create_index()

            # 加载集合
            self.collection = Collection(self.collection_name)
            self.collection.load()

            logger.info(f"✓ 集合 {self.collection_name} 初始化完成")

        except Exception as e:
            logger.error(f"初始化Milvus集合失败: {e}")
            self._update_stats("error")
            raise

    def _validate_collection_schema(self):
        """验证集合schema"""
        try:
            schema = self.collection.schema

            # 检查embedding字段维度
            for field in schema.fields:
                if field.name == "embedding":
                    embedding_dim = field.params.get("dim")
                    expected_dim = self.config.embedding_dimension

                    if embedding_dim != expected_dim:
                        logger.warning(
                            f"⚠ 集合维度不匹配(当前:{embedding_dim}, 期望:{expected_dim})"
                        )
                    break

        except Exception as e:
            logger.warning(f"验证集合schema失败: {e}")

    async def _create_collection(self):
        """创建集合"""
        dimension = self.config.embedding_dimension

        # 定义字段 - 统一schema
        fields = [
            # 主键
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),

            # 文档标识
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),

            # 内容
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),

            # 向量
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),

            # 基础元数据
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),

            # LLM提取的元数据
            FieldSchema(name="llm_metadata", dtype=DataType.JSON),

            # 基础元数据(兼容旧版)
            FieldSchema(name="metadata", dtype=DataType.JSON),

            # 时间戳
            FieldSchema(name="created_at", dtype=DataType.INT64)
        ]

        # 创建schema
        schema = CollectionSchema(
            fields=fields,
            description=f"Financial documents with full metadata (dimension: {dimension})"
        )

        # 创建集合
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        logger.info(f"✓ 创建集合 {self.collection_name} (向量维度: {dimension})")

    async def _create_index(self):
        """创建向量索引"""
        index_params = {
            "metric_type": self.config.metric_type,
            "index_type": self.config.index_type,
            "params": {"nlist": self.config.nlist}
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        logger.info(f"✓ 创建向量索引 (type: {self.config.index_type}, metric: {self.config.metric_type})")

    # ========================================================================
    # 数据操作
    # ========================================================================

    async def insert_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> List[int]:
        """
        插入文档块

        Args:
            document_id: 文档ID
            chunks: 文档块列表
            batch_size: 批量插入大小

        Returns:
            插入的记录ID列表
        """
        try:
            await self.ensure_connected()
            if self.collection is None:
                await self.init_collection()

            all_ids = []
            batch_size = batch_size or self.config.batch_size

            # 分批插入
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                entities = []

                for chunk in batch:
                    entity = {
                        "document_id": document_id,
                        "chunk_id": chunk.get("chunk_id", f"{document_id}_{chunk.get('chunk_index', i)}"),
                        "content": chunk.get("content", "")[:65535],
                        "embedding": chunk.get("embedding", []),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "page_number": chunk.get("page_number", 0),
                        "chunk_type": chunk.get("chunk_type", ChunkType.TEXT.value),
                        "llm_metadata": chunk.get("llm_metadata", {}),
                        "metadata": chunk.get("metadata", {}),
                        "created_at": int(time.time() * 1000)
                    }
                    entities.append(entity)

                # 批量插入
                result = self.collection.insert(entities)
                all_ids.extend(result.primary_keys)

            # 刷新使数据可见
            self.collection.flush()

            self._update_stats("insert")
            logger.info(f"✓ 插入 {len(chunks)} 个文档块 (文档ID: {document_id})")

            return all_ids

        except Exception as e:
            self._update_stats("error")
            logger.error(f"插入文档块失败: {e}")
            raise

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        document_ids: Optional[List[str]] = None,
        score_threshold: float = 0.0,
        chunk_types: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        向量搜索

        Args:
            query_embedding: 查询向量
            limit: 返回结果数量
            document_ids: 限定文档ID范围
            score_threshold: 相似度阈值
            chunk_types: 限定文档块类型
            output_fields: 输出字段列表

        Returns:
            搜索结果列表
        """
        start_time = time.time()

        try:
            await self.ensure_connected()
            if self.collection is None:
                await self.init_collection()

            # 构建搜索参数
            search_params = {
                "metric_type": self.config.metric_type,
                "params": {"nprobe": self.config.nprobe}
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

            # 默认输出字段
            if output_fields is None:
                output_fields = [
                    "document_id", "chunk_id", "content",
                    "chunk_index", "page_number", "chunk_type",
                    "llm_metadata", "metadata", "created_at"
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
                    result = SearchResult(
                        id=hit.id,
                        document_id=hit.entity.get("document_id", ""),
                        chunk_id=hit.entity.get("chunk_id", ""),
                        content=hit.entity.get("content", ""),
                        score=float(hit.score),
                        chunk_index=hit.entity.get("chunk_index", 0),
                        page_number=hit.entity.get("page_number", 0),
                        chunk_type=hit.entity.get("chunk_type", ChunkType.TEXT.value),
                        metadata=hit.entity.get("metadata", {}),
                        llm_metadata=hit.entity.get("llm_metadata", {}),
                        created_at=hit.entity.get("created_at", 0)
                    )
                    search_results.append(result)

            # 更新统计
            latency = time.time() - start_time
            self._update_stats("search", latency)

            logger.info(f"✓ 向量搜索完成 (返回: {len(search_results)} 个结果, 耗时: {latency:.3f}秒)")

            return search_results

        except Exception as e:
            self._update_stats("error")
            logger.error(f"向量搜索失败: {e}")
            raise

    async def delete_document(self, document_id: str) -> bool:
        """
        删除文档的所有向量

        Args:
            document_id: 文档ID

        Returns:
            是否成功
        """
        try:
            await self.ensure_connected()
            if self.collection is None:
                await self.init_collection()

            # 构建删除表达式
            expr = f'document_id == "{document_id}"'

            # 执行删除
            self.collection.delete(expr)

            # 刷新
            self.collection.flush()

            self._update_stats("delete")
            logger.info(f"✓ 删除文档 {document_id} 的向量数据")

            return True

        except Exception as e:
            self._update_stats("error")
            logger.error(f"删除文档失败: {e}")
            return False

    async def update_chunk_metadata(
        self,
        chunk_id: str,
        llm_metadata: Dict[str, Any]
    ) -> bool:
        """
        更新文档块的LLM元数据

        Args:
            chunk_id: 文档块ID
            llm_metadata: LLM提取的元数据

        Returns:
            是否成功
        """
        try:
            await self.ensure_connected()
            if self.collection is None:
                await self.init_collection()

            # 查询现有数据
            results = self.collection.query(
                expr=f'chunk_id == "{chunk_id}"',
                output_fields=["*"]
            )

            if not results:
                logger.warning(f"Chunk {chunk_id} 未找到")
                return False

            # 删除旧数据
            self.collection.delete(expr=f'chunk_id == "{chunk_id}"')

            # 重新插入（更新元数据）
            for old_data in results:
                old_data["llm_metadata"] = llm_metadata
                old_data["created_at"] = int(time.time() * 1000)
                self.collection.insert([old_data])

            self.collection.flush()
            logger.info(f"✓ 更新 chunk {chunk_id} 的元数据")

            return True

        except Exception as e:
            self._update_stats("error")
            logger.error(f"更新元数据失败: {e}")
            return False

    # ========================================================================
    # 统计信息
    # ========================================================================

    async def get_collection_stats(self) -> CollectionStats:
        """获取集合统计信息"""
        try:
            if self.collection is None:
                await self.init_collection()

            stats = CollectionStats(
                collection_name=self.collection_name,
                total_entities=self.collection.num_entities,
                embedding_dimension=self.config.embedding_dimension,
                index_type=self.config.index_type,
                metric_type=self.config.metric_type,
                total_inserts=self._stats["total_inserts"],
                total_searches=self._stats["total_searches"],
                total_deletes=self._stats["total_deletes"],
                total_errors=self._stats["total_errors"],
                avg_search_latency=self._stats["avg_search_latency"]
            )

            return stats

        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            raise


# ============================================================================
# 全局服务实例
# ============================================================================

_global_milvus_store: Optional[MilvusVectorStore] = None


def get_milvus_store(config: Optional[VectorStoreConfig] = None) -> MilvusVectorStore:
    """
    获取全局Milvus向量存储实例

    Args:
        config: 配置参数

    Returns:
        MilvusVectorStore实例
    """
    global _global_milvus_store

    if _global_milvus_store is None:
        _global_milvus_store = MilvusVectorStore(config)
        logger.info("全局Milvus向量存储已创建")

    return _global_milvus_store


# 导出
__all__ = [
    'MilvusVectorStore',
    'get_milvus_store'
]
