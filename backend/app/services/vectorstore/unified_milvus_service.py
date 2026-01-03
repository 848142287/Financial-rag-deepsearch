"""
统一Milvus向量存储服务 - 整合基础版和增强版功能
优化架构、性能和代码质量

核心优化:
1. 合并重复代码,减少维护成本
2. 支持灵活的元数据schema
3. 连接池和重试机制
4. 批量操作优化
5. 性能监控集成
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
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

logger = get_structured_logger(__name__)

class MilvusConfig:
    """Milvus配置"""
    def __init__(self):
        self.collection_name = getattr(settings, 'milvus_collection_name', 'financial_documents')
        self.host = getattr(settings, 'milvus_host', 'localhost')
        self.port = getattr(settings, 'milvus_port', 19530)
        self.embedding_model = getattr(settings, 'bge_embedding_model_name', 'bge-large-zh-v1.5')
        self.embedding_dim = getattr(settings, 'bge_embedding_dimension', 1024)

        # 索引配置
        self.index_type = getattr(settings, 'milvus_index_type', 'IVF_FLAT')
        self.metric_type = getattr(settings, 'milvus_metric_type', 'COSINE')
        self.nlist = getattr(settings, 'milvus_nlist', 128)
        self.nprobe = getattr(settings, 'milvus_nprobe', 10)

        # 连接池配置
        self.pool_size = getattr(settings, 'milvus_pool_size', 10)
        self.timeout = getattr(settings, 'milvus_timeout', 30)
        self.retry_max_attempts = 3
        self.retry_backoff_factor = 0.5
        self.health_check_interval = 60

class ChunkType(Enum):
    """文档块类型"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    MIXED = "mixed"

@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    id: int
    document_id: str
    chunk_id: str
    content: str
    score: float
    chunk_index: int
    page_number: int
    chunk_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    llm_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: int = 0

class UnifiedMilvusService:
    """统一Milvus服务 - 整合所有功能"""

    # 向量维度映射
    EMBEDDING_DIMENSIONS = {
        "qwen2.5-vl-embedding": 1024,
        "text-embedding-v4": 1536,
        "text-embedding-v3": 1024,
        "bge-large-zh-v1.5": 1024,
        "default": 1024
    }

    def __init__(self, config: MilvusConfig = None):
        self.config = config or MilvusConfig()
        self.collection_name = self.config.collection_name
        self.collection: Optional[Collection] = None

        # 连接状态
        self._is_connected = False
        self._last_health_check = 0

        # 性能统计
        self.stats = {
            "total_inserts": 0,
            "total_searches": 0,
            "total_deletes": 0,
            "total_errors": 0,
            "avg_search_latency": 0.0
        }

    async def connect(self) -> bool:
        """连接到Milvus（带重试机制）"""
        for attempt in range(self.config.retry_max_attempts):
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
                if attempt < self.config.retry_max_attempts - 1:
                    wait_time = self.config.retry_backoff_factor * (2 ** attempt)
                    logger.warning(f"连接Milvus失败(尝试{attempt+1}/{self.config.retry_max_attempts}): {e}, {wait_time:.1f}秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"✗ 连接Milvus失败(已重试{self.config.retry_max_attempts}次): {e}")
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

    async def ensure_connected(self):
        """确保连接可用（自动重连）"""
        if not self._is_connected:
            await self.connect()
            return

        is_healthy = await self.health_check()
        if not is_healthy:
            logger.warning("Milvus连接不健康，尝试重连...")
            await self.connect()

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
            raise

    def _validate_collection_schema(self):
        """验证集合schema"""
        try:
            schema = self.collection.schema

            # 检查embedding字段维度
            for field in schema.fields:
                if field.name == "embedding":
                    embedding_dim = field.params.get("dim")
                    expected_dim = self.config.embedding_dim

                    if embedding_dim != expected_dim:
                        logger.warning(
                            f"⚠ 集合维度不匹配(当前:{embedding_dim}, 期望:{expected_dim})"
                        )
                    break

        except Exception as e:
            logger.warning(f"验证集合schema失败: {e}")

    async def _create_collection(self):
        """创建集合"""
        dimension = self.config.embedding_dim

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

    async def insert_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100
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

            self.stats["total_inserts"] += len(chunks)
            logger.info(f"✓ 插入 {len(chunks)} 个文档块 (文档ID: {document_id})")

            return all_ids

        except Exception as e:
            self.stats["total_errors"] += 1
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
    ) -> List[VectorSearchResult]:
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
                    result = VectorSearchResult(
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
            self.stats["total_searches"] += 1
            self.stats["avg_search_latency"] = (
                (self.stats["avg_search_latency"] * (self.stats["total_searches"] - 1) + latency)
                / self.stats["total_searches"]
            )

            logger.info(f"✓ 向量搜索完成 (返回: {len(search_results)} 个结果, 耗时: {latency:.3f}秒)")

            return search_results

        except Exception as e:
            self.stats["total_errors"] += 1
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

            self.stats["total_deletes"] += 1
            logger.info(f"✓ 删除文档 {document_id} 的向量数据")

            return True

        except Exception as e:
            self.stats["total_errors"] += 1
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
            self.stats["total_errors"] += 1
            logger.error(f"更新元数据失败: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            if self.collection is None:
                await self.init_collection()

            stats = {
                "collection_name": self.collection_name,
                "total_entities": self.collection.num_entities,
                "embedding_dimension": self.config.embedding_dim,
                "index_type": self.config.index_type,
                "metric_type": self.config.metric_type,
                # 性能统计
                "total_inserts": self.stats["total_inserts"],
                "total_searches": self.stats["total_searches"],
                "total_deletes": self.stats["total_deletes"],
                "total_errors": self.stats["total_errors"],
                "avg_search_latency": f"{self.stats['avg_search_latency']:.3f}s"
            }

            return stats

        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            raise

    async def hybrid_search(
        self,
        query_embedding: List[float],
        keywords: List[str],
        limit: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[VectorSearchResult]:
        """
        混合搜索（向量+关键词过滤）

        Args:
            query_embedding: 查询向量
            keywords: 关键词列表
            limit: 返回结果数量
            document_ids: 限定文档ID范围

        Returns:
            搜索结果列表
        """
        try:
            # 先执行向量搜索（获取更多候选）
            vector_results = await self.search(
                query_embedding=query_embedding,
                limit=limit * 2,
                document_ids=document_ids
            )

            # 关键词过滤
            if keywords:
                filtered_results = []
                for result in vector_results:
                    content = result.content.lower()
                    if any(keyword.lower() in content for keyword in keywords):
                        filtered_results.append(result)
                vector_results = filtered_results

            # 限制结果数量
            return vector_results[:limit]

        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            raise

    async def batch_search(
        self,
        query_embeddings: List[List[float]],
        limit: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[List[VectorSearchResult]]:
        """
        批量向量搜索

        Args:
            query_embeddings: 查询向量列表
            limit: 每个查询返回的结果数量
            document_ids: 限定文档ID范围

        Returns:
            搜索结果列表的列表
        """
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
            expr = None
            if document_ids:
                doc_ids_str = ", ".join([f'"{did}"' for did in document_ids])
                expr = f"document_id in [{doc_ids_str}]"

            # 执行批量搜索
            results_list = self.collection.search(
                data=query_embeddings,
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=[
                    "document_id", "chunk_id", "content",
                    "chunk_index", "page_number", "chunk_type",
                    "llm_metadata", "metadata", "created_at"
                ]
            )

            # 处理搜索结果
            all_results = []
            for results in results_list:
                search_results = []
                for hit in results:
                    result = VectorSearchResult(
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
                all_results.append(search_results)

            self.stats["total_searches"] += len(query_embeddings)
            logger.info(f"✓ 批量搜索完成 ({len(query_embeddings)} 个查询)")

            return all_results

        except Exception as e:
            self.stats["total_errors"] += 1
            logger.error(f"批量搜索失败: {e}")
            raise

# 全局服务实例
_milvus_service: Optional[UnifiedMilvusService] = None

def get_milvus_service() -> UnifiedMilvusService:
    """获取全局Milvus服务实例（延迟初始化）"""
    global _milvus_service
    if _milvus_service is None:
        _milvus_service = UnifiedMilvusService()
    return _milvus_service

# 便捷函数
async def search_documents(
    query_embedding: List[float],
    limit: int = 10,
    document_ids: Optional[List[str]] = None
) -> List[VectorSearchResult]:
    """便捷的文档搜索函数"""
    service = get_milvus_service()
    return await service.search(
        query_embedding=query_embedding,
        limit=limit,
        document_ids=document_ids
    )
