"""
统一向量存储基类
定义所有向量存储服务的公共接口和数据结构

优化点：
- 清晰的抽象接口
- 统一的数据结构
- 标准化的错误处理
- 完善的配置管理
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class VectorStoreType(Enum):
    """向量存储类型"""
    MILVUS = "milvus"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEVIATE = "weviate"


class ChunkType(Enum):
    """文档块类型"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    MIXED = "mixed"


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    store_type: VectorStoreType
    collection_name: str
    embedding_dimension: int

    # 连接配置
    host: str = "localhost"
    port: int = 19530
    timeout: int = 30

    # 索引配置
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 128
    nprobe: int = 10

    # 批量操作配置
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 0.5

    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000


@dataclass
class ChunkData:
    """文档块数据（统一格式）"""
    chunk_id: str
    content: str
    embedding: List[float]
    chunk_index: int = 0
    page_number: int = 0
    chunk_type: str = ChunkType.TEXT.value
    metadata: Dict[str, Any] = field(default_factory=dict)
    llm_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "embedding": self.embedding,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata,
            "llm_metadata": self.llm_metadata,
            "created_at": self.created_at or int(datetime.now().timestamp() * 1000)
        }


@dataclass
class SearchResult:
    """搜索结果（统一格式）"""
    id: Any
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

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata,
            "llm_metadata": self.llm_metadata,
            "created_at": self.created_at
        }


@dataclass
class CollectionStats:
    """集合统计信息（统一格式）"""
    collection_name: str
    total_entities: int
    embedding_dimension: int
    index_type: str
    metric_type: str
    # 性能统计
    total_inserts: int = 0
    total_searches: int = 0
    total_deletes: int = 0
    total_errors: int = 0
    avg_search_latency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "collection_name": self.collection_name,
            "total_entities": self.total_entities,
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "total_inserts": self.total_inserts,
            "total_searches": self.total_searches,
            "total_deletes": self.total_deletes,
            "total_errors": self.total_errors,
            "avg_search_latency": f"{self.avg_search_latency:.3f}s"
        }


class BaseVectorStore(ABC):
    """
    向量存储抽象基类

    定义所有向量存储服务必须实现的接口
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.collection_name = config.collection_name
        self._is_connected = False
        self._stats = {
            "total_inserts": 0,
            "total_searches": 0,
            "total_deletes": 0,
            "total_errors": 0,
            "avg_search_latency": 0.0
        }

    # ========================================================================
    # 连接管理
    # ========================================================================

    @abstractmethod
    async def connect(self) -> bool:
        """
        连接到向量存储服务

        Returns:
            是否成功连接
        """
        pass

    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            是否健康
        """
        pass

    async def ensure_connected(self):
        """确保连接可用（自动重连）"""
        if not self._is_connected:
            await self.connect()
            return

        is_healthy = await self.health_check()
        if not is_healthy:
            await self.connect()

    # ========================================================================
    # 集合管理
    # ========================================================================

    @abstractmethod
    async def init_collection(self, force_recreate: bool = False):
        """
        初始化集合

        Args:
            force_recreate: 是否强制重新创建集合
        """
        pass

    @abstractmethod
    def _validate_collection_schema(self):
        """验证集合schema"""
        pass

    @abstractmethod
    async def _create_collection(self):
        """创建集合"""
        pass

    @abstractmethod
    async def _create_index(self):
        """创建向量索引"""
        pass

    # ========================================================================
    # 数据操作
    # ========================================================================

    @abstractmethod
    async def insert_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        插入文档块

        Args:
            document_id: 文档ID
            chunks: 文档块数据列表
            batch_size: 批量插入大小

        Returns:
            插入的记录ID列表
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """
        删除文档的所有向量

        Args:
            document_id: 文档ID

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    async def update_chunk_metadata(
        self,
        chunk_id: str,
        llm_metadata: Dict[str, Any]
    ) -> bool:
        """
        更新文档块的元数据

        Args:
            chunk_id: 文档块ID
            llm_metadata: LLM提取的元数据

        Returns:
            是否成功
        """
        pass

    # ========================================================================
    # 批量操作
    # ========================================================================

    async def batch_search(
        self,
        query_embeddings: List[List[float]],
        limit: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[List[SearchResult]]:
        """
        批量向量搜索

        Args:
            query_embeddings: 查询向量列表
            limit: 每个查询返回的结果数量
            document_ids: 限定文档ID范围

        Returns:
            搜索结果列表的列表
        """
        results = []
        for embedding in query_embeddings:
            result = await self.search(
                query_embedding=embedding,
                limit=limit,
                document_ids=document_ids
            )
            results.append(result)
        return results

    async def hybrid_search(
        self,
        query_embedding: List[float],
        keywords: List[str],
        limit: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
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

    # ========================================================================
    # 统计信息
    # ========================================================================

    @abstractmethod
    async def get_collection_stats(self) -> CollectionStats:
        """
        获取集合统计信息

        Returns:
            集合统计信息
        """
        pass

    def _update_stats(self, operation: str, latency: float = 0.0):
        """更新统计信息"""
        if operation == "insert":
            self._stats["total_inserts"] += 1
        elif operation == "search":
            self._stats["total_searches"] += 1
            # 更新平均延迟
            total = self._stats["total_searches"]
            self._stats["avg_search_latency"] = (
                (self._stats["avg_search_latency"] * (total - 1) + latency) / total
            )
        elif operation == "delete":
            self._stats["total_deletes"] += 1
        elif operation == "error":
            self._stats["total_errors"] += 1


# 导出
__all__ = [
    'VectorStoreType',
    'ChunkType',
    'VectorStoreConfig',
    'ChunkData',
    'SearchResult',
    'CollectionStats',
    'BaseVectorStore'
]
