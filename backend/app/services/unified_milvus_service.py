"""
统一的 Milvus 服务入口
整合所有 Milvus 相关功能，确保只有一个服务入口
"""
import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from app.services.enhanced_milvus_service import EnhancedMilvusService
from app.services.document_vector_storage import DocumentVectorStorage
from app.core.vector_config import vector_config, get_dimension

logger = logging.getLogger(__name__)


class MilvusServiceMode(str, Enum):
    """Milvus 服务模式"""
    BASIC = "basic"           # 基础模式（向后兼容）
    ENHANCED = "enhanced"     # 增强模式（完整元数据）


class UnifiedMilvusService:
    """
    统一的 Milvus 服务入口

    提供一致的接口，支持基础模式和增强模式
    """

    def __init__(self, mode: MilvusServiceMode = MilvusServiceMode.ENHANCED):
        """
        初始化统一服务

        Args:
            mode: 服务模式
        """
        self.mode = mode
        self.enhanced_service = EnhancedMilvusService()
        self.document_storage = DocumentVectorStorage()
        self._initialized = False

    async def initialize(self):
        """初始化服务"""
        if not self._initialized:
            await self.enhanced_service.init_collection()
            await self.document_storage.initialize()
            self._initialized = True
            logger.info(f"UnifiedMilvusService initialized in {self.mode} mode")

    async def insert_documents(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        include_llm_metadata: bool = True
    ) -> List[int]:
        """
        插入文档块

        Args:
            document_id: 文档ID
            chunks: 文档块列表
            include_llm_metadata: 是否包含 LLM 提取的元数据

        Returns:
            插入的记录ID列表
        """
        await self.initialize()

        if self.mode == MilvusServiceMode.ENHANCED and include_llm_metadata:
            # 使用增强服务，包含完整元数据
            return await self.enhanced_service.insert_chunks_with_full_metadata(
                document_id=document_id,
                chunks_data=chunks
            )
        else:
            # 使用基础模式（向后兼容）
            return await self._insert_basic(document_id, chunks)

    async def _insert_basic(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]]
    ) -> List[int]:
        """基础插入模式（向后兼容）"""
        # 移除 LLM 元数据，只保留基础字段
        basic_chunks = []
        for chunk in chunks:
            basic_chunk = {
                "chunk_id": chunk.get("chunk_id", ""),
                "content": chunk["content"],
                "embedding": chunk["embedding"],
                "chunk_index": chunk.get("chunk_index", 0),
                "page_number": chunk.get("page_number", 0),
                "chunk_type": chunk.get("chunk_type", "text"),
                "llm_metadata": {}  # 空元数据
            }
            basic_chunks.append(basic_chunk)

        return await self.enhanced_service.insert_chunks_with_full_metadata(
            document_id=document_id,
            chunks_data=basic_chunks
        )

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        document_ids: Optional[List[str]] = None,
        score_threshold: float = 0.0,
        chunk_types: Optional[List[str]] = None,
        include_full_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        向量搜索

        Args:
            query_embedding: 查询向量
            limit: 返回结果数量
            document_ids: 限定文档ID范围
            score_threshold: 相似度阈值
            chunk_types: 限定文档块类型
            include_full_metadata: 是否包含完整元数据

        Returns:
            搜索结果列表
        """
        await self.initialize()

        return await self.enhanced_service.search_with_metadata(
            query_embedding=query_embedding,
            limit=limit,
            document_ids=document_ids,
            score_threshold=score_threshold,
            chunk_types=chunk_types,
            output_full_metadata=include_full_metadata
        )

    async def delete_document(self, document_id: str) -> bool:
        """
        删除文档的所有向量

        Args:
            document_id: 文档ID

        Returns:
            是否成功
        """
        await self.initialize()

        # 通过删除表达式删除
        try:
            self.enhanced_service.collection.delete(
                expr=f'document_id == "{document_id}"'
            )
            self.enhanced_service.collection.flush()
            logger.info(f"Deleted document {document_id} from Milvus")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        await self.initialize()
        return await self.enhanced_service.get_collection_stats()

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            健康状态信息
        """
        try:
            await self.initialize()

            stats = await self.get_collection_stats()

            return {
                "status": "healthy",
                "mode": self.mode,
                "collection_name": stats["collection_name"],
                "total_entities": stats["total_entities"],
                "embedding_dimension": stats["embedding_dimension"],
                "vector_config": {
                    "primary_model": vector_config.primary_model,
                    "backup_model": vector_config.backup_model,
                    "dimension": get_dimension(),
                    "index_type": vector_config.index_type,
                    "metric_type": vector_config.metric_type
                }
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# 全局统一服务实例
unified_milvus_service = UnifiedMilvusService(
    mode=MilvusServiceMode.ENHANCED
)


# 便捷函数
async def get_milvus_service(
    mode: Optional[MilvusServiceMode] = None
) -> UnifiedMilvusService:
    """
    获取统一的 Milvus 服务实例

    Args:
        mode: 服务模式（None 则使用默认模式）

    Returns:
        UnifiedMilvusService 实例
    """
    if mode and mode != unified_milvus_service.mode:
        # 创建新实例
        service = UnifiedMilvusService(mode=mode)
        await service.initialize()
        return service

    await unified_milvus_service.initialize()
    return unified_milvus_service


# 兼容性别名（向后兼容）
MilvusService = UnifiedMilvusService
milvus_service_unified = unified_milvus_service


async def init_milvus_collection():
    """初始化 Milvus 集合（兼容旧代码）"""
    service = await get_milvus_service()
    return service


# 使用示例
"""
# 示例 1: 存储文档（增强模式）
from app.services.unified_milvus_service import get_milvus_service

milvus = await get_milvus_service()

# 准备文档块数据
chunks_data = [
    {
        "chunk_id": "doc1_chunk_0",
        "content": "这是文档内容",
        "embedding": [0.1, 0.2, ...],  # 1024维向量
        "chunk_index": 0,
        "page_number": 1,
        "chunk_type": "text",
        "llm_metadata": {
            "summary": "文档摘要",
            "keywords": ["关键词1", "关键词2"],
            "entities": ["实体1"],
            "topic": "主题",
            "importance_score": 0.85
        }
    }
]

# 插入文档
inserted_ids = await milvus.insert_documents(
    document_id="doc1",
    chunks=chunks_data,
    include_llm_metadata=True
)


# 示例 2: 搜索
results = await milvus.search(
    query_embedding=[0.1, 0.2, ...],  # 1024维查询向量
    limit=10,
    score_threshold=0.7,
    include_full_metadata=True
)

for result in results:
    print(f"Score: {result['score']}")
    print(f"Content: {result['content']}")
    print(f"Summary: {result['llm_metadata'].get('summary')}")


# 示例 3: 健康检查
health = await milvus.health_check()
print(health)
"""
