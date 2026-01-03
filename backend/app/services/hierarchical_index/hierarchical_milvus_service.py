"""
分层索引的Milvus存储服务
支持三个collection：文档摘要、章节索引、片段索引
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json

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
from app.schemas.hierarchical_index import (
    DocumentSummaryIndex,
    ChapterIndex,
    ChunkIndex,
    HierarchicalIndex
)

logger = get_structured_logger(__name__)


class HierarchicalMilvusService:
    """
    分层索引Milvus存储服务

    管理三个Collection:
    1. document_summaries - 文档摘要向量
    2. chapter_indexes - 章节摘要向量
    3. chunk_indexes - 片段内容向量
    """

    # Collection名称
    COLLECTION_DOC_SUMMARIES = "document_summaries"
    COLLECTION_CHAPTER_INDEXES = "chapter_indexes"
    COLLECTION_CHUNK_INDEXES = "chunk_indexes"

    # 向量维度
    EMBEDDING_DIM = 1024

    def __init__(self):
        """初始化服务"""
        self.config = {
            "host": getattr(settings, 'milvus_host', 'localhost'),
            "port": getattr(settings, 'milvus_port', 19530),
        }

        self.collections = {}
        self._is_connected = False

        logger.info("分层索引Milvus服务初始化完成")

    async def connect(self):
        """连接到Milvus并初始化collections"""
        try:
            connections.connect(
                alias="default",
                host=self.config["host"],
                port=self.config["port"]
            )
            self._is_connected = True

            logger.info(f"✓ 成功连接到Milvus: {self.config['host']}:{self.config['port']}")

            # 初始化三个collection
            await self._init_collections()

        except Exception as e:
            logger.error(f"连接Milvus失败: {str(e)}")
            raise

    async def disconnect(self):
        """断开连接"""
        try:
            connections.disconnect("default")
            self._is_connected = False
            logger.info("✓ 已断开Milvus连接")
        except Exception as e:
            logger.error(f"断开连接失败: {str(e)}")

    async def _init_collections(self):
        """初始化三个collection"""
        # 1. 文档摘要collection
        if not utility.has_collection(self.COLLECTION_DOC_SUMMARIES):
            await self._create_document_summary_collection()

        # 2. 章节索引collection
        if not utility.has_collection(self.COLLECTION_CHAPTER_INDEXES):
            await self._create_chapter_index_collection()

        # 3. 片段索引collection
        if not utility.has_collection(self.COLLECTION_CHUNK_INDEXES):
            await self._create_chunk_index_collection()

        # 加载collections
        self.collections["doc_summaries"] = Collection(self.COLLECTION_DOC_SUMMARIES)
        self.collections["chapters"] = Collection(self.COLLECTION_CHAPTER_INDEXES)
        self.collections["chunks"] = Collection(self.COLLECTION_CHUNK_INDEXES)

        logger.info("✓ 分层索引collections初始化完成")

    async def _create_document_summary_collection(self):
        """创建文档摘要collection"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="summary_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.EMBEDDING_DIM),
            FieldSchema(name="summary_text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=1000),  # JSON数组
            FieldSchema(name="entities", dtype=DataType.VARCHAR, max_length=1000),  # JSON数组
            FieldSchema(name="topics", dtype=DataType.VARCHAR, max_length=500),  # JSON数组
            FieldSchema(name="doc_length", dtype=DataType.INT64),
            FieldSchema(name="section_count", dtype=DataType.INT64),
            FieldSchema(name="chunk_count", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(fields=fields, description="文档摘要索引")
        collection = Collection(
            name=self.COLLECTION_DOC_SUMMARIES,
            schema=schema
        )

        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        collection.create_index(
            field_name="summary_embedding",
            index_params=index_params
        )

        logger.info(f"✓ 创建文档摘要collection: {self.COLLECTION_DOC_SUMMARIES}")

    async def _create_chapter_index_collection(self):
        """创建章节索引collection"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="chapter_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="summary_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.EMBEDDING_DIM),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="level", dtype=DataType.INT64),
            FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=500),  # JSON数组
            FieldSchema(name="parent_chapter_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="start_char", dtype=DataType.INT64),
            FieldSchema(name="end_char", dtype=DataType.INT64),
            FieldSchema(name="chunk_count", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(fields=fields, description="章节索引")
        collection = Collection(
            name=self.COLLECTION_CHAPTER_INDEXES,
            schema=schema
        )

        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        collection.create_index(
            field_name="summary_embedding",
            index_params=index_params
        )

        logger.info(f"✓ 创建章节索引collection: {self.COLLECTION_CHAPTER_INDEXES}")

    async def _create_chunk_index_collection(self):
        """创建片段索引collection"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chapter_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="content_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.EMBEDDING_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="start_char", dtype=DataType.INT64),
            FieldSchema(name="end_char", dtype=DataType.INT64),
            FieldSchema(name="page_number", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(fields=fields, description="片段索引")
        collection = Collection(
            name=self.COLLECTION_CHUNK_INDEXES,
            schema=schema
        )

        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        collection.create_index(
            field_name="content_embedding",
            index_params=index_params
        )

        logger.info(f"✓ 创建片段索引collection: {self.COLLECTION_CHUNK_INDEXES}")

    async def store_hierarchical_index(
        self,
        hierarchical_index: HierarchicalIndex,
        embedding_service
    ):
        """
        存储完整的分层索引

        Args:
            hierarchical_index: 分层索引结构
            embedding_service: 嵌入服务
        """
        import json

        try:
            # 1. 生成并存储文档摘要向量
            await self._store_document_summary(
                hierarchical_index.document_summary,
                embedding_service
            )

            # 2. 生成并存储章节向量
            for chapter in hierarchical_index.chapters:
                await self._store_chapter_index(
                    chapter,
                    embedding_service
                )

            # 3. 生成并存储片段向量
            for chunk in hierarchical_index.chunks:
                await self._store_chunk_index(
                    chunk,
                    embedding_service
                )

            # 刷新所有collections
            for collection in self.collections.values():
                collection.flush()

            logger.info(
                f"✓ 成功存储分层索引: "
                f"摘要=1, 章节={len(hierarchical_index.chapters)}, "
                f"片段={len(hierarchical_index.chunks)}"
            )

        except Exception as e:
            logger.error(f"存储分层索引失败: {str(e)}")
            raise

    async def _store_document_summary(
        self,
        summary: DocumentSummaryIndex,
        embedding_service
    ):
        """存储文档摘要"""
        import json

        # 生成摘要向量
        if not summary.embedding:
            embeddings = await embedding_service.get_embeddings([summary.summary_text])
            summary.embedding = embeddings[0]

        # 准备数据
        data = [{
            "id": f"doc_sum_{summary.document_id}",
            "document_id": summary.document_id,
            "summary_embedding": summary.embedding,
            "summary_text": summary.summary_text[:2000],
            "keywords": json.dumps(summary.keywords, ensure_ascii=False),
            "entities": json.dumps(summary.entities, ensure_ascii=False),
            "topics": json.dumps(summary.topics, ensure_ascii=False),
            "doc_length": summary.doc_length,
            "section_count": summary.section_count,
            "chunk_count": summary.chunk_count,
        }]

        # 插入数据
        collection = self.collections["doc_summaries"]
        collection.insert(data)
        logger.debug(f"  ✓ 存储文档摘要: {summary.document_id}")

    async def _store_chapter_index(
        self,
        chapter: ChapterIndex,
        embedding_service
    ):
        """存储章节索引"""
        import json

        # 生成章节摘要向量
        if not chapter.embedding:
            embeddings = await embedding_service.get_embeddings([chapter.summary])
            chapter.embedding = embeddings[0]

        # 准备数据
        data = [{
            "id": chapter.chapter_id,
            "chapter_id": chapter.chapter_id,
            "document_id": chapter.document_id,
            "summary_embedding": chapter.embedding,
            "title": chapter.title[:500],
            "summary": chapter.summary[:2000],
            "level": chapter.level,
            "keywords": json.dumps(chapter.keywords, ensure_ascii=False),
            "parent_chapter_id": chapter.parent_chapter_id or "",
            "start_char": chapter.start_char,
            "end_char": chapter.end_char,
            "chunk_count": chapter.chunk_count,
        }]

        # 插入数据
        collection = self.collections["chapters"]
        collection.insert(data)
        logger.debug(f"  ✓ 存储章节索引: {chapter.chapter_id}")

    async def _store_chunk_index(
        self,
        chunk: ChunkIndex,
        embedding_service
    ):
        """存储片段索引"""
        # 生成片段内容向量
        if not chunk.embedding:
            embeddings = await embedding_service.get_embeddings([chunk.content])
            chunk.embedding = embeddings[0]

        # 准备数据
        data = [{
            "id": chunk.chunk_id,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chapter_id": chunk.chapter_id or "",
            "content_embedding": chunk.embedding,
            "content": chunk.content[:5000],
            "chunk_type": chunk.chunk_type.value,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "page_number": chunk.page_number or 0,
        }]

        # 插入数据
        collection = self.collections["chunks"]
        collection.insert(data)
        logger.debug(f"  ✓ 存储片段索引: {chunk.chunk_id}")

    async def search_document_summaries(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索文档摘要

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            document_ids: 限定搜索的文档ID列表

        Returns:
            List[Dict]: 检索结果列表
        """
        import json

        collection = self.collections["doc_summaries"]
        collection.load()

        # 构建搜索表达式
        expr = None
        if document_ids:
            expr = f"document_id in {json.dumps(document_ids)}"

        # 执行搜索
        results = collection.search(
            data=[query_embedding],
            anns_field="summary_embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=expr,
            output_fields=[
                "document_id", "summary_text", "keywords", "entities",
                "topics", "doc_length", "section_count", "chunk_count"
            ]
        )

        # 解析结果
        retrieved = []
        for hit in results[0]:
            retrieved.append({
                "document_id": hit.entity.get("document_id"),
                "summary_text": hit.entity.get("summary_text"),
                "keywords": json.loads(hit.entity.get("keywords", "[]")),
                "entities": json.loads(hit.entity.get("entities", "[]")),
                "topics": json.loads(hit.entity.get("topics", "[]")),
                "score": hit.score,
                "metadata": {
                    "doc_length": hit.entity.get("doc_length"),
                    "section_count": hit.entity.get("section_count"),
                    "chunk_count": hit.entity.get("chunk_count")
                }
            })

        return retrieved

    async def search_chapter_indexes(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """搜索章节索引"""
        import json

        collection = self.collections["chapters"]
        collection.load()

        # 构建搜索表达式
        expr = None
        if document_ids:
            expr = f"document_id in {json.dumps(document_ids)}"

        # 执行搜索
        results = collection.search(
            data=[query_embedding],
            anns_field="summary_embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=expr,
            output_fields=[
                "chapter_id", "document_id", "title", "summary",
                "level", "keywords", "parent_chapter_id", "chunk_count"
            ]
        )

        # 解析结果
        retrieved = []
        for hit in results[0]:
            retrieved.append({
                "chapter_id": hit.entity.get("chapter_id"),
                "document_id": hit.entity.get("document_id"),
                "title": hit.entity.get("title"),
                "summary": hit.entity.get("summary"),
                "level": hit.entity.get("level"),
                "keywords": json.loads(hit.entity.get("keywords", "[]")),
                "parent_chapter_id": hit.entity.get("parent_chapter_id"),
                "chunk_count": hit.entity.get("chunk_count"),
                "score": hit.score
            })

        return retrieved

    async def search_chunk_indexes(
        self,
        query_embedding: List[float],
        top_k: int = 50,
        document_ids: Optional[List[str]] = None,
        chapter_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """搜索片段索引"""
        import json

        collection = self.collections["chunks"]
        collection.load()

        # 构建搜索表达式
        expr_parts = []
        if document_ids:
            expr_parts.append(f"document_id in {json.dumps(document_ids)}")
        if chapter_ids:
            expr_parts.append(f"chapter_id in {json.dumps(chapter_ids)}")

        expr = " and ".join(expr_parts) if expr_parts else None

        # 执行搜索
        results = collection.search(
            data=[query_embedding],
            anns_field="content_embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=expr,
            output_fields=[
                "chunk_id", "document_id", "chapter_id",
                "content", "chunk_type", "chunk_index", "page_number"
            ]
        )

        # 解析结果
        retrieved = []
        for hit in results[0]:
            retrieved.append({
                "chunk_id": hit.entity.get("chunk_id"),
                "document_id": hit.entity.get("document_id"),
                "chapter_id": hit.entity.get("chapter_id"),
                "content": hit.entity.get("content"),
                "chunk_type": hit.entity.get("chunk_type"),
                "chunk_index": hit.entity.get("chunk_index"),
                "page_number": hit.entity.get("page_number"),
                "score": hit.score
            })

        return retrieved

    async def delete_document_index(self, document_id: str):
        """
        删除文档的所有索引

        Args:
            document_id: 文档ID
        """
        try:
            # 1. 删除文档摘要
            doc_sum_collection = self.collections["doc_summaries"]
            doc_sum_collection.delete(
                expr=f"document_id == '{document_id}'"
            )

            # 2. 删除章节索引
            chapter_collection = self.collections["chapters"]
            chapter_collection.delete(
                expr=f"document_id == '{document_id}'"
            )

            # 3. 删除片段索引
            chunk_collection = self.collections["chunks"]
            chunk_collection.delete(
                expr=f"document_id == '{document_id}'"
            )

            # 刷新
            for collection in self.collections.values():
                collection.flush()

            logger.info(f"✓ 删除文档索引: {document_id}")

        except Exception as e:
            logger.error(f"删除文档索引失败: {str(e)}")
            raise


# 全局单例
_hierarchical_milvus_service = None


async def get_hierarchical_milvus_service() -> HierarchicalMilvusService:
    """获取分层索引Milvus服务单例"""
    global _hierarchical_milvus_service
    if _hierarchical_milvus_service is None:
        _hierarchical_milvus_service = HierarchicalMilvusService()
        await _hierarchical_milvus_service.connect()
    return _hierarchical_milvus_service
