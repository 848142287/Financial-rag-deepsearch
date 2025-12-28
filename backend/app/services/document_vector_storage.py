"""
文档向量存储服务 - 整合文档分析和向量存储
将 EnhancedDocumentAnalyzer 的结果存储到 Milvus，包含完整元数据
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from app.services.parsers.advanced.enhanced_document_analyzer import (
    DocumentAnalysisResult,
    ChunkAnalysisResult
)
from app.services.enhanced_milvus_service import EnhancedMilvusService
from app.services.parsers.advanced.enhanced_metadata_extractor import (
    ChunkMetadataExtraction
)

logger = logging.getLogger(__name__)


class DocumentVectorStorage:
    """文档向量存储服务"""

    def __init__(self):
        self.milvus_service = EnhancedMilvusService()

    async def initialize(self):
        """初始化服务"""
        await self.milvus_service.init_collection()
        logger.info("DocumentVectorStorage initialized")

    async def store_document_analysis(
        self,
        analysis_result: DocumentAnalysisResult,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        存储文档分析结果到 Milvus，包含完整元数据

        Args:
            analysis_result: 文档分析结果
            document_id: 文档ID（可选，默认使用 analysis_result.document_id）

        Returns:
            存储结果
        """
        try:
            document_id = document_id or analysis_result.document_id

            # 准备文档块数据
            chunks_data = []

            for i, chunk_analysis in enumerate(analysis_result.chunk_analyses):
                # 跳过没有向量嵌入的块
                if not chunk_analysis.embedding:
                    logger.warning(f"Chunk {i} has no embedding, skipping")
                    continue

                # 构建完整的元数据
                chunk_data = self._build_chunk_data(
                    chunk_analysis=chunk_analysis,
                    document_id=document_id,
                    chunk_index=i
                )

                chunks_data.append(chunk_data)

            # 存储到 Milvus
            if not chunks_data:
                logger.warning(f"No valid chunks to store for document {document_id}")
                return {
                    "status": "no_chunks",
                    "document_id": document_id,
                    "chunk_count": 0
                }

            inserted_ids = await self.milvus_service.insert_chunks_with_full_metadata(
                document_id=document_id,
                chunks_data=chunks_data
            )

            logger.info(
                f"Successfully stored {len(chunks_data)} chunks "
                f"with full metadata for document {document_id}"
            )

            return {
                "status": "success",
                "document_id": document_id,
                "chunk_count": len(chunks_data),
                "inserted_ids": inserted_ids
            }

        except Exception as e:
            logger.error(f"Failed to store document analysis: {e}")
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e)
            }

    def _build_chunk_data(
        self,
        chunk_analysis: ChunkAnalysisResult,
        document_id: str,
        chunk_index: int
    ) -> Dict[str, Any]:
        """
        构建文档块数据，包含完整元数据

        Args:
            chunk_analysis: 文档块分析结果
            document_id: 文档ID
            chunk_index: 文档块索引

        Returns:
            文档块数据字典
        """
        chunk = chunk_analysis.chunk
        metadata = chunk_analysis.metadata
        embedding = chunk_analysis.embedding

        # 基础数据
        chunk_data = {
            "chunk_id": f"{document_id}_chunk_{chunk_index}",
            "content": chunk.page_content,
            "embedding": embedding,
            "chunk_index": chunk_index,
            "page_number": chunk.metadata.get("page", 0),
            "chunk_type": chunk.metadata.get("type", "text"),
        }

        # LLM 提取的完整元数据
        llm_metadata = {}

        if metadata:
            # 文档块摘要
            if metadata.summary:
                llm_metadata["summary"] = metadata.summary

            # 关键词
            if metadata.keywords:
                llm_metadata["keywords"] = metadata.keywords

            # 实体
            if metadata.entities:
                llm_metadata["entities"] = metadata.entities

            # 主题
            if metadata.topic:
                llm_metadata["topic"] = metadata.topic

            # 重要性分数
            if metadata.importance_score is not None:
                llm_metadata["importance_score"] = metadata.importance_score

            # 文档块类型
            if metadata.chunk_type:
                llm_metadata["chunk_type"] = metadata.chunk_type

            # 章节ID
            if metadata.chapter_id:
                llm_metadata["chapter_id"] = metadata.chapter_id

            # 位置信息
            if metadata.position:
                llm_metadata["position"] = {
                    "start": metadata.position[0],
                    "end": metadata.position[1]
                }

        # 添加原始元数据中的额外信息
        for key, value in chunk.metadata.items():
            if key not in ["page", "type", "source", "filename"]:
                llm_metadata[f"raw_{key}"] = value

        chunk_data["llm_metadata"] = llm_metadata

        return chunk_data

    async def batch_store_documents(
        self,
        analysis_results: List[DocumentAnalysisResult]
    ) -> List[Dict[str, Any]]:
        """
        批量存储文档分析结果

        Args:
            analysis_results: 文档分析结果列表

        Returns:
            存储结果列表
        """
        results = []

        for analysis_result in analysis_results:
            try:
                result = await self.store_document_analysis(analysis_result)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed to store document {analysis_result.document_id}: {e}"
                )
                results.append({
                    "status": "error",
                    "document_id": analysis_result.document_id,
                    "error": str(e)
                })

        return results

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        limit: int = 10,
        document_ids: Optional[List[str]] = None,
        score_threshold: float = 0.0,
        chunk_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似的文档块

        Args:
            query_embedding: 查询向量
            limit: 返回结果数量
            document_ids: 限定文档ID范围
            score_threshold: 相似度阈值
            chunk_types: 限定文档块类型

        Returns:
            搜索结果列表
        """
        return await self.milvus_service.search_with_metadata(
            query_embedding=query_embedding,
            limit=limit,
            document_ids=document_ids,
            score_threshold=score_threshold,
            chunk_types=chunk_types,
            output_full_metadata=True
        )

    async def get_document_stats(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """
        获取文档的向量存储统计信息

        Args:
            document_id: 文档ID

        Returns:
            统计信息
        """
        try:
            # 查询该文档的所有文档块
            results = await self.milvus_service.search_with_metadata(
                query_embedding=[0.0] * len(self.milvus_service.collection.query(
                    expr=f"document_id == '{document_id}'",
                    output_fields=["embedding"]
                )[0]["embedding"]) if False else [],
                limit=10000,
                document_ids=[document_id],
                score_threshold=0.0
            )

            return {
                "document_id": document_id,
                "total_chunks": len(results),
                "chunks_with_summary": sum(
                    1 for r in results
                    if r.get("llm_metadata", {}).get("summary")
                ),
                "chunks_with_keywords": sum(
                    1 for r in results
                    if r.get("llm_metadata", {}).get("keywords")
                ),
                "chunks_with_entities": sum(
                    1 for r in results
                    if r.get("llm_metadata", {}).get("entities")
                )
            }

        except Exception as e:
            logger.error(f"Failed to get document stats: {e}")
            return {
                "document_id": document_id,
                "error": str(e)
            }


# 全局实例
document_vector_storage = DocumentVectorStorage()


async def get_document_vector_storage() -> DocumentVectorStorage:
    """获取文档向量存储服务实例"""
    if not document_vector_storage.milvus_service.collection:
        await document_vector_storage.initialize()
    return document_vector_storage
