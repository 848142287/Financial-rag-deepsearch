"""
分层检索服务
使用三层索引（文档摘要、章节、片段）进行智能检索
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.structured_logging import get_structured_logger
from app.schemas.hierarchical_index import (
    HierarchicalRetrievalRequest,
    HierarchicalRetrievalResult,
    RetrievedDocument,
    RetrievedChapter,
    RetrievedChunk
)
from app.services.llm_service import LLMService
from app.services.embeddings.unified_embedding_service import UnifiedEmbeddingService

logger = get_structured_logger(__name__)


class HierarchicalRetrievalService:
    """
    分层检索服务

    核心思想：
    1. 先在文档摘要层进行粗粒度检索，筛选出相关文档
    2. 在筛选出的文档的章节层进行中粒度检索，定位相关章节
    3. 在筛选出的章节的片段层进行细粒度检索，获取精确内容

    优势：
    - 减少检索范围，提高效率
    - 层层缩小范围，提高准确率
    - 提供多粒度的上下文信息
    """

    def __init__(
        self,
        embedding_service: UnifiedEmbeddingService = None,
        llm_service: LLMService = None
    ):
        """
        初始化分层检索服务

        Args:
            embedding_service: 嵌入服务
            llm_service: LLM服务
        """
        self.embedding_service = embedding_service or UnifiedEmbeddingService()
        self.llm_service = llm_service or LLMService()

        logger.info("分层检索服务初始化完成")

    async def retrieve(
        self,
        request: HierarchicalRetrievalRequest,
        hierarchical_indexes: Dict[str, Any]  # document_id -> HierarchicalIndex
    ) -> HierarchicalRetrievalResult:
        """
        执行分层检索

        Args:
            request: 检索请求
            hierarchical_indexes: 分层索引字典

        Returns:
            HierarchicalRetrievalResult: 检索结果
        """
        start_time = datetime.now()

        logger.info(f"🔍 开始分层检索: {request.query}")

        try:
            results = HierarchicalRetrievalResult(
                query=request.query,
                documents=[],
                chapters=[],
                chunks=[],
                merged_results=[]
            )

            # 第1层：文档级检索
            if request.use_summary:
                logger.info("  📄 第1层：文档摘要检索...")
                results.documents = await self._retrieve_from_documents(
                    request=request,
                    hierarchical_indexes=hierarchical_indexes
                )
                results.total_docs = len(results.documents)

                if not results.documents:
                    logger.warning("⚠️ 未找到相关文档")
                    return results

                logger.info(f"    ✓ 找到 {len(results.documents)} 个相关文档")

                # 过滤出相关文档的索引
                relevant_doc_ids = [doc.document_id for doc in results.documents]
                relevant_indexes = {
                    doc_id: idx
                    for doc_id, idx in hierarchical_indexes.items()
                    if doc_id in relevant_doc_ids
                }
            else:
                relevant_indexes = hierarchical_indexes

            # 第2层：章节级检索
            if request.use_chapters and relevant_indexes:
                logger.info("  📑 第2层：章节检索...")
                results.chapters = await self._retrieve_from_chapters(
                    request=request,
                    hierarchical_indexes=relevant_indexes,
                    max_chapters_per_doc=request.max_chapters_per_doc
                )
                results.total_chapters = len(results.chapters)

                if not results.chapters:
                    logger.warning("⚠️ 未找到相关章节")
                    return results

                logger.info(f"    ✓ 找到 {len(results.chapters)} 个相关章节")

                # 提取相关章节ID
                relevant_chapter_ids = [ch.chapter_id for ch in results.chapters]
            else:
                relevant_chapter_ids = None

            # 第3层：片段级检索
            if request.use_chunks and relevant_indexes:
                logger.info("  ✂️ 第3层：片段检索...")
                results.chunks = await self._retrieve_from_chunks(
                    request=request,
                    hierarchical_indexes=relevant_indexes,
                    chapter_ids=relevant_chapter_ids,
                    max_chunks_per_chapter=request.max_chunks_per_chapter
                )
                results.total_chunks = len(results.chunks)

                logger.info(f"    ✓ 找到 {len(results.chunks)} 个相关片段")

            # 合并结果
            results.merged_results = self._merge_retrieval_results(
                results.documents,
                results.chapters,
                results.chunks
            )

            # 限制最终结果数量
            results.merged_results = results.merged_results[:request.top_k]

            # 统计耗时
            results.retrieval_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"✅ 分层检索完成！"
                f"文档: {results.total_docs}, "
                f"章节: {results.total_chapters}, "
                f"片段: {results.total_chunks}, "
                f"耗时: {results.retrieval_time:.2f}秒"
            )

            return results

        except Exception as e:
            logger.error(f"❌ 分层检索失败: {str(e)}", exc_info=True)
            raise

    async def _retrieve_from_documents(
        self,
        request: HierarchicalRetrievalRequest,
        hierarchical_indexes: Dict[str, Any]
    ) -> List[RetrievedDocument]:
        """
        从文档摘要层检索

        Args:
            request: 检索请求
            hierarchical_indexes: 分层索引字典

        Returns:
            List[RetrievedDocument]: 相关文档列表
        """
        # 生成查询嵌入
        query_embedding = await self.embedding_service.embed_batch([request.query])
        query_embedding = query_embedding[0].tolist()

        retrieved_docs = []

        # 遍历所有文档的摘要索引
        for doc_id, index in hierarchical_indexes.items():
            document_summary = index.document_summary

            # 检查是否有限定的文档ID
            if request.document_ids and doc_id not in request.document_ids:
                continue

            # 如果有嵌入向量，计算相似度
            if document_summary.embedding:
                score = self._cosine_similarity(
                    query_embedding,
                    document_summary.embedding
                )

                if score >= request.doc_threshold:
                    retrieved_docs.append(RetrievedDocument(
                        document_id=doc_id,
                        summary_text=document_summary.summary_text,
                        score=score,
                        keywords=document_summary.keywords,
                        entities=document_summary.entities
                    ))

        # 按相似度排序，取前N个
        retrieved_docs.sort(key=lambda x: x.score, reverse=True)
        return retrieved_docs[:request.max_docs]

    async def _retrieve_from_chapters(
        self,
        request: HierarchicalRetrievalRequest,
        hierarchical_indexes: Dict[str, Any],
        max_chapters_per_doc: int
    ) -> List[RetrievedChapter]:
        """
        从章节层检索

        Args:
            request: 检索请求
            hierarchical_indexes: 分层索引字典
            max_chapters_per_doc: 每个文档最多返回的章节数

        Returns:
            List[RetrievedChapter]: 相关章节列表
        """
        # 生成查询嵌入
        query_embedding = await self.embedding_service.embed_batch([request.query])
        query_embedding = query_embedding[0].tolist()

        retrieved_chapters = []

        # 遍历所有文档的章节
        for doc_id, index in hierarchical_indexes.items():
            chapters_per_doc = 0

            for chapter in index.chapters:
                # 检查是否有限定的章节ID
                if request.chapter_ids and chapter.chapter_id not in request.chapter_ids:
                    continue

                # 如果有嵌入向量，计算相似度
                if chapter.embedding:
                    score = self._cosine_similarity(
                        query_embedding,
                        chapter.embedding
                    )

                    if score >= request.chapter_threshold:
                        retrieved_chapters.append(RetrievedChapter(
                            chapter_id=chapter.chapter_id,
                            document_id=doc_id,
                            title=chapter.title,
                            summary=chapter.summary,
                            score=score,
                            level=chapter.level,
                            chunk_count=chapter.chunk_count
                        ))

                        chapters_per_doc += 1
                        if chapters_per_doc >= max_chapters_per_doc:
                            break

        # 按相似度排序
        retrieved_chapters.sort(key=lambda x: x.score, reverse=True)
        return retrieved_chapters

    async def _retrieve_from_chunks(
        self,
        request: HierarchicalRetrievalRequest,
        hierarchical_indexes: Dict[str, Any],
        chapter_ids: Optional[List[str]] = None,
        max_chunks_per_chapter: int = 5
    ) -> List[RetrievedChunk]:
        """
        从片段层检索

        Args:
            request: 检索请求
            hierarchical_indexes: 分层索引字典
            chapter_ids: 限定检索的章节ID列表
            max_chunks_per_chapter: 每个章节最多返回的片段数

        Returns:
            List[RetrievedChunk]: 相关片段列表
        """
        # 生成查询嵌入
        query_embedding = await self.embedding_service.embed_batch([request.query])
        query_embedding = query_embedding[0].tolist()

        retrieved_chunks = []

        # 遍历所有文档的片段
        for doc_id, index in hierarchical_indexes.items():
            chunks_per_chapter = {}

            for chunk in index.chunks:
                # 检查是否限定在特定章节
                if chapter_ids and chunk.chapter_id not in chapter_ids:
                    continue

                # 如果有嵌入向量，计算相似度
                if chunk.embedding:
                    score = self._cosine_similarity(
                        query_embedding,
                        chunk.embedding
                    )

                    if score >= request.chunk_threshold:
                        # 查找章节标题
                        chapter_title = None
                        if chunk.chapter_id:
                            chapter = next(
                                (ch for ch in index.chapters
                                 if ch.chapter_id == chunk.chapter_id),
                                None
                            )
                            if chapter:
                                chapter_title = chapter.title

                        retrieved_chunks.append(RetrievedChunk(
                            chunk_id=chunk.chunk_id,
                            document_id=doc_id,
                            chapter_id=chunk.chapter_id,
                            content=chunk.content,
                            score=score,
                            chapter_title=chapter_title,
                            metadata=chunk.metadata
                        ))

                        # 统计每章节的片段数
                        if chunk.chapter_id:
                            chunks_per_chapter[chunk.chapter_id] = \
                                chunks_per_chapter.get(chunk.chapter_id, 0) + 1

                            # 限制每章节的片段数
                            if chunks_per_chapter[chunk.chapter_id] > max_chunks_per_chapter:
                                # 过滤超出限制的片段
                                retrieved_chunks = [
                                    c for c in retrieved_chunks
                                    if not (c.chapter_id == chunk.chapter_id and
                                           chunks_per_chapter[chunk.chapter_id] > max_chunks_per_chapter)
                                ]

        # 按相似度排序
        retrieved_chunks.sort(key=lambda x: x.score, reverse=True)
        return retrieved_chunks

    def _merge_retrieval_results(
        self,
        documents: List[RetrievedDocument],
        chapters: List[RetrievedChapter],
        chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """
        合并三层检索结果

        策略：
        - 优先使用片段级结果
        - 为片段添加章节和文档上下文
        - 重新计算综合得分
        """
        if not chunks:
            # 如果没有片段结果，从章节生成伪片段
            merged = []
            for chapter in chapters[:10]:
                merged.append(RetrievedChunk(
                    chunk_id=f"{chapter.chapter_id}_pseudo",
                    content=chapter.summary,
                    score=chapter.score * 0.9,  # 章节得分打折
                    chapter_title=chapter.title,
                    chapter_id=chapter.chapter_id,
                    metadata={"source": "chapter", "level": chapter.level}
                ))
            return merged

        # 为片段添加额外的上下文得分
        for chunk in chunks:
            # 如果片段所属章节得分高，提升片段得分
            if chunk.chapter_id:
                chapter = next((ch for ch in chapters if ch.chapter_id == chunk.chapter_id), None)
                if chapter and chapter.score > 0.7:
                    chunk.score = chunk.score * 1.1  # 提升得分

        return chunks

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        计算余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            float: 相似度得分 [0, 1]
        """
        import numpy as np

        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # 确保范围在[0, 1]
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logger.error(f"计算相似度失败: {str(e)}")
            return 0.0


# 全局单例
_hierarchical_retrieval_service = None


def get_hierarchical_retrieval_service() -> HierarchicalRetrievalService:
    """获取分层检索服务单例"""
    global _hierarchical_retrieval_service
    if _hierarchical_retrieval_service is None:
        _hierarchical_retrieval_service = HierarchicalRetrievalService()
    return _hierarchical_retrieval_service
