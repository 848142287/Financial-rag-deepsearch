"""
分层索引与文档处理流水线的集成
将分层索引构建整合到文档解析流程中
"""

from typing import Dict, Any
from datetime import datetime

from app.core.structured_logging import get_structured_logger
from app.services.hierarchical_index import (
    get_hierarchical_index_extractor,
    get_hierarchical_milvus_service
)
from app.services.embeddings.unified_embedding_service import UnifiedEmbeddingService
from app.schemas.hierarchical_index import HierarchicalIndex

logger = get_structured_logger(__name__)


class HierarchicalIndexPipelineIntegration:
    """
    分层索引流水线集成

    功能：
    1. 在文档处理完成后自动构建分层索引
    2. 将索引存储到Milvus
    3. 更新文档处理状态
    """

    def __init__(self):
        """初始化集成服务"""
        self.index_extractor = get_hierarchical_index_extractor()
        self.embedding_service = UnifiedEmbeddingService()

    async def build_index_from_pipeline(
        self,
        document_id: str,
        markdown_content: str,
        deepseek_summary: Dict[str, Any] = None
    ) -> HierarchicalIndex:
        """
        从文档处理流水线构建分层索引

        这个方法应该被添加到document_pipeline_service的处理流程中

        Args:
            document_id: 文档ID
            markdown_content: Markdown格式的文档内容
            deepseek_summary: Deepseek深度汇总结果

        Returns:
            HierarchicalIndex: 分层索引结构
        """
        try:
            logger.info(f"📚 开始为文档 {document_id} 构建分层索引")

            # 1. 抽取分层索引
            hierarchical_index = await self.index_extractor.extract_hierarchical_index(
                document_id=document_id,
                markdown_content=markdown_content,
                deepseek_summary=deepseek_summary
            )

            # 2. 生成嵌入向量
            logger.info("  🎯 生成嵌入向量...")
            await self._generate_embeddings(hierarchical_index)

            # 3. 存储到Milvus
            logger.info("  💾 存储到Milvus...")
            milvus_service = await get_hierarchical_milvus_service()
            await milvus_service.store_hierarchical_index(
                hierarchical_index=hierarchical_index,
                embedding_service=self.embedding_service
            )

            logger.info(
                f"✅ 分层索引构建完成！"
                f"摘要=1, 章节={len(hierarchical_index.chapters)}, "
                f"片段={len(hierarchical_index.chunks)}, "
                f"耗时={hierarchical_index.processing_time:.2f}秒"
            )

            return hierarchical_index

        except Exception as e:
            logger.error(f"❌ 构建分层索引失败: {str(e)}", exc_info=True)
            raise

    async def _generate_embeddings(self, hierarchical_index: HierarchicalIndex):
        """
        为分层索引生成嵌入向量

        Args:
            hierarchical_index: 分层索引结构
        """
        # 1. 为文档摘要生成向量
        if not hierarchical_index.document_summary.embedding:
            embeddings = await self.embedding_service.embed_batch(
                [hierarchical_index.document_summary.summary_text]
            )
            hierarchical_index.document_summary.embedding = embeddings[0].tolist()

        # 2. 为章节摘要生成向量
        chapter_summaries = [
            chapter.summary
            for chapter in hierarchical_index.chapters
            if not chapter.embedding
        ]

        if chapter_summaries:
            chapter_embeddings = await self.embedding_service.embed_batch(chapter_summaries)
            embed_idx = 0
            for chapter in hierarchical_index.chapters:
                if not chapter.embedding:
                    chapter.embedding = chapter_embeddings[embed_idx].tolist()
                    embed_idx += 1

        # 3. 为片段内容生成向量
        chunk_contents = [
            chunk.content
            for chunk in hierarchical_index.chunks
            if not chunk.embedding
        ]

        if chunk_contents:
            chunk_embeddings = await self.embedding_service.embed_batch(chunk_contents)
            embed_idx = 0
            for chunk in hierarchical_index.chunks:
                if not chunk.embedding:
                    chunk.embedding = chunk_embeddings[embed_idx].tolist()
                    embed_idx += 1


# 全局单例
_pipeline_integration = None


def get_hierarchical_index_pipeline_integration() -> HierarchicalIndexPipelineIntegration:
    """获取分层索引流水线集成单例"""
    global _pipeline_integration
    if _pipeline_integration is None:
        _pipeline_integration = HierarchicalIndexPipelineIntegration()
    return _pipeline_integration
