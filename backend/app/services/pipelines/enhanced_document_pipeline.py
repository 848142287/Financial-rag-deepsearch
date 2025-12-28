"""
统一文档图谱构建流水线
整合增强的 Embedding 和知识图谱功能，替代原有的基础实现
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.services.parsers.advanced.enhanced_document_analyzer import (
    EnhancedDocumentAnalyzer,
    DocumentAnalysisResult,
    AnalyzerConfig
)
from app.services.embeddings.enhanced_semantic_embedding import (
    EnhancedSemanticEmbeddingService,
    EnhancedEmbedding,
    EmbeddingLayer
)
from app.services.content.content_semantic_enhancer import (
    ContentSemanticEnhancer,
    SemanticEnhancement
)
from app.services.knowledge.deep_graph_extractor import (
    DeepKnowledgeGraphExtractor,
    EnrichedEntity,
    EnrichedRelation
)
from app.services.unified_knowledge_graph import UnifiedKnowledgeGraphService
from app.services.document_vector_storage import document_vector_storage

logger = logging.getLogger(__name__)


@dataclass
class EnhancedDocumentPipelineResult:
    """增强文档处理流水线结果"""
    document_id: str
    filename: str
    success: bool

    # 统计信息
    chunk_count: int
    entity_count: int
    relation_count: int
    embedding_layers_count: int

    # 质量指标
    avg_quality_score: float
    avg_confidence: float

    # 处理时间
    total_processing_time: float

    # 详细结果
    enhanced_embeddings: List[EnhancedEmbedding] = field(default_factory=list)
    semantic_enhancements: List[SemanticEnhancement] = field(default_factory=list)
    enriched_entities: List[EnrichedEntity] = field(default_factory=list)
    enriched_relations: List[EnrichedRelation] = field(default_factory=list)

    # 错误和警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class EnhancedDocumentPipeline:
    """
    增强文档处理流水线

    替代原有的基础实现，整合：
    1. 文档解析（PDF提取 + Markdown分割）
    2. 元数据提取（LLM增强）
    3. 多层次 Embedding
    4. 内容语义增强
    5. 深度知识图谱构建
    6. 统一存储（向量数据库 + 图数据库）
    """

    def __init__(self, config: Optional[Dict] = None):
        """初始化流水线"""
        self.config = config or {}

        # 初始化组件
        self.document_analyzer = EnhancedDocumentAnalyzer()
        self.embedding_service = EnhancedSemanticEmbeddingService()
        self.semantic_enhancer = ContentSemanticEnhancer()
        self.kg_extractor = DeepKnowledgeGraphExtractor()
        self.kg_service = None  # 延迟初始化

        logger.info("增强文档处理流水线初始化完成")

    async def initialize(self):
        """初始化知识图谱服务"""
        self.kg_service = await document_vector_storage.initialize()
        logger.info("知识图谱服务已初始化")

    async def process_document(
        self,
        file_path: str,
        original_filename: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> EnhancedDocumentPipelineResult:
        """
        处理文档的完整流水线

        Args:
            file_path: 文件路径
            original_filename: 原始文件名
            options: 处理选项

        Returns:
            EnhancedDocumentPipelineResult
        """
        start_time = datetime.now()
        options = options or {}

        result = EnhancedDocumentPipelineResult(
            document_id="",
            filename=original_filename or Path(file_path).name,
            success=False
        )

        try:
            logger.info(f"开始处理文档: {file_path}")

            # 步骤1: 文档解析（PDF提取 + 分割 + 元数据提取）
            logger.info("步骤1: 文档解析...")
            doc_result = await self.document_analyzer.analyze_document(
                file_path=file_path,
                original_filename=original_filename
            )

            if doc_result.status == "failed":
                result.errors.append("文档解析失败")
                return result

            result.document_id = doc_result.document_id
            result.chunk_count = doc_result.chunk_count

            # 步骤2: 多层次 Embedding
            logger.info("步骤2: 生成多层次 Embedding...")
            enhanced_embeddings = await self.embedding_service.generate_enhanced_embeddings(
                doc_result
            )

            result.enhanced_embeddings = enhanced_embeddings
            result.embedding_layers_count = sum(
                len(e.embeddings) for e in enhanced_embeddings
            )

            # 步骤3: 内容语义增强
            logger.info("步骤3: 内容语义增强...")
            semantic_enhancements = await self.semantic_enhancer.enhance_content(
                doc_result
            )

            result.semantic_enhancements = semantic_enhancements

            # 步骤4: 深度知识图谱提取
            logger.info("步骤4: 提取增强的实体和关系...")

            # 提取增强实体
            enriched_entities = await self.kg_extractor.extract_enriched_entities(
                doc_result
            )

            # 提取增强关系
            enriched_relations = await self.kg_extractor.extract_enriched_relations(
                doc_result,
                enriched_entities
            )

            result.enriched_entities = enriched_entities
            result.enriched_relations = enriched_relations
            result.entity_count = len(enriched_entities)
            result.relation_count = len(enriched_relations)

            # 步骤5: 存储到向量数据库（多层次 embeddings）
            logger.info("步骤5: 存储向量到 Milvus...")
            if hasattr(document_vector_storage, 'store_document_analysis'):
                store_result = await document_vector_storage.store_document_analysis(
                    doc_result
                )
                logger.info(f"向量存储结果: {store_result}")

            # 步骤6: 存储到知识图谱
            logger.info("步骤6: 存储到知识图谱...")
            # TODO: 实现知识图谱存储

            # 计算质量指标
            if enhanced_embeddings:
                result.avg_quality_score = sum(
                    e.quality_score for e in enhanced_embeddings
                ) / len(enhanced_embeddings)
                result.avg_confidence = sum(
                    e.confidence for e in enhanced_embeddings
                ) / len(enhanced_embeddings)

            result.success = True

        except Exception as e:
            error_msg = f"文档处理失败: {e}"
            result.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)

        finally:
            result.total_processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"文档处理完成: {result.document_id}, "
                f"成功: {result.success}, "
                f"块数: {result.chunk_count}, "
                f"实体: {result.entity_count}, "
                f"关系: {result.relation_count}, "
                f"embedding层: {result.embedding_layers_count}, "
                f"质量分: {result.avg_quality_score:.2f}, "
                f"耗时: {result.total_processing_time:.2f}s"
            )

        return result

    async def batch_process_documents(
        self,
        file_paths: List[str],
        options: Optional[Dict[str, Any]] = None
    ) -> List[EnhancedDocumentPipelineResult]:
        """
        批量处理文档

        Args:
            file_paths: 文件路径列表
            options: 处理选项

        Returns:
            结果列表
        """
        results = []

        for file_path in file_paths:
            try:
                result = await self.process_document(
                    file_path=file_path,
                    options=options
                )
                results.append(result)
            except Exception as e:
                logger.error(f"处理文档失败 {file_path}: {e}")
                results.append(EnhancedDocumentPipelineResult(
                    document_id="",
                    filename=Path(file_path).name,
                    success=False,
                    errors=[str(e)]
                ))

        # 统计
        success_count = sum(1 for r in results if r.success)
        logger.info(
            f"批量处理完成: {len(results)} 个文档, "
            f"成功: {success_count}, 失败: {len(results) - success_count}"
        )

        return results

    async def get_processing_statistics(
        self,
        results: List[EnhancedDocumentPipelineResult]
    ) -> Dict[str, Any]:
        """
        获取处理统计信息

        Args:
            results: 结果列表

        Returns:
            统计信息
        """
        if not results:
            return {}

        total_chunks = sum(r.chunk_count for r in results)
        total_entities = sum(r.entity_count for r in results)
        total_relations = sum(r.relation_count for r in results)
        total_embeddings = sum(r.embedding_layers_count for r in results)

        avg_quality = sum(r.avg_quality_score for r in results if r.success) / max(
            sum(1 for r in results if r.success), 1
        )
        avg_confidence = sum(r.avg_confidence for r in results if r.success) / max(
            sum(1 for r in results if r.success), 1
        )
        avg_time = sum(r.total_processing_time for r in results) / len(results)

        return {
            "total_documents": len(results),
            "successful_documents": sum(1 for r in results if r.success),
            "failed_documents": sum(1 for r in results if not r.success),
            "total_chunks": total_chunks,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "total_embedding_layers": total_embeddings,
            "avg_quality_score": round(avg_quality, 2),
            "avg_confidence": round(avg_confidence, 2),
            "avg_processing_time": round(avg_time, 2)
        }


# 全局实例
enhanced_document_pipeline = EnhancedDocumentPipeline()


# 便捷函数
async def process_document_with_enhancements(
    file_path: str,
    original_filename: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None
) -> EnhancedDocumentPipelineResult:
    """
    使用增强功能处理文档（便捷函数）

    Args:
        file_path: 文件路径
        original_filename: 原始文件名
        options: 处理选项

    Returns:
        EnhancedDocumentPipelineResult
    """
    pipeline = enhanced_document_pipeline
    if pipeline.kg_service is None:
        await pipeline.initialize()

    return await pipeline.process_document(
        file_path=file_path,
        original_filename=original_filename,
        options=options
    )
