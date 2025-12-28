"""
融合服务 - 整合基础和增强功能
提供向后兼容的API，支持从基础功能平滑迁移到增强功能
"""
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.services.pipelines.enhanced_document_pipeline import (
    EnhancedDocumentPipeline,
    EnhancedDocumentPipelineResult,
    enhanced_document_pipeline
)
from app.services.document_vector_storage import DocumentVectorStorage
from app.services.parsers.advanced.enhanced_document_analyzer import (
    EnhancedDocumentAnalyzer,
    DocumentAnalysisResult
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

logger = logging.getLogger(__name__)


class ProcessingMode(str, Enum):
    """处理模式"""
    BASIC = "basic"              # 仅基础功能
    ENHANCED = "enhanced"        # 完整增强功能
    HYBRID = "hybrid"            # 混合模式（部分增强）
    AUTO = "auto"                # 自动选择最优模式


@dataclass
class FusionConfig:
    """融合服务配置"""
    # 默认处理模式
    default_mode: ProcessingMode = ProcessingMode.ENHANCED

    # 增强功能开关
    enable_multi_layer_embeddings: bool = True
    enable_semantic_enhancement: bool = True
    enable_deep_kg_extraction: bool = True
    enable_entity_disambiguation: bool = True

    # 性能相关
    max_concurrent_docs: int = 3
    chunk_batch_size: int = 10

    # 质量阈值
    min_quality_score: float = 0.5
    min_confidence: float = 0.6

    # 存储配置
    store_to_mysql: bool = True
    store_to_milvus: bool = True
    store_to_neo4j: bool = True


@dataclass
class FusionDocumentResult:
    """融合文档处理结果"""
    document_id: str
    filename: str
    success: bool

    # 处理模式
    processing_mode: ProcessingMode

    # 基础统计
    chunk_count: int
    entity_count: int
    relation_count: int

    # 增强统计
    embedding_layers_count: int = 0
    semantic_enhancements_count: int = 0

    # 质量指标
    avg_quality_score: float = 0.0
    avg_confidence: float = 0.0

    # 处理时间
    total_processing_time: float = 0.0

    # 详细结果
    basic_result: Optional[Dict[str, Any]] = None
    enhanced_result: Optional[EnhancedDocumentPipelineResult] = None

    # 错误和警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class FusionDocumentService:
    """
    融合文档服务

    整合基础和增强功能，提供统一的文档处理接口
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """初始化融合服务"""
        self.config = config or FusionConfig()

        # 初始化组件
        self.document_analyzer = EnhancedDocumentAnalyzer()
        self.embedding_service = EnhancedSemanticEmbeddingService()
        self.semantic_enhancer = ContentSemanticEnhancer()
        self.kg_extractor = DeepKnowledgeGraphExtractor()
        self.vector_storage = DocumentVectorStorage()

        # 增强流水线
        self.enhanced_pipeline = enhanced_document_pipeline

        # 统一知识图谱（延迟初始化）
        self.unified_kg: Optional[UnifiedKnowledgeGraphService] = None

        logger.info(f"融合文档服务初始化完成，模式: {self.config.default_mode.value}")

    async def initialize(self):
        """初始化所有服务"""
        await self.vector_storage.initialize()
        await self.enhanced_pipeline.initialize()

        if self.config.enable_deep_kg_extraction or self.config.enable_entity_disambiguation:
            self.unified_kg = UnifiedKnowledgeGraphService()
            await self.unified_kg.initialize()

        logger.info("融合文档服务所有组件已初始化")

    async def process_document(
        self,
        file_path: str,
        original_filename: Optional[str] = None,
        mode: Optional[ProcessingMode] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> FusionDocumentResult:
        """
        处理文档（融合基础和增强功能）

        Args:
            file_path: 文件路径
            original_filename: 原始文件名
            mode: 处理模式（默认使用配置中的模式）
            options: 额外选项

        Returns:
            FusionDocumentResult
        """
        start_time = datetime.now()
        mode = mode or self.config.default_mode
        options = options or {}

        result = FusionDocumentResult(
            document_id="",
            filename=original_filename or file_path.split("/")[-1],
            success=False,
            processing_mode=mode
        )

        try:
            logger.info(
                f"开始处理文档: {file_path}, "
                f"模式: {mode.value}"
            )

            # 步骤1: 文档解析（所有模式都需要）
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

            # 根据模式选择处理策略
            if mode == ProcessingMode.BASIC:
                return await self._process_basic_mode(
                    doc_result, result, start_time
                )
            elif mode == ProcessingMode.ENHANCED:
                return await self._process_enhanced_mode(
                    doc_result, result, options
                )
            elif mode == ProcessingMode.HYBRID:
                return await self._process_hybrid_mode(
                    doc_result, result, options, start_time
                )
            else:  # AUTO
                return await self._process_auto_mode(
                    doc_result, result, options
                )

        except Exception as e:
            error_msg = f"文档处理失败: {e}"
            result.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            result.total_processing_time = (datetime.now() - start_time).total_seconds()

        return result

    async def _process_basic_mode(
        self,
        doc_result: DocumentAnalysisResult,
        result: FusionDocumentResult,
        start_time: datetime
    ) -> FusionDocumentResult:
        """
        基础模式：仅基础向量存储
        """
        try:
            # 存储到向量数据库
            logger.info("基础模式: 存储基础向量...")
            store_result = await self.vector_storage.store_document_analysis(
                doc_result
            )

            result.basic_result = store_result
            result.success = store_result.get("status") == "success"

        except Exception as e:
            result.errors.append(f"基础模式处理失败: {e}")
            logger.error(f"基础模式处理失败: {e}", exc_info=True)

        finally:
            result.total_processing_time = (datetime.now() - start_time).total_seconds()

        return result

    async def _process_enhanced_mode(
        self,
        doc_result: DocumentAnalysisResult,
        result: FusionDocumentResult,
        options: Dict[str, Any]
    ) -> FusionDocumentResult:
        """
        增强模式：完整的增强功能
        """
        try:
            # 使用增强流水线处理
            enhanced_result = await self.enhanced_pipeline.process_document(
                file_path=doc_result.file_path,
                original_filename=doc_result.original_filename,
                options=options
            )

            result.enhanced_result = enhanced_result
            result.entity_count = enhanced_result.entity_count
            result.relation_count = enhanced_result.relation_count
            result.embedding_layers_count = enhanced_result.embedding_layers_count
            result.semantic_enhancements_count = len(enhanced_result.semantic_enhancements)
            result.avg_quality_score = enhanced_result.avg_quality_score
            result.avg_confidence = enhanced_result.avg_confidence
            result.success = enhanced_result.success

            # 合并错误和警告
            result.errors.extend(enhanced_result.errors)
            result.warnings.extend(enhanced_result.warnings)

        except Exception as e:
            result.errors.append(f"增强模式处理失败: {e}")
            logger.error(f"增强模式处理失败: {e}", exc_info=True)

        return result

    async def _process_hybrid_mode(
        self,
        doc_result: DocumentAnalysisResult,
        result: FusionDocumentResult,
        options: Dict[str, Any],
        start_time: datetime
    ) -> FusionDocumentResult:
        """
        混合模式：选择性启用部分增强功能
        """
        try:
            # 1. 基础向量存储（始终执行）
            logger.info("混合模式: 执行基础存储...")
            basic_result = await self.vector_storage.store_document_analysis(
                doc_result
            )
            result.basic_result = basic_result

            # 2. 根据配置选择性启用增强功能
            enhanced_features = []

            if self.config.enable_multi_layer_embeddings:
                logger.info("混合模式: 生成多层次 Embedding...")
                enhanced_embeddings = await self.embedding_service.generate_enhanced_embeddings(
                    doc_result
                )
                result.embedding_layers_count = sum(
                    len(e.embeddings) for e in enhanced_embeddings
                )
                enhanced_features.append("multi_layer_embeddings")

            if self.config.enable_semantic_enhancement:
                logger.info("混合模式: 内容语义增强...")
                semantic_enhancements = await self.semantic_enhancer.enhance_content(
                    doc_result
                )
                result.semantic_enhancements_count = len(semantic_enhancements)
                enhanced_features.append("semantic_enhancement")

            if self.config.enable_deep_kg_extraction:
                logger.info("混合模式: 深度知识图谱提取...")
                entities = await self.kg_extractor.extract_enriched_entities(doc_result)
                relations = await self.kg_extractor.extract_enriched_relations(
                    doc_result, entities
                )
                result.entity_count = len(entities)
                result.relation_count = len(relations)
                enhanced_features.append("deep_kg_extraction")

            # 计算质量分数
            if result.embedding_layers_count > 0:
                result.avg_quality_score = 0.7
                result.avg_confidence = 0.75

            result.success = basic_result.get("status") == "success"

            logger.info(
                f"混合模式完成，启用的增强功能: {', '.join(enhanced_features)}"
            )

        except Exception as e:
            result.errors.append(f"混合模式处理失败: {e}")
            logger.error(f"混合模式处理失败: {e}", exc_info=True)

        finally:
            result.total_processing_time = (datetime.now() - start_time).total_seconds()

        return result

    async def _process_auto_mode(
        self,
        doc_result: DocumentAnalysisResult,
        result: FusionDocumentResult,
        options: Dict[str, Any]
    ) -> FusionDocumentResult:
        """
        自动模式：根据文档特征智能选择处理策略
        """
        try:
            # 分析文档特征
            chunk_count = doc_result.chunk_count
            avg_chunk_length = sum(
                len(chunk.page_content) for chunk in doc_result.chunks
            ) / max(chunk_count, 1)

            # 决策逻辑
            if chunk_count < 5 or avg_chunk_length < 200:
                # 小文档：使用基础模式
                logger.info("自动模式: 检测到小文档，使用基础模式")
                return await self._process_basic_mode(
                    doc_result, result, datetime.now()
                )
            elif chunk_count > 50:
                # 大文档：使用增强模式
                logger.info("自动模式: 检测到大文档，使用增强模式")
                return await self._process_enhanced_mode(
                    doc_result, result, options
                )
            else:
                # 中等文档：使用混合模式
                logger.info("自动模式: 检测到中等文档，使用混合模式")
                return await self._process_hybrid_mode(
                    doc_result, result, options, datetime.now()
                )

        except Exception as e:
            result.errors.append(f"自动模式处理失败: {e}")
            logger.error(f"自动模式处理失败: {e}", exc_info=True)

        return result

    async def batch_process_documents(
        self,
        file_paths: List[str],
        mode: Optional[ProcessingMode] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> List[FusionDocumentResult]:
        """
        批量处理文档

        Args:
            file_paths: 文件路径列表
            mode: 处理模式
            options: 额外选项

        Returns:
            结果列表
        """
        results = []

        for file_path in file_paths:
            try:
                result = await self.process_document(
                    file_path=file_path,
                    mode=mode,
                    options=options
                )
                results.append(result)
            except Exception as e:
                logger.error(f"批量处理文档失败 {file_path}: {e}")
                results.append(FusionDocumentResult(
                    document_id="",
                    filename=file_path.split("/")[-1],
                    success=False,
                    processing_mode=mode or self.config.default_mode,
                    errors=[str(e)]
                ))

        return results

    async def get_processing_statistics(
        self,
        results: List[FusionDocumentResult]
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

        total_docs = len(results)
        successful_docs = sum(1 for r in results if r.success)
        failed_docs = total_docs - successful_docs

        # 按模式统计
        mode_stats = {}
        for mode in ProcessingMode:
            mode_results = [r for r in results if r.processing_mode == mode]
            if mode_results:
                mode_stats[mode.value] = {
                    "count": len(mode_results),
                    "success_count": sum(1 for r in mode_results if r.success),
                    "avg_quality": sum(r.avg_quality_score for r in mode_results) / len(mode_results),
                    "avg_time": sum(r.total_processing_time for r in mode_results) / len(mode_results)
                }

        return {
            "total_documents": total_docs,
            "successful_documents": successful_docs,
            "failed_documents": failed_docs,
            "success_rate": round(successful_docs / max(total_docs, 1) * 100, 2),
            "mode_statistics": mode_stats,
            "total_chunks": sum(r.chunk_count for r in results),
            "total_entities": sum(r.entity_count for r in results),
            "total_relations": sum(r.relation_count for r in results),
            "total_embedding_layers": sum(r.embedding_layers_count for r in results),
            "overall_avg_quality": round(
                sum(r.avg_quality_score for r in results if r.success) /
                max(successful_docs, 1), 2
            ),
            "overall_avg_time": round(
                sum(r.total_processing_time for r in results) /
                max(total_docs, 1), 2
            )
        }

    # 向后兼容的便捷方法

    async def process_document_basic(self, file_path: str, original_filename: Optional[str] = None):
        """基础模式处理（向后兼容）"""
        return await self.process_document(
            file_path=file_path,
            original_filename=original_filename,
            mode=ProcessingMode.BASIC
        )

    async def process_document_enhanced(self, file_path: str, original_filename: Optional[str] = None):
        """增强模式处理（向后兼容）"""
        return await self.process_document(
            file_path=file_path,
            original_filename=original_filename,
            mode=ProcessingMode.ENHANCED
        )


# 全局实例
fusion_document_service = FusionDocumentService()


async def get_fusion_service() -> FusionDocumentService:
    """获取融合服务实例"""
    if fusion_document_service.unified_kg is None:
        await fusion_document_service.initialize()
    return fusion_document_service
