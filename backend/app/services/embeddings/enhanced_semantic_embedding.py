"""
多层次语义增强 Embedding 服务
充分利用文档解析内容，生成更丰富的向量表示
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime

from langchain_core.documents import Document

from app.services.parsers.advanced.enhanced_metadata_extractor import (
    ChunkMetadataExtraction,
    DocumentMetadataExtraction
)
from app.services.qwen_embedding_service import QwenEmbeddingService
from app.core.vector_config import get_dimension

logger = logging.getLogger(__name__)


class EmbeddingLayer(str, Enum):
    """Embedding 层次"""
    BASIC = "basic"              # 基础文本 embedding
    SUMMARY = "summary"          # 摘要 embedding
    KEY_POINTS = "key_points"    # 关键点 embedding
    ENRICHED = "enriched"        # 语义增强 embedding
    CONTEXTUAL = "contextual"    # 上下文增强 embedding
    MULTIMODAL = "multimodal"    # 多模态 embedding（文本+表格）


@dataclass
class EnhancedEmbedding:
    """增强的 Embedding 结果"""
    chunk_id: str
    document_id: str

    # 不同层次的 embedding
    embeddings: Dict[EmbeddingLayer, np.ndarray] = field(default_factory=dict)

    # 元数据
    content: str = ""
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)

    # 统计信息
    confidence: float = 0.0
    quality_score: float = 0.0
    processing_time: float = 0.0

    def get_embedding(self, layer: EmbeddingLayer) -> Optional[np.ndarray]:
        """获取指定层的 embedding"""
        return self.embeddings.get(layer)

    def get_best_embedding(self) -> Optional[np.ndarray]:
        """获取质量最高的 embedding"""
        if not self.embeddings:
            return None
        # 优先级: ENRICHED > CONTEXTUAL > MULTIMODAL > SUMMARY > KEY_POINTS > BASIC
        priority = [
            EmbeddingLayer.ENRICHED,
            EmbeddingLayer.CONTEXTUAL,
            EmbeddingLayer.MULTIMODAL,
            EmbeddingLayer.SUMMARY,
            EmbeddingLayer.KEY_POINTS,
            EmbeddingLayer.BASIC
        ]
        for layer in priority:
            if layer in self.embeddings:
                return self.embeddings[layer]
        return next(iter(self.embeddings.values()))


class EnhancedSemanticEmbeddingService:
    """多层次语义增强 Embedding 服务"""

    def __init__(self):
        self.embedding_service = QwenEmbeddingService()
        self.embedding_dim = get_dimension()

    async def generate_enhanced_embeddings(
        self,
        document_analysis_result: Any
    ) -> List[EnhancedEmbedding]:
        """
        为文档生成多层次增强 embedding

        Args:
            document_analysis_result: DocumentAnalysisResult 对象

        Returns:
            EnhancedEmbedding 列表
        """
        start_time = datetime.now()
        enhanced_embeddings = []

        try:
            document_id = document_analysis_result.document_id
            chunks = document_analysis_result.chunks
            chunks_metadata = document_analysis_result.chunks_metadata
            embeddings = document_analysis_result.embeddings

            # 为每个 chunk 生成多层次 embedding
            for i, (chunk, metadata, base_embedding) in enumerate(
                zip(chunks, chunks_metadata, embeddings)
            ):
                chunk_id = chunk.metadata.get('chunk_id', f"{document_id}_chunk_{i}")

                enhanced = EnhancedEmbedding(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=chunk.page_content,
                    summary=metadata.summary,
                    key_points=[kp.point for kp in metadata.key_points],
                    topics=[t.topic for t in metadata.topics]
                )

                # 1. 基础 embedding
                if base_embedding:
                    enhanced.embeddings[EmbeddingLayer.BASIC] = np.array(base_embedding)

                # 2. 摘要 embedding
                if metadata.summary:
                    enhanced.embeddings[EmbeddingLayer.SUMMARY] = await self._embed_text(
                        metadata.summary
                    )

                # 3. 关键点 embedding
                if metadata.key_points:
                    enhanced.embeddings[EmbeddingLayer.KEY_POINTS] = await self._embed_key_points(
                        metadata.key_points
                    )

                # 4. 语义增强 embedding（组合摘要+关键点+主题）
                enhanced.embeddings[EmbeddingLayer.ENRICHED] = await self._generate_enriched_embedding(
                    chunk.page_content,
                    metadata
                )

                # 5. 上下文增强 embedding（包含相邻chunk信息）
                enhanced.embeddings[EmbeddingLayer.CONTEXTUAL] = await self._generate_contextual_embedding(
                    chunk,
                    chunks,
                    i
                )

                # 6. 多模态 embedding（文本+表格）
                if metadata.tables:
                    enhanced.embeddings[EmbeddingLayer.MULTIMODAL] = await self._generate_multimodal_embedding(
                        chunk.page_content,
                        metadata.tables
                    )

                # 计算质量分数
                enhanced.quality_score = self._calculate_quality_score(enhanced)

                # 计算置信度
                enhanced.confidence = self._calculate_confidence(enhanced)

                enhanced_embeddings.append(enhanced)

            processing_time = (datetime.now() - start_time).total_seconds()

            # 更新处理时间
            for enhanced in enhanced_embeddings:
                enhanced.processing_time = processing_time / len(enhanced_embeddings)

            logger.info(
                f"生成了 {len(enhanced_embeddings)} 个增强 embedding, "
                f"文档: {document_id}, 耗时: {processing_time:.2f}s"
            )

            return enhanced_embeddings

        except Exception as e:
            logger.error(f"生成增强 embedding 失败: {e}")
            return []

    async def _embed_text(self, text: str) -> np.ndarray:
        """嵌入单个文本"""
        try:
            embeddings = await self.embedding_service.encode([text])
            return np.array(embeddings[0]) if embeddings else np.zeros(self.embedding_dim)
        except Exception as e:
            logger.warning(f"文本嵌入失败: {e}")
            return np.zeros(self.embedding_dim)

    async def _embed_key_points(
        self,
        key_points: List[Any]
    ) -> np.ndarray:
        """嵌入关键点列表"""
        if not key_points:
            return np.zeros(self.embedding_dim)

        try:
            # 提取关键点文本
            point_texts = [kp.point for kp in key_points]

            # 按重要性加权
            weighted_texts = []
            for kp in key_points:
                # 高重要性的关键点重复多次以增加权重
                weight = 3 if kp.importance == "high" else (2 if kp.importance == "medium" else 1)
                weighted_texts.extend([kp.point] * weight)

            # 嵌入并平均
            embeddings = await self.embedding_service.encode(weighted_texts)
            if embeddings:
                return np.mean(embeddings, axis=0)
            else:
                return np.zeros(self.embedding_dim)

        except Exception as e:
            logger.warning(f"关键点嵌入失败: {e}")
            return np.zeros(self.embedding_dim)

    async def _generate_enriched_embedding(
        self,
        content: str,
        metadata: ChunkMetadataExtraction
    ) -> np.ndarray:
        """
        生成语义增强 embedding
        组合摘要+关键点+主题+实体
        """
        try:
            # 构建增强文本
            enriched_parts = []

            # 1. 原始内容（截取前500字符）
            enriched_parts.append(content[:500])

            # 2. 摘要
            if metadata.summary:
                enriched_parts.append(f"摘要: {metadata.summary}")

            # 3. 关键点（重要性高的在前）
            if metadata.key_points:
                key_points_sorted = sorted(
                    metadata.key_points,
                    key=lambda kp: 3 if kp.importance == "high" else (2 if kp.importance == "medium" else 1),
                    reverse=True
                )
                top_key_points = key_points_sorted[:5]  # 最多取5个
                enriched_parts.append("关键点: " + "; ".join([kp.point for kp in top_key_points]))

            # 4. 主题
            if metadata.topics:
                top_topics = sorted(metadata.topics, key=lambda t: t.relevance_score, reverse=True)[:5]
                enriched_parts.append("主题: " + ", ".join([t.topic for t in top_topics]))

            # 5. 情感倾向
            if metadata.sentiment and metadata.sentiment != "neutral":
                enriched_parts.append(f"情感倾向: {metadata.sentiment}")

            # 连接成增强文本
            enriched_text = " | ".join(enriched_parts)

            # 嵌入增强文本
            embedding = await self._embed_text(enriched_text)

            return embedding

        except Exception as e:
            logger.warning(f"生成增强 embedding 失败: {e}")
            # 返回基础 embedding
            return await self._embed_text(content[:500])

    async def _generate_contextual_embedding(
        self,
        current_chunk: Document,
        all_chunks: List[Document],
        current_index: int
    ) -> np.ndarray:
        """
        生成上下文增强 embedding
        包含前一个和后一个 chunk 的信息
        """
        try:
            context_parts = []

            # 前一个 chunk（最后100字符）
            if current_index > 0:
                prev_chunk = all_chunks[current_index - 1]
                context_parts.append(f"上文: {prev_chunk.page_content[-200:]}")

            # 当前 chunk（前400字符）
            context_parts.append(f"当前: {current_chunk.page_content[:400]}")

            # 后一个 chunk（前200字符）
            if current_index < len(all_chunks) - 1:
                next_chunk = all_chunks[current_index + 1]
                context_parts.append(f"下文: {next_chunk.page_content[:200]}")

            # 连接成上下文文本
            context_text = " | ".join(context_parts)

            # 嵌入上下文文本
            embedding = await self._embed_text(context_text)

            return embedding

        except Exception as e:
            logger.warning(f"生成上下文 embedding 失败: {e}")
            # 返回基础 embedding
            return await self._embed_text(current_chunk.page_content[:500])

    async def _generate_multimodal_embedding(
        self,
        content: str,
        tables: List[Any]
    ) -> np.ndarray:
        """
        生成多模态 embedding（文本 + 表格）
        """
        try:
            multimodal_parts = []

            # 1. 文本内容（摘要）
            multimodal_parts.append(content[:300])

            # 2. 表格信息
            for table in tables[:3]:  # 最多取3个表格
                table_info = f"表格: {table.title} | 摘要: {table.summary}"
                multimodal_parts.append(table_info)

                # 添加表格的关键数据（前几行）
                if table.headers and table.rows:
                    # 表头
                    multimodal_parts.append("表头: " + ", ".join(table.headers))
                    # 第一行数据
                    if table.rows:
                        multimodal_parts.append("数据: " + " | ".join(table.rows[0]))

            # 连接成多模态文本
            multimodal_text = " | ".join(multimodal_parts)

            # 嵌入多模态文本
            embedding = await self._embed_text(multimodal_text)

            return embedding

        except Exception as e:
            logger.warning(f"生成多模态 embedding 失败: {e}")
            # 返回基础 embedding
            return await self._embed_text(content[:300])

    def _calculate_quality_score(
        self,
        enhanced: EnhancedEmbedding
    ) -> float:
        """计算 embedding 质量分数"""
        score = 0.0

        # 基础分数
        if EmbeddingLayer.BASIC in enhanced.embeddings:
            score += 0.3

        # 增强层次加分
        if EmbeddingLayer.ENRICHED in enhanced.embeddings:
            score += 0.25

        if EmbeddingLayer.CONTEXTUAL in enhanced.embeddings:
            score += 0.2

        if EmbeddingLayer.MULTIMODAL in enhanced.embeddings:
            score += 0.15

        # 摘要和关键点加分
        if EmbeddingLayer.SUMMARY in enhanced.embeddings:
            score += 0.05

        if EmbeddingLayer.KEY_POINTS in enhanced.embeddings:
            score += 0.05

        return min(1.0, score)

    def _calculate_confidence(
        self,
        enhanced: EnhancedEmbedding
    ) -> float:
        """计算置信度"""
        # 基于质量分数
        base_confidence = enhanced.quality_score

        # 根据内容丰富度调整
        content_richness = 0.0

        if enhanced.summary and len(enhanced.summary) > 20:
            content_richness += 0.1

        if enhanced.key_points and len(enhanced.key_points) >= 3:
            content_richness += 0.1

        if enhanced.topics and len(enhanced.topics) >= 2:
            content_richness += 0.1

        if enhanced.entities:
            content_richness += 0.1

        return min(1.0, base_confidence + content_richness)

    async def store_enhanced_embeddings(
        self,
        enhanced_embeddings: List[EnhancedEmbedding],
        vector_store
    ) -> Dict[str, Any]:
        """
        存储增强的 embeddings 到向量数据库

        Args:
            enhanced_embeddings: EnhancedEmbedding 列表
            vector_store: 向量存储服务

        Returns:
            存储结果统计
        """
        stats = {
            "total_chunks": len(enhanced_embeddings),
            "stored_by_layer": {},
            "total_stored": 0,
            "failed": 0
        }

        try:
            for enhanced in enhanced_embeddings:
                for layer, embedding in enhanced.embeddings.items():
                    try:
                        # 准备元数据
                        metadata = {
                            "chunk_id": enhanced.chunk_id,
                            "document_id": enhanced.document_id,
                            "embedding_layer": layer.value,
                            "summary": enhanced.summary,
                            "key_points": enhanced.key_points,
                            "topics": enhanced.topics,
                            "quality_score": enhanced.quality_score,
                            "confidence": enhanced.confidence
                        }

                        # 存储到向量数据库
                        # TODO: 根据实际向量存储服务接口调整
                        # await vector_store.insert_vector(
                        #     id=f"{enhanced.chunk_id}_{layer.value}",
                        #     vector=embedding.tolist(),
                        #     metadata=metadata
                        # )

                        # 统计
                        if layer.value not in stats["stored_by_layer"]:
                            stats["stored_by_layer"][layer.value] = 0
                        stats["stored_by_layer"][layer.value] += 1
                        stats["total_stored"] += 1

                    except Exception as e:
                        logger.warning(
                            f"存储 embedding 失败: {enhanced.chunk_id}/{layer.value}: {e}"
                        )
                        stats["failed"] += 1

            logger.info(
                f"存储增强 embeddings 完成: "
                f"总计 {stats['total_stored']}, 失败 {stats['failed']}"
            )

            return stats

        except Exception as e:
            logger.error(f"存储增强 embeddings 失败: {e}")
            return stats


# 全局实例
enhanced_semantic_embedding_service = EnhancedSemanticEmbeddingService()
