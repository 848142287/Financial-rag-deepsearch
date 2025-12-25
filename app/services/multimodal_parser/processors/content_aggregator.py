"""
多模态内容聚合器
聚合来自不同引擎的解析结果，保持语义完整性
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from collections import defaultdict

from ..core.multimodal_parser import ContentBlock, ContentType, Chapter

logger = logging.getLogger(__name__)


class ContentSource(Enum):
    """内容来源"""
    MINERU = "mineru"
    QWEN_OCR = "qwen_ocr"
    QWEN_MAX = "qwen_max"
    MERGED = "merged"


@dataclass
class AggregationRule:
    """聚合规则"""
    content_type: ContentType
    preferred_sources: List[ContentSource]
    merge_strategy: str  # "merge", "prefer_best", "combine"
    quality_threshold: float = 0.7


class ContentAggregator:
    """多模态内容聚合器"""

    def __init__(self):
        """初始化内容聚合器"""
        # 聚合规则配置
        self.aggregation_rules = {
            ContentType.TEXT: AggregationRule(
                ContentType.TEXT,
                [ContentSource.MINERU, ContentSource.QWEN_OCR],
                "combine",
                0.6
            ),
            ContentType.IMAGE: AggregationRule(
                ContentType.IMAGE,
                [ContentSource.MINERU, ContentSource.QWEN_MAX],
                "prefer_best",
                0.8
            ),
            ContentType.TABLE: AggregationRule(
                ContentType.TABLE,
                [ContentSource.MINERU, ContentSource.QWEN_MAX],
                "merge",
                0.7
            ),
            ContentType.FORMULA: AggregationRule(
                ContentType.FORMULA,
                [ContentSource.QWEN_MAX, ContentSource.MINERU],
                "prefer_best",
                0.8
            ),
            ContentType.CHAPTER: AggregationRule(
                ContentType.CHAPTER,
                [ContentSource.QWEN_MAX, ContentSource.MINERU],
                "prefer_best",
                0.8
            )
        }

        # 内容质量权重
        self.source_weights = {
            ContentSource.MINERU: 0.9,
            ContentSource.QWEN_OCR: 0.7,
            ContentSource.QWEN_MAX: 0.85,
            ContentSource.MERGED: 1.0
        }

        # 相似度阈值
        self.similarity_threshold = 0.8

        logger.info("内容聚合器初始化完成")

    async def aggregate_content(
        self,
        raw_results: Dict[str, Any],
        chapters: List[Chapter]
    ) -> List[ContentBlock]:
        """
        聚合来自不同引擎的内容

        Args:
            raw_results: 原始解析结果
            chapters: 章节列表

        Returns:
            聚合后的内容块列表
        """
        try:
            logger.info("开始聚合多模态内容")

            # 从各个引擎提取内容块
            mineru_blocks = self._extract_blocks_from_mineru(raw_results.get("mineru_results", {}))
            qwen_ocr_blocks = self._extract_blocks_from_qwen_ocr(raw_results.get("qwen_ocr_results", {}))
            qwen_max_blocks = self._extract_blocks_from_qwen_max(raw_results.get("qwen_max_results", {}))

            # 按页面和类型分组
            grouped_blocks = self._group_blocks_by_page_and_type(
                mineru_blocks, qwen_ocr_blocks, qwen_max_blocks
            )

            # 聚合每个组的内容
            aggregated_blocks = []
            for page_num, type_groups in grouped_blocks.items():
                for content_type, blocks in type_groups.items():
                    aggregated_block = await self._aggregate_blocks_group(
                        blocks, content_type, page_num, chapters
                    )
                    if aggregated_block:
                        aggregated_blocks.append(aggregated_block)

            # 排序和去重
            sorted_blocks = self._sort_and_deduplicate_blocks(aggregated_blocks)

            logger.info(f"内容聚合完成: {len(sorted_blocks)}个内容块")
            return sorted_blocks

        except Exception as e:
            logger.error(f"内容聚合失败: {str(e)}")
            raise

    def _extract_blocks_from_mineru(self, mineru_results: Dict[str, Any]) -> List[ContentBlock]:
        """从Mineru结果提取内容块"""
        blocks = []
        content_blocks = mineru_results.get("content_blocks", [])

        for block_data in content_blocks:
            try:
                content_type = self._map_content_type(block_data.get("content_type", "text"))

                block = ContentBlock(
                    id=f"mineru_{block_data.get('id', '')}",
                    content_type=content_type,
                    content=block_data.get("content", ""),
                    bbox=self._normalize_bbox(block_data.get("bbox")),
                    page_number=block_data.get("page_number", 1),
                    confidence=block_data.get("confidence", 1.0),
                    metadata={
                        "source_engine": ContentSource.MINERU.value,
                        **block_data.get("metadata", {})
                    }
                )
                blocks.append(block)

            except Exception as e:
                logger.warning(f"提取Mineru块失败: {str(e)}")
                continue

        return blocks

    def _extract_blocks_from_qwen_ocr(self, qwen_ocr_results: Dict[str, Any]) -> List[ContentBlock]:
        """从Qwen-VL-OCR结果提取内容块"""
        blocks = []
        text_blocks = qwen_ocr_results.get("text_blocks", [])

        for block_data in text_blocks:
            try:
                block = ContentBlock(
                    id=f"qwen_ocr_{block_data.get('id', '')}",
                    content_type=ContentType.TEXT,
                    content=block_data.get("content", ""),
                    bbox=None,
                    page_number=block_data.get("page_number", 1),
                    confidence=block_data.get("confidence", 0.9),
                    metadata={
                        "source_engine": ContentSource.QWEN_OCR.value,
                        "image_path": block_data.get("image_path")
                    }
                )
                blocks.append(block)

            except Exception as e:
                logger.warning(f"提取Qwen-OCR块失败: {str(e)}")
                continue

        return blocks

    def _extract_blocks_from_qwen_max(self, qwen_max_results: Dict[str, Any]) -> List[ContentBlock]:
        """从Qwen-VL-Max结果提取内容块"""
        blocks = []
        analysis_results = qwen_max_results.get("analysis_results", [])

        for block_data in analysis_results:
            try:
                content_type = self._map_content_type(block_data.get("content_type", "text"))

                block = ContentBlock(
                    id=f"qwen_max_{block_data.get('id', '')}",
                    content_type=content_type,
                    content=block_data.get("content", ""),
                    bbox=None,
                    page_number=block_data.get("page_number", 1),
                    confidence=block_data.get("confidence", 0.85),
                    metadata={
                        "source_engine": ContentSource.QWEN_MAX.value,
                        "structured_data": block_data.get("structured_data", {})
                    }
                )
                blocks.append(block)

            except Exception as e:
                logger.warning(f"提取Qwen-Max块失败: {str(e)}")
                continue

        return blocks

    def _map_content_type(self, type_str: str) -> ContentType:
        """映射内容类型"""
        type_mapping = {
            "text": ContentType.TEXT,
            "image": ContentType.IMAGE,
            "table": ContentType.TABLE,
            "formula": ContentType.FORMULA,
            "figure": ContentType.IMAGE,
            "title": ContentType.HEADER,
            "key_point": ContentType.TEXT,
            "table_info": ContentType.TABLE,
            "figure_info": ContentType.IMAGE,
            "formula_info": ContentType.FORMULA,
            "summary": ContentType.TEXT
        }
        return type_mapping.get(type_str.lower(), ContentType.TEXT)

    def _normalize_bbox(self, bbox) -> Optional[Tuple[float, float, float, float]]:
        """标准化边界框"""
        if not bbox or len(bbox) != 4:
            return None
        try:
            return tuple(float(coord) for coord in bbox)
        except (ValueError, TypeError):
            return None

    def _group_blocks_by_page_and_type(
        self,
        mineru_blocks: List[ContentBlock],
        qwen_ocr_blocks: List[ContentBlock],
        qwen_max_blocks: List[ContentBlock]
    ) -> Dict[int, Dict[ContentType, List[ContentBlock]]]:
        """按页面和类型分组内容块"""
        grouped = defaultdict(lambda: defaultdict(list))

        # 添加Mineru块
        for block in mineru_blocks:
            page_num = block.page_number
            grouped[page_num][block.content_type].append(block)

        # 添加Qwen-OCR块
        for block in qwen_ocr_blocks:
            page_num = block.page_number
            grouped[page_num][block.content_type].append(block)

        # 添加Qwen-Max块
        for block in qwen_max_blocks:
            page_num = block.page_number
            grouped[page_num][block.content_type].append(block)

        return dict(grouped)

    async def _aggregate_blocks_group(
        self,
        blocks: List[ContentBlock],
        content_type: ContentType,
        page_number: int,
        chapters: List[Chapter]
    ) -> Optional[ContentBlock]:
        """聚合一组内容块"""
        try:
            if not blocks:
                return None

            # 获取聚合规则
            rule = self.aggregation_rules.get(content_type)
            if not rule:
                # 默认选择最佳质量的块
                return max(blocks, key=lambda x: x.confidence)

            # 根据策略聚合
            if rule.merge_strategy == "prefer_best":
                return await self._prefer_best_aggregation(blocks, rule)
            elif rule.merge_strategy == "merge":
                return await self._merge_aggregation(blocks, rule, page_number, chapters)
            elif rule.merge_strategy == "combine":
                return await self._combine_aggregation(blocks, rule, page_number, chapters)
            else:
                return max(blocks, key=lambda x: x.confidence)

        except Exception as e:
            logger.warning(f"内容块聚合失败: {str(e)}")
            return max(blocks, key=lambda x: x.confidence) if blocks else None

    async def _prefer_best_aggregation(
        self,
        blocks: List[ContentBlock],
        rule: AggregationRule
    ) -> ContentBlock:
        """选择最佳质量的内容块"""
        # 按优先级和质量排序
        def block_priority(block):
            source_priority = len(rule.preferred_sources)
            try:
                source = ContentSource(block.metadata.get("source_engine"))
                if source in rule.preferred_sources:
                    source_priority = rule.preferred_sources.index(source)
            except ValueError:
                pass

            return (source_priority, block.confidence)

        best_block = max(blocks, key=block_priority)

        # 标记为聚合结果
        aggregated_block = ContentBlock(
            id=f"aggregated_{best_block.id}",
            content_type=best_block.content_type,
            content=best_block.content,
            bbox=best_block.bbox,
            page_number=best_block.page_number,
            confidence=best_block.confidence,
            metadata={
                **best_block.metadata,
                "aggregation_strategy": "prefer_best",
                "source_engines": [block.metadata.get("source_engine") for block in blocks],
                "total_sources": len(blocks)
            }
        )

        return aggregated_block

    async def _merge_aggregation(
        self,
        blocks: List[ContentBlock],
        rule: AggregationRule,
        page_number: int,
        chapters: List[Chapter]
    ) -> ContentBlock:
        """合并多个内容块"""
        # 按相似性分组
        similarity_groups = self._group_by_similarity(blocks)

        merged_content = []
        all_metadata = {}
        total_confidence = 0
        all_sources = set()

        for group in similarity_groups:
            if len(group) == 1:
                # 单个块，直接添加
                block = group[0]
                merged_content.append(block.content)
            else:
                # 多个相似块，合并内容
                merged_text = await self._merge_similar_blocks(group)
                merged_content.append(merged_text)

            # 收集元数据和置信度
            for block in group:
                all_metadata.update(block.metadata)
                total_confidence += block.confidence
                all_sources.add(block.metadata.get("source_engine", "unknown"))

        # 计算平均置信度
        avg_confidence = total_confidence / len(blocks) if blocks else 0

        # 合并边界框
        merged_bbox = self._merge_bboxes([block.bbox for block in blocks if block.bbox])

        # 查找所属章节
        chapter_id = await self._find_chapter_for_page(page_number, chapters)

        merged_block = ContentBlock(
            id=f"merged_{page_number}_{len(blocks)}",
            content_type=rule.content_type,
            content="\n\n".join(merged_content),
            bbox=merged_bbox,
            page_number=page_number,
            chapter_id=chapter_id,
            confidence=min(avg_confidence * 1.1, 1.0),  # 略微提升合并后的置信度
            metadata={
                **all_metadata,
                "aggregation_strategy": "merge",
                "source_engines": list(all_sources),
                "total_sources": len(all_sources),
                "merged_from": len(blocks)
            }
        )

        return merged_block

    async def _combine_aggregation(
        self,
        blocks: List[ContentBlock],
        rule: AggregationRule,
        page_number: int,
        chapters: List[Chapter]
    ) -> ContentBlock:
        """组合多个内容块"""
        # 按来源分类
        source_groups = defaultdict(list)
        for block in blocks:
            source = block.metadata.get("source_engine", "unknown")
            source_groups[source].append(block)

        combined_content = []
        all_metadata = {}
        weighted_confidence = 0
        total_weight = 0
        all_sources = set()

        # 按优先级处理各来源的内容
        for source in rule.preferred_sources:
            if source.value in source_groups:
                source_blocks = source_groups[source.value]
                for block in source_blocks:
                    combined_content.append(f"[{source.value}]: {block.content}")

                    # 加权计算置信度
                    weight = self.source_weights.get(ContentSource(source.value), 0.5)
                    weighted_confidence += block.confidence * weight
                    total_weight += weight

                    all_metadata.update(block.metadata)
                    all_sources.add(source.value)

        # 计算加权平均置信度
        avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0

        # 合并边界框
        merged_bbox = self._merge_bboxes([block.bbox for block in blocks if block.bbox])

        # 查找所属章节
        chapter_id = await self._find_chapter_for_page(page_number, chapters)

        combined_block = ContentBlock(
            id=f"combined_{page_number}_{len(blocks)}",
            content_type=rule.content_type,
            content="\n\n".join(combined_content),
            bbox=merged_bbox,
            page_number=page_number,
            chapter_id=chapter_id,
            confidence=avg_confidence,
            metadata={
                **all_metadata,
                "aggregation_strategy": "combine",
                "source_engines": list(all_sources),
                "total_sources": len(all_sources)
            }
        )

        return combined_block

    def _group_by_similarity(self, blocks: List[ContentBlock]) -> List[List[ContentBlock]]:
        """按相似性分组内容块"""
        if len(blocks) <= 1:
            return [blocks]

        groups = []
        used_blocks = set()

        for i, block1 in enumerate(blocks):
            if i in used_blocks:
                continue

            current_group = [block1]
            used_blocks.add(i)

            for j, block2 in enumerate(blocks[i+1:], i+1):
                if j in used_blocks:
                    continue

                similarity = self._calculate_similarity(block1, block2)
                if similarity >= self.similarity_threshold:
                    current_group.append(block2)
                    used_blocks.add(j)

            groups.append(current_group)

        return groups

    def _calculate_similarity(self, block1: ContentBlock, block2: ContentBlock) -> float:
        """计算两个内容块的相似度"""
        try:
            # 内容相似度
            content1 = block1.content.lower().strip()
            content2 = block2.content.lower().strip()

            if content1 == content2:
                return 1.0

            # 简单的文本相似度计算
            words1 = set(content1.split())
            words2 = set(content2.split())

            if not words1 or not words2:
                return 0.0

            intersection = len(words1 & words2)
            union = len(words1 | words2)

            content_similarity = intersection / union if union > 0 else 0.0

            # 位置相似度
            position_similarity = 0.0
            if block1.bbox and block2.bbox:
                # 计算边界框的重叠度
                bbox1 = block1.bbox
                bbox2 = block2.bbox

                # 简化的重叠度计算
                x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
                y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))

                if x_overlap > 0 and y_overlap > 0:
                    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                    overlap_area = x_overlap * y_overlap
                    position_similarity = overlap_area / max(area1, area2)

            # 综合相似度
            total_similarity = 0.7 * content_similarity + 0.3 * position_similarity

            return total_similarity

        except Exception as e:
            logger.warning(f"相似度计算失败: {str(e)}")
            return 0.0

    async def _merge_similar_blocks(self, blocks: List[ContentBlock]) -> str:
        """合并相似的内容块"""
        try:
            # 选择最长的内容作为基础
            base_block = max(blocks, key=lambda x: len(x.content))
            base_content = base_block.content

            # 检查其他块是否有额外信息
            additional_info = []
            for block in blocks:
                if block.id != base_block.id:
                    # 查找基础内容中没有的部分
                    if block.content not in base_content:
                        additional_info.append(block.content)

            # 合并内容
            if additional_info:
                return f"{base_content}\n\n补充信息:\n" + "\n".join(additional_info)
            else:
                return base_content

        except Exception as e:
            logger.warning(f"相似块合并失败: {str(e)}")
            return blocks[0].content if blocks else ""

    def _merge_bboxes(self, bboxes: List[Optional[Tuple[float, float, float, float]]]) -> Optional[Tuple[float, float, float, float]]:
        """合并边界框"""
        valid_bboxes = [bbox for bbox in bboxes if bbox and len(bbox) == 4]

        if not valid_bboxes:
            return None
        elif len(valid_bboxes) == 1:
            return valid_bboxes[0]

        # 计算包含所有边界框的最小边界框
        min_x = min(bbox[0] for bbox in valid_bboxes)
        min_y = min(bbox[1] for bbox in valid_bboxes)
        max_x = max(bbox[2] for bbox in valid_bboxes)
        max_y = max(bbox[3] for bbox in valid_bboxes)

        return (min_x, min_y, max_x, max_y)

    async def _find_chapter_for_page(self, page_number: int, chapters: List[Chapter]) -> Optional[str]:
        """为页面查找对应的章节"""
        for chapter in chapters:
            if chapter.start_page <= page_number <= chapter.end_page:
                return chapter.id

        # 如果没有找到精确匹配，找最近的章节
        closest_chapter = None
        min_distance = float('inf')

        for chapter in chapters:
            distance = abs(chapter.start_page - page_number)
            if distance < min_distance:
                min_distance = distance
                closest_chapter = chapter

        return closest_chapter.id if closest_chapter else None

    def _sort_and_deduplicate_blocks(self, blocks: List[ContentBlock]) -> List[ContentBlock]:
        """排序和去重内容块"""
        try:
            # 按页面、类型和位置排序
            sorted_blocks = sorted(blocks, key=lambda x: (
                x.page_number,
                x.content_type.value,
                x.bbox[1] if x.bbox else 0,  # y坐标
                x.bbox[0] if x.bbox else 0   # x坐标
            ))

            # 去重（基于内容相似性）
            deduplicated_blocks = []
            seen_contents = set()

            for block in sorted_blocks:
                # 生成内容指纹用于去重
                content_fingerprint = self._generate_content_fingerprint(block.content)

                if content_fingerprint not in seen_contents:
                    deduplicated_blocks.append(block)
                    seen_contents.add(content_fingerprint)

            return deduplicated_blocks

        except Exception as e:
            logger.warning(f"排序去重失败: {str(e)}")
            return blocks

    def _generate_content_fingerprint(self, content: str) -> str:
        """生成内容指纹用于去重"""
        # 简化实现：去除空白字符并截取前100个字符
        normalized = re.sub(r'\s+', '', content[:100].lower())
        return normalized

    def get_aggregation_statistics(self, blocks: List[ContentBlock]) -> Dict[str, Any]:
        """获取聚合统计信息"""
        try:
            stats = {
                "total_blocks": len(blocks),
                "content_type_distribution": {},
                "source_engine_distribution": {},
                "confidence_distribution": {
                    "min": 0.0,
                    "max": 0.0,
                    "avg": 0.0
                },
                "page_distribution": {},
                "aggregation_strategies": {}
            }

            # 统计内容类型分布
            for block in blocks:
                content_type = block.content_type.value
                stats["content_type_distribution"][content_type] = stats["content_type_distribution"].get(content_type, 0) + 1

                # 统计来源引擎分布
                source_engine = block.metadata.get("source_engine", "unknown")
                stats["source_engine_distribution"][source_engine] = stats["source_engine_distribution"].get(source_engine, 0) + 1

                # 统计置信度
                confidence = block.confidence
                stats["confidence_distribution"]["min"] = min(stats["confidence_distribution"]["min"], confidence)
                stats["confidence_distribution"]["max"] = max(stats["confidence_distribution"]["max"], confidence)

                # 统计页面分布
                page = block.page_number
                stats["page_distribution"][page] = stats["page_distribution"].get(page, 0) + 1

                # 统计聚合策略
                strategy = block.metadata.get("aggregation_strategy", "single")
                stats["aggregation_strategies"][strategy] = stats["aggregation_strategies"].get(strategy, 0) + 1

            # 计算平均置信度
            if blocks:
                total_confidence = sum(block.confidence for block in blocks)
                stats["confidence_distribution"]["avg"] = total_confidence / len(blocks)

            return stats

        except Exception as e:
            logger.error(f"聚合统计失败: {str(e)}")
            return {}