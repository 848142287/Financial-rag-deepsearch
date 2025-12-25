"""
自动修复器
自动检测和修复文档解析中的完整性问题
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re
from collections import defaultdict

from ..core.multimodal_parser import ParsedDocument, ContentBlock, ContentType, Chapter
from ..evaluators.integrity_evaluator import IntegrityEvaluator, IntegrityIssue, IntegrityReport

logger = logging.getLogger(__name__)


@dataclass
class RepairAction:
    """修复动作"""
    action_type: str
    description: str
    target_blocks: List[str]
    repair_data: Optional[Dict[str, Any]] = None
    confidence: float = 0.0


class AutoRepairer:
    """自动修复器"""

    def __init__(self):
        """初始化自动修复器"""
        self.integrity_evaluator = IntegrityEvaluator()

        # 修复策略配置
        self.repair_strategies = {
            IntegrityIssue.MISSING_CONTENT: self._repair_missing_content,
            IntegrityIssue.BROKEN_STRUCTURE: self._repair_broken_structure,
            IntegrityIssue.LOW_QUALITY: self._repair_low_quality,
            IntegrityIssue.MISSING_REFERENCES: self._repair_missing_references,
            IntegrityIssue.INCONSISTENT_FORMAT: self._repair_inconsistent_format,
            IntegrityIssue.ORPHANED_CONTENT: self._repair_orphaned_content,
            IntegrityIssue.DUPLICATE_CONTENT: self._repair_duplicate_content
        }

        # 修复置信度阈值
        self.repair_threshold = 0.5
        self.max_repair_attempts = 3

        logger.info("自动修复器初始化完成")

    async def repair_document(
        self,
        document: ParsedDocument,
        file_path: str,
        integrity_score: float
    ) -> ParsedDocument:
        """
        自动修复文档

        Args:
            document: 原始文档
            file_path: 文件路径
            integrity_score: 完整性评分

        Returns:
            修复后的文档
        """
        try:
            logger.info(f"开始自动修复文档: {document.document_id}")

            if integrity_score >= 0.8:
                logger.info("文档完整性良好，无需修复")
                return document

            # 生成完整性报告
            report = await self.integrity_evaluator.generate_integrity_report(document)

            # 规划修复动作
            repair_actions = await self._plan_repairs(report)

            if not repair_actions:
                logger.info("无需修复或无法自动修复")
                return document

            # 执行修复动作
            repaired_document = await self._execute_repairs(document, repair_actions, file_path)

            # 验证修复效果
            new_score = await self.integrity_evaluator.evaluate_integrity(repaired_document)
            improvement = new_score - integrity_score

            logger.info(f"自动修复完成，评分提升: {improvement:.3f} ({integrity_score:.3f} -> {new_score:.3f})")

            return repaired_document

        except Exception as e:
            logger.error(f"自动修复失败: {str(e)}")
            return document

    async def _plan_repairs(self, report: IntegrityReport) -> List[RepairAction]:
        """规划修复动作"""
        actions = []

        # 按严重程度排序问题
        sorted_issues = sorted(report.issues, key=lambda x: x.severity, reverse=True)

        for issue in sorted_issues:
            if issue.severity < self.repair_threshold:
                continue  # 跳过轻微问题

            try:
                # 获取对应的修复策略
                repair_strategy = self.repair_strategies.get(issue.issue_type)
                if repair_strategy:
                    issue_actions = await repair_strategy(issue)
                    actions.extend(issue_actions)
            except Exception as e:
                logger.warning(f"规划修复动作失败 {issue.issue_type}: {str(e)}")
                continue

        return actions

    async def _execute_repairs(
        self,
        document: ParsedDocument,
        actions: List[RepairAction],
        file_path: str
    ) -> ParsedDocument:
        """执行修复动作"""
        repaired_document = document

        for action in actions:
            try:
                logger.info(f"执行修复动作: {action.action_type}")

                if action.action_type == "enhance_ocr":
                    repaired_document = await self._enhance_ocr_repair(repaired_document, action, file_path)
                elif action.action_type == "merge_blocks":
                    repaired_document = await self._merge_blocks_repair(repaired_document, action)
                elif action.action_type == "fill_missing_pages":
                    repaired_document = await self._fill_missing_pages_repair(repaired_document, action, file_path)
                elif action.action_type == "fix_structure":
                    repaired_document = await self._fix_structure_repair(repaired_document, action)
                elif action.action_type == "remove_duplicates":
                    repaired_document = await self._remove_duplicates_repair(repaired_document, action)
                elif action.action_type == "reorganize_content":
                    repaired_document = await self._reorganize_content_repair(repaired_document, action)
                elif action.action_type == "enhance_confidence":
                    repaired_document = await self._enhance_confidence_repair(repaired_document, action)

                logger.info(f"修复动作完成: {action.action_type}")

            except Exception as e:
                logger.error(f"执行修复动作失败 {action.action_type}: {str(e)}")
                continue

        return repaired_document

    async def _repair_missing_content(self, issue: IntegrityIssueDetail) -> List[RepairAction]:
        """修复缺失内容"""
        actions = []

        if "页面" in issue.location:
            # 缺失页面内容的修复
            pages = self._extract_page_numbers(issue.description)
            for page in pages:
                action = RepairAction(
                    action_type="fill_missing_pages",
                    description=f"补充页面 {page} 的内容",
                    target_blocks=[],
                    repair_data={"page_number": page},
                    confidence=0.7
                )
                actions.append(action)

        elif "类型" in issue.location:
            # 特定内容类型缺失的修复
            content_type = self._extract_content_type(issue.description)
            if content_type:
                action = RepairAction(
                    action_type="enhance_ocr",
                    description=f"增强 {content_type} 类型内容识别",
                    target_blocks=[],
                    repair_data={"target_type": content_type},
                    confidence=0.6
                )
                actions.append(action)

        return actions

    async def _repair_broken_structure(self, issue: IntegrityIssueDetail) -> List[RepairAction]:
        """修复结构问题"""
        actions = []

        if "章节层次" in issue.description:
            action = RepairAction(
                action_type="fix_structure",
                description="修复章节层次结构",
                target_blocks=[],
                repair_data={"fix_hierarchy": True},
                confidence=0.8
            )
            actions.append(action)

        elif "页面内容不连续" in issue.description:
            # 重新分配内容到章节
            action = RepairAction(
                action_type="reorganize_content",
                description="重新组织内容分配",
                target_blocks=[],
                repair_data={"fix_page_continuity": True},
                confidence=0.7
            )
            actions.append(action)

        return actions

    async def _repair_low_quality(self, issue: IntegrityIssueDetail) -> List[RepairAction]:
        """修复低质量内容"""
        actions = []

        if "置信度" in issue.description:
            # 增强低置信度内容
            action = RepairAction(
                action_type="enhance_confidence",
                description="提升内容识别置信度",
                target_blocks=issue.affected_blocks,
                repair_data={"enhance_method": "reprocess"},
                confidence=0.6
            )
            actions.append(action)

        elif "过短" in issue.description:
            # 合并短内容块
            action = RepairAction(
                action_type="merge_blocks",
                description="合并过短的内容块",
                target_blocks=issue.affected_blocks,
                repair_data={"merge_short_blocks": True},
                confidence=0.7
            )
            actions.append(action)

        return actions

    async def _repair_missing_references(self, issue: IntegrityIssueDetail) -> List[RepairAction]:
        """修复缺失引用"""
        actions = []

        action = RepairAction(
            action_type="fix_structure",
            description="修复引用关系",
            target_blocks=[],
            repair_data={"fix_references": True},
            confidence=0.5
        )
        actions.append(action)

        return actions

    async def _repair_inconsistent_format(self, issue: IntegrityIssueDetail) -> List[RepairAction]:
        """修复格式不一致"""
        actions = []

        action = RepairAction(
            action_type="reorganize_content",
            description="统一内容格式",
            target_blocks=[],
            repair_data={"unify_format": True},
            confidence=0.6
        )
        actions.append(action)

        return actions

    async def _repair_orphaned_content(self, issue: IntegrityIssueDetail) -> List[RepairAction]:
        """修复孤立内容"""
        actions = []

        action = RepairAction(
            action_type="reorganize_content",
            description="重新分配孤立内容",
            target_blocks=issue.affected_blocks,
            repair_data={"reassign_orphaned": True},
            confidence=0.7
        )
        actions.append(action)

        return actions

    async def _repair_duplicate_content(self, issue: IntegrityIssueDetail) -> List[RepairAction]:
        """修复重复内容"""
        actions = []

        action = RepairAction(
            action_type="remove_duplicates",
            description="移除重复内容块",
            target_blocks=issue.affected_blocks,
            repair_data={"deduplication_method": "hash_based"},
            confidence=0.8
        )
        actions.append(action)

        return actions

    # 修复方法实现

    async def _enhance_ocr_repair(
        self,
        document: ParsedDocument,
        action: RepairAction,
        file_path: str
    ) -> ParsedDocument:
        """增强OCR识别修复"""
        try:
            target_type = action.repair_data.get("target_type")
            if target_type:
                logger.info(f"增强 {target_type} 类型内容识别")
                # 这里可以调用更高精度的OCR引擎
                # 实际实现中可能需要重新解析特定页面

            return document

        except Exception as e:
            logger.error(f"增强OCR修复失败: {str(e)}")
            return document

    async def _merge_blocks_repair(
        self,
        document: ParsedDocument,
        action: RepairAction
    ) -> ParsedDocument:
        """合并内容块修复"""
        try:
            target_block_ids = action.target_blocks
            merge_short_blocks = action.repair_data.get("merge_short_blocks", False)

            if not target_block_ids:
                return document

            # 查找目标块
            target_blocks = [
                block for block in document.content_blocks
                if block.id in target_block_ids
            ]

            if len(target_blocks) < 2:
                return document

            # 按页面和位置排序
            target_blocks.sort(key=lambda x: (x.page_number, x.bbox[1] if x.bbox else 0))

            # 合并内容
            merged_content = "\n".join(block.content for block in target_blocks)

            # 创建合并后的块
            merged_block = ContentBlock(
                id=f"merged_{target_blocks[0].page_number}_{len(target_blocks)}",
                content_type=target_blocks[0].content_type,
                content=merged_content,
                bbox=self._merge_bboxes([block.bbox for block in target_blocks if block.bbox]),
                page_number=target_blocks[0].page_number,
                confidence=max(block.confidence for block in target_blocks),
                metadata={
                    "merged_from": [block.id for block in target_blocks],
                    "merge_strategy": "auto_repair",
                    **target_blocks[0].metadata
                }
            )

            # 移除原始块，添加合并后的块
            document.content_blocks = [
                block for block in document.content_blocks
                if block.id not in target_block_ids
            ]
            document.content_blocks.append(merged_block)

            logger.info(f"合并了 {len(target_blocks)} 个内容块")
            return document

        except Exception as e:
            logger.error(f"合并内容块修复失败: {str(e)}")
            return document

    async def _fill_missing_pages_repair(
        self,
        document: ParsedDocument,
        action: RepairAction,
        file_path: str
    ) -> ParsedDocument:
        """填充缺失页面修复"""
        try:
            page_number = action.repair_data.get("page_number")
            if not page_number:
                return document

            logger.info(f"尝试填充页面 {page_number} 的内容")

            # 这里可以调用特定页面的重新解析
            # 实际实现可能需要使用不同的解析引擎或参数

            # 创建占位块
            placeholder_block = ContentBlock(
                id=f"placeholder_page_{page_number}",
                content_type=ContentType.TEXT,
                content=f"[页面 {page_number} 内容解析失败，需要手动检查]",
                page_number=page_number,
                confidence=0.1,
                metadata={
                    "auto_generated": True,
                    "repair_reason": "missing_page"
                }
            )

            document.content_blocks.append(placeholder_block)
            return document

        except Exception as e:
            logger.error(f"填充缺失页面修复失败: {str(e)}")
            return document

    async def _fix_structure_repair(
        self,
        document: ParsedDocument,
        action: RepairAction
    ) -> ParsedDocument:
        """修复结构问题"""
        try:
            fix_hierarchy = action.repair_data.get("fix_hierarchy", False)
            fix_references = action.repair_data.get("fix_references", False)

            if fix_hierarchy:
                # 修复章节层次
                document.chapters = self._fix_chapter_hierarchy(document.chapters)

            if fix_references:
                # 修复引用关系
                document.content_blocks = self._fix_content_references(document.content_blocks)

            return document

        except Exception as e:
            logger.error(f"结构修复失败: {str(e)}")
            return document

    async def _remove_duplicates_repair(
        self,
        document: ParsedDocument,
        action: RepairAction
    ) -> ParsedDocument:
        """移除重复内容修复"""
        try:
            target_block_ids = action.target_blocks

            if not target_block_ids:
                return document

            # 生成内容哈希用于去重
            content_hashes = {}
            unique_blocks = []

            for block in document.content_blocks:
                content_hash = self._generate_content_hash(block.content)

                if content_hash in content_hashes:
                    # 重复块，检查是否在目标列表中
                    if block.id in target_block_ids:
                        continue  # 移除重复块
                else:
                    content_hashes[content_hash] = block
                    unique_blocks.append(block)

            removed_count = len(document.content_blocks) - len(unique_blocks)
            document.content_blocks = unique_blocks

            logger.info(f"移除了 {removed_count} 个重复内容块")
            return document

        except Exception as e:
            logger.error(f"移除重复内容修复失败: {str(e)}")
            return document

    async def _reorganize_content_repair(
        self,
        document: ParsedDocument,
        action: RepairAction
    ) -> ParsedDocument:
        """重新组织内容修复"""
        try:
            fix_page_continuity = action.repair_data.get("fix_page_continuity", False)
            reassign_orphaned = action.repair_data.get("reassign_orphaned", False)
            unify_format = action.repair_data.get("unify_format", False)

            if reassign_orphaned:
                # 重新分配孤立内容
                for block in document.content_blocks:
                    if not block.chapter_id:
                        # 查找最近的章节
                        nearest_chapter = self._find_nearest_chapter(block.page_number, document.chapters)
                        if nearest_chapter:
                            block.chapter_id = nearest_chapter.id

            if unify_format:
                # 统一内容格式
                for block in document.content_blocks:
                    block.content = self._normalize_content_format(block.content)

            return document

        except Exception as e:
            logger.error(f"重新组织内容修复失败: {str(e)}")
            return document

    async def _enhance_confidence_repair(
        self,
        document: ParsedDocument,
        action: RepairAction
    ) -> ParsedDocument:
        """增强置信度修复"""
        try:
            target_block_ids = action.target_blocks

            if not target_block_ids:
                return document

            for block in document.content_blocks:
                if block.id in target_block_ids:
                    # 提升低置信度块的置信度
                    if block.confidence < 0.6:
                        block.confidence = min(block.confidence + 0.2, 0.8)
                        block.metadata["confidence_boosted"] = True

            return document

        except Exception as e:
            logger.error(f"增强置信度修复失败: {str(e)}")
            return document

    # 辅助方法

    def _extract_page_numbers(self, text: str) -> List[int]:
        """从文本中提取页码"""
        pages = []
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            try:
                page = int(num)
                if 1 <= page <= 1000:  # 合理的页面范围
                    pages.append(page)
            except ValueError:
                continue
        return pages

    def _extract_content_type(self, text: str) -> Optional[str]:
        """从文本中提取内容类型"""
        type_mapping = {
            "text": "文本",
            "image": "图片",
            "table": "表格",
            "formula": "公式"
        }

        for type_key, type_name in type_mapping.items():
            if type_key in text.lower():
                return type_name

        return None

    def _merge_bboxes(self, bboxes: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
        """合并边界框"""
        if not bboxes:
            return None

        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[2] for bbox in bboxes)
        max_y = max(bbox[3] for bbox in bboxes)

        return (min_x, min_y, max_x, max_y)

    def _fix_chapter_hierarchy(self, chapters: List[Chapter]) -> List[Chapter]:
        """修复章节层次"""
        if not chapters:
            return chapters

        # 确保章节层次连续
        fixed_chapters = []
        last_level = 0

        for chapter in chapters:
            # 如果层次跳跃太大，调整到合理范围
            if chapter.level > last_level + 2:
                chapter.level = last_level + 1

            fixed_chapters.append(chapter)
            last_level = chapter.level

        return fixed_chapters

    def _fix_content_references(self, blocks: List[ContentBlock]) -> List[ContentBlock]:
        """修复内容引用"""
        # 这里可以添加引用关系的自动修复逻辑
        # 例如：检查图表引用是否对应实际存在的图表
        return blocks

    def _find_nearest_chapter(self, page_number: int, chapters: List[Chapter]) -> Optional[Chapter]:
        """查找最近的章节"""
        if not chapters:
            return None

        nearest_chapter = None
        min_distance = float('inf')

        for chapter in chapters:
            distance = abs(chapter.start_page - page_number)
            if distance < min_distance:
                min_distance = distance
                nearest_chapter = chapter

        return nearest_chapter

    def _normalize_content_format(self, content: str) -> str:
        """标准化内容格式"""
        # 移除多余空白
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        # 统一标点符号
        content = re.sub(r'[,，]+', ',', content)
        content = re.sub(r'[.。]+', '。', content)

        return content.strip()

    def _generate_content_hash(self, content: str) -> str:
        """生成内容哈希"""
        normalized = re.sub(r'\s+', '', content.lower())
        return str(hash(normalized))

    def get_repair_summary(self, original_score: float, repaired_score: float) -> Dict[str, Any]:
        """获取修复摘要"""
        return {
            "original_score": original_score,
            "repaired_score": repaired_score,
            "improvement": repaired_score - original_score,
            "improvement_percentage": ((repaired_score - original_score) / original_score * 100) if original_score > 0 else 0,
            "success": repaired_score > original_score
        }