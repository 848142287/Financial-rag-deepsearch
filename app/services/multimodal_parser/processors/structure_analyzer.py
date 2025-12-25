"""
文档结构分析器
分析文档的章节结构和层次关系
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class StructurePattern:
    """结构模式"""
    pattern: str
    level: int
    description: str


class StructureAnalyzer:
    """文档结构分析器"""

    def __init__(self):
        """初始化结构分析器"""
        # 章节标题模式
        self.chapter_patterns = [
            # 中文标题模式
            StructurePattern(r"^\s*第[一二三四五六七八九十百千万\d]+章[^\n]*", 1, "第X章"),
            StructurePattern(r"^\s*\d+[、\.\s]+[^\n]*", 2, "数字序号"),
            StructurePattern(r"^\s*[一二三四五六七八九十百千万]+[、\.\s]+[^\n]*", 2, "中文序号"),
            StructurePattern(r"^\s*[\(（]\d+[)）][^\n]*", 3, "括号数字序号"),
            StructurePattern(r"^\s*[A-Za-z]\.[^\n]*", 3, "字母序号"),

            # 英文标题模式
            StructurePattern(r"^\s*Chapter\s+\d+[^\n]*", 1, "Chapter"),
            StructurePattern(r"^\s*\d+\.[^\n]*", 2, "数字序号"),
            StructurePattern(r"^\s*\d+\.\d+[^\n]*", 3, "多级数字序号"),
            StructurePattern(r"^\s*[A-Z]\.[^\n]*", 3, "大写字母序号"),

            # 特殊模式
            StructurePattern(r"^\s*[§][^\n]*", 4, "特殊符号序号"),
            StructurePattern(r"^\s*▪[^\n]*", 4, "项目符号"),
        ]

        # 内容类型识别模式
        self.content_patterns = {
            "table": [r"^\s*表\s*\d+[^\n]*", r"^\s*Table\s*\d+[^\n]*"],
            "figure": [r"^\s*图\s*\d+[^\n]*", r"^\s*Figure\s*\d+[^\n]*", r"^\s*插图\s*\d+[^\n]*"],
            "formula": [r"^\s*公式\s*\d+[^\n]*", r"^\s*Equation\s*\d+[^\n]*"],
            "footnote": [r"^\[\d+\]", r"^\s*\(\d+\)", r"^\s*注：", r"^\s*Note:"],
            "header": [r"^\s*页眉[^\n]*", r"^\s*Header[^\n]*"],
            "footer": [r"^\s*页脚[^\n]*", r"^\s*Footer[^\n]*", r"^\s*第\d+页共\d+页"],
        }

        # 章节结束标识
        self.chapter_end_patterns = [
            r"^\s*本章小结",
            r"^\s*本章总结",
            r"^\s*Chapter\s+Summary",
            r"^\s*参考文献",
            r"^\s*References",
            r"^\s*附录",
            r"^\s*Appendix"
        ]

        logger.info("结构分析器初始化完成")

    async def analyze_structure(self, raw_results: Dict[str, Any]) -> List[Any]:
        """
        分析文档结构

        Args:
            raw_results: 原始解析结果

        Returns:
            章节列表
        """
        try:
            logger.info("开始分析文档结构")

            # 收集所有文本内容
            all_blocks = self._collect_all_text_blocks(raw_results)

            # 按页面和位置排序
            sorted_blocks = self._sort_blocks_by_position(all_blocks)

            # 提取章节标题
            chapters = await self._extract_chapters(sorted_blocks)

            # 构建章节层次结构
            structured_chapters = self._build_chapter_hierarchy(chapters)

            # 分配内容块到章节
            self._assign_blocks_to_chapters(sorted_blocks, structured_chapters)

            logger.info(f"文档结构分析完成: {len(structured_chapters)}个章节")
            return structured_chapters

        except Exception as e:
            logger.error(f"文档结构分析失败: {str(e)}")
            return []

    def _collect_all_text_blocks(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """收集所有文本块"""
        blocks = []

        # 从Mineru结果收集
        if raw_results.get("mineru_results"):
            mineru_result = raw_results["mineru_results"]
            blocks.extend(mineru_result.get("content_blocks", []))

        # 从Qwen-VL-OCR结果收集
        if raw_results.get("qwen_ocr_results"):
            ocr_result = raw_results["qwen_ocr_results"]
            blocks.extend(ocr_result.get("text_blocks", []))

        # 从Qwen-VL-Max结果收集
        if raw_results.get("qwen_max_results"):
            max_result = raw_results["qwen_max_results"]
            blocks.extend(max_result.get("analysis_results", []))

        return blocks

    def _sort_blocks_by_position(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按页面和位置排序内容块"""
        try:
            # 首先按页面排序
            sorted_blocks = sorted(blocks, key=lambda x: (
                x.get("page_number", 0),
                # 如果有bbox信息，按y坐标排序
                x.get("bbox", [0, 0, 0, 0])[1] if x.get("bbox") else 0
            ))

            return sorted_blocks

        except Exception as e:
            logger.warning(f"内容块排序失败: {str(e)}")
            return blocks

    async def _extract_chapters(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取章节标题"""
        chapters = []

        for i, block in enumerate(blocks):
            content = block.get("content", "").strip()
            page_number = block.get("page_number", 1)

            if not content:
                continue

            # 检查是否匹配章节模式
            chapter_info = self._match_chapter_pattern(content)
            if chapter_info:
                chapter = {
                    "id": f"chapter_{len(chapters) + 1}",
                    "title": content,
                    "level": chapter_info["level"],
                    "page_number": page_number,
                    "block_index": i,
                    "pattern": chapter_info["description"],
                    "content": content,
                    "sub_chapters": [],
                    "blocks": []
                }
                chapters.append(chapter)

        logger.info(f"提取到 {len(chapters)} 个章节标题")
        return chapters

    def _match_chapter_pattern(self, text: str) -> Optional[Dict[str, Any]]:
        """匹配章节模式"""
        for pattern_info in self.chapter_patterns:
            if re.match(pattern_info.pattern, text, re.IGNORECASE):
                return {
                    "level": pattern_info.level,
                    "pattern": pattern_info.pattern,
                    "description": pattern_info.description
                }
        return None

    def _build_chapter_hierarchy(self, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """构建章节层次结构"""
        try:
            if not chapters:
                return []

            # 按页面和位置排序章节
            chapters.sort(key=lambda x: (x["page_number"], x["block_index"]))

            # 构建层次结构
            chapter_stack = []  # 当前打开的章节栈

            for chapter in chapters:
                current_level = chapter["level"]

                # 找到合适的父章节
                while chapter_stack and chapter_stack[-1]["level"] >= current_level:
                    chapter_stack.pop()

                # 添加到父章节或根级别
                if chapter_stack:
                    parent_chapter = chapter_stack[-1]
                    parent_chapter["sub_chapters"].append(chapter["id"])
                else:
                    # 顶级章节，不做特殊处理
                    pass

                chapter_stack.append(chapter)

            return chapters

        except Exception as e:
            logger.error(f"构建章节层次结构失败: {str(e)}")
            return chapters

    def _assign_blocks_to_chapters(self, blocks: List[Dict[str, Any]], chapters: List[Dict[str, Any]]):
        """将内容块分配到章节"""
        try:
            if not chapters:
                return

            # 创建章节ID到章节对象的映射
            chapter_map = {chapter["id"]: chapter for chapter in chapters}

            # 为每个章节创建块列表
            for chapter in chapters:
                chapter["blocks"] = []

            # 遍历所有块，分配到对应章节
            current_chapter_id = None

            for block in blocks:
                content = block.get("content", "").strip()
                page_number = block.get("page_number", 1)

                # 检查是否是章节标题
                is_chapter_title = any(
                    chapter["block_index"] == i for i, chapter in enumerate(chapters)
                    if chapter.get("page_number") == page_number and chapter["content"] == content
                )

                if is_chapter_title:
                    # 找到对应的章节
                    for chapter in chapters:
                        if (chapter.get("page_number") == page_number and
                            chapter["content"] == content):
                            current_chapter_id = chapter["id"]
                            break

                # 将块添加到当前章节
                if current_chapter_id and current_chapter_id in chapter_map:
                    chapter_map[current_chapter_id]["blocks"].append(block.get("id", ""))

        except Exception as e:
            logger.error(f"分配内容块到章节失败: {str(e)}")

    def identify_content_types(self, content: str) -> str:
        """识别内容类型"""
        content_lower = content.lower()

        # 检查特殊内容类型
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.match(pattern, content, re.IGNORECASE):
                    return content_type

        # 检查是否是标题
        if self._match_chapter_pattern(content):
            return "title"

        # 检查是否是列表项
        if content.startswith(('•', '-', '*', '1.', '2.', '3.')):
            return "list_item"

        # 检查是否是段落（默认）
        if len(content) > 20:
            return "paragraph"

        # 短文本可能是其他类型
        return "text"

    def extract_document_outline(self, chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取文档大纲"""
        try:
            outline = {
                "title": "",
                "total_chapters": len(chapters),
                "max_level": 0,
                "chapter_distribution": defaultdict(int),
                "structure_tree": []
            }

            # 构建大纲树
            for chapter in chapters:
                level = chapter.get("level", 1)
                outline["max_level"] = max(outline["max_level"], level)
                outline["chapter_distribution"][level] += 1

                outline_item = {
                    "id": chapter["id"],
                    "title": chapter["title"],
                    "level": level,
                    "page_number": chapter.get("page_number", 0),
                    "sub_chapters_count": len(chapter.get("sub_chapters", []))
                }

                outline["structure_tree"].append(outline_item)

                # 如果是顶级章节，设为文档标题
                if level == 1 and not outline["title"]:
                    outline["title"] = chapter["title"]

            return dict(outline)

        except Exception as e:
            logger.error(f"提取文档大纲失败: {str(e)}")
            return {}

    def validate_structure(self, chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证文档结构"""
        try:
            validation_result = {
                "is_valid": True,
                "issues": [],
                "suggestions": []
            }

            # 检查是否有章节
            if not chapters:
                validation_result["is_valid"] = False
                validation_result["issues"].append("未检测到章节结构")
                return validation_result

            # 检查章节层次是否合理
            levels = [chapter.get("level", 1) for chapter in chapters]
            if levels != sorted(levels):
                validation_result["issues"].append("章节层次不连续")
                validation_result["suggestions"].append("建议检查章节编号格式")

            # 检查是否有重复的章节编号
            level_counts = {}
            for level in levels:
                level_counts[level] = level_counts.get(level, 0) + 1

            for level, count in level_counts.items():
                if count > 10:  # 假设同一级别章节不超过10个
                    validation_result["suggestions"].append(f"第{level}级章节数量较多({count}个)，请检查结构")

            # 检查页面顺序
            pages = [chapter.get("page_number", 0) for chapter in chapters]
            if pages != sorted(pages):
                validation_result["issues"].append("章节页面顺序不正确")
                validation_result["suggestions"].append("请检查文档页面顺序")

            return validation_result

        except Exception as e:
            logger.error(f"文档结构验证失败: {str(e)}")
            return {"is_valid": False, "issues": [str(e)], "suggestions": []}