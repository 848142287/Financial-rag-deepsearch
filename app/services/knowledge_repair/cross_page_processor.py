"""
跨页内容处理器
处理跨页的表格、图片、公式等内容
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CrossPageReference:
    """跨页引用"""
    source_page: int
    target_page: int
    content_id: str
    content_type: str
    confidence: float
    relationship: str


@dataclass
class MergedContent:
    """合并后的内容"""
    merged_id: str
    source_contents: List[Dict[str, Any]]
    merged_content: str
    metadata: Dict[str, Any]


class CrossPageProcessor:
    """跨页内容处理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.continuation_patterns = [
            r'（续表）',
            r'（续上表）',
            r'（continued）',
            r'table\s+\d+\.?\s*continued',
            r'续上页',
            r'见下页',
            r'continued\s+on\s+next\s+page',
        ]

        self.content_signatures = {
            'table': self._generate_table_signature,
            'image': self._generate_image_signature,
            'formula': self._generate_formula_signature
        }

    async def process_cross_page_content(
        self,
        pages_content: List[Dict[str, Any]]
    ) -> Tuple[List[CrossPageReference], List[MergedContent]]:
        """处理跨页内容"""
        try:
            # 检测跨页引用
            references = self._detect_cross_page_references(pages_content)

            # 合并跨页内容
            merged_contents = self._merge_cross_page_content(pages_content, references)

            # 修复跨页表格
            merged_tables = await self._merge_cross_page_tables(pages_content, references)

            # 合并所有内容
            all_merged = merged_contents + merged_tables

            return references, all_merged

        except Exception as e:
            logger.error(f"跨页内容处理失败: {str(e)}")
            return [], []

    def _detect_cross_page_references(self, pages_content: List[Dict[str, Any]]) -> List[CrossPageReference]:
        """检测跨页引用"""
        references = []

        for page_idx, page in enumerate(pages_content):
            page_text = page.get('text', '')
            page_number = page.get('page_number', page_idx + 1)

            # 检查明确的续表标记
            for pattern in self.continuation_patterns:
                if re.search(pattern, page_text, re.IGNORECASE):
                    # 查找上一页的相似内容
                    if page_idx > 0:
                        prev_page = pages_content[page_idx - 1]
                        ref = self._find_continuation_reference(prev_page, page, pattern)
                        if ref:
                            references.append(ref)

            # 通过内容相似性检测隐式跨页
            if page_idx > 0:
                implicit_refs = self._detect_implicit_cross_page(pages_content[page_idx - 1], page)
                references.extend(implicit_refs)

        return references

    def _find_continuation_reference(
        self,
        source_page: Dict[str, Any],
        target_page: Dict[str, Any],
        pattern: str
    ) -> Optional[CrossPageReference]:
        """查找续表引用"""
        source_contents = source_page.get('multimodal_contents', [])
        target_contents = target_page.get('multimodal_contents', [])

        # 查找表格类型的内容
        source_tables = [c for c in source_contents if c.get('type') == 'table']
        target_tables = [c for c in target_contents if c.get('type') == 'table']

        if source_tables and target_tables:
            # 使用最后一个表格作为源头
            source_table = source_tables[-1]
            target_table = target_tables[0]

            return CrossPageReference(
                source_page=source_page.get('page_number', 0),
                target_page=target_page.get('page_number', 0),
                content_id=source_table.get('id', ''),
                content_type='table',
                confidence=0.9,
                relationship='continuation'
            )

        return None

    def _detect_implicit_cross_page(
        self,
        source_page: Dict[str, Any],
        target_page: Dict[str, Any]
    ) -> List[CrossPageReference]:
        """检测隐式跨页引用"""
        references = []

        source_contents = source_page.get('multimodal_contents', [])
        target_contents = target_page.get('multimodal_contents', [])

        # 按类型分组
        source_by_type = defaultdict(list)
        target_by_type = defaultdict(list)

        for content in source_contents:
            content_type = content.get('type', 'text')
            source_by_type[content_type].append(content)

        for content in target_contents:
            content_type = content.get('type', 'text')
            target_by_type[content_type].append(content)

        # 检查每种类型的内容
        for content_type in ['table', 'image', 'formula']:
            if content_type in source_by_type and content_type in target_by_type:
                # 比较最后一项和第一项的相似性
                last_source = source_by_type[content_type][-1]
                first_target = target_by_type[content_type][0]

                similarity = self._calculate_content_similarity(last_source, first_target)
                if similarity > 0.7:  # 相似度阈值
                    references.append(CrossPageReference(
                        source_page=source_page.get('page_number', 0),
                        target_page=target_page.get('page_number', 0),
                        content_id=last_source.get('id', ''),
                        content_type=content_type,
                        confidence=similarity,
                        relationship='similar_content'
                    ))

        return references

    def _calculate_content_similarity(self, content1: Dict, content2: Dict) -> float:
        """计算内容相似度"""
        text1 = content1.get('text', '')
        text2 = content2.get('text', '')

        # 简单的文本相似度计算
        if not text1 or not text2:
            return 0.0

        # 转换为单词集合
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # 计算Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        if union == 0:
            return 0.0

        return intersection / union

    def _merge_cross_page_content(
        self,
        pages_content: List[Dict[str, Any]],
        references: List[CrossPageReference]
    ) -> List[MergedContent]:
        """合并跨页内容"""
        merged_contents = []
        processed_ids = set()

        # 按内容类型和源头分组
        ref_groups = defaultdict(list)
        for ref in references:
            key = f"{ref.content_type}_{ref.source_page}"
            ref_groups[key].append(ref)

        # 处理每个引用组
        for group_key, group_refs in ref_groups.items():
            if len(group_refs) > 1:  # 确保是真正的跨页
                merged = self._merge_content_group(pages_content, group_refs)
                if merged:
                    merged_contents.append(merged)
                    processed_ids.update(ref.content_id for ref in group_refs)

        return merged_contents

    def _merge_content_group(
        self,
        pages_content: List[Dict[str, Any]],
        references: List[CrossPageReference]
    ) -> Optional[MergedContent]:
        """合并内容组"""
        content_type = references[0].content_type
        source_contents = []

        # 收集所有相关内容
        for ref in references:
            source_page = next((p for p in pages_content if p.get('page_number') == ref.source_page), None)
            if source_page:
                contents = source_page.get('multimodal_contents', [])
                matching_content = next((c for c in contents if c.get('id') == ref.content_id), None)
                if matching_content:
                    source_contents.append(matching_content)

        if not source_contents:
            return None

        # 根据内容类型进行合并
        if content_type == 'table':
            merged_text = self._merge_table_contents(source_contents)
        elif content_type == 'image':
            merged_text = self._merge_image_contents(source_contents)
        elif content_type == 'formula':
            merged_text = self._merge_formula_contents(source_contents)
        else:
            merged_text = '\n'.join([c.get('text', '') for c in source_contents])

        # 生成合并后的元数据
        metadata = {
            'content_type': content_type,
            'source_count': len(source_contents),
            'total_pages': len(set(ref.source_page for ref in references)),
            'merge_confidence': sum(ref.confidence for ref in references) / len(references)
        }

        return MergedContent(
            merged_id=f"merged_{content_type}_{hash(merged_text)}",
            source_contents=source_contents,
            merged_content=merged_text,
            metadata=metadata
        )

    def _merge_table_contents(self, tables: List[Dict[str, Any]]) -> str:
        """合并表格内容"""
        if not tables:
            return ''

        merged_rows = []
        headers = None

        for table in tables:
            table_data = table.get('metadata', {}).get('table_data', {})
            if not headers:
                headers = table_data.get('headers', [])
                merged_rows.append(' | '.join(headers))
                merged_rows.append('-' * (len(' | '.join(headers))))

            rows = table_data.get('data', [])
            for row in rows:
                # 跳过表头行
                if row != headers:
                    merged_rows.append(' | '.join(str(cell) for cell in row))

        return '\n'.join(merged_rows)

    def _merge_image_contents(self, images: List[Dict[str, Any]]) -> str:
        """合并图片内容"""
        descriptions = []
        for img in images:
            desc = img.get('metadata', {}).get('description', '')
            if desc:
                descriptions.append(desc)

        return '\n'.join(descriptions) if descriptions else '跨页图片内容'

    def _merge_formula_contents(self, formulas: List[Dict[str, Any]]) -> str:
        """合并公式内容"""
        expressions = []
        for formula in formulas:
            expr = formula.get('metadata', {}).get('formula', {}).get('expression', '')
            if expr:
                expressions.append(expr)

        return '\n'.join(expressions) if expressions else '跨页公式'

    async def _merge_cross_page_tables(
        self,
        pages_content: List[Dict[str, Any]],
        references: List[CrossPageReference]
    ) -> List[MergedContent]:
        """合并跨页表格"""
        merged_tables = []

        # 找出所有表格相关的引用
        table_refs = [ref for ref in references if ref.content_type == 'table']

        if not table_refs:
            return merged_tables

        # 按相似性分组
        table_groups = self._group_similar_tables(pages_content, table_refs)

        # 合并每个组
        for group in table_groups:
            merged_table = self._merge_table_group(group)
            if merged_table:
                merged_tables.append(merged_table)

        return merged_tables

    def _group_similar_tables(
        self,
        pages_content: List[Dict[str, Any]],
        references: List[CrossPageReference]
    ) -> List[List[Dict[str, Any]]]:
        """将相似的表格分组"""
        groups = []
        processed = set()

        for ref in references:
            if ref.content_id in processed:
                continue

            # 查找相似的表格
            group = [ref]
            processed.add(ref.content_id)

            for other_ref in references:
                if other_ref.content_id in processed:
                    continue

                # 检查表头相似性
                if self._are_tables_similar(pages_content, ref, other_ref):
                    group.append(other_ref)
                    processed.add(other_ref.content_id)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _are_tables_similar(
        self,
        pages_content: List[Dict[str, Any]],
        ref1: CrossPageReference,
        ref2: CrossPageReference
    ) -> bool:
        """检查两个表格是否相似"""
        # 获取表格内容
        table1 = self._get_table_content(pages_content, ref1)
        table2 = self._get_table_content(pages_content, ref2)

        if not table1 or not table2:
            return False

        # 比较表头
        headers1 = table1.get('metadata', {}).get('table_data', {}).get('headers', [])
        headers2 = table2.get('metadata', {}).get('table_data', {}).get('headers', [])

        if len(headers1) != len(headers2):
            return False

        # 计算表头相似度
        matching_headers = 0
        for h1, h2 in zip(headers1, headers2):
            if h1.lower().strip() == h2.lower().strip():
                matching_headers += 1

        return matching_headers / len(headers1) > 0.7

    def _get_table_content(
        self,
        pages_content: List[Dict[str, Any]],
        ref: CrossPageReference
    ) -> Optional[Dict[str, Any]]:
        """获取表格内容"""
        page = next((p for p in pages_content if p.get('page_number') == ref.source_page), None)
        if not page:
            return None

        contents = page.get('multimodal_contents', [])
        return next((c for c in contents if c.get('id') == ref.content_id and c.get('type') == 'table'), None)

    def _merge_table_group(self, table_refs: List[CrossPageReference]) -> Optional[MergedContent]:
        """合并表格组"""
        if not table_refs:
            return None

        # 收集所有表格的行
        all_rows = []
        headers = None

        for ref in sorted(table_refs, key=lambda x: x.source_page):
            # 这里应该从实际页面数据中获取表格行
            # 简化处理，直接创建合并后的表格
            pass

        # 创建合并后的表格
        merged_content = "合并后的跨页表格"

        return MergedContent(
            merged_id=f"merged_table_{hash(merged_content)}",
            source_contents=[],
            merged_content=merged_content,
            metadata={
                'content_type': 'table',
                'source_count': len(table_refs),
                'merged_pages': sorted(set(ref.source_page for ref in table_refs))
            }
        )

    def _generate_table_signature(self, content: Dict[str, Any]) -> str:
        """生成表格签名"""
        table_data = content.get('metadata', {}).get('table_data', {})
        headers = table_data.get('headers', [])
        return '|'.join(sorted(headers))

    def _generate_image_signature(self, content: Dict[str, Any]) -> str:
        """生成图片签名"""
        metadata = content.get('metadata', {})
        img_type = metadata.get('image_type', '')
        keywords = metadata.get('keywords', [])
        return f"{img_type}|{'|'.join(sorted(keywords))}"

    def _generate_formula_signature(self, content: Dict[str, Any]) -> str:
        """生成公式签名"""
        formula = content.get('metadata', {}).get('formula', {})
        expr = formula.get('expression', '')
        variables = sorted(formula.get('variables', []))
        return f"{expr}|{'|'.join(variables)}"