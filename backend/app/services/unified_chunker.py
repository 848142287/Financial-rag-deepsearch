"""
统一分块器（优化版）
整合所有分块功能，提供统一的分块接口
"""

import re
from app.core.structured_logging import get_structured_logger
from typing import List, Dict, Any, Optional

from .unified_document_service import UnifiedChunk

logger = get_structured_logger(__name__)

class UnifiedChunker:
    """
    统一分块器

    整合以下分块器功能：
    - FinancialDocumentChunker (金融文档分块)
    - SemanticChunker (语义分块)
    - EnhancedMarkdownSplitter (Markdown分块)

    特点：
    - 自动检测文档类型
    - 智能选择分块策略
    - 保持表格完整性
    - 支持语义边界
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 512)
        self.chunk_overlap = self.config.get('chunk_overlap', 50)
        self.semantic = self.config.get('semantic', True)

    async def chunk(
        self,
        texts: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[UnifiedChunk]:
        """
        智能分块主方法

        Args:
            texts: 文本列表
            metadata: 元数据

        Returns:
            UnifiedChunk列表
        """
        metadata = metadata or {}

        # 检测文档类型
        doc_type = self._detect_document_type(texts, metadata)

        # 根据文档类型选择策略
        if doc_type == 'financial_report':
            return await self._chunk_financial(texts, metadata)
        elif doc_type == 'markdown':
            return await self._chunk_markdown(texts, metadata)
        elif doc_type == 'academic_paper':
            return await self._chunk_by_sections(texts, metadata)
        else:
            return await self._chunk_semantic(texts, metadata)

    def _detect_document_type(
        self,
        texts: List[str],
        metadata: Dict[str, Any]
    ) -> str:
        """检测文档类型"""
        # 从元数据获取
        if 'document_type' in metadata:
            return metadata['document_type']

        # 基于内容检测
        full_text = ' '.join(texts)

        # 检测Markdown
        if re.search(r'^#{1,6}\s', full_text, re.MULTILINE):
            return 'markdown'

        # 检测财务报告
        financial_keywords = ['财务报表', '资产负债表', '利润表', '现金流量表',
                           'Financial Statement', 'Balance Sheet', 'Income Statement']
        if any(keyword in full_text for keyword in financial_keywords):
            return 'financial_report'

        # 检测学术论文
        academic_keywords = ['摘要', 'Abstract', '关键词', 'Keywords',
                           '参考文献', 'References', '引言', 'Introduction']
        if any(keyword in full_text for keyword in academic_keywords):
            return 'academic_paper'

        # 默认
        return 'general'

    async def _chunk_financial(
        self,
        texts: List[str],
        metadata: Dict[str, Any]
    ) -> List[UnifiedChunk]:
        """
        金融文档分块

        策略：
        - 保持表格完整
        - 按章节分块
        - 不拆分财务指标
        """
        chunks = []
        current_chunk = ""
        chunk_index = 0
        current_page = 0

        for text in texts:
            # 检测表格
            tables = re.finditer(r'\|.*\|', text)
            table_positions = [(m.start(), m.end()) for m in tables]

            # 如果有表格，保持表格完整
            if table_positions:
                # 提取表格
                for start, end in table_positions:
                    table_text = text[start:end]

                    # 如果当前chunk不为空，先保存
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            current_chunk, chunk_index, current_page, metadata
                        ))
                        chunk_index += 1
                        current_chunk = ""

                    # 表格作为独立chunk
                    chunks.append(self._create_chunk(
                        table_text, chunk_index, current_page,
                        {**metadata, 'content_type': 'table'}
                    ))
                    chunk_index += 1

                # 处理表格之间的文本
                remaining_text = text
                for start, end in sorted(table_positions, reverse=True):
                    remaining_text = remaining_text[:start] + remaining_text[end:]

                if remaining_text.strip():
                    current_chunk += remaining_text + "\n"
            else:
                # 普通文本
                if len(current_chunk) + len(text) > self.chunk_size:
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            current_chunk, chunk_index, current_page, metadata
                        ))
                        chunk_index += 1
                        current_chunk = text + "\n"
                    else:
                        # 单个文本太长，需要切分
                        sub_chunks = self._split_long_text(text, self.chunk_size)
                        for sub_chunk in sub_chunks:
                            chunks.append(self._create_chunk(
                                sub_chunk, chunk_index, current_page, metadata
                            ))
                            chunk_index += 1
                else:
                    current_chunk += text + "\n"

            current_page += 1

        # 保存最后一个chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, chunk_index, current_page, metadata
            ))

        return chunks

    async def _chunk_markdown(
        self,
        texts: List[str],
        metadata: Dict[str, Any]
    ) -> List[UnifiedChunk]:
        """
        Markdown文档分块

        策略：
        - 按标题层级分块
        - 保持代码块完整
        - 保持表格完整
        """
        chunks = []
        chunk_index = 0

        full_text = '\n'.join(texts)

        # 按标题分割
        sections = re.split(r'\n^(#{1,6})\s+', full_text, flags=re.MULTILINE)

        current_section = ""
        current_level = 0

        for i, section in enumerate(sections):
            if i == 0:
                # 第一个部分是文档开头（没有标题）
                current_section = section
                continue

            # section[0]是标题符号，section[1:]是标题和内容
            if len(section) > 0 and section[0] in '#':
                parts = section.split('\n', 1)
                heading_marker = parts[0]
                heading_content = parts[1] if len(parts) > 1 else ""

                # 计算标题级别
                level = len(heading_marker)

                # 如果是更高层级的标题，保存当前section
                if level <= current_level and current_section.strip():
                    chunks.append(self._create_chunk(
                        current_section, chunk_index, 0, metadata
                    ))
                    chunk_index += 1
                    current_section = f"{heading_marker} {heading_content}"
                    current_level = level
                else:
                    current_section += f"\n{heading_marker} {heading_content}"
            else:
                current_section += "\n" + section

            # 检查section长度，如果太长则分块
            if len(current_section) > self.chunk_size * 1.5:
                sub_chunks = self._split_long_text(current_section, self.chunk_size)
                for j, sub_chunk in enumerate(sub_chunks):
                    if j > 0:
                        chunk_index += 1
                    chunks.append(self._create_chunk(
                        sub_chunk, chunk_index, 0, metadata
                    ))
                current_section = ""
                chunk_index += 1

        # 保存最后一个section
        if current_section.strip():
            chunks.append(self._create_chunk(
                current_section, chunk_index, 0, metadata
            ))

        return chunks

    async def _chunk_by_sections(
        self,
        texts: List[str],
        metadata: Dict[str, Any]
    ) -> List[UnifiedChunk]:
        """
        按章节分块（学术论文）

        策略：
        - 识别章节结构
        - 保持引用完整
        - 不拆分段落
        """
        chunks = []
        chunk_index = 0

        full_text = '\n'.join(texts)

        # 识别章节标题
        section_patterns = [
            r'\n\s*(摘要|Abstract)\s*\n',
            r'\n\s*(关键词|Keywords)\s*\n',
            r'\n\s*(引言|Introduction)\s*\n',
            r'\n\s*(相关工作|Related Work)\s*\n',
            r'\n\s*(方法|Methodology|Methods)\s*\n',
            r'\n\s*(实验|Experiments|Experimental)\s*\n',
            r'\n\s*(结果|Results)\s*\n',
            r'\n\s*(讨论|Discussion)\s*\n',
            r'\n\s*(结论|Conclusion)\s*\n',
            r'\n\s*(参考文献|References)\s*\n'
        ]

        # 按章节分割
        split_positions = [(0, None)]
        for pattern in section_patterns:
            for match in re.finditer(pattern, full_text):
                split_positions.append((match.start(), match.group().strip()))

        # 排序并去重
        split_positions = sorted(set(split_positions), key=lambda x: x[0])

        # 生成分块
        for i in range(len(split_positions) - 1):
            start, title = split_positions[i]
            end = split_positions[i + 1][0]

            section_text = full_text[start:end].strip()

            if section_text:
                # 如果section太长，按段落分割
                if len(section_text) > self.chunk_size * 1.5:
                    paragraphs = section_text.split('\n\n')
                    current_chunk_text = ""

                    for para in paragraphs:
                        if len(current_chunk_text) + len(para) > self.chunk_size:
                            if current_chunk_text:
                                chunks.append(self._create_chunk(
                                    current_chunk_text, chunk_index, 0,
                                    {**metadata, 'section': title}
                                ))
                                chunk_index += 1
                                current_chunk_text = para + "\n\n"
                            else:
                                # 单个段落太长
                                sub_chunks = self._split_long_text(para, self.chunk_size)
                                for sub_chunk in sub_chunks:
                                    chunks.append(self._create_chunk(
                                        sub_chunk, chunk_index, 0,
                                        {**metadata, 'section': title}
                                    ))
                                    chunk_index += 1
                        else:
                            current_chunk_text += para + "\n\n"

                    if current_chunk_text.strip():
                        chunks.append(self._create_chunk(
                            current_chunk_text, chunk_index, 0,
                            {**metadata, 'section': title}
                        ))
                        chunk_index += 1
                else:
                    chunks.append(self._create_chunk(
                        section_text, chunk_index, 0,
                        {**metadata, 'section': title}
                    ))
                    chunk_index += 1

        return chunks

    async def _chunk_semantic(
        self,
        texts: List[str],
        metadata: Dict[str, Any]
    ) -> List[UnifiedChunk]:
        """
        语义分块（通用）

        策略：
        - 按段落分割
        - 保持语义完整性
        - 可配置的重叠
        """
        chunks = []
        chunk_index = 0

        full_text = '\n'.join(texts)

        # 按段落分割
        paragraphs = full_text.split('\n\n')

        current_chunk = ""
        overlap_buffer = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 检查是否需要开始新的chunk
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    # 保存当前chunk
                    chunks.append(self._create_chunk(
                        current_chunk, chunk_index, 0, metadata
                    ))
                    chunk_index += 1

                    # 保存重叠部分
                    if self.chunk_overlap > 0:
                        overlap_buffer = current_chunk[-self.chunk_overlap:]
                    else:
                        overlap_buffer = ""

                    current_chunk = overlap_buffer + "\n\n" + para + "\n\n"
                else:
                    # 单个段落太长
                    sub_chunks = self._split_long_text(para, self.chunk_size)
                    for sub_chunk in sub_chunks:
                        chunks.append(self._create_chunk(
                            sub_chunk, chunk_index, 0, metadata
                        ))
                        chunk_index += 1
                    current_chunk = ""
            else:
                current_chunk += para + "\n\n"

        # 保存最后一个chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk, chunk_index, 0, metadata
            ))

        return chunks

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _create_chunk(
        self,
        content: str,
        chunk_index: int,
        page_number: int,
        metadata: Dict[str, Any]
    ) -> UnifiedChunk:
        """创建UnifiedChunk对象"""
        return UnifiedChunk(
            content=content.strip(),
            chunk_id=f"chunk_{chunk_index}",
            document_id=metadata.get('document_id', ''),
            chunk_index=chunk_index,
            page_numbers=[page_number],
            content_type=metadata.get('content_type', 'text'),
            metadata=metadata
        )

    def _split_long_text(
        self,
        text: str,
        max_length: int
    ) -> List[str]:
        """分割过长的文本"""
        chunks = []
        current_chunk = ""

        sentences = re.split(r'([。！？.!?])', text)

        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')

            if len(current_chunk) + len(sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
