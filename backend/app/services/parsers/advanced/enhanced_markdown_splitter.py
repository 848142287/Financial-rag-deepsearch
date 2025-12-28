"""
增强的 Markdown 分割器 - 基于标题的智能分割

功能特点：
1. 基于标题层级智能分割文档
2. 维护标题路径元数据 (Header_1 > Header_2 > Header_3)
3. 保留完整的文档结构信息
4. 支持自定义标题识别规则

集成来源: 03_DataAnalysis_main/backend/core/analysis/data_analyzer.py
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from langchain_core.documents import Document


@dataclass
class SplitConfig:
    """分割配置"""
    max_chunk_size: int = 2000  # 最大chunk大小（字符数）
    chunk_overlap: int = 200    # chunk重叠大小
    min_chunk_size: int = 100   # 最小chunk大小
    title_patterns: List[str] = field(default_factory=lambda: [
        r'^#\s+(.+)$',          # 一级标题
        r'^##\s+(.+)$',         # 二级标题
        r'^###\s+(.+)$',        # 三级标题
        r'^####\s+(.+)$',       # 四级标题
        r'^#####\s+(.+)$',      # 五级标题
        r'^######\s+(.+)$',     # 六级标题
    ])
    title_marker: str = "{{第{page}页}}"  # 页码标记格式


@dataclass
class ChunkMetadata:
    """Chunk元数据"""
    chunk_id: str
    title_path: str              # 标题路径: "Header_1 > Header_2 > Header_3"
    title_level_1: Optional[str] = None
    title_level_2: Optional[str] = None
    title_level_3: Optional[str] = None
    title_level_4: Optional[str] = None
    title_level_5: Optional[str] = None
    title_level_6: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    char_start: int = 0
    char_end: int = 0
    is_title_chunk: bool = False  # 是否是标题开头chunk


class EnhancedMarkdownSplitter:
    """
    增强的 Markdown 分割器

    基于文档标题结构进行智能分割，保留完整的结构上下文信息
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        """
        初始化分割器

        Args:
            config: 分割配置，如果为None则使用默认配置
        """
        self.config = config or SplitConfig()
        self._compile_patterns()

    def _compile_patterns(self):
        """编译标题正则表达式"""
        self.title_regex = re.compile(
            '|'.join(f'(?P<h{i}>{pattern})' for i, pattern in enumerate(self.config.title_patterns)),
            re.MULTILINE
        )

    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        分割文本为多个Document

        Args:
            text: 输入的markdown文本
            metadata: 原始元数据

        Returns:
            分割后的Document列表
        """
        if not text or not text.strip():
            return []

        # 1. 查找所有标题分割点
        split_points = self._find_title_split_points(text)

        if not split_points:
            # 没有找到标题，使用基础分割
            return self._fallback_split(text, metadata)

        # 2. 根据分割点创建chunks
        chunks = self._create_chunks_from_split_points(text, split_points, metadata or {})

        # 3. 后处理：合并过小的chunks
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _find_title_split_points(self, text: str) -> List[Dict[str, Any]]:
        """
        查找所有标题分割点

        Args:
            text: 输入文本

        Returns:
            分割点列表，每个元素包含位置、标题、层级等信息
        """
        split_points = []
        title_stack = []  # [(level, title), ...]

        for match in self.title_regex.finditer(text):
            # 确定标题层级
            for i in range(6):
                if match.group(f'h{i}'):
                    level = i
                    title = match.group(i + 1)  # group(0)是整个匹配，group(1+)是捕获组
                    break
            else:
                continue

            # 更新标题栈
            # 弹出比当前层级高或相等的标题
            while title_stack and title_stack[-1][0] >= level:
                title_stack.pop()
            title_stack.append((level, title))

            split_points.append({
                'position': match.start(),
                'title': title.strip(),
                'level': level,
                'title_path': ' > '.join([t for _, t in title_stack]),
                'title_stack': {f'title_level_{i+1}': t if i < len(title_stack) else None
                               for i in range(6)}
            })

        return split_points

    def _create_chunks_from_split_points(
        self,
        text: str,
        split_points: List[Dict[str, Any]],
        base_metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        根据分割点创建chunks

        Args:
            text: 输入文本
            split_points: 分割点列表
            base_metadata: 基础元数据

        Returns:
            Document列表
        """
        chunks = []

        for i, point in enumerate(split_points):
            # 确定chunk的起始和结束位置
            start_pos = point['position']
            end_pos = split_points[i + 1]['position'] if i + 1 < len(split_points) else len(text)

            # 提取chunk文本
            chunk_text = text[start_pos:end_pos].strip()

            if not chunk_text:
                continue

            # 如果chunk太小，跳过（会在后续合并中处理）
            if len(chunk_text) < self.config.min_chunk_size:
                continue

            # 如果chunk太大，需要进一步分割
            if len(chunk_text) > self.config.max_chunk_size:
                sub_chunks = self._split_large_chunk(
                    chunk_text,
                    point,
                    base_metadata
                )
                chunks.extend(sub_chunks)
            else:
                # 创建正常大小的chunk
                chunk_metadata = self._create_chunk_metadata(
                    point,
                    start_pos,
                    end_pos,
                    base_metadata,
                    is_title_chunk=True
                )
                chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return chunks

    def _split_large_chunk(
        self,
        chunk_text: str,
        split_point: Dict[str, Any],
        base_metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        分割过大的chunk

        Args:
            chunk_text: 过大的chunk文本
            split_point: 原始分割点信息
            base_metadata: 基础元数据

        Returns:
            分割后的Document列表
        """
        chunks = []
        # 使用重叠分割
        start = 0
        chunk_id = 0

        while start < len(chunk_text):
            end = start + self.config.max_chunk_size

            # 尝试在段落边界分割
            if end < len(chunk_text):
                # 查找最近的换行符
                newline_pos = chunk_text.rfind('\n', start, end)
                if newline_pos > start + self.config.min_chunk_size:
                    end = newline_pos + 1

            sub_chunk_text = chunk_text[start:end].strip()

            if not sub_chunk_text:
                break

            chunk_metadata = self._create_chunk_metadata(
                split_point,
                start,
                end,
                base_metadata,
                is_title_chunk=(chunk_id == 0)
            )
            chunk_metadata['chunk_id'] = f"{chunk_metadata.get('chunk_id', '0')}_sub{chunk_id}"

            chunks.append(Document(page_content=sub_chunk_text, metadata=chunk_metadata))

            start = end - self.config.chunk_overlap
            chunk_id += 1

        return chunks

    def _create_chunk_metadata(
        self,
        split_point: Dict[str, Any],
        start_pos: int,
        end_pos: int,
        base_metadata: Dict[str, Any],
        is_title_chunk: bool
    ) -> Dict[str, Any]:
        """
        创建chunk元数据

        Args:
            split_point: 分割点信息
            start_pos: 起始位置
            end_pos: 结束位置
            base_metadata: 基础元数据
            is_title_chunk: 是否是标题开头chunk

        Returns:
            完整的元数据字典
        """
        # 提取页码信息（如果有页码标记）
        page_start, page_end = self._extract_pages_from_text(
            split_point.get('title_path', ''),
            start_pos,
            end_pos
        )

        metadata = {
            'chunk_id': f"chunk_{start_pos}_{end_pos}",
            'title_path': split_point.get('title_path', ''),
            'char_start': start_pos,
            'char_end': end_pos,
            'is_title_chunk': is_title_chunk,
        }

        # 添加标题层级信息
        metadata.update(split_point.get('title_stack', {}))

        # 添加页码信息（如果有）
        if page_start is not None:
            metadata['page_start'] = page_start
        if page_end is not None:
            metadata['page_end'] = page_end

        # 合并基础元数据
        metadata.update(base_metadata)

        return metadata

    def _extract_pages_from_text(
        self,
        text: str,
        start_pos: int,
        end_pos: int
    ) -> tuple[Optional[int], Optional[int]]:
        """
        从文本中提取页码信息

        Args:
            text: 输入文本
            start_pos: 起始位置
            end_pos: 结束位置

        Returns:
            (page_start, page_end) 元组
        """
        # 查找页码标记
        page_pattern = re.compile(self.config.title_marker.replace('{page}', r'(\d+)'))
        pages = page_pattern.findall(text)

        if pages:
            try:
                return int(pages[0]), int(pages[-1])
            except (ValueError, IndexError):
                pass

        return None, None

    def _merge_small_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        合并过小的chunks到前一个chunk

        Args:
            chunks: Document列表

        Returns:
            合并后的Document列表
        """
        if not chunks:
            return []

        merged = [chunks[0]]

        for chunk in chunks[1:]:
            if len(chunk.page_content) < self.config.min_chunk_size:
                # 合并到前一个chunk
                last_chunk = merged[-1]
                merged_content = last_chunk.page_content + '\n\n' + chunk.page_content
                merged_metadata = last_chunk.metadata.copy()
                merged_metadata['char_end'] = chunk.metadata.get('char_end', 0)

                merged[-1] = Document(page_content=merged_content, metadata=merged_metadata)
            else:
                merged.append(chunk)

        return merged

    def _fallback_split(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        当没有找到标题时的后备分割方法

        Args:
            text: 输入文本
            metadata: 元数据

        Returns:
            分割后的Document列表
        """
        # 使用简单的基于大小的分割
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.config.max_chunk_size

            # 尝试在段落边界分割
            if end < len(text):
                newline_pos = text.rfind('\n', start, end)
                if newline_pos > start + self.config.min_chunk_size:
                    end = newline_pos + 1

            chunk_text = text[start:end].strip()

            if not chunk_text:
                break

            chunk_metadata = {
                'chunk_id': f'chunk_{chunk_id}',
                'char_start': start,
                'char_end': end,
                'is_title_chunk': False,
            }
            if metadata:
                chunk_metadata.update(metadata)

            chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))

            start = end - self.config.chunk_overlap
            chunk_id += 1

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        批量分割文档

        Args:
            documents: Document列表

        Returns:
            分割后的Document列表
        """
        all_chunks = []

        for doc in documents:
            chunks = self.split_text(doc.page_content, doc.metadata)
            all_chunks.extend(chunks)

        return all_chunks


# 创建全局实例
enhanced_markdown_splitter = EnhancedMarkdownSplitter()


# 便捷函数
def split_markdown(text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
    """
    分割markdown文本

    Args:
        text: markdown文本
        metadata: 元数据

    Returns:
        分割后的Document列表
    """
    return enhanced_markdown_splitter.split_text(text, metadata)


def split_markdown_documents(documents: List[Document]) -> List[Document]:
    """
    批量分割markdown文档

    Args:
        documents: Document列表

    Returns:
        分割后的Document列表
    """
    return enhanced_markdown_splitter.split_documents(documents)
