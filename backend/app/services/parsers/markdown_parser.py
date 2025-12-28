"""
Markdown 文件解析器
支持 .md 和 .markdown 文件的解析，提取结构化内容和元数据

功能特点：
1. 解析Markdown文件为结构化数据
2. 提取元数据（标题、作者、日期等）
3. 保留文档结构（标题层级）
4. 支持代码块、表格、列表等元素提取
5. 集成EnhancedMarkdownSplitter进行智能分块
"""

import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from .base import BaseFileParser, ParseResult, DocumentChunk
from .advanced.enhanced_markdown_splitter import EnhancedMarkdownSplitter, SplitConfig

logger = logging.getLogger(__name__)


@dataclass
class MarkdownMetadata:
    """Markdown文档元数据"""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    tags: List[str] = None
    categories: List[str] = None
    description: Optional[str] = None

    # 统计信息
    heading_count: int = 0
    code_block_count: int = 0
    table_count: int = 0
    link_count: int = 0
    image_count: int = 0

    # 结构信息
    heading_structure: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'title': self.title,
            'author': self.author,
            'date': self.date,
            'tags': self.tags or [],
            'categories': self.categories or [],
            'description': self.description,
            'heading_count': self.heading_count,
            'code_block_count': self.code_block_count,
            'table_count': self.table_count,
            'link_count': self.link_count,
            'image_count': self.image_count,
            'heading_structure': self.heading_structure or []
        }


class MarkdownParser(BaseFileParser):
    """
    Markdown 文件解析器

    支持解析 .md 和 .markdown 文件，提取结构化内容和元数据
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Markdown解析器

        Args:
            config: 配置字典
                - extract_metadata: 是否提取元数据 (默认True)
                - preserve_html: 是否保留HTML标签 (默认False)
                - chunk_config: 分块配置 (SplitConfig对象)
        """
        super().__init__(config)
        self.extract_metadata = self.config.get('extract_metadata', True)
        self.preserve_html = self.config.get('preserve_html', False)

        # 初始化分块器
        chunk_config = self.config.get('chunk_config')
        if isinstance(chunk_config, dict):
            chunk_config = SplitConfig(**chunk_config)
        self.splitter = EnhancedMarkdownSplitter(chunk_config)

    @property
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名"""
        return ['.md', '.markdown', '.mdown', '.mkd']

    @property
    def parser_name(self) -> str:
        """解析器名称"""
        return "markdown_parser"

    def can_parse(self, file_path: str, file_extension: str = None) -> bool:
        """
        检查是否能解析指定文件

        Args:
            file_path: 文件路径
            file_extension: 文件扩展名

        Returns:
            bool: 是否能解析
        """
        if file_extension:
            return file_extension.lower() in self.supported_extensions

        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions

    async def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        解析Markdown文件

        Args:
            file_path: 文件路径
            **kwargs: 额外参数
                - extract_headings: 是否提取标题结构
                - include_html: 是否包含HTML内容

        Returns:
            ParseResult: 解析结果
        """
        import time
        start_time = time.time()

        # 验证文件
        is_valid, error_msg = self.validate_file(file_path)
        if not is_valid:
            return ParseResult(
                content="",
                metadata={'error': error_msg},
                success=False,
                error_message=error_msg,
                parse_time=time.time() - start_time
            )

        try:
            # 读取文件内容
            encoding = self.detect_encoding(file_path) or 'utf-8'
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                raw_content = f.read()

            if not raw_content.strip():
                return ParseResult(
                    content="",
                    metadata={'warning': 'File is empty'},
                    success=True,
                    parse_time=time.time() - start_time
                )

            # 提取元数据
            md_metadata = self._extract_markdown_metadata(raw_content)

            # 清理和处理内容
            processed_content = self._process_markdown_content(raw_content)

            # 提取文档结构
            heading_structure = self._extract_heading_structure(raw_content)

            # 统计信息
            stats = self._collect_statistics(raw_content)

            # 构建完整元数据
            metadata = {
                'file_type': 'markdown',
                'encoding': encoding,
                'metadata': md_metadata.to_dict(),
                'heading_structure': heading_structure,
                'statistics': stats,
                'parser_version': '2.0',
                'has_frontmatter': md_metadata.title is not None
            }

            # 如果需要，提取标题到顶层
            if md_metadata.title:
                metadata['title'] = md_metadata.title

            parse_time = time.time() - start_time

            self.logger.info(f"Successfully parsed Markdown file: {file_path}")

            return ParseResult(
                content=processed_content,
                metadata=metadata,
                success=True,
                parse_time=parse_time,
                encoding=encoding
            )

        except Exception as e:
            parse_time = time.time() - start_time
            error_msg = f"Failed to parse Markdown file {file_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return ParseResult(
                content="",
                metadata={'error_details': str(e)},
                success=False,
                error_message=error_msg,
                parse_time=parse_time
            )

    def _extract_markdown_metadata(self, content: str) -> MarkdownMetadata:
        """
        提取Markdown元数据

        Args:
            content: Markdown内容

        Returns:
            MarkdownMetadata: 元数据对象
        """
        metadata = MarkdownMetadata()

        # 提取YAML front matter
        frontmatter = self._extract_yaml_frontmatter(content)
        if frontmatter:
            metadata.title = frontmatter.get('title')
            metadata.author = frontmatter.get('author')
            metadata.date = frontmatter.get('date')
            metadata.tags = frontmatter.get('tags', [])
            metadata.categories = frontmatter.get('categories', [])
            metadata.description = frontmatter.get('description')

        # 如果没有从front matter提取到标题，尝试从第一个标题提取
        if not metadata.title:
            metadata.title = self._extract_first_heading(content)

        return metadata

    def _extract_yaml_frontmatter(self, content: str) -> Optional[Dict[str, Any]]:
        """
        提取YAML front matter

        Args:
            content: Markdown内容

        Returns:
            Optional[Dict]: front matter数据
        """
        # 检查是否有YAML front matter
        if not content.startswith('---'):
            return None

        # 查找结束标记
        end_pos = content.find('\n---', 4)
        if end_pos == -1:
            return None

        # 提取YAML内容
        yaml_content = content[4:end_pos]

        try:
            import yaml
            return yaml.safe_load(yaml_content)
        except ImportError:
            self.logger.warning("PyYAML not installed, cannot parse YAML front matter")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to parse YAML front matter: {e}")
            return None

    def _extract_first_heading(self, content: str) -> Optional[str]:
        """
        提取第一个一级标题

        Args:
            content: Markdown内容

        Returns:
            Optional[str]: 标题文本
        """
        # 匹配第一个 # 标题
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        return None

    def _process_markdown_content(self, content: str) -> str:
        """
        处理Markdown内容

        Args:
            content: 原始Markdown内容

        Returns:
            str: 处理后的内容
        """
        processed = content

        # 移除YAML front matter
        if processed.startswith('---'):
            end_pos = processed.find('\n---', 4)
            if end_pos != -1:
                processed = processed[end_pos + 5:]

        # 如果不保留HTML，移除HTML标签
        if not self.preserve_html:
            # 保留代码块中的HTML
            processed = self._remove_html_except_code_blocks(processed)

        # 清理多余的空白行
        processed = re.sub(r'\n\s*\n\s*\n', '\n\n', processed)

        return processed.strip()

    def _remove_html_except_code_blocks(self, content: str) -> str:
        """
        移除HTML标签（保留代码块中的）

        Args:
            content: Markdown内容

        Returns:
            str: 处理后的内容
        """
        # 简单实现：移除常见HTML标签
        # 注意：这不完美，但对于大多数情况足够了
        html_pattern = re.compile(r'<[^>]+>')

        # 分割内容，保留代码块
        parts = []
        in_code_block = False
        code_block_delimiter = None

        for line in content.split('\n'):
            # 检查代码块开始/结束
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_block_delimiter = '```'
                elif code_block_delimiter == '```':
                    in_code_block = False
                    code_block_delimiter = None
            elif line.strip().startswith('~~~'):
                if not in_code_block:
                    in_code_block = True
                    code_block_delimiter = '~~~'
                elif code_block_delimiter == '~~~':
                    in_code_block = False
                    code_block_delimiter = None

            # 如果在代码块中，保留原样
            if in_code_block:
                parts.append(line)
            else:
                # 移除HTML标签
                cleaned_line = html_pattern.sub('', line)
                parts.append(cleaned_line)

        return '\n'.join(parts)

    def _extract_heading_structure(self, content: str) -> List[Dict[str, Any]]:
        """
        提取标题结构

        Args:
            content: Markdown内容

        Returns:
            List[Dict]: 标题结构列表
        """
        headings = []
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

        for match in heading_pattern.finditer(content):
            level = len(match.group(1))
            title = match.group(2).strip()
            position = match.start()

            headings.append({
                'level': level,
                'title': title,
                'position': position
            })

        return headings

    def _collect_statistics(self, content: str) -> Dict[str, int]:
        """
        收集文档统计信息

        Args:
            content: Markdown内容

        Returns:
            Dict: 统计信息
        """
        stats = {}

        # 标题数量
        stats['heading_count'] = len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE))

        # 代码块数量
        stats['code_block_count'] = len(re.findall(r'```', content)) // 2

        # 表格数量（简单检测）
        stats['table_count'] = len(re.findall(r'\|.*\|', content))

        # 链接数量
        stats['link_count'] = len(re.findall(r'\[.+?\]\(.+?\)', content))

        # 图片数量
        stats['image_count'] = len(re.findall(r'!\[.*?\]\(.*?\)', content))

        # 列表数量
        stats['list_item_count'] = len(re.findall(r'^[\*\-\+]\s+', content, re.MULTILINE))
        stats['ordered_list_count'] = len(re.findall(r'^\d+\.\s+', content, re.MULTILINE))

        # 基本统计
        stats['line_count'] = len(content.split('\n'))
        stats['word_count'] = len(content.split())
        stats['char_count'] = len(content)

        return stats

    def chunk_content(
        self,
        content: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        将Markdown内容分割成块（使用EnhancedMarkdownSplitter）

        Args:
            content: Markdown内容
            chunk_size: chunk大小（字符数）
            chunk_overlap: chunk重叠大小
            metadata: 元数据

        Returns:
            List[DocumentChunk]: 分块列表
        """
        # 更新分块器配置
        self.splitter.config.max_chunk_size = chunk_size
        self.splitter.config.chunk_overlap = chunk_overlap

        # 使用增强分割器
        from langchain_core.documents import Document as LangchainDocument
        langchain_docs = self.splitter.split_text(content, metadata)

        # 转换为DocumentChunk
        chunks = []
        for i, lc_doc in enumerate(langchain_docs):
            chunk = DocumentChunk(
                content=lc_doc.page_content,
                chunk_index=i,
                metadata=lc_doc.metadata,
                source_type=self.parser_name
            )
            chunks.append(chunk)

        return chunks

    def extract_tables(self, content: str) -> List[Dict[str, Any]]:
        """
        提取Markdown表格

        Args:
            content: Markdown内容

        Returns:
            List[Dict]: 表格列表
        """
        tables = []
        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # 检测表格开始（包含 | 分隔符）
            if '|' in line and line.startswith('|'):
                table_lines = [line]

                # 收集表格行
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if '|' in next_line:
                        table_lines.append(next_line)
                        i += 1
                    else:
                        break

                # 解析表格
                if len(table_lines) >= 2:
                    table = self._parse_table(table_lines)
                    if table:
                        tables.append(table)

            i += 1

        return tables

    def _parse_table(self, table_lines: List[str]) -> Optional[Dict[str, Any]]:
        """
        解析表格

        Args:
            table_lines: 表格行列表

        Returns:
            Optional[Dict]: 表格数据
        """
        try:
            # 第一行是表头
            headers = [h.strip() for h in table_lines[0].split('|')[1:-1]]

            # 第二行是分隔符（跳过）
            if len(table_lines) > 2:
                # 数据行
                rows = []
                for line in table_lines[2:]:
                    cells = [c.strip() for c in line.split('|')[1:-1]]
                    if cells:
                        rows.append(cells)

                return {
                    'headers': headers,
                    'rows': rows,
                    'row_count': len(rows),
                    'column_count': len(headers)
                }
        except Exception as e:
            self.logger.warning(f"Failed to parse table: {e}")

        return None

    def extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """
        提取代码块

        Args:
            content: Markdown内容

        Returns:
            List[Dict]: 代码块列表
        """
        code_blocks = []
        pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)

        for match in pattern.finditer(content):
            language = match.group(1) or 'text'
            code = match.group(2)

            code_blocks.append({
                'language': language,
                'code': code,
                'length': len(code)
            })

        return code_blocks

    def convert_to_html(self, content: str) -> str:
        """
        将Markdown转换为HTML

        Args:
            content: Markdown内容

        Returns:
            str: HTML内容
        """
        try:
            import markdown
            md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
            return md.convert(content)
        except ImportError:
            self.logger.warning("markdown library not installed")
            return content
        except Exception as e:
            self.logger.error(f"Failed to convert Markdown to HTML: {e}")
            return content

    def get_parser_info(self) -> Dict[str, Any]:
        """获取解析器信息"""
        info = super().get_parser_info()
        info.update({
            'extract_metadata': self.extract_metadata,
            'preserve_html': self.preserve_html,
            'supported_features': [
                'yaml_frontmatter',
                'heading_extraction',
                'table_extraction',
                'code_block_extraction',
                'html_conversion',
                'intelligent_chunking'
            ]
        })
        return info
