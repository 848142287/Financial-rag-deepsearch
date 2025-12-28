"""
文本文件解析器
支持 .txt 和纯文本文件的解析

功能特点：
1. 解析纯文本文件
2. 自动检测编码
3. 智能分块（基于段落和句子）
4. 提取基本元数据
"""

import re
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .base import BaseFileParser, ParseResult, DocumentChunk
from .advanced.enhanced_markdown_splitter import EnhancedMarkdownSplitter, SplitConfig

logger = logging.getLogger(__name__)


class TextParser(BaseFileParser):
    """
    文本文件解析器

    支持解析 .txt 和其他纯文本文件
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化文本解析器

        Args:
            config: 配置字典
                - detect_language: 是否检测语言 (默认True)
                - preserve_whitespace: 是否保留空白字符 (默认False)
                - chunk_by_paragraph: 是否按段落分块 (默认True)
                - chunk_config: 分块配置
        """
        super().__init__(config)
        self.detect_language = self.config.get('detect_language', True)
        self.preserve_whitespace = self.config.get('preserve_whitespace', False)
        self.chunk_by_paragraph = self.config.get('chunk_by_paragraph', True)

        # 初始化分块器
        chunk_config = self.config.get('chunk_config')
        if isinstance(chunk_config, dict):
            chunk_config = SplitConfig(**chunk_config)
        self.splitter = EnhancedMarkdownSplitter(chunk_config)

    @property
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名"""
        return ['.txt', '.text', '.log', '.json', '.xml', '.csv']

    @property
    def parser_name(self) -> str:
        """解析器名称"""
        return "text_parser"

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
        解析文本文件

        Args:
            file_path: 文件路径
            **kwargs: 额外参数

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

            # 处理内容
            processed_content = self._process_text_content(raw_content)

            # 提取元数据
            metadata = self._extract_text_metadata(raw_content, processed_content)

            # 添加基础信息
            metadata.update({
                'file_type': 'text',
                'encoding': encoding,
                'parser_version': '2.0'
            })

            # 检测语言
            if self.detect_language:
                language = self._detect_language(processed_content)
                if language:
                    metadata['detected_language'] = language

            parse_time = time.time() - start_time

            self.logger.info(f"Successfully parsed text file: {file_path}")

            return ParseResult(
                content=processed_content,
                metadata=metadata,
                success=True,
                parse_time=parse_time,
                encoding=encoding
            )

        except Exception as e:
            parse_time = time.time() - start_time
            error_msg = f"Failed to parse text file {file_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return ParseResult(
                content="",
                metadata={'error_details': str(e)},
                success=False,
                error_message=error_msg,
                parse_time=parse_time
            )

    def _process_text_content(self, content: str) -> str:
        """
        处理文本内容

        Args:
            content: 原始文本内容

        Returns:
            str: 处理后的内容
        """
        processed = content

        # 如果不保留空白，进行清理
        if not self.preserve_whitespace:
            # 移除多余的空白行
            processed = re.sub(r'\n\s*\n\s*\n', '\n\n', processed)

            # 移除行首行尾空白
            lines = [line.strip() for line in processed.split('\n')]
            processed = '\n'.join(lines)

            # 移除多余的空格
            processed = re.sub(r' +', ' ', processed)

        return processed.strip()

    def _extract_text_metadata(self, raw_content: str, processed_content: str) -> Dict[str, Any]:
        """
        提取文本元数据

        Args:
            raw_content: 原始内容
            processed_content: 处理后的内容

        Returns:
            Dict: 元数据字典
        """
        metadata = {}

        # 基本统计
        lines = processed_content.split('\n')
        metadata['line_count'] = len(lines)
        metadata['char_count'] = len(processed_content)
        metadata['word_count'] = len(processed_content.split())
        metadata['paragraph_count'] = len([l for l in lines if l.strip()])

        # 非空行数
        non_empty_lines = [l for l in lines if l.strip()]
        metadata['non_empty_line_count'] = len(non_empty_lines)

        # 平均行长度
        if non_empty_lines:
            avg_line_length = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
            metadata['avg_line_length'] = avg_line_length

        # 最大行长度
        max_line_length = max(len(l) for l in lines) if lines else 0
        metadata['max_line_length'] = max_line_length

        # 特殊字符统计
        metadata['digit_count'] = sum(c.isdigit() for c in processed_content)
        metadata['alpha_count'] = sum(c.isalpha() for c in processed_content)
        metadata['punctuation_count'] = sum(
            not c.isspace() and not c.isalnum()
            for c in processed_content
        )

        # 检测文件类型（根据内容特征）
        metadata['content_type_hint'] = self._detect_content_type(processed_content)

        return metadata

    def _detect_content_type(self, content: str) -> str:
        """
        检测内容类型

        Args:
            content: 文本内容

        Returns:
            str: 内容类型提示
        """
        # 检测JSON
        if content.strip().startswith('{') and content.strip().endswith('}'):
            return 'json'

        # 检测XML
        if content.strip().startswith('<') and content.strip().endswith('>'):
            return 'xml'

        # 检测CSV
        if ',' in content.split('\n')[0]:
            return 'possibly_csv'

        # 检测代码（通过常见编程语言特征）
        code_indicators = [
            'def ', 'function ', 'class ', 'import ', '#include',
            'if (', 'for (', 'while (', 'return '
        ]
        if any(indicator in content for indicator in code_indicators):
            return 'possibly_code'

        return 'plain_text'

    def _detect_language(self, content: str) -> Optional[str]:
        """
        检测文本语言

        Args:
            content: 文本内容

        Returns:
            Optional[str]: 语言代码
        """
        try:
            from langdetect import detect, DetectorFactory
            # 设置种子以获得一致的结果
            DetectorFactory.seed = 0

            # 使用前1000个字符检测
            sample = content[:1000]
            if len(sample) > 50:  # 确保有足够的内容
                language = detect(sample)
                return language
        except ImportError:
            self.logger.warning("langdetect not installed")
        except Exception as e:
            self.logger.warning(f"Failed to detect language: {e}")

        return None

    def chunk_content(
        self,
        content: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        将文本内容分割成块

        Args:
            content: 文本内容
            chunk_size: chunk大小
            chunk_overlap: chunk重叠
            metadata: 元数据

        Returns:
            List[DocumentChunk]: 分块列表
        """
        if self.chunk_by_paragraph:
            # 按段落分块
            return self._chunk_by_paragraph(content, chunk_size, chunk_overlap, metadata)
        else:
            # 使用基础分块方法
            return super().chunk_content(content, chunk_size, chunk_overlap, metadata)

    def _chunk_by_paragraph(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: Optional[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """
        按段落分块

        Args:
            content: 文本内容
            chunk_size: 最大chunk大小
            chunk_overlap: chunk重叠
            metadata: 元数据

        Returns:
            List[DocumentChunk]: 分块列表
        """
        # 分割段落
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = ""
        chunk_index = 0
        char_start = 0

        for para in paragraphs:
            # 如果添加这个段落会超过chunk_size
            if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                # 保存当前chunk
                chunk_metadata = {
                    'chunk_size': len(current_chunk),
                    'chunk_start': char_start,
                    'chunk_end': char_start + len(current_chunk),
                    'paragraph_count': current_chunk.count('\n\n') + 1,
                    'chunking_method': 'paragraph',
                    **(metadata or {})
                }

                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    metadata=chunk_metadata,
                    source_type=self.parser_name
                ))

                # 更新位置
                char_start += len(current_chunk)
                chunk_index += 1

                # 开始新chunk（保留重叠）
                if chunk_overlap > 0:
                    # 简单的重叠策略：保留最后一个段落
                    last_para_start = current_chunk.rfind('\n\n')
                    if last_para_start != -1:
                        current_chunk = current_chunk[last_para_start + 2:]
                    else:
                        current_chunk = ""
                else:
                    current_chunk = ""

            # 添加段落
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

        # 添加最后一个chunk
        if current_chunk.strip():
            chunk_metadata = {
                'chunk_size': len(current_chunk),
                'chunk_start': char_start,
                'chunk_end': char_start + len(current_chunk),
                'paragraph_count': current_chunk.count('\n\n') + 1,
                'chunking_method': 'paragraph',
                **(metadata or {})
            }

            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                metadata=chunk_metadata,
                source_type=self.parser_name
            ))

        return chunks

    def get_parser_info(self) -> Dict[str, Any]:
        """获取解析器信息"""
        info = super().get_parser_info()
        info.update({
            'detect_language': self.detect_language,
            'preserve_whitespace': self.preserve_whitespace,
            'chunk_by_paragraph': self.chunk_by_paragraph,
            'supported_features': [
                'encoding_detection',
                'language_detection',
                'content_type_detection',
                'paragraph_based_chunking',
                'statistics_collection'
            ]
        })
        return info
