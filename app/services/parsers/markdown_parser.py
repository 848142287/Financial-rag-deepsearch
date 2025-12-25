"""
Markdown文档解析器
处理.md格式的文档
"""

import logging
import re
from typing import Dict, List, Any, Optional
from .base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)


class MarkdownParser(BaseFileParser):
    """Markdown文档解析器"""

    def __init__(self):
        self.supported_extensions = ['.md', '.markdown']
        self.supported_mime_types = ['text/markdown', 'text/x-markdown']

    def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        解析Markdown文档

        Args:
            file_path: 文件路径
            **kwargs: 其他参数

        Returns:
            ParseResult: 解析结果
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析Markdown结构
            parsed_content = self._parse_markdown_structure(content)

            # 构建元数据
            metadata = {
                'file_type': 'markdown',
                'line_count': len(content.splitlines()),
                'char_count': len(content),
                'word_count': len(content.split()),
                'headers': self._extract_headers(content),
                'links': self._extract_links(content),
                'images': self._extract_images(content),
                'code_blocks': self._extract_code_blocks(content)
            }

            return ParseResult(
                content=parsed_content,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error(f"Markdown文档解析失败: {e}")
            return ParseResult(
                content="",
                metadata={'file_type': 'markdown', 'error': str(e)},
                success=False,
                error_message=str(e)
            )

    def _parse_markdown_structure(self, content: str) -> str:
        """解析Markdown结构，转换为纯文本"""
        lines = content.splitlines()
        processed_lines = []

        in_code_block = False
        code_block_lang = ""

        for line in lines:
            # 处理代码块
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_block_lang = line.strip()[3:].strip()
                    processed_lines.append(f"[代码块开始: {code_block_lang}]")
                else:
                    in_code_block = False
                    processed_lines.append("[代码块结束]")
                continue

            if in_code_block:
                processed_lines.append(f"代码: {line}")
                continue

            # 处理标题
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2)
                processed_lines.append(f"标题{level}: {title}")
                continue

            # 处理列表
            if line.strip().startswith(('- ', '* ', '+ ')):
                item = line.strip()[2:]
                processed_lines.append(f"列表项: {item}")
                continue
            elif re.match(r'^\d+\.\s+', line.strip()):
                item = re.sub(r'^\d+\.\s+', '', line.strip())
                processed_lines.append(f"有序列表项: {item}")
                continue

            # 处理链接
            line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'链接: \1 -> \2', line)

            # 处理图片
            line = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'图片: \1 -> \2', line)

            # 处理粗体和斜体
            line = re.sub(r'\*\*([^*]+)\*\*', r'\1（粗体）', line)
            line = re.sub(r'\*([^*]+)\*', r'\1（斜体）', line)

            # 移除多余的空行，保留内容
            if line.strip():
                processed_lines.append(line.strip())

        return "\n".join(processed_lines)

    def _extract_headers(self, content: str) -> List[Dict[str, Any]]:
        """提取标题"""
        headers = []
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if header_match:
                headers.append({
                    'level': len(header_match.group(1)),
                    'text': header_match.group(2),
                    'line': line_num
                })

        return headers

    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """提取链接"""
        links = []
        matches = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)

        for text, url in matches:
            links.append({
                'text': text,
                'url': url
            })

        return links

    def _extract_images(self, content: str) -> List[Dict[str, str]]:
        """提取图片"""
        images = []
        matches = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)

        for alt_text, url in matches:
            images.append({
                'alt_text': alt_text,
                'url': url
            })

        return images

    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """提取代码块"""
        code_blocks = []
        lines = content.splitlines()
        in_block = False
        block_start = 0
        lang = ""

        for line_num, line in enumerate(lines, 1):
            if line.strip().startswith('```'):
                if not in_block:
                    in_block = True
                    block_start = line_num
                    lang = line.strip()[3:].strip()
                else:
                    in_block = False
                    code_blocks.append({
                        'language': lang,
                        'start_line': block_start,
                        'end_line': line_num
                    })
                    lang = ""

        return code_blocks

    def is_supported(self, file_path: str, mime_type: str = None) -> bool:
        """
        检查是否支持解析该文件

        Args:
            file_path: 文件路径
            mime_type: MIME类型（可选）

        Returns:
            bool: 是否支持
        """
        import os
        _, ext = os.path.splitext(file_path.lower())
        if ext in self.supported_extensions:
            return True

        if mime_type and mime_type in self.supported_mime_types:
            return True

        return False