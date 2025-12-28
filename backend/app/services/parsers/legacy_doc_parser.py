"""
旧版Word文档解析器 - 支持 .doc 格式 (Word 97-2003)
使用 docx2txt 库进行文本提取
"""

import logging
import os
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import time

from .base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)

try:
    import docx2txt
    DOCX2TXT_AVAILABLE = True
except ImportError:
    DOCX2TXT_AVAILABLE = False
    logger.warning("docx2txt not available, .doc parsing will not work")


@dataclass
class DocMetadata:
    """.doc 文档元数据"""
    title: str = ""
    author: str = ""
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    page_count: int = 0
    word_count: int = 0
    char_count: int = 0


class LegacyDocParser(BaseFileParser):
    """旧版Word文档解析器 - 支持 .doc 格式"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.extract_images = self.config.get('extract_images', False)
        self.enable_ocr = self.config.get('enable_ocr', False)
        self.preserve_formatting = self.config.get('preserve_formatting', False)

    @property
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名列表"""
        return ['.doc']

    @property
    def parser_name(self) -> str:
        """解析器名称"""
        return "LegacyDocParser"

    def can_parse(self, file_path: str, file_extension: str = None) -> bool:
        """检查是否能解析指定文件"""
        if not DOCX2TXT_AVAILABLE:
            return False

        if file_extension:
            return file_extension.lower() in self.supported_extensions

        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_extensions

    async def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        解析旧版Word文档

        Args:
            file_path: Word文件路径 (.doc)
            **kwargs: 其他参数
                - extract_images: 是否提取图像（默认False）
                - enable_ocr: 是否启用OCR（默认False）
                - output_format: 输出格式（'markdown', 'text', 'json'，默认'markdown'）

        Returns:
            ParseResult: 解析结果
        """
        start_time = time.time()

        try:
            if not DOCX2TXT_AVAILABLE:
                return ParseResult(
                    content="",
                    metadata={'file_type': 'doc', 'error': 'docx2txt library not available'},
                    success=False,
                    error_message='docx2txt library not installed. Run: pip install docx2txt'
                )

            # 获取参数
            extract_images = kwargs.get('extract_images', self.extract_images)
            output_format = kwargs.get('output_format', 'markdown')

            # 创建临时目录用于提取图片
            temp_dir = None
            if extract_images:
                temp_dir = tempfile.mkdtemp(prefix='doc_images_')

            # 1. 提取文本内容
            text_content = await self._extract_text(file_path, temp_dir)

            # 2. 提取图片（如果需要）
            extracted_images = []
            if extract_images and temp_dir:
                extracted_images = await self._extract_images(file_path, temp_dir)

            # 3. 处理文本内容
            if output_format == 'markdown':
                content = self._format_as_markdown(text_content, extracted_images)
            elif output_format == 'json':
                content = self._format_as_json(text_content, extracted_images)
            else:  # text
                content = text_content

            # 4. 清理临时目录
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

            # 5. 构建元数据
            metadata = self._build_metadata(file_path, text_content, extracted_images)
            metadata['parse_time'] = time.time() - start_time

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True,
                parse_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Legacy .doc parsing failed: {e}", exc_info=True)

            # 清理临时目录
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

            return ParseResult(
                content="",
                metadata={'file_type': 'doc', 'error': str(e)},
                success=False,
                error_message=str(e),
                parse_time=time.time() - start_time
            )

    async def _extract_text(self, file_path: str, temp_dir: Optional[str] = None) -> str:
        """
        提取文档文本内容

        Args:
            file_path: 文件路径
            temp_dir: 临时目录（用于提取图片）

        Returns:
            str: 提取的文本内容
        """
        try:
            # 使用 docx2txt 提取文本
            if temp_dir:
                # 如果提供了临时目录,docx2txt 会将图片提取到该目录
                text = docx2txt.process(file_path, temp_dir)
            else:
                # 只提取文本,不提取图片
                text = docx2txt.process(file_path)

            # 清理文本
            text = self._clean_text(text)

            return text

        except Exception as e:
            logger.error(f"Failed to extract text from .doc file: {e}")
            raise

    async def _extract_images(self, file_path: str, temp_dir: str) -> List[Dict[str, Any]]:
        """
        提取文档中的图片

        Args:
            file_path: 文件路径
            temp_dir: 临时目录

        Returns:
            List[Dict]: 图片信息列表
        """
        images = []

        try:
            # docx2txt 在提取文本时会自动将图片保存到 temp_dir
            # 我们只需要扫描该目录找到图片文件
            if os.path.exists(temp_dir):
                for filename in os.listdir(temp_dir):
                    file_path_full = os.path.join(temp_dir, filename)

                    # 检查是否为图片文件
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.wmf', '.emf')):
                        # 获取文件大小
                        file_size = os.path.getsize(file_path_full)

                        images.append({
                            'filename': filename,
                            'temp_path': file_path_full,
                            'file_size': file_size,
                            'type': 'embedded'
                        })

        except Exception as e:
            logger.warning(f"Failed to extract images from .doc file: {e}")

        return images

    def _clean_text(self, text: str) -> str:
        """
        清理提取的文本

        Args:
            text: 原始文本

        Returns:
            str: 清理后的文本
        """
        if not text:
            return ""

        # 去除首尾空白
        text = text.strip()

        # 将连续的多个空行替换为单个空行
        import re
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # 去除过多的空格
        text = re.sub(r' +', ' ', text)

        return text

    def _format_as_markdown(self, text: str, images: List[Dict[str, Any]] = None) -> str:
        """
        将内容格式化为 Markdown

        Args:
            text: 文本内容
            images: 图片列表

        Returns:
            str: Markdown 格式的内容
        """
        markdown_parts = []

        # 添加文本内容
        if text:
            markdown_parts.append(text)

        # 添加图片引用
        if images:
            markdown_parts.append("\n\n## 文档中的图片\n")
            for idx, img in enumerate(images, 1):
                markdown_parts.append(f"\n![图片 {idx}]({img['filename']})\n")
                markdown_parts.append(f"*图片信息*: {img['filename']} ({img['file_size']} bytes)\n")

        return "\n".join(markdown_parts)

    def _format_as_json(self, text: str, images: List[Dict[str, Any]] = None) -> str:
        """
        将内容格式化为 JSON

        Args:
            text: 文本内容
            images: 图片列表

        Returns:
            str: JSON 格式的内容
        """
        import json

        output = {
            'text': text,
            'images': images,
            'image_count': len(images) if images else 0
        }

        return json.dumps(output, ensure_ascii=False, indent=2)

    def _build_metadata(
        self,
        file_path: str,
        text_content: str,
        images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        构建元数据

        Args:
            file_path: 文件路径
            text_content: 文本内容
            images: 图片列表

        Returns:
            Dict: 元数据字典
        """
        file_path_obj = Path(file_path)

        # 统计文本信息
        words = text_content.split() if text_content else []
        chars = len(text_content) if text_content else 0
        lines = text_content.count('\n') + 1 if text_content else 0

        metadata = {
            'file_type': 'doc',
            'parser_version': '1.0',
            'format': 'Legacy Word (Word 97-2003)',
            'word_count': len(words),
            'char_count': chars,
            'line_count': lines,
            'image_count': len(images),
            'has_images': len(images) > 0,
            'file_size': file_path_obj.stat().st_size if file_path_obj.exists() else 0,
            'images_extracted': len(images)
        }

        # 尝试从文件名提取标题
        metadata['title'] = file_path_obj.stem

        return metadata

    def get_parser_info(self) -> Dict[str, Any]:
        """
        获取解析器信息

        Returns:
            Dict: 解析器信息
        """
        return {
            'name': self.parser_name,
            'supported_extensions': self.supported_extensions,
            'description': 'Legacy Word document parser for .doc format (Word 97-2003)',
            'library': 'docx2txt',
            'library_available': DOCX2TXT_AVAILABLE,
            'capabilities': {
                'extract_text': True,
                'extract_images': True,
                'preserve_formatting': False,
                'support_old_format': True
            },
            'config': {
                'extract_images': self.extract_images,
                'enable_ocr': self.enable_ocr,
                'preserve_formatting': self.preserve_formatting
            }
        }
