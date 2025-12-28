"""
PyMuPDF4LLM解析器
专为RAG优化的PDF解析器，自动生成结构化Markdown内容

安装依赖: pip install pymupdf4llm==0.0.7
"""

import os
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)

try:
    import pymupdf4llm
    import fitz  # PyMuPDF
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False
    logger.warning("pymupdf4llm未安装，请运行: pip install pymupdf4llm")


class PyMuPDF4LLMParser(BaseFileParser):
    """
    PyMuPDF4LLM文档解析器

    特点：
    - 专为LLM优化，自动生成GitHub兼容的Markdown
    - 智能内容排序，自动按正确阅读顺序组织文本、表格、图片
    - 识别标题层级、列表、代码块等
    - 速度约0.12秒/页
    - 自动提取和引用图片
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.parser_name = "PyMuPDF4LLM"
        self.supported_extensions = ['.pdf']

        # PyMuPDF4LLM配置
        self.extract_images = self.config.get('extract_images', True)
        self.image_format = self.config.get('image_format', 'png')
        self.image_dpi = self.config.get('image_dpi', 150)
        self.page_chunks = self.config.get('page_chunks', True)
        self.enable_page_screenshots = self.config.get('enable_page_screenshots', False)
        self.output_dir = self.config.get('output_dir', './pymupdf4llm_output')

    def can_parse(self, file_path: str, file_extension: str = None) -> bool:
        """检查是否能解析指定文件"""
        ext = file_extension or Path(file_path).suffix.lower()

        # 检查PyMuPDF4LLM是否可用
        if not PYMUPDF4LLM_AVAILABLE:
            logger.warning("PyMuPDF4LLM not available")
            return False

        return ext in self.supported_extensions

    async def parse(self, file_path: str, **kwargs) -> ParseResult:
        """解析文件内容"""
        if not PYMUPDF4LLM_AVAILABLE:
            error_msg = "PyMuPDF4LLM not installed. Please run: pip install pymupdf4llm"
            return ParseResult(
                content="",
                metadata={'error': error_msg},
                success=False,
                error_message=error_msg
            )

        try:
            # 验证文件
            valid, error_msg = self.validate_file(file_path)
            if not valid:
                return ParseResult(
                    content="",
                    metadata={'error': error_msg},
                    success=False,
                    error_message=error_msg
                )

            start_time = time.time()

            # 创建输出目录（如果需要提取图片）
            images_dir = None
            if self.extract_images:
                output_path = Path(self.output_dir)
                output_path.mkdir(exist_ok=True, parents=True)
                images_dir = output_path / "images"
                images_dir.mkdir(exist_ok=True, parents=True)

            # 使用PyMuPDF4LLM提取Markdown内容
            md_text = await self._extract_markdown(file_path, images_dir)

            # 生成页面截图（如果启用）
            page_screenshots = []
            if self.enable_page_screenshots and images_dir:
                page_screenshots = await self._generate_page_screenshots(file_path, images_dir)

            # 提取文档元数据
            metadata = await self._extract_document_metadata(file_path)
            metadata['images_dir'] = str(images_dir) if images_dir else None
            metadata['page_screenshots'] = page_screenshots
            metadata['extraction_config'] = {
                'extract_images': self.extract_images,
                'image_dpi': self.image_dpi,
                'page_chunks': self.page_chunks
            }

            parse_time = time.time() - start_time
            metadata['parse_time'] = parse_time

            # 清理内容
            content = self.clean_content(md_text)

            self.logger.info(f"Successfully parsed PDF with PyMuPDF4LLM: {file_path}")

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True,
                parse_time=parse_time
            )

        except Exception as e:
            error_msg = f"PyMuPDF4LLM parsing error for {file_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return ParseResult(
                content="",
                metadata={'error': str(e)},
                success=False,
                error_message=error_msg
            )

    async def _extract_markdown(self, pdf_path: str, images_dir: Optional[Path]) -> str:
        """使用PyMuPDF4LLM提取Markdown内容"""
        try:
            # 准备参数
            kwargs = {
                'page_chunks': self.page_chunks,
                'write_images': self.extract_images and images_dir is not None,
                'image_format': self.image_format,
                'dpi': self.image_dpi
            }

            # 如果启用图片提取，设置图片路径
            if self.extract_images and images_dir is not None:
                kwargs['image_path'] = str(images_dir)

            # 调用PyMuPDF4LLM
            md_data = pymupdf4llm.to_markdown(pdf_path, **kwargs)

            # 处理返回的数据
            markdown_content = []

            if isinstance(md_data, list):
                # 如果是列表，遍历每页
                for page_data in md_data:
                    if isinstance(page_data, dict):
                        text = page_data.get('text', '')
                        # 将绝对路径转换为相对路径
                        if images_dir:
                            text = text.replace(str(images_dir.absolute()), 'images')
                        markdown_content.append(text)
                    else:
                        markdown_content.append(str(page_data))
            else:
                # 如果是字符串，直接使用
                text = str(md_data)
                # 将绝对路径转换为相对路径
                if images_dir:
                    text = text.replace(str(images_dir.absolute()), 'images')
                markdown_content.append(text)

            return "".join(markdown_content)

        except Exception as e:
            self.logger.error(f"Failed to extract markdown: {str(e)}")
            raise

    async def _generate_page_screenshots(self, pdf_path: str, images_dir: Path) -> List[Dict[str, Any]]:
        """生成每页的完整截图"""
        screenshots = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]

                try:
                    # 生成2倍分辨率的截图
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_filename = f"page_{page_num + 1}_full.png"
                    img_path = images_dir / img_filename
                    pix.save(str(img_path))

                    screenshots.append({
                        'page_num': page_num + 1,
                        'image_path': f"images/{img_filename}",
                        'width': pix.width,
                        'height': pix.height
                    })

                    # 释放内存
                    pix = None

                except Exception as e:
                    self.logger.warning(f"Failed to generate screenshot for page {page_num + 1}: {e}")

            doc.close()

        except Exception as e:
            self.logger.error(f"Failed to generate page screenshots: {e}")

        return screenshots

    async def _extract_document_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """提取PDF文档元数据"""
        metadata = {}

        try:
            doc = fitz.open(pdf_path)

            # 基本文档信息
            metadata['total_pages'] = len(doc)

            # 提取PDF元数据
            pdf_metadata = doc.metadata
            metadata.update({
                'title': pdf_metadata.get('title', ''),
                'author': pdf_metadata.get('author', ''),
                'subject': pdf_metadata.get('subject', ''),
                'keywords': pdf_metadata.get('keywords', ''),
                'creator': pdf_metadata.get('creator', ''),
                'producer': pdf_metadata.get('producer', ''),
                'creation_date': pdf_metadata.get('creationDate', ''),
                'mod_date': pdf_metadata.get('modDate', ''),
            })

            # 统计信息
            total_chars = 0
            total_images = 0
            total_tables = 0

            for page_num in range(len(doc)):
                page = doc[page_num]

                # 统计文本
                text = page.get_text()
                total_chars += len(text)

                # 统计图片
                try:
                    image_list = page.get_images()
                    total_images += len(image_list)
                except:
                    pass

                # 统计表格
                try:
                    tables = page.find_tables()
                    if tables and hasattr(tables, 'tables'):
                        total_tables += len(tables.tables)
                except:
                    pass

            metadata.update({
                'total_characters': total_chars,
                'total_images': total_images,
                'total_tables': total_tables,
                'avg_chars_per_page': total_chars / len(doc) if len(doc) > 0 else 0
            })

            doc.close()

        except Exception as e:
            self.logger.warning(f"Failed to extract document metadata: {e}")

        return metadata

    def get_parser_info(self) -> Dict[str, Any]:
        """获取解析器信息"""
        info = super().get_parser_info()
        info.update({
            'library_version': 'pymupdf4llm',
            'available': PYMUPDF4LLM_AVAILABLE,
            'features': {
                'markdown_output': True,
                'image_extraction': True,
                'table_extraction': True,
                'page_screenshots': True,
                'automatic_formatting': True,
                'rag_optimized': True
            },
            'performance': {
                'avg_time_per_page': '0.12s',
                'recommended_for': 'RAG applications, document indexing'
            }
        })
        return info
