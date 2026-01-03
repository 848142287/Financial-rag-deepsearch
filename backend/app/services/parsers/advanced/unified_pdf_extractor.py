"""
统一 PDF 提取器
提供快速和精确两种 PDF 解析模式
"""

import os
import fitz  # PyMuPDF
from dataclasses import dataclass
import pypdf

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

@dataclass
class PDFExtractionResult:
    """PDF 提取结果"""
    success: bool
    text: str = ""
    pages: int = 0
    metadata: Dict[str, Any] = None
    images: List[Dict[str, Any]] = None
    tables: List[Dict[str, Any]] = None
    error: str = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.images is None:
            self.images = []
        if self.tables is None:
            self.tables = []

class UnifiedPDFExtractor:
    """统一 PDF 提取器"""

    def __init__(self):
        self.logger = logger

    async def extract_fast(
        self,
        file_path: str,
        extract_images: bool = False,
        extract_tables: bool = False
    ) -> PDFExtractionResult:
        """
        快速提取模式 - 使用 PyPDF2，速度快但精度一般

        Args:
            file_path: PDF 文件路径
            extract_images: 是否提取图片
            extract_tables: 是否提取表格

        Returns:
            PDFExtractionResult: 提取结果
        """
        try:
            self.logger.info(f"快速提取模式: {file_path}")

            # 使用 pypdf 快速提取文本
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)

                # 获取元数据
                metadata = self._extract_metadata_pypdf(pdf_reader)

                # 提取文本
                text_content = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            text_content.append({
                                'page': page_num + 1,
                                'text': text
                            })
                    except Exception as e:
                        self.logger.warning(f"提取第 {page_num + 1} 页失败: {e}")

                # 合并文本
                full_text = "\n\n".join([page['text'] for page in text_content])

                return PDFExtractionResult(
                    success=True,
                    text=full_text,
                    pages=len(pdf_reader.pages),
                    metadata=metadata
                )

        except Exception as e:
            self.logger.error(f"快速提取失败: {e}")
            return PDFExtractionResult(
                success=False,
                error=str(e)
            )

    async def extract_full(
        self,
        file_path: str,
        extract_images: bool = True,
        extract_tables: bool = True,
        ocr_enabled: bool = False
    ) -> PDFExtractionResult:
        """
        完整提取模式 - 使用 PyMuPDF，精度高且支持更多功能

        Args:
            file_path: PDF 文件路径
            extract_images: 是否提取图片
            extract_tables: 是否提取表格
            ocr_enabled: 是否启用 OCR

        Returns:
            PDFExtractionResult: 提取结果
        """
        try:
            self.logger.info(f"完整提取模式: {file_path}")

            # 使用 PyMuPDF (fitz) 提取
            doc = fitz.open(file_path)

            # 提取元数据
            metadata = self._extract_metadata_fitx(doc)

            # 提取文本和图片
            text_content = []
            images = []
            tables = []

            for page_num in range(doc.page_count):
                page = doc[page_num]

                # 提取文本
                text = page.get_text("text")
                if text.strip():
                    text_content.append({
                        'page': page_num + 1,
                        'text': text
                    })

                # 提取图片
                if extract_images:
                    page_images = self._extract_images_from_page(page, page_num + 1)
                    images.extend(page_images)

                # 提取表格（简单实现）
                if extract_tables:
                    page_tables = self._extract_tables_from_page(page, page_num + 1)
                    tables.extend(page_tables)

            # 合并文本
            full_text = "\n\n".join([page['text'] for page in text_content])

            doc.close()

            return PDFExtractionResult(
                success=True,
                text=full_text,
                pages=doc.page_count,
                metadata=metadata,
                images=images if extract_images else [],
                tables=tables if extract_tables else []
            )

        except Exception as e:
            self.logger.error(f"完整提取失败: {e}")
            return PDFExtractionResult(
                success=False,
                error=str(e)
            )

    def _extract_metadata_pypdf(self, pdf_reader) -> Dict[str, Any]:
        """从 PyPDF 提取元数据"""
        metadata = {}

        if pdf_reader.metadata:
            metadata = {
                'title': pdf_reader.metadata.get('/Title', ''),
                'author': pdf_reader.metadata.get('/Author', ''),
                'subject': pdf_reader.metadata.get('/Subject', ''),
                'creator': pdf_reader.metadata.get('/Creator', ''),
                'producer': pdf_reader.metadata.get('/Producer', ''),
                'creation_date': pdf_reader.metadata.get('/CreationDate', ''),
            }

        metadata['pages'] = len(pdf_reader.pages)
        return metadata

    def _extract_metadata_fitx(self, doc) -> Dict[str, Any]:
        """从 PyMuPDF 提取元数据"""
        metadata = {
            'pages': doc.page_count,
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'keywords': doc.metadata.get('keywords', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
        }
        return metadata

    def _extract_images_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """从页面提取图片"""
        images = []
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)

                if base_image:
                    image_info = {
                        'page': page_num,
                        'index': img_index,
                        'xref': xref,
                        'width': base_image.get('width', 0),
                        'height': base_image.get('height', 0),
                        'format': base_image.get('ext', ''),
                        'size': len(base_image.get('image', b''))
                    }
                    images.append(image_info)
            except Exception as e:
                self.logger.warning(f"提取图片失败: {e}")

        return images

    def _extract_tables_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """从页面提取表格（简单实现）"""
        # PyMuPDF 不直接支持表格提取，这里返回空列表
        # 实际项目中可以集成 pdfplumber 或 camelot
        return []

    async def extract_from_bytes(
        self,
        file_bytes: bytes,
        mode: str = 'fast',
        **kwargs
    ) -> PDFExtractionResult:
        """
        从字节流提取 PDF 内容

        Args:
            file_bytes: PDF 文件字节流
            mode: 提取模式 ('fast' 或 'full')
            **kwargs: 其他参数

        Returns:
            PDFExtractionResult: 提取结果
        """
        import tempfile

        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        try:
            # 根据模式选择提取方法
            if mode == 'fast':
                result = await self.extract_fast(tmp_file_path, **kwargs)
            else:
                result = await self.extract_full(tmp_file_path, **kwargs)

            return result

        finally:
            # 删除临时文件
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

# 全局实例
unified_pdf_extractor = UnifiedPDFExtractor()

__all__ = ['UnifiedPDFExtractor', 'unified_pdf_extractor', 'PDFExtractionResult']
