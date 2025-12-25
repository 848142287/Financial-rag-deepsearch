"""
PDF文档解析器
处理.pdf格式的文档
"""

import logging
import os
import re
from typing import Dict, List, Any, Optional
from .base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not available, PDF parsing will be limited")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available, enhanced PDF parsing will be limited")


class PDFParser(BaseFileParser):
    """PDF文档解析器"""

    def __init__(self):
        super().__init__()
        self._supported_extensions = ['.pdf']
        self.supported_mime_types = ['application/pdf']

    @property
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名列表"""
        return self._supported_extensions

    @property
    def parser_name(self) -> str:
        """解析器名称"""
        return "PDFParser"

    def can_parse(self, file_path: str, file_extension: str = None) -> bool:
        """检查是否能解析指定文件"""
        if file_extension:
            return file_extension.lower() in self._supported_extensions
        return any(file_path.lower().endswith(ext) for ext in self._supported_extensions)

    def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        解析PDF文档

        Args:
            file_path: 文件路径
            **kwargs: 其他参数

        Returns:
            ParseResult: 解析结果
        """
        try:
            import os
            file_size = os.path.getsize(file_path)

            # 优先使用pdfplumber（功能更强）
            if PDFPLUMBER_AVAILABLE:
                return self._parse_with_pdfplumber(file_path, file_size)
            elif PYPDF2_AVAILABLE:
                return self._parse_with_pypdf2(file_path, file_size)
            else:
                return self._parse_basic(file_path, file_size)

        except Exception as e:
            logger.error(f"PDF文档解析失败: {e}")
            return ParseResult(
                content="",
                metadata={'file_type': 'pdf', 'error': str(e)},
                success=False,
                error_message=str(e)
            )

    def _parse_with_pdfplumber(self, file_path: str, file_size: int) -> ParseResult:
        """使用pdfplumber解析PDF"""
        try:
            import pdfplumber

            content_parts = []
            pages_data = []
            tables_data = []
            images_data = []

            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # 提取文本
                        text = page.extract_text()
                        if text and text.strip():
                            content_parts.append(f"第{page_num}页:\n{text}")

                        # 分析页面数据
                        page_data = {
                            'page_number': page_num,
                            'text_length': len(text) if text else 0,
                            'bbox': page.bbox,
                            'width': page.width,
                            'height': page.height
                        }

                        # 检查表格
                        tables = page.extract_tables()
                        if tables:
                            page_data['tables_count'] = len(tables)
                            for i, table in enumerate(tables):
                                table_text = self._format_table(table)
                                tables_data.append({
                                    'page': page_num,
                                    'table_number': i + 1,
                                    'content': table_text
                                })

                        # 检查图片（简化版）
                        if hasattr(page, 'images'):
                            images = page.images
                            if images:
                                page_data['images_count'] = len(images)
                                images_data.append({
                                    'page': page_num,
                                    'images_count': len(images)
                                })

                        pages_data.append(page_data)

                    except Exception as e:
                        logger.warning(f"解析第{page_num}页时出错: {e}")
                        content_parts.append(f"第{page_num}页: 解析失败 - {str(e)}")

            content = "\n\n".join(content_parts)

            # 提取文档元数据
            metadata = {
                'file_type': 'pdf',
                'file_size': file_size,
                'pages': len(pages_data),
                'total_text_length': sum(p['text_length'] for p in pages_data),
                'pages_data': pages_data,
                'tables': tables_data,
                'images': images_data,
                'parsing_method': 'pdfplumber'
            }

            # 尝试提取文档信息
            metadata.update(self._extract_pdf_metadata(file_path))

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error(f"pdfplumber解析失败: {e}")
            # 降级到PyPDF2
            if PYPDF2_AVAILABLE:
                return self._parse_with_pypdf2(file_path, file_size)
            else:
                return self._parse_basic(file_path, file_size)

    def _parse_with_pypdf2(self, file_path: str, file_size: int) -> ParseResult:
        """使用PyPDF2解析PDF"""
        try:
            import PyPDF2

            content_parts = []
            pages_data = []

            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()

                        if text and text.strip():
                            content_parts.append(f"第{page_num + 1}页:\n{text}")

                        pages_data.append({
                            'page_number': page_num + 1,
                            'text_length': len(text) if text else 0,
                            'rotation': page.rotation if hasattr(page, 'rotation') else 0
                        })

                    except Exception as e:
                        logger.warning(f"PyPDF2解析第{page_num + 1}页时出错: {e}")
                        content_parts.append(f"第{page_num + 1}页: 解析失败 - {str(e)}")

            content = "\n\n".join(content_parts)

            # 提取元数据
            metadata = {
                'file_type': 'pdf',
                'file_size': file_size,
                'pages': len(pages_data),
                'total_text_length': sum(p['text_length'] for p in pages_data),
                'pages_data': pages_data,
                'tables': [],  # PyPDF2不支持表格提取
                'images': [],  # PyPDF2不支持图片提取
                'parsing_method': 'PyPDF2'
            }

            # 提取PDF元数据
            if pdf_reader.metadata:
                metadata['pdf_metadata'] = dict(pdf_reader.metadata)

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error(f"PyPDF2解析失败: {e}")
            return self._parse_basic(file_path, file_size)

    def _parse_basic(self, file_path: str, file_size: int) -> ParseResult:
        """基础PDF解析（当其他库不可用时）"""
        import os

        content = f"PDF文档: {os.path.basename(file_path)}\n"
        content += f"文件大小: {file_size} 字节\n"
        content += "注意: 由于缺少PDF解析库（PyPDF2或pdfplumber），无法提取内容\n"
        content += "建议安装依赖:\n"
        content += "  pip install PyPDF2 pdfplumber\n"
        content += "  或者: pip install PyMuPDF"

        metadata = {
            'file_type': 'pdf',
            'file_size': file_size,
            'parsing_limited': True,
            'recommendation': 'Install PDF parsing libraries',
            'missing_libraries': []
        }

        if not PYPDF2_AVAILABLE:
            metadata['missing_libraries'].append('PyPDF2')
        if not PDFPLUMBER_AVAILABLE:
            metadata['missing_libraries'].append('pdfplumber')

        return ParseResult(
            content=content,
            metadata=metadata,
            success=True
        )

    def _format_table(self, table: List[List[str]]) -> str:
        """格式化表格内容"""
        if not table:
            return ""

        formatted_lines = []
        for row in table:
            # 清理和格式化行数据
            clean_row = []
            for cell in row:
                if cell is not None:
                    clean_cell = str(cell).strip()
                    # 处理换行符
                    clean_cell = clean_cell.replace('\n', ' ').replace('\r', '')
                    clean_row.append(clean_cell)
                else:
                    clean_row.append('')
            formatted_lines.append(" | ".join(clean_row))

        return "\n".join(formatted_lines)

    def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取PDF文件元数据"""
        metadata = {}

        try:
            import os
            from datetime import datetime

            # 文件基本信息
            stat = os.stat(file_path)
            metadata['file_created'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
            metadata['file_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()

            # 尝试从文件名提取信息
            filename = os.path.basename(file_path)
            metadata['filename'] = filename

            # 尝试从文件名提取日期
            date_patterns = [
                r'(\d{4})\D*(\d{1,2})\D*(\d{1,2})',  # YYYYMMDD
                r'(\d{4})\D*(\d{1,2})',  # YYYYMM
                r'(\d{4})'  # YYYY
            ]

            for pattern in date_patterns:
                match = re.search(pattern, filename)
                if match:
                    metadata['extracted_date'] = match.group(0)
                    break

            # 尝试提取券商信息
            broker_names = [
                '国泰君安', '国信证券', '华泰证券', '招商证券', '海通证券',
                '中信证券', '申万宏源', '中金公司', '广发证券', '安信证券',
                '光大证券', '东方证券', '兴业证券', '华安证券', '民生证券',
                '平安证券', '国金证券', '华西证券', '中泰证券', '长江证券'
            ]

            for broker in broker_names:
                if broker in filename:
                    metadata['extracted_broker'] = broker
                    break

            # 尝试提取报告类型
            report_types = [
                '研报', '报告', '策略', '分析', '研究', '周报', '月报',
                '年报', '季报', '深度报告', '专题研究'
            ]

            extracted_types = []
            for report_type in report_types:
                if report_type in filename:
                    extracted_types.append(report_type)

            if extracted_types:
                metadata['extracted_types'] = extracted_types

        except Exception as e:
            logger.warning(f"提取PDF元数据失败: {e}")

        return metadata

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