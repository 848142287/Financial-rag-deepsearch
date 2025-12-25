"""
增强PDF处理器
专为金融文档优化的PDF解析服务，支持高级OCR、图表提取、公式解析等
"""

import os
import re
import tempfile
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import easyocr
import pandas as pd

# 导入专用解析器
from .specialized_parsers import (
    FinancialChartAnalyzer, FinancialFormulaAnalyzer,
    analyze_financial_chart, analyze_financial_formula
)
from .financial_ocr_enhancer import (
    FinancialOCREnhancer, DocumentType,
    enhance_financial_document_ocr
)

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """内容类型"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    FORMULA = "formula"
    HEADER = "header"
    FOOTER = "footer"


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    pages_processed: int = 0
    total_pages: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    contents: List['ContentBlock'] = field(default_factory=list)


@dataclass
class ContentBlock:
    """内容块"""
    content_type: ContentType
    page_number: int
    text: str
    bbox: List[float]  # [x0, y0, x1, y1]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_path: Optional[str] = None
    table_data: Optional[Dict[str, Any]] = None


class EnhancedPDFProcessor:
    """增强PDF处理器"""

    def __init__(self):
        self.ocr_reader = None
        self.temp_dir = None
        self.chart_detector = None

    async def initialize(self):
        """初始化处理器"""
        try:
            # 初始化OCR
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'])

            # 创建临时目录
            self.temp_dir = tempfile.mkdtemp(prefix="enhanced_pdf_")

            # 初始化图表检测器
            self.chart_detector = ChartDetector()
            await self.chart_detector.initialize()

            logger.info("Enhanced PDF processor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize enhanced PDF processor: {e}")
            raise

    async def process_pdf_document(
        self,
        pdf_path: str,
        config: Dict[str, Any]
    ) -> ProcessingResult:
        """
        处理PDF文档

        Args:
            pdf_path: PDF文件路径
            config: 处理配置

        Returns:
            ProcessingResult: 处理结果
        """
        import time
        start_time = time.time()

        try:
            if not os.path.exists(pdf_path):
                return ProcessingResult(
                    success=False,
                    errors=[f"PDF file not found: {pdf_path}"]
                )

            # 打开PDF
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)

            result = ProcessingResult(
                success=True,
                total_pages=total_pages
            )

            # 处理每一页
            for page_num in range(total_pages):
                try:
                    page_content = await self._process_page(
                        pdf_document, page_num, config
                    )
                    result.contents.extend(page_content)
                    result.pages_processed += 1

                except Exception as e:
                    error_msg = f"Error processing page {page_num + 1}: {e}"
                    result.errors.append(error_msg)
                    logger.warning(error_msg)

            # 提取文档元数据
            result.metadata = await self._extract_metadata(pdf_document, config)
            result.processing_time = time.time() - start_time

            pdf_document.close()

            logger.info(f"PDF processing completed: {result.pages_processed}/{result.total_pages} pages, "
                       f"{len(result.contents)} content blocks, {result.processing_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return ProcessingResult(
                success=False,
                errors=[str(e)],
                processing_time=time.time() - start_time
            )

    async def _process_page(
        self,
        pdf_document: fitz.Document,
        page_num: int,
        config: Dict[str, Any]
    ) -> List[ContentBlock]:
        """处理单个PDF页面"""
        page = pdf_document[page_num]
        page_size = page.rect  # 页面尺寸

        content_blocks = []

        try:
            # 1. 提取原始文本
            text_blocks = await self._extract_text_blocks(page, page_num)
            content_blocks.extend(text_blocks)

            # 2. 提取表格
            if config.get("extract_tables", True):
                table_blocks = await self._extract_tables(page, page_num)
                content_blocks.extend(table_blocks)

            # 3. 提取图像和图表
            if config.get("extract_images", True):
                image_blocks = await self._extract_images(
                    page, page_num, pdf_document, config
                )
                content_blocks.extend(image_blocks)

            # 4. 检测公式
            if config.get("extract_formulas", True):
                formula_blocks = await self._detect_formulas(page, page_num)
                content_blocks.extend(formula_blocks)

            # 5. 检测图表
            if config.get("extract_charts", True):
                chart_blocks = await self._detect_charts(page, page_num)
                content_blocks.extend(chart_blocks)

        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")

        return content_blocks

    async def _extract_text_blocks(
        self, page: fitz.Page, page_num: int
    ) -> List[ContentBlock]:
        """提取文本块"""
        content_blocks = []

        try:
            # 获取文本块
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                bbox = list(span["bbox"])

                                # 判断是否是标题
                                is_header = self._is_header(span)
                                content_type = ContentType.HEADER if is_header else ContentType.TEXT

                                content_block = ContentBlock(
                                    content_type=content_type,
                                    page_number=page_num + 1,
                                    text=text,
                                    bbox=bbox,
                                    confidence=1.0,
                                    metadata={
                                        "font_size": span.get("size", 0),
                                        "font_flags": span.get("flags", 0),
                                        "font_name": span.get("font", ""),
                                        "color": span.get("color", 0)
                                    }
                                )
                                content_blocks.append(content_block)

        except Exception as e:
            logger.error(f"Error extracting text blocks: {e}")

        return content_blocks

    async def _extract_tables(
        self, page: fitz.Page, page_num: int
    ) -> List[ContentBlock]:
        """提取表格"""
        content_blocks = []

        try:
            # 查找表格
            tables = page.find_tables()

            for table in tables:
                try:
                    # 提取表格数据
                    table_data = table.extract()

                    # 转换为DataFrame
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])

                    # 生成表格文本
                    table_text = self._table_to_text(df)

                    content_block = ContentBlock(
                        content_type=ContentType.TABLE,
                        page_number=page_num + 1,
                        text=table_text,
                        bbox=list(table.bbox),
                        confidence=0.9,
                        table_data={
                            "headers": table_data[0] if table_data else [],
                            "rows": len(table_data) - 1 if table_data else 0,
                            "cols": len(table_data[0]) if table_data else 0,
                            "data": df.to_dict('records') if not df.empty else []
                        },
                        metadata={
                            "extraction_method": "pymupdf_table_finder"
                        }
                    )
                    content_blocks.append(content_block)

                except Exception as e:
                    logger.warning(f"Error extracting table: {e}")

        except Exception as e:
            logger.error(f"Error in table extraction: {e}")

        return content_blocks

    async def _extract_images(
        self,
        page: fitz.Page,
        page_num: int,
        pdf_document: fitz.Document,
        config: Dict[str, Any]
    ) -> List[ContentBlock]:
        """提取图像"""
        content_blocks = []

        try:
            # 获取图像列表
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                try:
                    # 获取图像
                    xref = img_info[0]
                    pix = fitz.Pixmap(pdf_document, xref)

                    if pix.n - pix.alpha < 4:  # 确保是RGB或灰度图像
                        # 保存图像
                        image_path = os.path.join(
                            self.temp_dir,
                            f"page_{page_num + 1}_img_{img_index}.png"
                        )
                        pix.save(image_path)

                        # 判断是否是图表
                        is_chart = await self._is_chart_image(image_path, config)
                        content_type = ContentType.CHART if is_chart else ContentType.IMAGE

                        content_block = ContentBlock(
                            content_type=content_type,
                            page_number=page_num + 1,
                            text=f"[{'图表' if is_chart else '图片'} 第{img_index + 1}个]",
                            bbox=[0, 0, 0, 0],  # 图像的位置需要单独计算
                            confidence=0.8,
                            image_path=image_path,
                            metadata={
                                "width": pix.width,
                                "height": pix.height,
                                "colorspace": pix.colorspace,
                                "is_chart": is_chart,
                                "image_type": "extracted_from_pdf"
                            }
                        )
                        content_blocks.append(content_block)

                    pix = None

                except Exception as e:
                    logger.warning(f"Error extracting image {img_index}: {e}")

        except Exception as e:
            logger.error(f"Error in image extraction: {e}")

        return content_blocks

    async def _detect_formulas(
        self, page: fitz.Page, page_num: int
    ) -> List[ContentBlock]:
        """检测公式"""
        content_blocks = []

        try:
            # 获取页面文本
            text = page.get_text()

            # 公式正则表达式模式
            formula_patterns = [
                # LaTeX风格公式
                r'\$.*?\$',
                r'\\\(.*?\\\)',
                r'\\\[.*?\\\]',
                # 数学表达式
                r'[a-zA-Z]+\s*[=≠<>≤≥]\s*[^.]+\n',
                r'[∂∫∑∏√±±∓×÷≤≥≠∞∂∇∆]',
                # 金融公式模式
                r'[A-Z]+\s*[=]\s*[^.]+\s*[+\-*/]\s*[^.]+',
                r'[ROC|ROI|P/E|EPS|EBITDA].*?[=].*?[%]',
            ]

            for pattern in formula_patterns:
                matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)

                for match in matches:
                    formula_text = match.group().strip()

                    # 过滤太短的匹配
                    if len(formula_text) < 3:
                        continue

                    # 检查是否真的是公式
                    if self._is_formula(formula_text):
                        content_block = ContentBlock(
                            content_type=ContentType.FORMULA,
                            page_number=page_num + 1,
                            text=formula_text,
                            bbox=[0, 0, 0, 0],  # 需要通过文本搜索获取精确位置
                            confidence=0.7,
                            metadata={
                                "formula_type": "mathematical",
                                "extraction_method": "regex_pattern"
                            }
                        )
                        content_blocks.append(content_block)

        except Exception as e:
            logger.error(f"Error in formula detection: {e}")

        return content_blocks

    async def _detect_charts(
        self, page: fitz.Page, page_num: int
    ) -> List[ContentBlock]:
        """检测图表"""
        content_blocks = []

        try:
            # 获取页面文本
            text = page.get_text()

            # 图表关键词模式
            chart_keywords = [
                '图表', '图形', '走势图', '柱状图', '饼图', '折线图',
                '散点图', '雷达图', '热力图', '趋势', '对比', '分布',
                'Figure', 'Chart', 'Graph', 'Plot', 'Diagram'
            ]

            lines = text.split('\n')
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # 检查是否包含图表关键词
                for keyword in chart_keywords:
                    if keyword in line:
                        # 检查周围文本是否描述图表
                        context = self._get_chart_context(lines, line_num)
                        if self._is_chart_description(context):
                            content_block = ContentBlock(
                                content_type=ContentType.CHART,
                                page_number=page_num + 1,
                                text=line,
                                bbox=[0, 0, 0, 0],
                                confidence=0.6,
                                metadata={
                                    "chart_keyword": keyword,
                                    "context": context,
                                    "extraction_method": "text_pattern"
                                }
                            )
                            content_blocks.append(content_block)
                        break

        except Exception as e:
            logger.error(f"Error in chart detection: {e}")

        return content_blocks

    def _is_header(self, span: Dict[str, Any]) -> bool:
        """判断是否是标题"""
        font_size = span.get("size", 0)
        font_flags = span.get("flags", 0)

        # 根据字体大小和样式判断
        is_large_font = font_size > 14
        is_bold = (font_flags & 2**4) != 0  # 粗体标志

        return is_large_font or is_bold

    def _table_to_text(self, df: pd.DataFrame) -> str:
        """将DataFrame转换为文本"""
        if df.empty:
            return ""

        # 创建表格文本
        lines = []

        # 表头
        headers = df.columns.tolist()
        lines.append(" | ".join(headers))
        lines.append("-" * len(" | ".join(headers)))

        # 数据行
        for _, row in df.iterrows():
            row_text = " | ".join([str(cell) for cell in row])
            lines.append(row_text)

        return "\n".join(lines)

    async def _is_chart_image(self, image_path: str, config: Dict[str, Any]) -> bool:
        """判断图像是否是图表"""
        try:
            # 使用图表检测器
            return await self.chart_detector.is_chart(image_path)

        except Exception as e:
            logger.warning(f"Error in chart detection: {e}")
            return False

    def _is_formula(self, text: str) -> bool:
        """判断文本是否是公式"""
        # 公式特征检查
        formula_indicators = [
            '=', '+', '-', '*', '/', '^', '√', '∑', '∫', '∂', '∞',
            'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'max', 'min',
            'α', 'β', 'γ', 'δ', 'θ', 'λ', 'μ', 'σ', 'π', 'φ'
        ]

        indicator_count = sum(1 for indicator in formula_indicators if indicator in text)

        # 如果包含多个公式指示符，认为是公式
        return indicator_count >= 2

    def _get_chart_context(self, lines: List[str], line_num: int) -> str:
        """获取图表的上下文"""
        # 获取前后几行作为上下文
        start = max(0, line_num - 2)
        end = min(len(lines), line_num + 3)

        context_lines = lines[start:end]
        return "\n".join(context_lines)

    def _is_chart_description(self, context: str) -> bool:
        """判断是否是图表描述"""
        chart_description_indicators = [
            '显示', '表明', '展示', '说明', '反映', '对比',
            '增长', '下降', '上升', '趋势', '变化', '分布',
            'shows', 'indicates', 'demonstrates', 'illustrates'
        ]

        return any(indicator in context for indicator in chart_description_indicators)

    async def _extract_metadata(
        self, pdf_document: fitz.Document, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """提取PDF元数据"""
        try:
            metadata = pdf_document.metadata

            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": len(pdf_document),
                "encrypted": pdf_document.is_encrypted,
                "pdf_version": pdf_document.pdf_version(),
                "file_size": os.path.getsize(pdf_document.name) if hasattr(pdf_document, 'name') else 0
            }

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}

    def cleanup(self):
        """清理临时文件"""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class ChartDetector:
    """图表检测器"""

    def __init__(self):
        self.initialized = False

    async def initialize(self):
        """初始化图表检测器"""
        try:
            # 这里可以加载预训练的图表检测模型
            self.initialized = True
            logger.info("Chart detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize chart detector: {e}")

    async def is_chart(self, image_path: str) -> bool:
        """判断图像是否是图表"""
        try:
            if not self.initialized:
                return False

            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                return False

            # 简单的图表检测启发式规则
            return self._detect_chart_heuristics(image)

        except Exception as e:
            logger.error(f"Error in chart detection: {e}")
            return False

    def _detect_chart_heuristics(self, image: np.ndarray) -> bool:
        """基于启发式规则的图表检测"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)

            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 统计轮廓特征
            straight_lines = 0
            curves = 0

            for contour in contours:
                # 近似轮廓
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) <= 4:  # 直线多边形
                    straight_lines += 1
                else:  # 曲线
                    curves += 1

            # 图表通常包含较多的直线和曲线
            total_contours = len(contours)
            if total_contours > 10:
                line_ratio = straight_lines / total_contours
                curve_ratio = curves / total_contours

                # 如果直线或曲线比例较高，可能是图表
                return line_ratio > 0.3 or curve_ratio > 0.2

            return False

        except Exception as e:
            logger.error(f"Error in heuristic chart detection: {e}")
            return False


# 全局实例
enhanced_pdf_processor = EnhancedPDFProcessor()


async def get_enhanced_pdf_processor() -> EnhancedPDFProcessor:
    """获取增强PDF处理器实例"""
    if not enhanced_pdf_processor.initialized:
        await enhanced_pdf_processor.initialize()
    return enhanced_pdf_processor


# 便捷函数
async def process_pdf_document(
    pdf_path: str,
    config: Dict[str, Any]
) -> ProcessingResult:
    """便捷的PDF处理函数"""
    processor = await get_enhanced_pdf_processor()
    return await processor.process_pdf_document(pdf_path, config)