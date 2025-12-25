"""
ReportLab文档转换器
主要用于生成PDF文档和处理复杂的文档布局
"""

import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
from io import BytesIO

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("ReportLab未安装，相关功能不可用")

from .base_converter import BaseConverter, ConversionResult, ConversionStatus

logger = logging.getLogger(__name__)


class ReportLabConverter(BaseConverter):
    """ReportLab转换器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab未安装，请运行: pip install reportlab")

        super().__init__(config)

        # PDF生成参数
        self.page_size = getattr(self, 'page_size', A4)
        self.margin = self.config.get('margin', 0.75 * inch)
        self.font_size = self.config.get('font_size', 12)

        # 样式设置
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """设置自定义样式"""
        # 标题样式
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        # 正文样式
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=self.font_size,
            spaceAfter=12,
            alignment=TA_LEFT
        ))

    async def convert(
        self,
        input_path: str,
        output_format: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> ConversionResult:
        """转换文档"""
        start_time = time.time()

        try:
            # 验证输入文件
            if not self.validate_input_file(input_path):
                return ConversionResult(
                    status=ConversionStatus.FAILED,
                    error_message="输入文件验证失败"
                )

            # 生成输出路径
            output_path = self.generate_output_path(input_path, output_format, output_path)

            # 根据输入格式选择转换方法
            input_ext = Path(input_path).suffix.lower()

            if output_format.lower() == 'pdf':
                if input_ext in ['.txt', '.md']:
                    await self._text_to_pdf(input_path, output_path, **kwargs)
                elif input_ext in ['.html', '.htm']:
                    await self._html_to_pdf(input_path, output_path, **kwargs)
                else:
                    return ConversionResult(
                        status=ConversionStatus.FAILED,
                        error_message=f"不支持从 {input_ext} 转换为PDF"
                    )
            else:
                return ConversionResult(
                    status=ConversionStatus.FAILED,
                    error_message=f"不支持的输出格式: {output_format}"
                )

            conversion_time = time.time() - start_time

            if os.path.exists(output_path):
                logger.info(f"转换成功: {output_path}")

                return ConversionResult(
                    status=ConversionStatus.COMPLETED,
                    output_path=output_path,
                    output_format=output_format,
                    metadata=self.get_file_metadata(output_path),
                    conversion_time=conversion_time,
                    file_size=os.path.getsize(output_path)
                )
            else:
                return ConversionResult(
                    status=ConversionStatus.FAILED,
                    error_message="输出文件未生成",
                    conversion_time=conversion_time
                )

        except Exception as e:
            logger.error(f"转换失败: {str(e)}")
            return ConversionResult(
                status=ConversionStatus.FAILED,
                error_message=str(e),
                conversion_time=time.time() - start_time
            )

    async def _text_to_pdf(self, input_path: str, output_path: str, **kwargs):
        """将文本文件转换为PDF"""
        # 读取文本内容
        with open(input_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # 创建PDF文档
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            leftMargin=self.margin,
            rightMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin
        )

        # 构建内容
        story = []

        # 添加标题（如果提供）
        title = kwargs.get('title', Path(input_path).stem)
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 12))

        # 处理文本内容
        paragraphs = text_content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # 处理Markdown格式（如果有）
                if input_path.endswith('.md'):
                    para = self._process_markdown(para)

                story.append(Paragraph(para, self.styles['CustomBody']))
                story.append(Spacer(1, 6))

        # 生成PDF
        doc.build(story)

    async def _html_to_pdf(self, input_path: str, output_path: str, **kwargs):
        """将HTML文件转换为PDF"""
        # 读取HTML内容
        with open(input_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # 简单的HTML到PDF转换（这里简化处理）
        # 实际项目中可能需要使用更专业的HTML解析器
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
        except ImportError:
            # 如果没有BeautifulSoup，直接提取文本
            import re
            text = re.sub('<[^<]+?>', '', html_content)
            await self._text_to_pdf(input_path, output_path, content=text)
            return

        # 创建PDF文档
        doc = SimpleDocTemplate(output_path, pagesize=self.page_size)

        story = []

        # 处理HTML元素
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'img']):
            if element.name == 'h1':
                story.append(Paragraph(element.get_text(), self.styles['CustomTitle']))
            elif element.name == 'h2':
                story.append(Paragraph(element.get_text(), self.styles['Heading2']))
            elif element.name == 'h3':
                story.append(Paragraph(element.get_text(), self.styles['Heading3']))
            elif element.name == 'p':
                story.append(Paragraph(element.get_text(), self.styles['CustomBody']))
            elif element.name in ['ul', 'ol']:
                # 处理列表
                list_items = element.find_all('li')
                for li in list_items:
                    story.append(Paragraph(f"• {li.get_text()}", self.styles['CustomBody']))
            elif element.name == 'img':
                # 处理图片
                src = element.get('src')
                if src and os.path.exists(src):
                    img = Image(src, width=4*inch, height=3*inch)
                    story.append(img)

            story.append(Spacer(1, 6))

        # 生成PDF
        doc.build(story)

    def _process_markdown(self, text: str) -> str:
        """简单的Markdown处理"""
        import re

        # 处理标题
        text = re.sub(r'^# (.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

        # 处理粗体
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

        # 处理斜体
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)

        return text

    async def create_pdf_from_content(
        self,
        content: List[Dict[str, Any]],
        output_path: str,
        **kwargs
    ) -> ConversionResult:
        """从内容创建PDF文档"""
        start_time = time.time()

        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=self.page_size,
                leftMargin=self.margin,
                rightMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin
            )

            story = []

            for item in content:
                content_type = item.get('type', 'text')

                if content_type == 'text':
                    text = item.get('content', '')
                    style_name = item.get('style', 'CustomBody')
                    story.append(Paragraph(text, self.styles[style_name]))

                elif content_type == 'title':
                    title = item.get('content', '')
                    story.append(Paragraph(title, self.styles['CustomTitle']))

                elif content_type == 'image':
                    img_path = item.get('path')
                    if img_path and os.path.exists(img_path):
                        width = item.get('width', 4*inch)
                        height = item.get('height', 3*inch)
                        img = Image(img_path, width=width, height=height)
                        story.append(img)

                elif content_type == 'table':
                    table_data = item.get('data', [])
                    if table_data:
                        table = Table(table_data)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 14),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(table)

                story.append(Spacer(1, 12))

            # 生成PDF
            doc.build(story)

            conversion_time = time.time() - start_time

            if os.path.exists(output_path):
                return ConversionResult(
                    status=ConversionStatus.COMPLETED,
                    output_path=output_path,
                    output_format='pdf',
                    conversion_time=conversion_time,
                    file_size=os.path.getsize(output_path)
                )
            else:
                return ConversionResult(
                    status=ConversionStatus.FAILED,
                    error_message="PDF生成失败",
                    conversion_time=conversion_time
                )

        except Exception as e:
            logger.error(f"PDF创建失败: {str(e)}")
            return ConversionResult(
                status=ConversionStatus.FAILED,
                error_message=str(e),
                conversion_time=time.time() - start_time
            )

    def get_supported_input_formats(self) -> List[str]:
        """获取支持的输入格式"""
        return [
            '.txt',  # 纯文本
            '.md',   # Markdown
            '.html', '.htm',  # HTML文档
            '.json',  # JSON文件（作为内容数据）
        ]

    def get_supported_output_formats(self) -> List[str]:
        """获取支持的输出格式"""
        return [
            'pdf',  # 主要输出格式
        ]