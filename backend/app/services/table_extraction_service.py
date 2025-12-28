"""
表格提取服务
从PDF文档和图片中提取结构化表格数据
"""

import logging
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TableFormat(Enum):
    """表格格式类型"""
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    HTML = "html"


@dataclass
class TableExtractionConfig:
    """表格提取配置"""
    extract_from_pdf: bool = True
    extract_from_images: bool = True
    use_ocr_for_tables: bool = True
    table_format: TableFormat = TableFormat.MARKDOWN
    max_tables: int = 20


class TableExtractionService:
    """表格提取服务"""

    def __init__(self, config: Optional[TableExtractionConfig] = None):
        self.config = config or TableExtractionConfig()
        self._ocr_service = None

    def _get_ocr_service(self):
        """获取OCR服务"""
        if self._ocr_service is None:
            from app.services.ocr_service import get_ocr_service
            self._ocr_service = get_ocr_service()
        return self._ocr_service

    async def extract_tables_from_pdf(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """从PDF中提取表格"""
        logger.info("从PDF中提取表格...")

        tables = []

        try:
            # 方法1: 使用pdfplumber提取表格
            import pdfplumber
            import io

            pdf_file = io.BytesIO(pdf_bytes)

            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # 提取当前页的表格
                        page_tables = page.extract_tables()

                        if page_tables:
                            for table_idx, table in enumerate(page_tables):
                                # 转换表格数据
                                formatted_table = await self._format_table_data(
                                    table,
                                    page_num=page_num + 1,
                                    table_index=table_idx + 1
                                )

                                if formatted_table:
                                    tables.append(formatted_table)

                    except Exception as e:
                        logger.warning(f"第 {page_num + 1} 页表格提取失败: {e}")
                        continue

            logger.info(f"✅ 使用pdfplumber提取到 {len(tables)} 个表格")
            return tables

        except ImportError:
            logger.warning("pdfplumber未安装")
            return []
        except Exception as e:
            logger.error(f"PDF表格提取失败: {e}")
            # 回退到OCR方法
            return await self._extract_tables_via_ocr(pdf_bytes)

    async def _extract_tables_via_ocr(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """使用OCR提取表格"""
        logger.info("使用OCR提取表格...")

        try:
            # 将PDF转换为图片并使用OCR识别表格
            from pdf2image import convert_from_bytes

            # 转换PDF为图片(限制前5页以节省时间)
            images = await asyncio.to_thread(
                convert_from_bytes,
                pdf_bytes,
                dpi=300,
                first_page=1,
                last_page=5
            )

            ocr_service = self._get_ocr_service()
            tables = []

            for page_num, image in enumerate(images):
                # 转换图片为字节
                from io import BytesIO
                img_buffer = BytesIO()
                image.save(img_buffer, format='PNG')
                image_bytes = img_buffer.getvalue()

                # 使用OCR识别表格
                prompt = """请识别图片中的所有表格数据。

请以JSON格式返回结果：
{
  "tables": [
    {
      "page": 页码,
      "rows": [
        ["列1", "列2", "列3"],
        ["数据1", "数据2", "数据3"]
      ]
    }
  ]
}

注意：
- 保持表格的行列结构
- 空单元格用空字符串表示
- 合并单元格只在第一个单元格保留内容
"""

                result = await ocr_service.extract_text_from_image(image_bytes, prompt)

                if result['success']:
                    try:
                        # 解析JSON结果
                        result_text = result['text']
                        if result_text.startswith('```json'):
                            result_text = result_text[7:]
                        elif result_text.startswith('```'):
                            result_text = result_text[3:]
                        if result_text.endswith('```'):
                            result_text = result_text[:-3]
                        result_text = result_text.strip()

                        data = json.loads(result_text)
                        ocr_tables = data.get('tables', [])

                        for table_data in ocr_tables:
                            tables.append({
                                'page': table_data.get('page', page_num + 1),
                                'rows': table_data.get('rows', []),
                                'method': 'OCR',
                                'format': 'JSON'
                            })

                    except json.JSONDecodeError:
                        logger.warning(f"第 {page_num + 1} 页OCR结果解析失败")

            logger.info(f"✅ OCR提取到 {len(tables)} 个表格")
            return tables[:self.config.max_tables]

        except Exception as e:
            logger.error(f"OCR表格提取失败: {e}")
            return []

    async def _format_table_data(self, table_data, page_num: int, table_index: int) -> Optional[Dict[str, Any]]:
        """格式化表格数据"""
        try:
            if not table_data or len(table_data) == 0:
                return None

            # 转换为列表格式
            rows = []
            for row in table_data:
                if row:
                    formatted_row = []
                    for cell in row:
                        if cell is not None:
                            formatted_row.append(str(cell).strip())
                        else:
                            formatted_row.append("")
                    rows.append(formatted_row)

            if len(rows) == 0:
                return None

            # 检测是否有表头(第一行)
            has_header = len(rows) > 1

            # 生成表格的markdown格式
            markdown_table = self._convert_to_markdown(rows, has_header)

            # 生成CSV格式
            csv_table = self._convert_to_csv(rows)

            return {
                'page': page_num,
                'table_index': table_index,
                'rows': rows,
                'row_count': len(rows),
                'column_count': len(rows[0]) if rows else 0,
                'has_header': has_header,
                'formats': {
                    'markdown': markdown_table,
                    'csv': csv_table,
                    'json': json.dumps(rows, ensure_ascii=False)
                },
                'method': 'pdfplumber'
            }

        except Exception as e:
            logger.error(f"表格格式化失败: {e}")
            return None

    def _convert_to_markdown(self, rows: List[List[str]], has_header: bool = True) -> str:
        """转换为Markdown表格格式"""
        if not rows:
            return ""

        markdown_lines = []
        separator = " | "

        # 表头行
        markdown_lines.append(separator.join(rows[0]))

        # 分隔符行
        markdown_lines.append(separator.join(["---"] * len(rows[0])))

        # 数据行
        for row in rows[1:]:
            markdown_lines.append(separator.join(row))

        return "\n".join(markdown_lines)

    def _convert_to_csv(self, rows: List[List[str]]) -> str:
        """转换为CSV格式"""
        if not rows:
            return ""

        csv_lines = []
        for row in rows:
            csv_lines.append(",".join([f'"{cell}"' for cell in row]))

        return "\n".join(csv_lines)

    async def extract_tables_from_markdown(self, markdown_text: str) -> List[Dict[str, Any]]:
        """从Markdown文本中提取表格"""
        logger.info("从Markdown中提取表格...")

        tables = []
        lines = markdown_text.split('\n')

        current_table_rows = []
        in_table = False

        for i, line in enumerate(lines):
            # 检测表格开始(包含 | 的行)
            if '|' in line and line.strip().startswith('|'):
                in_table = True
                # 移除首尾的 |
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                current_table_rows.append(cells)

            # 检测分隔符行
            elif in_table and set(line.strip()) == {'-', '|', ' '}:
                continue

            # 表格结束
            elif in_table and '|' not in line:
                if current_table_rows:
                    tables.append({
                        'rows': current_table_rows,
                        'row_count': len(current_table_rows),
                        'column_count': len(current_table_rows[0]) if current_table_rows else 0,
                        'method': 'markdown'
                    })
                current_table_rows = []
                in_table = False

        # 处理最后一个表格
        if current_table_rows:
            tables.append({
                'rows': current_table_rows,
                'row_count': len(current_table_rows),
                'column_count': len(current_table_rows[0]) if current_table_rows else 0,
                'method': 'markdown'
            })

        logger.info(f"✅ 从Markdown提取到 {len(tables)} 个表格")
        return tables

    async def extract_tables_from_text(self, text_content: str) -> List[Dict[str, Any]]:
        """从纯文本中智能识别表格"""
        logger.info("从文本中智能识别表格...")

        tables = []

        try:
            # 检测表格模式
            lines = text_content.split('\n')

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # 检测可能的表格行(包含多个制表符或连续空格)
                if '\t' in line or (line.count('  ') >= 3 and not line.startswith(' ')):
                    # 开始收集表格行
                    table_rows = []

                    while i < len(lines):
                        current_line = lines[i].strip()

                        # 判断是否还是表格行
                        if '\t' in current_line or (current_line.count('  ') >= 3):
                            # 分割列
                            if '\t' in current_line:
                                cells = [cell.strip() for cell in current_line.split('\t')]
                            else:
                                # 使用连续空格分割
                                cells = re.split(r'\s{2,}', current_line)

                            if len(cells) >= 2:  # 至少2列才算是表格
                                table_rows.append(cells)
                        else:
                            # 表格结束
                            break

                        i += 1

                    # 如果收集到足够行(至少2行),认为是表格
                    if len(table_rows) >= 2:
                        tables.append({
                            'rows': table_rows,
                            'row_count': len(table_rows),
                            'column_count': len(table_rows[0]),
                            'method': 'text_pattern'
                        })

                i += 1

            logger.info(f"✅ 从文本识别到 {len(tables)} 个表格")
            return tables[:self.config.max_tables]

        except Exception as e:
            logger.error(f"文本表格识别失败: {e}")
            return []

    async def analyze_table_structure(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析表格结构"""
        rows = table_data.get('rows', [])

        if not rows:
            return {}

        analysis = {
            'row_count': len(rows),
            'column_count': len(rows[0]) if rows else 0,
            'has_header': table_data.get('has_header', True),
            'empty_cells': 0,
            'numeric_columns': 0,
            'column_types': []
        }

        # 分析列类型
        if rows:
            for col_idx in range(len(rows[0])):
                column_values = []
                for row in rows[1:]:
                    if col_idx < len(row):
                        val = row[col_idx].strip()
                        if val:
                            column_values.append(val)
                        else:
                            analysis['empty_cells'] += 1

                # 判断列类型
                if column_values:
                    numeric_count = sum(1 for v in column_values if re.match(r'^[\d.]+%?$', v))
                    if numeric_count / len(column_values) > 0.8:
                        analysis['numeric_columns'] += 1
                        analysis['column_types'].append('numeric')
                    else:
                        analysis['column_types'].append('text')
                else:
                    analysis['column_types'].append('empty')

        return analysis


# 全局服务实例
_table_extraction_service = None


def get_table_extraction_service() -> TableExtractionService:
    """获取表格提取服务实例"""
    global _table_extraction_service
    if _table_extraction_service is None:
        _table_extraction_service = TableExtractionService()
    return _table_extraction_service
