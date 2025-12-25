"""
Excel文档解析器
处理.xlsx和.xls格式的文档
"""

import logging
from typing import Dict, List, Any, Optional
from .base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available, Excel parsing will be limited")

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logger.warning("openpyxl not available, .xlsx parsing will be limited")

try:
    import xlrd
    XLDR_AVAILABLE = True
except ImportError:
    XLDR_AVAILABLE = False
    logger.warning("xlrd not available, .xls parsing will be limited")


class ExcelParser(BaseFileParser):
    """Excel文档解析器"""

    def __init__(self):
        self.supported_extensions = ['.xlsx', '.xls']
        self.supported_mime_types = [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel'
        ]

    def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        解析Excel文档

        Args:
            file_path: 文件路径
            **kwargs: 其他参数

        Returns:
            ParseResult: 解析结果
        """
        try:
            import os
            _, ext = os.path.splitext(file_path.lower())

            if not PANDAS_AVAILABLE:
                return ParseResult(
                    content="",
                    metadata={'file_type': 'excel', 'error': 'pandas library not available'},
                    success=False,
                    error_message='pandas library not installed'
                )

            if ext == '.xlsx' and not OPENPYXL_AVAILABLE:
                return ParseResult(
                    content="",
                    metadata={'file_type': 'xlsx', 'error': 'openpyxl library not available'},
                    success=False,
                    error_message='openpyxl library not installed for .xlsx files'
                )

            if ext == '.xls' and not XLDR_AVAILABLE:
                return ParseResult(
                    content="",
                    metadata={'file_type': 'xls', 'error': 'xlrd library not available'},
                    success=False,
                    error_message='xlrd library not installed for .xls files'
                )

            # 尝试读取Excel文件
            try:
                # 读取所有工作表
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names

                content_parts = []
                sheet_data = {}

                for sheet_name in sheet_names:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)

                        # 转换为文本内容
                        sheet_text = f"工作表: {sheet_name}\n"
                        sheet_text += f"行数: {len(df)}, 列数: {len(df.columns)}\n"
                        sheet_text += "列名: " + ", ".join(str(col) for col in df.columns) + "\n"

                        # 添加前几行数据
                        if not df.empty:
                            sheet_text += "数据预览:\n"
                            for idx, row in df.head(10).iterrows():
                                row_text = " | ".join(str(val) if pd.notna(val) else "" for val in row)
                                sheet_text += f"{row_text}\n"

                        content_parts.append(sheet_text)
                        sheet_data[sheet_name] = {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'column_names': list(df.columns),
                            'has_data': not df.empty
                        }

                    except Exception as e:
                        logger.warning(f"读取工作表 {sheet_name} 失败: {e}")
                        content_parts.append(f"工作表 {sheet_name}: 读取失败 - {str(e)}")

                content = "\n\n".join(content_parts)

                # 构建元数据
                metadata = {
                    'file_type': ext[1:],  # 去掉点号
                    'sheets': len(sheet_names),
                    'sheet_names': sheet_names,
                    'sheet_data': sheet_data,
                    'total_rows': sum(data['rows'] for data in sheet_data.values()),
                    'total_columns': sum(data['columns'] for data in sheet_data.values())
                }

                return ParseResult(
                    content=content,
                    metadata=metadata,
                    success=True
                )

            except Exception as e:
                logger.error(f"Excel文件读取失败: {e}")
                # 尝试使用更基础的方法
                return self._parse_basic(file_path, ext)

        except Exception as e:
            logger.error(f"Excel文档解析失败: {e}")
            return ParseResult(
                content="",
                metadata={'file_type': 'excel', 'error': str(e)},
                success=False,
                error_message=str(e)
            )

    def _parse_basic(self, file_path: str, ext: str) -> ParseResult:
        """基础Excel文件解析"""
        import os

        file_size = os.path.getsize(file_path)

        content = f"Excel文档: {os.path.basename(file_path)}\n"
        content += f"文件类型: {ext}\n"
        content += f"文件大小: {file_size} 字节\n"
        content += "注意: 由于缺少相关依赖库，无法提取详细内容"

        if ext == '.xlsx' and not OPENPYXL_AVAILABLE:
            content += "\n建议安装: pip install openpyxl"
        elif ext == '.xls' and not XLDR_AVAILABLE:
            content += "\n建议安装: pip install xlrd"

        metadata = {
            'file_type': ext[1:],
            'file_size': file_size,
            'parsing_limited': True,
            'recommendation': f"Install required dependencies for {ext} files"
        }

        return ParseResult(
            content=content,
            metadata=metadata,
            success=True
        )

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