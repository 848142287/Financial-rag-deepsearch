"""
CSV文件解析器
处理CSV格式的数据文件
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from .base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)


class CSVParser(BaseFileParser):
    """CSV文件解析器"""

    def __init__(self):
        self.supported_extensions = ['.csv']
        self.supported_mime_types = ['text/csv', 'application/csv']

    def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        解析CSV文件

        Args:
            file_path: 文件路径
            **kwargs: 其他参数
                - encoding: 文件编码，默认utf-8
                - delimiter: 分隔符，默认逗号
                - max_rows: 最大读取行数，默认None（读取全部）

        Returns:
            ParseResult: 解析结果
        """
        try:
            encoding = kwargs.get('encoding', 'utf-8')
            delimiter = kwargs.get('delimiter', ',')
            max_rows = kwargs.get('max_rows', None)

            # 读取CSV文件
            if max_rows:
                df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, nrows=max_rows)
            else:
                df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)

            # 转换为文本内容
            content = self._dataframe_to_text(df)

            # 构建元数据
            metadata = {
                'file_type': 'csv',
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'data_types': df.dtypes.to_dict(),
                'encoding': encoding,
                'delimiter': delimiter,
                'sample_data': df.head(5).to_dict() if len(df) > 0 else {}
            }

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error(f"CSV解析失败: {e}")
            return ParseResult(
                content="",
                metadata={'file_type': 'csv', 'error': str(e)},
                success=False,
                error_message=str(e)
            )

    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """
        将DataFrame转换为文本格式

        Args:
            df: pandas DataFrame

        Returns:
            str: 文本内容
        """
        lines = []

        # 添加列标题
        headers = df.columns.tolist()
        lines.append("列标题: " + ", ".join(headers))

        # 添加数据行（限制行数避免过长）
        max_display_rows = min(100, len(df))
        for i, row in df.head(max_display_rows).iterrows():
            row_text = ", ".join(str(val) if pd.notna(val) else "" for val in row)
            lines.append(f"第{i+1}行: {row_text}")

        # 如果数据超过显示限制，添加说明
        if len(df) > max_display_rows:
            lines.append(f"... (还有 {len(df) - max_display_rows} 行数据未显示)")

        # 添加数据统计信息
        lines.append("\n数据统计:")
        lines.append(f"总行数: {len(df)}")
        lines.append(f"总列数: {len(df.columns)}")

        # 添加每列的数据类型和基本信息
        lines.append("\n列信息:")
        for col in df.columns:
            dtype = df[col].dtype
            non_null_count = df[col].count()
            unique_count = df[col].nunique()

            lines.append(f"- {col}: 类型={dtype}, 非空值={non_null_count}, 唯一值={unique_count}")

            # 如果是数值型列，添加统计信息
            if pd.api.types.is_numeric_dtype(df[col]):
                lines.append(f"  统计: 最小值={df[col].min()}, 最大值={df[col].max()}, 平均值={df[col].mean():.2f}")

        return "\n".join(lines)

    def is_supported(self, file_path: str, mime_type: str = None) -> bool:
        """
        检查是否支持解析该文件

        Args:
            file_path: 文件路径
            mime_type: MIME类型（可选）

        Returns:
            bool: 是否支持
        """
        # 检查文件扩展名
        import os
        _, ext = os.path.splitext(file_path.lower())
        if ext in self.supported_extensions:
            return True

        # 检查MIME类型
        if mime_type and mime_type in self.supported_mime_types:
            return True

        return False