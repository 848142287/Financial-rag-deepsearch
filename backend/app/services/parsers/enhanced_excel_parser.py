"""
增强Excel文档解析器 - 支持语义分块、RAG优化
处理.xlsx和.xls格式的文档，提供面向RAG的增强解析能力
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

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


class TableRegionType(Enum):
    """表格区域类型"""
    HEADER = "header"          # 表头区域
    DATA = "data"              # 数据区域
    SUMMARY = "summary"        # 汇总区域
    NOTE = "note"              # 备注区域
    FORMULA = "formula"        # 公式区域
    UNKNOWN = "unknown"


@dataclass
class TableBlock:
    """表格语义块"""
    block_id: str
    block_type: TableRegionType
    content: str
    sheet_name: str
    range_str: str = ""        # 如 "A1:D10"
    row_count: int = 0
    col_count: int = 0
    semantic_context: str = ""  # 语义上下文
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'block_id': self.block_id,
            'block_type': self.block_type.value,
            'content': self.content,
            'sheet_name': self.sheet_name,
            'range_str': self.range_str,
            'row_count': self.row_count,
            'col_count': self.col_count,
            'semantic_context': self.semantic_context,
            'keywords': self.keywords
        }


class EnhancedExcelParser(BaseFileParser):
    """增强Excel文档解析器 - 支持语义分块和RAG优化"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_data_rows = self.config.get('min_data_rows', 2)
        self.header_threshold = self.config.get('header_threshold', 0.7)

    @property
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名列表"""
        return ['.xlsx', '.xls']

    @property
    def parser_name(self) -> str:
        """解析器名称"""
        return "EnhancedExcelParser"

    def can_parse(self, file_path: str, file_extension: str = None) -> bool:
        """检查是否能解析指定文件"""
        if file_extension:
            return file_extension.lower() in self.supported_extensions

        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_extensions

    async def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        解析Excel文档（增强版 - 支持语义分块）

        Args:
            file_path: Excel文件路径
            **kwargs: 其他参数
                - extract_semantic_blocks: 是否提取语义块（默认True）
                - output_format: 输出格式（'markdown', 'json', 默认'markdown'）

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

            # 获取参数
            extract_semantic_blocks = kwargs.get('extract_semantic_blocks', True)
            output_format = kwargs.get('output_format', 'markdown')

            # 读取Excel文件
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names

            all_blocks = []
            sheet_metadata = {}

            # 逐个sheet解析
            for sheet_name in sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, dtype=str)

                # 提取语义块
                if extract_semantic_blocks:
                    blocks = self._extract_semantic_blocks(df, sheet_name)
                    all_blocks.extend(blocks)
                else:
                    # 基础提取
                    blocks = [self._create_basic_block(df, sheet_name)]
                    all_blocks.extend(blocks)

                # 收集sheet元数据
                sheet_metadata[sheet_name] = {
                    'name': sheet_name,
                    'rows': len(df),
                    'columns': len(df.columns) if len(df) > 0 else 0,
                    'has_data': not df.empty,
                    'block_count': len(blocks)
                }

            # 生成输出
            if output_format == 'json':
                content = self._generate_json_output(all_blocks, sheet_metadata)
            else:
                content = self._generate_markdown_output(all_blocks, sheet_metadata)

            # 构建元数据
            metadata = {
                'file_type': ext[1:],  # 去掉点号
                'parser_version': '2.0',
                'total_sheets': len(sheet_names),
                'sheet_names': sheet_names,
                'sheet_metadata': sheet_metadata,
                'total_blocks': len(all_blocks),
                'semantic_extraction': extract_semantic_blocks,
                'block_types': self._count_block_types(all_blocks)
            }

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error(f"Enhanced Excel parsing failed: {e}", exc_info=True)
            return ParseResult(
                content="",
                metadata={'file_type': 'excel', 'error': str(e)},
                success=False,
                error_message=str(e)
            )

    def _extract_semantic_blocks(self, df: pd.DataFrame, sheet_name: str) -> List[TableBlock]:
        """提取语义块"""
        blocks = []

        if df.empty:
            return blocks

        # 检测表格区域
        regions = self._detect_table_regions(df)

        if not regions:
            # 没有检测到区域，将整个sheet作为一个块
            blocks.append(self._create_fallback_block(df, sheet_name))
            return blocks

        # 为每个区域创建块
        for idx, region in enumerate(regions):
            block = self._extract_table_block(df, region, sheet_name, idx)
            if block:
                blocks.append(block)

        return blocks

    def _detect_table_regions(self, df: pd.DataFrame) -> List[Dict]:
        """检测表格区域"""
        regions = []
        rows, cols = df.shape

        if rows == 0 or cols == 0:
            return regions

        # 简化实现：使用固定大小的窗口检测
        window_size = min(20, rows)  # 窗口大小
        min_region_size = 3  # 最小区域大小

        # 检测非空区域
        non_empty_rows = df.notna().any(axis=1)
        non_empty_cols = df.notna().any(axis=0)

        if not non_empty_rows.any() or not non_empty_cols.any():
            return regions

        # 找到数据的边界
        first_row = non_empty_rows.idxmax() if non_empty_rows.any() else 0
        last_row = len(df) - 1 - non_empty_rows[::-1].idxmax() if non_empty_rows.any() else len(df)
        first_col = non_empty_cols.idxmax() if non_empty_cols.any() else 0
        last_col = len(df.columns) - 1 - non_empty_cols[::-1].idxmax() if non_empty_cols.any() else len(df.columns)

        # 创建主要数据区域
        main_region = {
            "start_row": int(first_row),
            "end_row": int(last_row) + 1,
            "start_col": int(first_col),
            "end_col": int(last_col) + 1,
            "rows": int(last_row - first_row + 1),
            "cols": int(last_col - first_col + 1)
        }

        regions.append(main_region)

        return regions

    def _extract_table_block(
        self,
        df: pd.DataFrame,
        region: Dict,
        sheet_name: str,
        block_idx: int
    ) -> Optional[TableBlock]:
        """提取表格块"""
        start_row = int(region["start_row"])
        end_row = int(region["end_row"])
        start_col = int(region["start_col"])
        end_col = int(region["end_col"])

        # 提取区域数据
        region_df = df.iloc[start_row:end_row, start_col:end_col]

        # 识别区域类型
        block_type = self._identify_region_type(region_df)

        # 生成块ID
        block_id = f"{sheet_name}_block{block_idx}"

        # 生成内容文本
        content = self._generate_block_content(region_df, block_type)

        # 提取关键词
        keywords = self._extract_keywords(region_df, content)

        # 生成语义上下文
        semantic_context = self._generate_semantic_context(region_df, block_type)

        # 生成范围字符串
        range_str = self._get_range_str(start_row, start_col, end_row, end_col)

        block = TableBlock(
            block_id=block_id,
            block_type=block_type,
            content=content,
            sheet_name=sheet_name,
            range_str=range_str,
            row_count=end_row - start_row,
            col_count=end_col - start_col,
            semantic_context=semantic_context,
            keywords=keywords
        )

        return block

    def _identify_region_type(self, df: pd.DataFrame) -> TableRegionType:
        """识别区域类型"""
        if df.empty or len(df) < 2:
            return TableRegionType.UNKNOWN

        # 检查是否为表头
        if self._is_likely_header(df.iloc[:2]):
            return TableRegionType.HEADER

        # 检查是否为数据区域
        if len(df) > self.min_data_rows:
            numeric_ratio = df.applymap(self._is_numeric).sum().sum() / (df.shape[0] * df.shape[1])
            if numeric_ratio > 0.3:
                return TableRegionType.DATA

        # 检查是否为汇总区域
        if self._is_summary_region(df):
            return TableRegionType.SUMMARY

        # 检查是否为备注区域
        if self._is_note_region(df):
            return TableRegionType.NOTE

        return TableRegionType.UNKNOWN

    def _is_likely_header(self, df) -> bool:
        """判断是否为表头"""
        if len(df) < 2:
            return False

        # 第一行有更多文本，第二行有更多数值
        row1_text = df.iloc[0].apply(lambda x: isinstance(x, str) and x.strip() != "").sum()
        row2_numeric = df.iloc[1].apply(self._is_numeric).sum()

        return row1_text > 0 and row2_numeric > 0

    def _is_summary_region(self, df) -> bool:
        """判断是否为汇总区域"""
        summary_keywords = {"总计", "合计", "汇总", "小计", "平均", "最大", "最小", "Total", "Sum", "Average"}

        for cell in df.values.flatten():
            if isinstance(cell, str):
                for keyword in summary_keywords:
                    if keyword in cell:
                        return True

        return False

    def _is_note_region(self, df) -> bool:
        """判断是否为备注区域"""
        if df.shape[0] > 5:
            return False

        note_keywords = {"备注", "说明", "注意", "注解", "注：", "Note", "Remark"}

        for cell in df.values.flatten():
            if isinstance(cell, str):
                for keyword in note_keywords:
                    if keyword in cell:
                        return True

        return False

    def _is_numeric(self, value) -> bool:
        """判断是否为数值"""
        if pd.isna(value):
            return False
        try:
            float(str(value))
            return True
        except:
            return False

    def _generate_block_content(self, df: pd.DataFrame, block_type: TableRegionType) -> str:
        """生成块内容"""
        if df.empty:
            return ""

        if block_type == TableRegionType.HEADER:
            return self._format_header_content(df)
        elif block_type == TableRegionType.DATA:
            return self._format_data_content(df)
        elif block_type == TableRegionType.SUMMARY:
            return self._format_summary_content(df)
        else:
            return self._format_general_content(df)

    def _format_header_content(self, df: pd.DataFrame) -> str:
        """格式化表头内容"""
        lines = []

        # 表头行
        if len(df) > 0:
            headers = [str(cell) for cell in df.iloc[0] if pd.notna(cell)]
            if headers:
                lines.append("表头: " + " | ".join(headers))

        # 示例数据
        if len(df) > 1:
            for i in range(1, min(4, len(df))):
                row_data = [str(cell) for cell in df.iloc[i] if pd.notna(cell)]
                if row_data:
                    lines.append(f"示例{i}: " + " | ".join(row_data))

        return "\n".join(lines)

    def _format_data_content(self, df: pd.DataFrame) -> str:
        """格式化数据内容"""
        lines = []

        # 检查是否有表头
        data_start = 0
        headers = []

        if len(df) > 1 and self._is_likely_header(df.iloc[:2]):
            headers = [str(cell) for cell in df.iloc[0] if pd.notna(cell)]
            if headers:
                lines.append("表头: " + " | ".join(headers))
                data_start = 1

        # 数据行
        for i in range(data_start, min(len(df), 15)):  # 最多15行
            row_data = []
            for j, cell in enumerate(df.iloc[i]):
                if pd.notna(cell):
                    cell_str = str(cell)
                    if headers and j < len(headers):
                        row_data.append(f"{headers[j]}: {cell_str}")
                    else:
                        row_data.append(cell_str)

            if row_data:
                lines.append(" | ".join(row_data))

        if len(df) > 15:
            lines.append(f"... 还有 {len(df) - 15} 行数据")

        # 添加统计信息
        stats = self._extract_statistics(df)
        if stats:
            lines.append("\n统计信息:")
            lines.append(stats)

        return "\n".join(lines)

    def _format_summary_content(self, df: pd.DataFrame) -> str:
        """格式化汇总内容"""
        lines = ["**汇总数据**"]

        for i, row in df.iterrows():
            row_data = [str(cell) for cell in row if pd.notna(cell)]
            if row_data:
                lines.append(" | ".join(row_data))

        return "\n".join(lines)

    def _format_general_content(self, df: pd.DataFrame) -> str:
        """格式化通用内容"""
        lines = []

        for i in range(min(len(df), 20)):
            row_data = [str(cell) for cell in df.iloc[i] if pd.notna(cell)]
            if row_data:
                lines.append(f"行{i+1}: " + " | ".join(row_data))

        if len(df) > 20:
            lines.append(f"... 还有 {len(df) - 20} 行数据")

        return "\n".join(lines)

    def _extract_statistics(self, df: pd.DataFrame) -> str:
        """提取统计信息"""
        stats = []

        try:
            numeric_df = df.apply(pd.to_numeric, errors='coerce')

            for col in numeric_df.columns:
                col_data = numeric_df[col].dropna()
                if len(col_data) > 2:  # 至少3个数值才计算统计
                    col_stats = f"列{col}: 平均={col_data.mean():.2f}, 最小={col_data.min():.2f}, 最大={col_data.max():.2f}"
                    stats.append(col_stats)
        except:
            pass

        return "; ".join(stats)

    def _extract_keywords(self, df: pd.DataFrame, content: str) -> List[str]:
        """提取关键词"""
        keywords = set()

        # 从内容提取
        words = re.findall(r'[\u4e00-\u9fff\w]{2,}', content)
        keywords.update([w for w in words if len(w) >= 2])

        # 从第一行提取（通常是表头）
        if len(df) > 0:
            for cell in df.iloc[0]:
                if isinstance(cell, str) and cell.strip():
                    keywords.update(cell.strip().split())

        # 过滤停用词
        stop_words = {"的", "了", "在", "是", "和", "与", "及", "或", "行", "列"}
        keywords = [kw for kw in keywords if kw not in stop_words and len(kw) >= 2]

        return list(keywords)[:10]  # 最多10个关键词

    def _generate_semantic_context(self, df: pd.DataFrame, block_type: TableRegionType) -> str:
        """生成语义上下文"""
        contexts = {
            TableRegionType.DATA: "数据表格区域，包含结构化数据记录",
            TableRegionType.HEADER: "表格表头区域，定义数据列的含义",
            TableRegionType.SUMMARY: "汇总统计区域，包含总计、平均等计算结果",
            TableRegionType.NOTE: "备注说明区域，提供额外信息或解释",
            TableRegionType.UNKNOWN: "通用内容区域"
        }
        return contexts.get(block_type, "未知区域")

    def _get_range_str(self, start_row: int, start_col: int, end_row: int, end_col: int) -> str:
        """获取范围字符串（如 A1:D10）"""
        def col_to_letter(col_idx):
            result = ""
            while col_idx >= 0:
                result = chr(65 + (col_idx % 26)) + result
                col_idx = col_idx // 26 - 1
            return result

        start_letter = col_to_letter(start_col)
        end_letter = col_to_letter(end_col - 1)

        return f"{start_letter}{start_row+1}:{end_letter}{end_row}"

    def _create_fallback_block(self, df: pd.DataFrame, sheet_name: str) -> TableBlock:
        """创建回退块"""
        content = self._format_general_content(df)

        return TableBlock(
            block_id=f"{sheet_name}_full",
            block_type=TableRegionType.UNKNOWN,
            content=content,
            sheet_name=sheet_name,
            range_str=f"A1:{self._get_range_str(0, 0, len(df), len(df.columns))}",
            row_count=len(df),
            col_count=len(df.columns),
            semantic_context="完整工作表内容",
            keywords=self._extract_keywords(df, content)
        )

    def _create_basic_block(self, df: pd.DataFrame, sheet_name: str) -> TableBlock:
        """创建基础块"""
        content = []
        content.append(f"## 工作表: {sheet_name}")
        content.append(f"行数: {len(df)}, 列数: {len(df.columns)}")

        if len(df.columns) > 0:
            content.append("\n列名: " + ", ".join(str(col) for col in df.columns))

        # 添加前几行数据
        if not df.empty:
            content.append("\n数据预览:")
            for idx, row in df.head(10).iterrows():
                row_text = " | ".join(str(val) if pd.notna(val) else "" for val in row)
                content.append(row_text)

        return TableBlock(
            block_id=f"{sheet_name}_basic",
            block_type=TableRegionType.UNKNOWN,
            content="\n".join(content),
            sheet_name=sheet_name,
            range_str=f"A1:{self._get_range_str(0, 0, len(df), len(df.columns))}",
            row_count=len(df),
            col_count=len(df.columns),
            semantic_context="工作表基础信息",
            keywords=[]
        )

    def _generate_markdown_output(
        self,
        blocks: List[TableBlock],
        sheet_metadata: Dict[str, Any]
    ) -> str:
        """生成Markdown输出"""
        lines = []

        # 文档标题
        lines.append(f"# Excel文档解析\n")
        lines.append(f"总工作表数: {len(sheet_metadata)}")
        lines.append(f"总语义块数: {len(blocks)}\n")

        # 按工作表组织
        current_sheet = None
        for block in blocks:
            if block.sheet_name != current_sheet:
                current_sheet = block.sheet_name
                sheet_info = sheet_metadata.get(current_sheet, {})
                lines.append(f"\n## 工作表: {current_sheet}")
                lines.append(f"行数: {sheet_info.get('rows', 0)}, 列数: {sheet_info.get('columns', 0)}")
                lines.append(f"块数: {sheet_info.get('block_count', 0)}\n")

            # 块内容
            lines.append(f"### {block.block_type.value.upper()} - {block.range_str}")
            if block.semantic_context:
                lines.append(f"*{block.semantic_context}*")
            lines.append(f"\n{block.content}\n")

            # 关键词
            if block.keywords:
                lines.append(f"**关键词**: {', '.join(block.keywords[:5])}\n")

        return "\n".join(lines)

    def _generate_json_output(
        self,
        blocks: List[TableBlock],
        sheet_metadata: Dict[str, Any]
    ) -> str:
        """生成JSON输出"""
        import json

        output = {
            'file_type': 'excel',
            'total_sheets': len(sheet_metadata),
            'total_blocks': len(blocks),
            'sheets': sheet_metadata,
            'blocks': [block.to_dict() for block in blocks]
        }

        return json.dumps(output, ensure_ascii=False, indent=2)

    def _count_block_types(self, blocks: List[TableBlock]) -> Dict[str, int]:
        """统计块类型数量"""
        type_counts = {}
        for block in blocks:
            block_type = block.block_type.value
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
        return type_counts
