"""
增强表格提取服务 - 支持多种表格格式
完善数据覆盖，提升识别率
"""

import re
from dataclasses import dataclass, field

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

@dataclass
class TableData:
    """表格数据"""
    headers: List[str]
    rows: List[List[str]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    table_type: str = "markdown"  # markdown, html, csv, excel, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            'headers': self.headers,
            'rows': self.rows,
            'row_count': len(self.rows),
            'column_count': len(self.headers),
            'table_type': self.table_type,
            'metadata': self.metadata
        }

class EnhancedTableExtractionService:
    """
    增强表格提取服务

    支持格式：
    1. Markdown表格
    2. HTML表格
    3. CSV格式表格
    4. 管道分隔表格
    5. 制表符分隔表格
    6. Excel复制格式（TSV）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化表格提取服务"""
        self.config = config or {}
        self.logger = logger

    def extract_all_tables(self, content: str,
                          content_type: str = "markdown") -> List[TableData]:
        """
        提取所有表格

        Args:
            content: 文档内容
            content_type: 内容类型

        Returns:
            List[TableData]: 表格列表
        """
        tables = []

        # 根据内容类型选择提取方法
        if content_type == "markdown":
            tables.extend(self._extract_markdown_tables(content))
        elif content_type == "html":
            tables.extend(self._extract_html_tables(content))

        # 通用格式提取
        tables.extend(self._extract_csv_tables(content))
        tables.extend(self._extract_pipe_tables(content))
        tables.extend(self._extract_tsv_tables(content))

        return tables

    def _extract_markdown_tables(self, content: str) -> List[TableData]:
        """提取Markdown表格"""
        tables = []
        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # 检测表格开始
            if self._is_markdown_table_line(line):
                table_lines = [line]

                # 收集表格行
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if self._is_markdown_table_line(next_line):
                        table_lines.append(next_line)
                        i += 1
                    elif next_line == '':  # 空行可能表示表格结束
                        break
                    else:
                        # 检查是否是分隔行
                        if self._is_separator_row(next_line):
                            table_lines.append(next_line)
                            i += 1
                        else:
                            break

                # 解析表格
                if len(table_lines) >= 2:
                    table = self._parse_markdown_table(table_lines)
                    if table:
                        tables.append(table)

            i += 1

        return tables

    def _extract_html_tables(self, content: str) -> List[TableData]:
        """提取HTML表格"""
        tables = []

        # 简单实现：使用正则表达式提取table标签
        # 注意：生产环境建议使用BeautifulSoup或lxml
        table_pattern = re.compile(
            r'<table[^>]*>(.*?)</table>',
            re.DOTALL | re.IGNORECASE
        )

        for match in table_pattern.finditer(content):
            table_html = match.group(1)
            table = self._parse_html_table(table_html)
            if table:
                tables.append(table)

        return tables

    def _extract_csv_tables(self, content: str) -> List[TableData]:
        """提取CSV格式表格"""
        tables = []

        # 检测CSV块（连续的CSV格式行）
        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # 检测CSV行（包含逗号分隔）
            if ',' in line and self._count_commas(line) >= 2:
                csv_lines = [line]

                # 收集CSV行
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line and ',' in next_line:
                        csv_lines.append(next_line)
                        i += 1
                    else:
                        break

                # 解析CSV表格
                if len(csv_lines) >= 2:
                    table = self._parse_csv_table(csv_lines)
                    if table:
                        tables.append(table)

            i += 1

        return tables

    def _extract_pipe_tables(self, content: str) -> List[TableData]:
        """提取管道分隔表格（不带Markdown格式）"""
        tables = []

        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # 检测管道分隔行（不一定是Markdown表格）
            if '|' in line and not line.startswith('#'):
                pipe_lines = [line]

                # 收集连续的管道分隔行
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if '|' in next_line and next_line != '':
                        pipe_lines.append(next_line)
                        i += 1
                    else:
                        break

                # 解析管道表格
                if len(pipe_lines) >= 2:
                    table = self._parse_pipe_table(pipe_lines)
                    if table:
                        tables.append(table)

            i += 1

        return tables

    def _extract_tsv_tables(self, content: str) -> List[TableData]:
        """提取制表符分隔表格（TSV/Excel复制格式）"""
        tables = []

        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # 检测制表符分隔行
            if '\t' in line and self._count_tabs(line) >= 1:
                tsv_lines = [line]

                # 收集TSV行
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if '\t' in next_line and next_line.strip():
                        tsv_lines.append(next_line)
                        i += 1
                    else:
                        break

                # 解析TSV表格
                if len(tsv_lines) >= 2:
                    table = self._parse_tsv_table(tsv_lines)
                    if table:

                        tables.append(table)

            i += 1

        return tables

    def _is_markdown_table_line(self, line: str) -> bool:
        """检查是否是Markdown表格行"""
        return line.startswith('|') and line.endswith('|')

    def _is_separator_row(self, line: str) -> bool:
        """检查是否是分隔行"""
        # Markdown表格分隔行: |---|---|
        separator_pattern = r'^\|[\s\-\:]+\|$'
        return bool(re.match(separator_pattern, line))

    def _parse_markdown_table(self, table_lines: List[str]) -> Optional[TableData]:
        """解析Markdown表格"""
        if len(table_lines) < 2:
            return None

        try:
            # 第一行是表头
            headers = self._split_table_row(table_lines[0])

            # 检查第二行是否是分隔行
            start_idx = 1
            if self._is_separator_row(table_lines[1]):
                start_idx = 2

            # 解析数据行
            rows = []
            for i in range(start_idx, len(table_lines)):
                row = self._split_table_row(table_lines[i])
                if row:
                    rows.append(row)

            return TableData(
                headers=headers,
                rows=rows,
                table_type="markdown",
                metadata={
                    'line_count': len(table_lines),
                    'has_separator': start_idx == 2
                }
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse Markdown table: {e}")
            return None

    def _parse_html_table(self, table_html: str) -> Optional[TableData]:
        """解析HTML表格"""
        try:
            # 简单实现：提取th和td
            headers = []
            rows = []

            # 提取表头
            th_pattern = re.compile(r'<th[^>]*>(.*?)</th>', re.IGNORECASE | re.DOTALL)
            for match in th_pattern.finditer(table_html):
                header_text = re.sub(r'<[^>]+>', '', match.group(1))  # 移除内部标签
                headers.append(header_text.strip())

            # 提取行
            tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.IGNORECASE | re.DOTALL)
            for tr_match in tr_pattern.finditer(table_html):
                row_html = tr_match.group(1)

                # 提取单元格
                cell_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.IGNORECASE | re.DOTALL)
                row = []
                for cell_match in cell_pattern.finditer(row_html):
                    cell_text = re.sub(r'<[^>]+>', '', cell_match.group(1))
                    row.append(cell_text.strip())

                if row:
                    rows.append(row)

            if headers or rows:
                return TableData(
                    headers=headers,
                    rows=rows,
                    table_type="html",
                    metadata={'raw_html_length': len(table_html)}
                )

        except Exception as e:
            self.logger.warning(f"Failed to parse HTML table: {e}")

        return None

    def _parse_csv_table(self, csv_lines: List[str]) -> Optional[TableData]:
        """解析CSV表格"""
        try:
            # 简单CSV解析（注意：不支持引号包裹的逗号）
            rows = []
            for line in csv_lines:
                # 处理引号
                row = self._parse_csv_line(line)
                if row:
                    rows.append(row)

            if not rows:
                return None

            # 第一行作为表头
            headers = rows[0]
            data_rows = rows[1:] if len(rows) > 1 else []

            return TableData(
                headers=headers,
                rows=data_rows,
                table_type="csv",
                metadata={'total_lines': len(csv_lines)}
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse CSV table: {e}")
            return None

    def _parse_pipe_table(self, pipe_lines: List[str]) -> Optional[TableData]:
        """解析管道分隔表格"""
        try:
            rows = []
            for line in pipe_lines:
                # 分割管道符
                cells = [cell.strip() for cell in line.split('|')]
                # 移除空的首尾元素
                cells = [c for c in cells if c]
                if cells:
                    rows.append(cells)

            if not rows:
                return None

            headers = rows[0]
            data_rows = rows[1:] if len(rows) > 1 else []

            return TableData(
                headers=headers,
                rows=data_rows,
                table_type="pipe",
                metadata={'total_lines': len(pipe_lines)}
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse pipe table: {e}")
            return None

    def _parse_tsv_table(self, tsv_lines: List[str]) -> Optional[TableData]:
        """解析TSV表格"""
        try:
            rows = []
            for line in tsv_lines:
                cells = [cell.strip() for cell in line.split('\t')]
                cells = [c for c in cells if c]  # 移除空单元格
                if cells:
                    rows.append(cells)

            if not rows:
                return None

            headers = rows[0]
            data_rows = rows[1:] if len(rows) > 1 else []

            return TableData(
                headers=headers,
                rows=data_rows,
                table_type="tsv",
                metadata={'total_lines': len(tsv_lines)}
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse TSV table: {e}")
            return None

    def _split_table_row(self, row: str) -> List[str]:
        """分割表格行"""
        # 移除首尾管道符
        if row.startswith('|'):
            row = row[1:]
        if row.endswith('|'):
            row = row[:-1]

        # 分割并清理
        cells = [cell.strip() for cell in row.split('|')]
        return cells

    def _parse_csv_line(self, line: str) -> List[str]:
        """解析CSV行（简单版本）"""
        # 处理引号包裹的情况
        if '"' in line:
            # 引号包裹的逗号不分割
            pattern = r'("(?:[^"]*"")*[^"]*"|[^,]+)'
            matches = re.findall(pattern, line)
            result = []
            for match in matches:
                # 移除引号
                if match.startswith('"') and match.endswith('"'):
                    match = match[1:-1]
                    # 处理转义的引号
                    match = match.replace('""', '"')
                result.append(match.strip())
            return result
        else:
            # 简单逗号分割
            return [cell.strip() for cell in line.split(',')]

    def _count_commas(self, line: str) -> int:
        """计算逗号数量"""
        return line.count(',')

    def _count_tabs(self, line: str) -> int:
        """计算制表符数量"""
        return line.count('\t')

# 全局实例
enhanced_table_extraction_service = EnhancedTableExtractionService()
