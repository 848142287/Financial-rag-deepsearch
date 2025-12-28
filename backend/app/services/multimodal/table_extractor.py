"""
表格内容提取器
提取表格结构化数据，识别表格类型
"""

import logging
import re
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class TableExtractor:
    """表格内容提取器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.table_types = {
            'financial_report': ['财务报表', '资产负债表', '利润表', '现金流量表'],
            'ratio_analysis': ['财务指标', '比率分析', 'ROE', 'ROA'],
            'business_metrics': ['业务指标', '运营指标', 'KPI'],
            'comparison': ['对比', '比较', '同比', '环比'],
            'forecast': ['预测', '预估', '预期', 'forecast'],
            'standard': []  # 默认类型
        }

    async def extract(self, table_text: str) -> Dict[str, Any]:
        """提取表格数据"""
        try:
            # 解析表格结构
            table_structure = self._parse_table_structure(table_text)

            # 识别表格类型
            table_type = self._classify_table_type(table_text, table_structure)

            # 提取数据
            table_data = self._extract_table_data(table_structure)

            # 清洗数据
            cleaned_data = self._clean_table_data(table_data)

            # 计算统计信息
            statistics = self._calculate_statistics(cleaned_data)

            return {
                'type': table_type,
                'headers': cleaned_data.get('headers', []),
                'data': cleaned_data.get('rows', []),
                'statistics': statistics,
                'metadata': {
                    'row_count': len(cleaned_data.get('rows', [])),
                    'column_count': len(cleaned_data.get('headers', [])),
                    'has_numeric_data': self._has_numeric_data(cleaned_data),
                    'completeness': self._calculate_completeness(cleaned_data)
                },
                'confidence': self._calculate_confidence(cleaned_data, table_type)
            }

        except Exception as e:
            logger.error(f"表格提取失败: {str(e)}")
            return {
                'type': 'standard',
                'headers': [],
                'data': [],
                'statistics': {},
                'metadata': {},
                'confidence': 0.0
            }

    def _parse_table_structure(self, table_text: str) -> List[List[str]]:
        """解析表格结构"""
        rows = []
        lines = table_text.strip().split('\n')

        for line in lines:
            if line.strip():
                # 尝试不同的分隔符
                if '|' in line:
                    row = [cell.strip() for cell in line.split('|')]
                elif '\t' in line:
                    row = [cell.strip() for cell in line.split('\t')]
                elif ',' in line:
                    row = [cell.strip() for cell in line.split(',')]
                else:
                    # 尝试按空格分割，但保留数字和单位
                    row = re.split(r'\s{2,}', line.strip())  # 两个以上空格作为分隔符

                if len(row) > 1:  # 至少两列
                    rows.append(row)

        return rows

    def _classify_table_type(self, table_text: str, structure: List[List[str]]) -> str:
        """分类表格类型"""
        text = (table_text + ' ' + ' '.join([' '.join(row) for row in structure])).lower()

        # 检查每种表格类型
        for table_type, keywords in self.table_types.items():
            if keywords and any(keyword in text for keyword in keywords):
                return table_type

        # 根据表格结构特征判断
        if self._is_financial_table(structure):
            return 'financial_report'
        elif self._is_ratio_table(structure):
            return 'ratio_analysis'
        elif self._is_comparison_table(structure):
            return 'comparison'

        return 'standard'

    def _is_financial_table(self, structure: List[List[str]]) -> bool:
        """判断是否为财务表格"""
        financial_keywords = [
            '资产', '负债', '收入', '成本', '利润', '现金流',
            'asset', 'liability', 'revenue', 'cost', 'profit', 'cash flow'
        ]

        for row in structure[:5]:  # 检查前5行
            for cell in row:
                if any(keyword in cell.lower() for keyword in financial_keywords):
                    return True

        return False

    def _is_ratio_table(self, structure: List[List[str]]) -> bool:
        """判断是否为比率分析表格"""
        ratio_keywords = [
            'ROE', 'ROA', '毛利率', '净利率', '周转率',
            'ratio', 'rate', 'margin', 'turnover'
        ]

        for row in structure[:5]:
            for cell in row:
                if any(keyword in cell.lower() for keyword in ratio_keywords):
                    return True

        return False

    def _is_comparison_table(self, structure: List[List[str]]) -> bool:
        """判断是否为对比表格"""
        comparison_keywords = [
            '同比', '环比', '增长', '变化', '差异',
            'yoy', 'mom', 'growth', 'change', 'diff'
        ]

        for row in structure[:5]:
            for cell in row:
                if any(keyword in cell.lower() for keyword in comparison_keywords):
                    return True

        return False

    def _extract_table_data(self, structure: List[List[str]]) -> Dict[str, Any]:
        """提取表格数据"""
        if not structure:
            return {'headers': [], 'rows': []}

        # 第一行作为表头
        headers = structure[0] if len(structure) > 0 else []
        rows = structure[1:] if len(structure) > 1 else []

        # 统一列数
        max_cols = max(len(headers), max(len(row) for row in rows) if rows else 0)

        # 补齐表头
        while len(headers) < max_cols:
            headers.append(f'Column_{len(headers)+1}')

        # 补齐数据行
        for row in rows:
            while len(row) < max_cols:
                row.append('')

        return {
            'headers': headers,
            'rows': rows
        }

    def _clean_table_data(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """清洗表格数据"""
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])

        # 清洗表头
        cleaned_headers = [self._clean_cell(header) for header in headers]

        # 清洗数据行
        cleaned_rows = []
        for row in rows:
            cleaned_row = [self._clean_cell(cell) for cell in row]
            if any(cell for cell in cleaned_row):  # 保留非空行
                cleaned_rows.append(cleaned_row)

        return {
            'headers': cleaned_headers,
            'rows': cleaned_rows
        }

    def _clean_cell(self, cell: str) -> str:
        """清洗单元格数据"""
        if not cell:
            return ''

        # 去除多余空格
        cell = re.sub(r'\s+', ' ', cell.strip())

        # 处理数字格式
        if re.match(r'^[\d,.-]+$', cell):
            # 移除千位分隔符
            cell = cell.replace(',', '')

        return cell

    def _calculate_statistics(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算统计信息"""
        rows = table_data.get('rows', [])
        statistics = {
            'numeric_columns': [],
            'text_columns': [],
            'column_stats': {}
        }

        if not rows or not table_data.get('headers'):
            return statistics

        headers = table_data['headers']
        num_columns = len(headers)

        # 分析每列
        for col_idx in range(num_columns):
            column_data = [row[col_idx] if col_idx < len(row) else '' for row in rows]
            column_stats = self._analyze_column(column_data)
            statistics['column_stats'][headers[col_idx]] = column_stats

            if column_stats['type'] == 'numeric':
                statistics['numeric_columns'].append(headers[col_idx])
            else:
                statistics['text_columns'].append(headers[col_idx])

        return statistics

    def _analyze_column(self, column_data: List[str]) -> Dict[str, Any]:
        """分析列数据"""
        stats = {
            'type': 'text',
            'null_count': 0,
            'unique_count': 0,
            'numeric_stats': {}
        }

        # 过滤非空值
        non_empty_values = [val for val in column_data if val.strip()]
        stats['null_count'] = len(column_data) - len(non_empty_values)
        stats['unique_count'] = len(set(non_empty_values))

        # 尝试转换为数字
        numeric_values = []
        for val in non_empty_values:
            num_val = self._parse_number(val)
            if num_val is not None:
                numeric_values.append(num_val)

        # 如果超过70%的值可以转为数字，认为是数字列
        if len(numeric_values) / max(len(non_empty_values), 1) > 0.7:
            stats['type'] = 'numeric'
            if numeric_values:
                stats['numeric_stats'] = {
                    'count': len(numeric_values),
                    'mean': sum(numeric_values) / len(numeric_values),
                    'min': min(numeric_values),
                    'max': max(numeric_values)
                }

        return stats

    def _parse_number(self, value: str) -> Optional[float]:
        """解析数字"""
        if not value:
            return None

        # 移除常见的非数字字符
        cleaned = re.sub(r'[^\d.-]', '', value)

        try:
            return float(cleaned)
        except ValueError:
            return None

    def _has_numeric_data(self, table_data: Dict[str, Any]) -> bool:
        """检查是否有数字数据"""
        stats = table_data.get('statistics', {})
        return len(stats.get('numeric_columns', [])) > 0

    def _calculate_completeness(self, table_data: Dict[str, Any]) -> float:
        """计算数据完整性"""
        rows = table_data.get('rows', [])
        if not rows:
            return 0.0

        total_cells = 0
        filled_cells = 0

        for row in rows:
            for cell in row:
                total_cells += 1
                if cell.strip():
                    filled_cells += 1

        return filled_cells / total_cells if total_cells > 0 else 0.0

    def _calculate_confidence(self, table_data: Dict[str, Any], table_type: str) -> float:
        """计算置信度"""
        confidence = 0.5

        # 根据数据完整性调整
        completeness = table_data.get('metadata', {}).get('completeness', 0)
        confidence += completeness * 0.3

        # 根据表格类型调整
        if table_type != 'standard':
            confidence += 0.1

        # 根据数字数据比例调整
        if self._has_numeric_data(table_data):
            confidence += 0.1

        return min(max(confidence, 0.0), 1.0)