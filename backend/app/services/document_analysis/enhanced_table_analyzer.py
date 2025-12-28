"""
增强表格分析器 - Enhanced Table Analyzer
深度分析表格数据，生成业务洞察和决策建议
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


@dataclass
class TableInsight:
    """表格洞察"""
    insight_type: str  # highest_value, lowest_value, trend, anomaly, comparison
    description: str
    location: str  # 位置描述，如"Q4营收"
    value: Any
    context: str  # 上下文解释
    importance: str  # 高/中/低


@dataclass
class TableRecommendation:
    """表格建议"""
    action_type: str  # advantage_area, problem_area, action_suggestion
    title: str
    description: str
    metrics: List[str]
    priority: str  # 高/中/低


class EnhancedTableAnalyzer:
    """增强表格分析器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化分析器

        Args:
            config: 配置字典
        """
        self.config = config or {}

    def analyze_table_deeply(
        self,
        table_data: List[List[str]],
        table_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        深度分析表格数据

        Args:
            table_data: 表格数据（二维列表）
            table_context: 表格上下文信息
                - title: 表格标题
                - page_number: 页码
                - section: 所属章节

        Returns:
            Dict: 深度分析结果
        """
        if not table_data or len(table_data) == 0:
            return self._empty_analysis()

        try:
            analysis = {
                'structure': self._analyze_structure(table_data),
                'statistics': self._calculate_statistics(table_data),
                'patterns': self._identify_patterns(table_data),
                'insights': self._generate_insights(table_data),
                'business_logic': self._extract_business_logic(table_data),
                'data_quality': self._assess_data_quality(table_data)
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing table: {e}")
            return self._empty_analysis()

    def _empty_analysis(self) -> Dict[str, Any]:
        """返回空分析结果"""
        return {
            'structure': {},
            'statistics': {},
            'patterns': [],
            'insights': [],
            'business_logic': {},
            'data_quality': {}
        }

    def _analyze_structure(self, table_data: List[List[str]]) -> Dict[str, Any]:
        """分析表格结构"""
        if not table_data:
            return {}

        rows = len(table_data)
        cols = max(len(row) for row in table_data) if table_data else 0

        # 假设第一行是表头
        headers = table_data[0] if table_data else []

        return {
            'rows': rows,
            'columns': cols,
            'headers': headers,
            'data_rows': rows - 1,  # 减去表头
            'has_header': len(headers) > 0,
            'is_empty': rows == 0 or cols == 0
        }

    def _calculate_statistics(self, table_data: List[List[str]]) -> Dict[str, Any]:
        """计算统计数据"""
        if not table_data or len(table_data) <= 1:  # 只有表头或空表
            return {}

        # 提取数值数据（跳过表头）
        numeric_values = []
        for row in table_data[1:]:  # 跳过表头
            for cell in row:
                # 提取数值（支持百分比、货币等）
                value = self._extract_numeric_value(cell)
                if value is not None:
                    numeric_values.append(value)

        if not numeric_values:
            return {'numeric_cells': 0}

        # 计算统计指标
        return {
            'numeric_cells': len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'mean': statistics.mean(numeric_values),
            'median': statistics.median(numeric_values),
            'stdev': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
            'total': sum(numeric_values),
            'range': max(numeric_values) - min(numeric_values)
        }

    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """从文本中提取数值"""
        if not text or not isinstance(text, str):
            return None

        # 移除常见的非数字字符
        text_clean = re.sub(r'[,%\s￥$€¥£]', '', text)

        # 处理百分号
        if '%' in text:
            text_clean = text_clean.replace('%', '')
            try:
                return float(text_clean) / 100
            except ValueError:
                return None

        # 提取数字
        match = re.search(r'-?\d+\.?\d*', text_clean)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

        return None

    def _identify_patterns(self, table_data: List[List[str]]) -> List[Dict[str, Any]]:
        """识别数据模式"""
        patterns = []

        if len(table_data) <= 1:
            return patterns

        # 分析列模式
        for col_idx in range(len(table_data[0])):
            column = []
            for row in table_data[1:]:  # 跳过表头
                if col_idx < len(row):
                    column.append(row[col_idx])

            # 检查是否为时间序列
            if self._is_time_series(column):
                patterns.append({
                    'type': 'time_series',
                    'location': f'column_{col_idx}',
                    'description': '时间序列数据'
                })

                # 检查趋势
                trend = self._detect_trend(column)
                if trend:
                    patterns.append({
                        'type': 'trend',
                        'location': f'column_{col_idx}',
                        'trend_type': trend['direction'],
                        'strength': trend['strength']
                    })

        return patterns

    def _is_time_series(self, column: List[str]) -> bool:
        """判断是否为时间序列"""
        # 检查是否包含时间关键词
        time_keywords = ['年', '月', '季度', 'Q1', 'Q2', 'Q3', 'Q4',
                        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for cell in column[:5]:  # 检查前5个
            if any(keyword in str(cell) for keyword in time_keywords):
                return True

        return False

    def _detect_trend(self, column: List[str]) -> Optional[Dict[str, str]]:
        """检测趋势"""
        # 提取数值
        values = []
        for cell in column:
            value = self._extract_numeric_value(str(cell))
            if value is not None:
                values.append(value)

        if len(values) < 3:
            return None

        # 简单趋势检测：比较首尾
        first_avg = statistics.mean(values[:len(values)//3])
        last_avg = statistics.mean(values[-len(values)//3:])

        change_percent = ((last_avg - first_avg) / first_avg * 100) if first_avg != 0 else 0

        direction = "上升" if change_percent > 5 else "下降" if change_percent < -5 else "平稳"
        strength = "强" if abs(change_percent) > 20 else "中" if abs(change_percent) > 10 else "弱"

        return {
            'direction': direction,
            'strength': strength,
            'change_percent': change_percent
        }

    def _generate_insights(self, table_data: List[List[str]]) -> List[Dict[str, Any]]:
        """生成业务洞察"""
        insights = []

        if len(table_data) <= 1:
            return insights

        # 转换为更易处理的格式
        data_rows = table_data[1:]
        headers = table_data[0] if table_data else []

        # 1. 查找最高值
        max_insight = self._find_extreme_values(data_rows, headers, find_max=True)
        if max_insight:
            insights.append(max_insight)

        # 2. 查找最低值
        min_insight = self._find_extreme_values(data_rows, headers, find_max=False)
        if min_insight:
            insights.append(min_insight)

        # 3. 识别关键差异
        comparison_insights = self._identify_key_differences(data_rows, headers)
        insights.extend(comparison_insights)

        return insights

    def _find_extreme_values(
        self,
        data_rows: List[List[str]],
        headers: List[str],
        find_max: bool = True
    ) -> Optional[Dict[str, Any]]:
        """查找极值"""
        extreme_value = None
        extreme_location = None
        extreme_row_idx = None
        extreme_col_idx = None

        for row_idx, row in enumerate(data_rows):
            for col_idx, cell in enumerate(row):
                value = self._extract_numeric_value(str(cell))

                if value is not None:
                    if extreme_value is None:
                        extreme_value = value
                        extreme_location = f"行{row_idx+2}, 列{col_idx+1}"
                        extreme_row_idx = row_idx
                        extreme_col_idx = col_idx
                    else:
                        if find_max and value > extreme_value:
                            extreme_value = value
                            extreme_location = f"行{row_idx+2}, 列{col_idx+1}"
                            extreme_row_idx = row_idx
                            extreme_col_idx = col_idx
                        elif not find_max and value < extreme_value:
                            extreme_value = value
                            extreme_location = f"行{row_idx+2}, 列{col_idx+1}"
                            extreme_row_idx = row_idx
                            extreme_col_idx = col_idx

        if extreme_value is not None:
            # 获取行列标签
            row_label = data_rows[extreme_row_idx][0] if extreme_row_idx < len(data_rows) else f"行{extreme_row_idx+2}"
            col_label = headers[extreme_col_idx] if extreme_col_idx < len(headers) else f"列{extreme_col_idx+1}"

            return {
                'insight_type': 'highest_value' if find_max else 'lowest_value',
                'description': f"{'最高' if find_max else '最低'}值出现在{row_label}的{col_label}",
                'location': f"{row_label} - {col_label}",
                'value': extreme_value,
                'context': f"该{'值' if find_max else '值'}为{extreme_value}，位于{extreme_location}",
                'importance': '高' if find_max else '中'
            }

        return None

    def _identify_key_differences(
        self,
        data_rows: List[List[str]],
        headers: List[str]
    ) -> List[Dict[str, Any]]:
        """识别关键差异"""
        differences = []

        # 简单实现：比较相邻行
        if len(data_rows) < 2:
            return differences

        for i in range(len(data_rows) - 1):
            row1 = data_rows[i]
            row2 = data_rows[i + 1]

            # 比较数值列
            for col_idx in range(1, min(len(row1), len(row2))):
                val1 = self._extract_numeric_value(str(row1[col_idx]))
                val2 = self._extract_numeric_value(str(row2[col_idx]))

                if val1 is not None and val2 is not None and val1 != 0:
                    diff_percent = abs((val2 - val1) / val1 * 100)

                    if diff_percent > 50:  # 差异超过50%
                        col_label = headers[col_idx] if col_idx < len(headers) else f"列{col_idx+1}"
                        row1_label = row1[0] if row1 else f"行{i+2}"
                        row2_label = row2[0] if row2 else f"行{i+3}"

                        differences.append({
                            'insight_type': 'comparison',
                            'description': f"{row2_label}相对于{row1_label}在{col_label}上有显著差异",
                            'location': f"{row1_label} vs {row2_label} - {col_label}",
                            'value': f"{val2:.2f} vs {val1:.2f} ({diff_percent:.1f}%)",
                            'context': f"变化率为{diff_percent:.1f}%",
                            'importance': '高' if diff_percent > 100 else '中'
                        })

        return differences[:5]  # 返回最多5个差异

    def _extract_business_logic(self, table_data: List[List[str]]) -> Dict[str, Any]:
        """提取业务逻辑"""
        logic = {
            'advantage_areas': [],
            'problem_areas': [],
            'action_suggestions': []
        }

        if len(table_data) <= 1:
            return logic

        # 简单的业务逻辑提取
        data_rows = table_data[1:]
        headers = table_data[0] if table_data else []

        # 识别优势领域（高值区域）
        for row_idx, row in enumerate(data_rows):
            row_label = row[0] if row else f"行{row_idx+2}"
            row_values = []

            for col_idx, cell in enumerate(row[1:], 1):
                value = self._extract_numeric_value(str(cell))
                if value is not None:
                    row_values.append((value, headers[col_idx] if col_idx < len(headers) else f"列{col_idx+1}"))

            if row_values:
                # 找出该行的最高值
                max_val, max_col = max(row_values, key=lambda x: x[0])

                # 如果明显高于其他值，标记为优势
                avg_val = sum(v for v, _ in row_values) / len(row_values)
                if max_val > avg_val * 1.5:
                    logic['advantage_areas'].append({
                        'area': f"{row_label} - {max_col}",
                        'description': f"{max_col}表现突出，值为{max_val}，高于平均值{avg_val:.2f}",
                        'metric': max_col,
                        'value': max_val
                    })

        return logic

    def _assess_data_quality(self, table_data: List[List[str]]) -> Dict[str, Any]:
        """评估数据质量"""
        quality = {
            'completeness': 0,
            'consistency': 0,
            'empty_cells': 0,
            'total_cells': 0
        }

        if not table_data:
            return quality

        total_cells = sum(len(row) for row in table_data)
        empty_cells = 0

        for row in table_data:
            for cell in row:
                if not cell or str(cell).strip() == '':
                    empty_cells += 1

        quality['total_cells'] = total_cells
        quality['empty_cells'] = empty_cells
        quality['completeness'] = (total_cells - empty_cells) / total_cells * 100 if total_cells > 0 else 0

        return quality
