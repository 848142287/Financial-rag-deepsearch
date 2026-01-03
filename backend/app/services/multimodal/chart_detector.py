"""
图表检测器
识别和分析各种图表类型
"""

from app.core.structured_logging import get_structured_logger
import re
from typing import Dict, Any, List

logger = get_structured_logger(__name__)


class ChartDetector:
    """图表检测器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chart_types = {
            'bar': ['柱状图', '条形图', '柱形图', 'bar chart', 'column chart'],
            'line': ['折线图', '曲线图', '线形图', 'line chart', 'curve'],
            'pie': ['饼图', '扇形图', '圆饼图', 'pie chart'],
            'scatter': ['散点图', '点图', 'scatter plot', 'dot plot'],
            'area': ['面积图', '区域图', 'area chart'],
            'radar': ['雷达图', '蜘蛛图', 'radar chart', 'spider chart'],
            'heatmap': ['热力图', '热图', 'heatmap'],
            'histogram': ['直方图', '分布图', 'histogram'],
            'box': ['箱线图', '盒图', 'box plot'],
            'waterfall': ['瀑布图', 'waterfall chart']
        }

        self.chart_elements = {
            'x_axis': ['X轴', '横轴', '横坐标', 'X-axis', 'horizontal'],
            'y_axis': ['Y轴', '纵轴', '纵坐标', 'Y-axis', 'vertical'],
            'legend': ['图例', '说明', 'legend'],
            'title': ['标题', '主题', 'title'],
            'grid': ['网格', '网格线', 'grid'],
            'trend': ['趋势线', 'trend line'],
            'annotation': ['标注', '注释', 'annotation']
        }

    async def detect(self, chart_text: str) -> Dict[str, Any]:
        """检测和分析图表"""
        try:
            # 检测图表类型
            chart_type = self._detect_chart_type(chart_text)

            # 提取数据点
            data_points = self._extract_data_points(chart_text, chart_type)

            # 提取图表元素
            elements = self._extract_chart_elements(chart_text)

            # 识别数据趋势
            trends = self._identify_trends(chart_text, data_points, chart_type)

            # 提取图表描述
            description = self._extract_description(chart_text)

            # 检测时间序列
            time_series = self._detect_time_series(chart_text)

            return {
                'type': chart_type,
                'data': data_points,
                'elements': elements,
                'trends': trends,
                'description': description,
                'time_series': time_series,
                'metadata': {
                    'data_count': len(data_points),
                    'has_annotation': 'annotation' in elements,
                    'has_grid': 'grid' in elements,
                    'chart_subtype': self._detect_chart_subtype(chart_text, chart_type)
                },
                'confidence': self._calculate_confidence(chart_text, chart_type, data_points)
            }

        except Exception as e:
            logger.error(f"图表检测失败: {str(e)}")
            return {
                'type': 'unknown',
                'data': [],
                'elements': [],
                'trends': [],
                'description': '',
                'time_series': False,
                'metadata': {},
                'confidence': 0.0
            }

    def _detect_chart_type(self, text: str) -> str:
        """检测图表类型"""
        text_lower = text.lower()

        # 按优先级检查每种图表类型
        for chart_type, keywords in self.chart_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return chart_type

        # 根据文本特征推断
        if self._has_percentage_data(text):
            return 'pie'  # 百分比数据可能是饼图

        if self._has_time_series_data(text):
            return 'line'  # 时间序列数据可能是折线图

        if self._has_categorical_data(text):
            return 'bar'  # 分类数据可能是柱状图

        return 'unknown'

    def _has_percentage_data(self, text: str) -> bool:
        """检查是否包含百分比数据"""
        percentages = re.findall(r'\d+\.?\d*%', text)
        return len(percentages) >= 2

    def _has_time_series_data(self, str) -> bool:
        """检查是否包含时间序列数据"""
        time_patterns = [
            r'\d{4}年', r'\d{1,2}月', r'Q[1-4]',
            r'20\d{2}', r'19\d{2}',
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
        ]

        for pattern in time_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _has_categorical_data(self, text: str) -> bool:
        """检查是否包含分类数据"""
        # 简单检查：是否有多个类别名称
        categories = re.findall(r'[A-Za-z\u4e00-\u9fa5]{2,}[:：]\s*\d+', text)
        return len(categories) >= 2

    def _extract_data_points(self, text: str, chart_type: str) -> List[Dict[str, Any]]:
        """提取数据点"""
        data_points = []

        # 提取数值数据
        number_patterns = [
            r'([A-Za-z\u4e00-\u9fa5]+)[:：]\s*([\d,.-]+)',  # 标签: 数值
            r'([\d,.-]+)\s*\(([A-Za-z\u4e00-\u9fa5]+)\)',  # 数值 (标签)
            r'(\d+\.?\d*)%\s*([A-Za-z\u4e00-\u9fa5]+)',     # 百分比 标签
        ]

        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:
                    label = match[0].strip()
                    value = match[1].strip().replace(',', '')

                    # 尝试转换为数值
                    try:
                        if '%' in value:
                            numeric_value = float(value.replace('%', ''))
                            unit = '%'
                        else:
                            numeric_value = float(value)
                            unit = ''
                    except ValueError:
                        continue

                    data_points.append({
                        'label': label,
                        'value': numeric_value,
                        'unit': unit,
                        'type': chart_type
                    })

        # 如果没有提取到结构化数据，尝试提取纯数字
        if not data_points:
            numbers = re.findall(r'(\d+\.?\d*)', text)
            if len(numbers) >= 2:
                for i, num in enumerate(numbers[:10]):  # 限制前10个
                    data_points.append({
                        'label': f'Data_{i+1}',
                        'value': float(num),
                        'unit': '',
                        'type': chart_type
                    })

        return data_points

    def _extract_chart_elements(self, text: str) -> List[str]:
        """提取图表元素"""
        elements = []

        for element, keywords in self.chart_elements.items():
            if any(keyword in text.lower() for keyword in keywords):
                elements.append(element)

        return elements

    def _identify_trends(self, text: str, data_points: List[Dict], chart_type: str) -> Dict[str, Any]:
        """识别数据趋势"""
        trends = {
            'direction': 'unknown',
            'pattern': None,
            'summary': ''
        }

        # 从文本中提取趋势描述
        trend_keywords = {
            'increasing': ['上升', '增长', '增加', '提高', '上升', 'up', 'increase', 'growth'],
            'decreasing': ['下降', '减少', '降低', '下跌', 'down', 'decrease', 'decline'],
            'stable': ['稳定', '持平', '不变', 'stable', 'flat'],
            'fluctuating': ['波动', '震荡', '起伏', 'fluctuate', 'volatile']
        }

        text_lower = text.lower()
        for direction, keywords in trend_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                trends['direction'] = direction
                break

        # 从数据分析趋势
        if len(data_points) >= 3:
            values = [dp['value'] for dp in data_points]
            if all(values[i] <= values[i+1] for i in range(len(values)-1)):
                trends['direction'] = 'increasing'
                trends['pattern'] = 'monotonic_increase'
            elif all(values[i] >= values[i+1] for i in range(len(values)-1)):
                trends['direction'] = 'decreasing'
                trends['pattern'] = 'monotonic_decrease'

        # 生成趋势摘要
        if trends['direction'] == 'increasing':
            trends['summary'] = '数据呈上升趋势'
        elif trends['direction'] == 'decreasing':
            trends['summary'] = '数据呈下降趋势'
        elif trends['direction'] == 'stable':
            trends['summary'] = '数据保持稳定'
        elif trends['direction'] == 'fluctuating':
            trends['summary'] = '数据呈现波动'
        else:
            trends['summary'] = '趋势不明显'

        return trends

    def _extract_description(self, text: str) -> str:
        """提取图表描述"""
        # 查找图表相关的描述句子
        desc_patterns = [
            r'图表显示[:：]\s*([^\n]+)',
            r'如图所示[:：]\s*([^\n]+)',
            r'Figure\s*\d+[:：]\s*([^\n]+)',
            r'Chart\s*\d+[:：]\s*([^\n]+)',
            r'图表说明[:：]\s*([^\n]+)'
        ]

        for pattern in desc_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # 如果没有找到明确的描述，提取关键句子
        sentences = re.split(r'[。！？.!?]', text)
        for sentence in sentences:
            if any(keyword in sentence for keyword in ['图表', '数据', '显示', '表明']):
                return sentence.strip()

        return ''

    def _detect_time_series(self, text: str) -> bool:
        """检测是否为时间序列图表"""
        return self._has_time_series_data(text)

    def _detect_chart_subtype(self, text: str, chart_type: str) -> str:
        """检测图表子类型"""
        text_lower = text.lower()

        if chart_type == 'bar':
            if any(kw in text_lower for kw in ['stacked', '堆叠', '堆积']):
                return 'stacked_bar'
            elif any(kw in text_lower for kw in ['grouped', '分组', '并列']):
                return 'grouped_bar'
            elif any(kw in text_lower for kw in ['horizontal', '水平', '横向']):
                return 'horizontal_bar'

        elif chart_type == 'line':
            if any(kw in text_lower for kw in ['multi', '多条', '多线']):
                return 'multi_line'
            elif any(kw in text_lower for kw in ['smooth', '平滑', '曲线']):
                return 'smooth_line'

        elif chart_type == 'pie':
            if any(kw in text_lower for kw in ['exploded', '分离', '展开']):
                return 'exploded_pie'
            elif any(kw in text_lower for kw in ['donut', '环形', '圆环']):
                return 'donut_chart'

        return 'standard'

    def _calculate_confidence(self, text: str, chart_type: str, data_points: List[Dict]) -> float:
        """计算置信度"""
        confidence = 0.5

        # 根据图表类型确定性调整
        if chart_type != 'unknown':
            confidence += 0.2

        # 根据数据点数量调整
        if len(data_points) >= 3:
            confidence += 0.1
        elif len(data_points) >= 5:
            confidence += 0.2

        # 根据图表元素识别调整
        elements = self._extract_chart_elements(text)
        if len(elements) >= 2:
            confidence += 0.1

        # 根据描述文本调整
        if self._extract_description(text):
            confidence += 0.1

        return min(max(confidence, 0.0), 1.0)