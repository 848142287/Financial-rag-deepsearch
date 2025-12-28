"""
统计图表分析器 - Statistical Chart Analyzer
分析统计图表，识别趋势、相关性、关键点位等
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


class StatisticalChartAnalyzer:
    """统计图表分析器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化分析器

        Args:
            config: 配置字典
        """
        self.config = config or {}

    def analyze_statistical_chart(
        self,
        chart_info: Dict[str, Any],
        x_data: List[Any] = None,
        y_data: List[float] = None
    ) -> Dict[str, Any]:
        """
        分析统计图表

        Args:
            chart_info: 图表信息
            x_data: X轴数据
            y_data: Y轴数据

        Returns:
            Dict: 统计分析结果
        """
        try:
            if not y_data or len(y_data) < 3:
                return {'status': 'insufficient_data'}

            analysis = {
                'trend_analysis': self._analyze_trend(y_data),
                'correlation_analysis': self._analyze_correlation(x_data, y_data),
                'key_points': self._identify_key_points(y_data),
                'statistical_summary': self._generate_statistical_summary(y_data)
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing statistical chart: {e}")
            return {}

    def _analyze_trend(self, y_data: List[float]) -> Dict[str, Any]:
        """分析趋势"""
        if not y_data or len(y_data) < 3:
            return {'status': 'insufficient_data'}

        # 计算移动平均
        window_size = min(5, len(y_data) // 3)
        moving_avg = []
        for i in range(len(y_data) - window_size + 1):
            avg = sum(y_data[i:i+window_size]) / window_size
            moving_avg.append(avg)

        # 判断趋势方向
        if len(moving_avg) >= 2:
            first_avg = moving_avg[0]
            last_avg = moving_avg[-1]

            change_percent = ((last_avg - first_avg) / first_avg * 100) if first_avg != 0 else 0

            if change_percent > 5:
                direction = "上升"
                strength = "强" if change_percent > 20 else "中" if change_percent > 10 else "弱"
            elif change_percent < -5:
                direction = "下降"
                strength = "强" if change_percent < -20 else "中" if change_percent < -10 else "弱"
            else:
                direction = "平稳"
                strength = "无"

            # 检测波动性
            if len(y_data) > 2:
                volatility = statistics.stdev(y_data) if len(y_data) > 1 else 0
                mean_val = statistics.mean(y_data)
                cv = (volatility / mean_val * 100) if mean_val != 0 else 0

                volatility_level = "高" if cv > 30 else "中" if cv > 15 else "低"
            else:
                volatility_level = "无法判断"

            return {
                'direction': direction,
                'strength': strength,
                'change_percent': round(change_percent, 2),
                'volatility_level': volatility_level,
                'description': f"数据呈{direction}趋势，{strength}波动性{volatility_level}"
            }

        return {'status': 'insufficient_data'}

    def _analyze_correlation(
        self,
        x_data: List[Any],
        y_data: List[float]
    ) -> Dict[str, Any]:
        """分析相关性"""
        if not x_data or not y_data or len(x_data) != len(y_data) or len(y_data) < 3:
            return {'status': 'insufficient_data'}

        # 尝试将X数据转换为数值
        x_numeric = []
        y_numeric = []

        for i, x in enumerate(x_data):
            try:
                x_val = float(x)
                y_val = float(y_data[i])

                # 过滤无效值
                if x_val == x_val and y_val == y_val:  # 检查NaN
                    x_numeric.append(x_val)
                    y_numeric.append(y_val)
            except (ValueError, TypeError):
                continue

        if len(x_numeric) < 3:
            return {'status': 'insufficient_numeric_data'}

        # 计算皮尔逊相关系数
        correlation = self._calculate_correlation(x_numeric, y_numeric)

        # 判断相关性强度和方向
        abs_corr = abs(correlation)

        if abs_corr > 0.8:
            strength = "强"
        elif abs_corr > 0.5:
            strength = "中等"
        elif abs_corr > 0.3:
            strength = "弱"
        else:
            strength = "极弱或无相关"

        direction = "正相关" if correlation > 0 else "负相关"

        return {
            'correlation_coefficient': round(correlation, 3),
            'strength': strength,
            'direction': direction,
            'significance': "显著" if abs_corr > 0.5 else "不显著",
            'description': f"变量间存在{direction}（r={correlation:.3f}），{strength}"
        }

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """计算皮尔逊相关系数"""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

        denominator = (sum_sq_x * sum_sq_y) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _identify_key_points(self, y_data: List[float]) -> List[Dict[str, Any]]:
        """识别关键点位"""
        key_points = []

        if not y_data or len(y_data) < 3:
            return key_points

        # 1. 极值点
        for i in range(1, len(y_data) - 1):
            prev_val = y_data[i - 1]
            curr_val = y_data[i]
            next_val = y_data[i + 1]

            # 局部最大值
            if curr_val > prev_val and curr_val > next_val:
                key_points.append({
                    'point_type': 'local_maximum',
                    'position': i,
                    'value': curr_val,
                    'description': f"位置{i}出现局部峰值{curr_val:.2f}"
                })

            # 局部最小值
            elif curr_val < prev_val and curr_val < next_val:
                key_points.append({
                    'point_type': 'local_minimum',
                    'position': i,
                    'value': curr_val,
                    'description': f"位置{i}出现局部谷值{curr_val:.2f}"
                })

        # 2. 全局极值
        if y_data:
            max_val = max(y_data)
            min_val = min(y_data)
            max_pos = y_data.index(max_val)
            min_pos = y_data.index(min_val)

            key_points.append({
                'point_type': 'global_maximum',
                'position': max_pos,
                'value': max_val,
                'description': f"全局最大值{max_val:.2f}出现在位置{max_pos}"
            })

            key_points.append({
                'point_type': 'global_minimum',
                'position': min_pos,
                'value': min_val,
                'description': f"全局最小值{min_val:.2f}出现在位置{min_pos}"
            })

        # 3. 拐点（简化的拐点检测）
        for i in range(1, len(y_data) - 1):
            slope1 = y_data[i] - y_data[i - 1]
            slope2 = y_data[i + 1] - y_data[i]

            # 斜率符号改变
            if (slope1 > 0 and slope2 < 0) or (slope1 < 0 and slope2 > 0):
                key_points.append({
                    'point_type': 'inflection_point',
                    'position': i,
                    'value': y_data[i],
                    'description': f"位置{i}出现拐点，趋势发生改变"
                })

        return key_points[:10]  # 返回最多10个关键点

    def _generate_statistical_summary(self, y_data: List[float]) -> Dict[str, Any]:
        """生成统计摘要"""
        if not y_data:
            return {}

        return {
            'count': len(y_data),
            'mean': round(statistics.mean(y_data), 2),
            'median': round(statistics.median(y_data), 2),
            'mode': "N/A",  # 需要额外计算
            'min': round(min(y_data), 2),
            'max': round(max(y_data), 2),
            'range': round(max(y_data) - min(y_data), 2),
            'stdev': round(statistics.stdev(y_data), 2) if len(y_data) > 1 else 0,
            'variance': round(statistics.variance(y_data), 2) if len(y_data) > 1 else 0,
            'quartiles': {
                'q1': round(statistics.quantiles(y_data, n=4)[0], 2) if len(y_data) >= 4 else "N/A",
                'q2': round(statistics.quantiles(y_data, n=4)[1], 2) if len(y_data) >= 4 else "N/A",
                'q3': round(statistics.quantiles(y_data, n=4)[2], 2) if len(y_data) >= 4 else "N/A"
            }
        }
