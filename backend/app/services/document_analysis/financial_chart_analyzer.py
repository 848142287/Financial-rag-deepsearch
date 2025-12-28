"""
金融图表分析器 - Financial Chart Analyzer
专业分析金融投资图表，提供收益性、风险性等深度分析
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


@dataclass
class FinancialMetrics:
    """金融指标"""
    total_return: float  # 总收益率
    annualized_return: float  # 年化收益率
    volatility: float  # 波动率
    sharpe_ratio: float  # 夏普比率
    max_drawdown: float  # 最大回撤
    beta: float  # 贝塔系数
    alpha: float  # 阿尔法


class FinancialChartAnalyzer:
    """金融图表分析器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化分析器

        Args:
            config: 配置字典
                - risk_free_rate: 无风险利率 (默认0.03)
        """
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.03)

    def analyze_financial_chart(
        self,
        chart_data: Dict[str, Any],
        data_points: List[float] = None
    ) -> Dict[str, Any]:
        """
        分析金融图表

        Args:
            chart_data: 图表数据
                - title: 图表标题
                - x_axis: X轴标签
                - y_axis: Y轴标签
                - series: 数据序列
            data_points: 数据点列表

        Returns:
            Dict: 金融分析结果
        """
        try:
            analysis = {
                'return_analysis': self._analyze_returns(data_points or []),
                'risk_analysis': self._analyze_risk(data_points or []),
                'risk_adjusted_return': self._calculate_risk_adjusted_metrics(data_points or []),
                'interpretation': self._generate_interpretation(data_points or [])
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing financial chart: {e}")
            return {}

    def _analyze_returns(self, data_points: List[float]) -> Dict[str, Any]:
        """收益性分析"""
        if not data_points or len(data_points) < 2:
            return {'status': 'insufficient_data'}

        # 计算收益率序列
        returns = []
        for i in range(1, len(data_points)):
            if data_points[i-1] != 0:
                ret = (data_points[i] - data_points[i-1]) / data_points[i-1]
                returns.append(ret)

        if not returns:
            return {'status': 'no_valid_returns'}

        # 总收益率
        total_return = (data_points[-1] - data_points[0]) / data_points[0] * 100 if data_points[0] != 0 else 0

        # 年化收益率（假设数据为月度）
        periods = len(returns)
        annualized_return = ((1 + total_return/100) ** (12/periods) - 1) * 100 if periods > 0 else 0

        # 收益质量指标
        return_quality = {
            'volatility': statistics.stdev(returns) * 100 if len(returns) > 1 else 0,
            'positive_periods': sum(1 for r in returns if r > 0),
            'negative_periods': sum(1 for r in returns if r < 0),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns) * 100 if returns else 0
        }

        return {
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return, 2),
            'return_quality': {
                'volatility': round(return_quality['volatility'], 2),
                'win_rate': round(return_quality['win_rate'], 1),
                'positive_periods': return_quality['positive_periods'],
                'negative_periods': return_quality['negative_periods']
            }
        }

    def _analyze_risk(self, data_points: List[float]) -> Dict[str, Any]:
        """风险分析"""
        if not data_points or len(data_points) < 2:
            return {'status': 'insufficient_data'}

        # 计算收益率序列
        returns = []
        for i in range(1, len(data_points)):
            if data_points[i-1] != 0:
                ret = (data_points[i] - data_points[i-1]) / data_points[i-1]
                returns.append(ret)

        if not returns:
            return {'status': 'no_valid_returns'}

        # 波动率
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0

        # 最大回撤
        max_drawdown = 0
        peak = data_points[0]

        for value in data_points:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100 if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        # 下行风险（只考虑负收益）
        negative_returns = [r for r in returns if r < 0]
        downside_risk = statistics.stdev(negative_returns) if len(negative_returns) > 1 else 0

        return {
            'volatility': round(volatility * 100, 2),
            'max_drawdown': round(max_drawdown, 2),
            'downside_risk': round(downside_risk * 100, 2),
            'risk_level': self._assess_risk_level(volatility, max_drawdown)
        }

    def _calculate_risk_adjusted_metrics(self, data_points: List[float]) -> Dict[str, Any]:
        """计算风险调整后收益指标"""
        if not data_points or len(data_points) < 2:
            return {'status': 'insufficient_data'}

        # 计算收益率序列
        returns = []
        for i in range(1, len(data_points)):
            if data_points[i-1] != 0:
                ret = (data_points[i] - data_points[i-1]) / data_points[i-1]
                returns.append(ret)

        if not returns or len(returns) < 2:
            return {'status': 'insufficient_data'}

        # 计算平均收益率和波动率
        avg_return = statistics.mean(returns)
        volatility = statistics.stdev(returns)

        # 夏普比率 (年化)
        sharpe = (avg_return * 12 - self.risk_free_rate) / (volatility * (12 ** 0.5)) if volatility > 0 else 0

        # 索提诺比率（只考虑下行风险）
        negative_returns = [r for r in returns if r < 0]
        downside_deviation = statistics.stdev(negative_returns) if len(negative_returns) > 1 else 0

        sortino = (avg_return * 12 - self.risk_free_rate) / (downside_deviation * (12 ** 0.5)) if downside_deviation > 0 else 0

        # 卡玛比率 (收益 / 最大回撤)
        max_drawdown = 0
        peak = data_points[0]
        for value in data_points:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        total_return = (data_points[-1] - data_points[0]) / data_points[0] if data_points[0] != 0 else 0
        calmar = total_return / max_drawdown if max_drawdown != 0 else 0

        return {
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'calmar_ratio': round(calmar, 2),
            'evaluation': self._evaluate_risk_adjusted_performance(sharpe)
        }

    def _assess_risk_level(self, volatility: float, max_drawdown: float) -> str:
        """评估风险等级"""
        if volatility < 0.1 and max_drawdown < 10:
            return "低风险"
        elif volatility < 0.2 and max_drawdown < 20:
            return "中等风险"
        else:
            return "高风险"

    def _evaluate_risk_adjusted_performance(self, sharpe: float) -> str:
        """评估风险调整后收益表现"""
        if sharpe > 2:
            return "优秀"
        elif sharpe > 1:
            return "良好"
        elif sharpe > 0.5:
            return "一般"
        else:
            return "较差"

    def _generate_interpretation(self, data_points: List[float]) -> List[str]:
        """生成分析解读"""
        interpretations = []

        if not data_points or len(data_points) < 2:
            return ["数据不足，无法生成解读"]

        total_return = (data_points[-1] - data_points[0]) / data_points[0] * 100 if data_points[0] != 0 else 0

        # 收益表现解读
        if total_return > 20:
            interpretations.append(f"投资收益表现优异，总收益率达{total_return:.1f}%")
        elif total_return > 0:
            interpretations.append(f"投资获得正收益，总收益率为{total_return:.1f}%")
        elif total_return > -10:
            interpretations.append(f"投资小幅亏损，总收益率为{total_return:.1f}%")
        else:
            interpretations.append(f"投资出现较大亏损，总收益率为{total_return:.1f}%")

        # 趋势解读
        if len(data_points) > 3:
            first_third = data_points[:len(data_points)//3]
            last_third = data_points[-len(data_points)//3:]

            avg_first = sum(first_third) / len(first_third)
            avg_last = sum(last_third) / len(last_third)

            if avg_last > avg_first * 1.1:
                interpretations.append("整体呈上升趋势，后期表现优于前期")
            elif avg_last < avg_first * 0.9:
                interpretations.append("整体呈下降趋势，后期表现弱于前期")
            else:
                interpretations.append("整体表现相对平稳")

        return interpretations
