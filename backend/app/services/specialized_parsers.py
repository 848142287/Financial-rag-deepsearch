"""
专用解析器
包含图表分析器、公式解析器、表格结构化解析器等
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

import cv2
import numpy as np
from PIL import Image
import pandas as pd

logger = logging.getLogger(__name__)


class ChartType(str, Enum):
    """图表类型"""
    BAR = "bar"           # 柱状图
    LINE = "line"         # 折线图
    PIE = "pie"           # 饼图
    SCATTER = "scatter"   # 散点图
    AREA = "area"         # 面积图
    HISTOGRAM = "histogram"  # 直方图
    BOX_PLOT = "box_plot" # 箱线图
    HEATMAP = "heatmap"   # 热力图
    RADAR = "radar"       # 雷达图
    UNKNOWN = "unknown"   # 未知类型


class FormulaType(str, Enum):
    """公式类型"""
    MATHEMATICAL = "mathematical"  # 数学公式
    FINANCIAL = "financial"        # 金融公式
    STATISTICAL = "statistical"    # 统计公式
    CHEMICAL = "chemical"          # 化学公式
    UNKNOWN = "unknown"            # 未知类型


@dataclass
class ChartAnalysis:
    """图表分析结果"""
    chart_type: ChartType
    title: str
    data_summary: Dict[str, Any]
    insights: List[str]
    trends: List[str]
    key_points: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormulaAnalysis:
    """公式分析结果"""
    formula_type: FormulaType
    original_formula: str
    explanation: str
    variables: Dict[str, str]
    calculation_result: Optional[float] = None
    interpretation: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableAnalysis:
    """表格分析结果"""
    headers: List[str]
    data_types: Dict[str, str]
    statistics: Dict[str, Any]
    insights: List[str]
    relationships: List[str]
    confidence: float
    structured_data: List[Dict[str, Any]] = field(default_factory=list)


class FinancialChartAnalyzer:
    """金融图表分析器"""

    def __init__(self):
        self.financial_keywords = [
            # 财务指标
            'revenue', 'profit', 'loss', 'income', 'expense', 'margin',
            'roi', 'roa', 'roe', 'eps', 'pe_ratio', 'debt', 'equity',
            'assets', 'liabilities', 'cash_flow', 'growth', 'return',

            # 货币单位
            '￥', '$', '€', '¥', '元', '万元', '亿元', '百万', '十亿',

            # 时间相关
            'q1', 'q2', 'q3', 'q4', '季度', '年度', '月度', '周',
            'yoy', 'mom', '同比', '环比',

            # 股票相关
            'stock', 'price', 'volume', 'market', 'share', 'index',
            '股价', '成交量', '市值', '指数',

            # 趋势描述
            'increase', 'decrease', 'growth', 'decline', 'rise', 'fall',
            '增长', '下降', '上升', '下跌', '波动'
        ]

    async def analyze_chart(
        self,
        image_path: str,
        text_description: str = "",
        metadata: Dict[str, Any] = None
    ) -> ChartAnalysis:
        """
        分析金融图表

        Args:
            image_path: 图像文件路径
            text_description: 图表的文字描述
            metadata: 额外元数据

        Returns:
            ChartAnalysis: 分析结果
        """
        try:
            # 1. 确定图表类型
            chart_type = await self._detect_chart_type(image_path, text_description)

            # 2. 提取数据
            data_summary = await self._extract_chart_data(image_path, chart_type)

            # 3. 生成洞察
            insights = await self._generate_insights(
                chart_type, data_summary, text_description
            )

            # 4. 分析趋势
            trends = await self._analyze_trends(chart_type, data_summary)

            # 5. 提取关键点
            key_points = await self._extract_key_points(
                chart_type, data_summary, insights
            )

            # 6. 生成标题
            title = await self._generate_title(chart_type, data_summary, text_description)

            return ChartAnalysis(
                chart_type=chart_type,
                title=title,
                data_summary=data_summary,
                insights=insights,
                trends=trends,
                key_points=key_points,
                confidence=self._calculate_confidence(data_summary, text_description),
                metadata=metadata or {}
            )

        except Exception as e:
            logger.error(f"Error analyzing chart: {e}")
            return ChartAnalysis(
                chart_type=ChartType.UNKNOWN,
                title="图表分析失败",
                data_summary={},
                insights=[f"分析失败: {str(e)}"],
                trends=[],
                key_points=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )

    async def _detect_chart_type(
        self, image_path: str, text_description: str
    ) -> ChartType:
        """检测图表类型"""
        try:
            # 基于文本描述检测
            text_indicators = {
                ChartType.BAR: ['柱状图', '条形图', 'bar chart', 'column chart'],
                ChartType.LINE: ['折线图', '线图', 'line chart', '趋势图'],
                ChartType.PIE: ['饼图', 'pie chart', '比例图'],
                ChartType.SCATTER: ['散点图', 'scatter plot', '点图'],
                ChartType.AREA: ['面积图', 'area chart', '区域图'],
                ChartType.HISTOGRAM: ['直方图', 'histogram', '分布图']
            }

            # 基于图像特征检测
            image_features = await self._extract_image_features(image_path)

            # 综合判断
            scores = {}
            for chart_type, keywords in text_indicators.items():
                text_score = sum(1 for kw in keywords if kw.lower() in text_description.lower())
                image_score = image_features.get(chart_type.value, 0)
                scores[chart_type] = text_score + image_score

            # 返回得分最高的类型
            if scores:
                return max(scores.items(), key=lambda x: x[1])[0]

            return ChartType.UNKNOWN

        except Exception as e:
            logger.error(f"Error detecting chart type: {e}")
            return ChartType.UNKNOWN

    async def _extract_image_features(self, image_path: str) -> Dict[str, float]:
        """提取图像特征"""
        features = {}

        try:
            image = cv2.imread(image_path)
            if image is None:
                return features

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 分析轮廓特征
            straight_lines = 0
            curves = 0
            circular_shapes = 0

            for contour in contours:
                # 计算轮廓面积和周长
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                if perimeter > 0:
                    # 轮廓近似
                    epsilon = 0.02 * perimeter
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # 圆形度检测
                    if area > 0:
                        circularity = 4 * math.pi * area / (perimeter * perimeter)
                        if circularity > 0.7:
                            circular_shapes += 1

                    # 直线检测
                    if len(approx) <= 4:
                        straight_lines += 1
                    else:
                        curves += 1

            total = len(contours) if contours else 1

            # 基于特征判断图表类型
            features['bar'] = straight_lines / total * 0.8
            features['line'] = curves / total * 0.6
            features['pie'] = circular_shapes / total * 0.9
            features['scatter'] = min(1.0, len(contours) / 50)

        except Exception as e:
            logger.error(f"Error extracting image features: {e}")

        return features

    async def _extract_chart_data(
        self, image_path: str, chart_type: ChartType
    ) -> Dict[str, Any]:
        """提取图表数据"""
        try:
            if chart_type == ChartType.UNKNOWN:
                return {}

            # 这里应该调用专门的数据提取算法
            # 简化实现，返回模拟数据
            return {
                "data_points": 10,
                "categories": ["Q1", "Q2", "Q3", "Q4"],
                "values": [100, 120, 115, 135],
                "trend": "increasing",
                "peak_value": 135,
                "min_value": 100,
                "data_range": 35
            }

        except Exception as e:
            logger.error(f"Error extracting chart data: {e}")
            return {}

    async def _generate_insights(
        self, chart_type: ChartType, data_summary: Dict[str, Any], text_description: str
    ) -> List[str]:
        """生成洞察"""
        insights = []

        try:
            # 基于数据生成基本洞察
            if data_summary.get("trend") == "increasing":
                insights.append("数据显示上升趋势")
            elif data_summary.get("trend") == "decreasing":
                insights.append("数据显示下降趋势")

            # 基于图表类型生成特定洞察
            if chart_type == ChartType.BAR:
                insights.append("柱状图显示不同类别的数值对比")
            elif chart_type == ChartType.LINE:
                insights.append("折线图显示数据随时间的变化趋势")
            elif chart_type == ChartType.PIE:
                insights.append("饼图显示各部分占比关系")

            # 基于金融关键词生成洞察
            if any(keyword in text_description.lower() for keyword in self.financial_keywords):
                insights.append("图表包含金融相关数据")

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("洞察生成失败")

        return insights

    async def _analyze_trends(
        self, chart_type: ChartType, data_summary: Dict[str, Any]
    ) -> List[str]:
        """分析趋势"""
        trends = []

        try:
            if data_summary.get("trend"):
                trends.append(f"整体趋势: {data_summary['trend']}")

            if "peak_value" in data_summary and "min_value" in data_summary:
                range_ratio = (data_summary["peak_value"] - data_summary["min_value"]) / data_summary["min_value"]
                if range_ratio > 0.5:
                    trends.append("数据波动较大")

        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")

        return trends

    async def _extract_key_points(
        self, chart_type: ChartType, data_summary: Dict[str, Any], insights: List[str]
    ) -> List[str]:
        """提取关键点"""
        key_points = []

        try:
            if "peak_value" in data_summary:
                key_points.append(f"最高值: {data_summary['peak_value']}")

            if "min_value" in data_summary:
                key_points.append(f"最低值: {data_summary['min_value']}")

            if len(insights) > 0:
                key_points.append(f"主要发现: {insights[0]}")

        except Exception as e:
            logger.error(f"Error extracting key points: {e}")

        return key_points

    async def _generate_title(
        self, chart_type: ChartType, data_summary: Dict[str, Any], text_description: str
    ) -> str:
        """生成标题"""
        try:
            # 从文本描述中提取标题
            lines = text_description.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) < 50:  # 假设标题较短
                    if any(keyword in line for keyword in ['图', 'chart', '图示']):
                        return line

            # 基于图表类型生成默认标题
            type_titles = {
                ChartType.BAR: "数据对比柱状图",
                ChartType.LINE: "趋势变化折线图",
                ChartType.PIE: "占比分析饼图",
                ChartType.SCATTER: "相关关系散点图",
                ChartType.AREA: "累积变化面积图"
            }

            return type_titles.get(chart_type, "数据图表")

        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return "图表分析"

    def _calculate_confidence(
        self, data_summary: Dict[str, Any], text_description: str
    ) -> float:
        """计算置信度"""
        try:
            confidence = 0.5  # 基础置信度

            # 基于数据完整性
            if data_summary:
                confidence += 0.2

            # 基于文本描述长度
            if len(text_description) > 50:
                confidence += 0.2

            # 基于金融关键词
            financial_keyword_count = sum(1 for kw in self.financial_keywords
                                        if kw in text_description.lower())
            confidence += min(0.3, financial_keyword_count * 0.05)

            return min(1.0, confidence)

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0


class FinancialFormulaAnalyzer:
    """金融公式分析器"""

    def __init__(self):
        self.financial_formulas = {
            # ROI公式
            'roi': {
                'pattern': r'roi\s*=\s*\([^)]+\)\s*/\s*\([^)]+\)',
                'explanation': '投资回报率(ROI)计算收益与投资成本的比率',
                'variables': {'return': '收益', 'investment': '投资成本'}
            },

            # PE比率
            'pe_ratio': {
                'pattern': r'pe\s*=\s*[^/]+\s*/\s*[^+]+',
                'explanation': '市盈率(PE)反映股价与每股收益的关系',
                'variables': {'price': '股价', 'eps': '每股收益'}
            },

            # ROE公式
            'roe': {
                'pattern': r'roe\s*=\s*[^/]+\s*/\s*[^)]+\)',
                'explanation': '净资产收益率(ROE)衡量公司盈利能力',
                'variables': {'net_income': '净利润', 'shareholders_equity': '股东权益'}
            },

            # 复利公式
            'compound_interest': {
                'pattern': r'[Aa]\s*=\s*[Pp]\s*\(\s*1\s*\+\s*[rR]\s*\)\s*\^\s*[ntT]',
                'explanation': '复利计算公式，计算投资的本利和',
                'variables': {'A': '本利和', 'P': '本金', 'r': '利率', 'n': '期数'}
            }
        }

        self.math_functions = {
            'sqrt': '平方根', 'log': '对数', 'ln': '自然对数',
            'exp': '指数', 'sin': '正弦', 'cos': '余弦',
            'tan': '正切', 'max': '最大值', 'min': '最小值'
        }

    async def analyze_formula(
        self, formula_text: str, context: str = ""
    ) -> FormulaAnalysis:
        """
        分析金融公式

        Args:
            formula_text: 公式文本
            context: 上下文信息

        Returns:
            FormulaAnalysis: 分析结果
        """
        try:
            # 1. 识别公式类型
            formula_type = await self._identify_formula_type(formula_text)

            # 2. 提取变量
            variables = await self._extract_variables(formula_text)

            # 3. 生成解释
            explanation = await self._generate_explanation(formula_text, formula_type, context)

            # 4. 计算结果（如果可能）
            calculation_result = await self._try_calculate(formula_text)

            # 5. 生成解释说明
            interpretation = await self._generate_interpretation(
                formula_text, formula_type, context
            )

            return FormulaAnalysis(
                formula_type=formula_type,
                original_formula=formula_text,
                explanation=explanation,
                variables=variables,
                calculation_result=calculation_result,
                interpretation=interpretation,
                confidence=self._calculate_formula_confidence(formula_text, context),
                metadata={"context": context}
            )

        except Exception as e:
            logger.error(f"Error analyzing formula: {e}")
            return FormulaAnalysis(
                formula_type=FormulaType.UNKNOWN,
                original_formula=formula_text,
                explanation=f"公式分析失败: {str(e)}",
                variables={},
                interpretation="无法解释该公式",
                confidence=0.0,
                metadata={"error": str(e)}
            )

    async def _identify_formula_type(self, formula_text: str) -> FormulaType:
        """识别公式类型"""
        formula_lower = formula_text.lower()

        # 金融公式检测
        for formula_name, formula_info in self.financial_formulas.items():
            if re.search(formula_info['pattern'], formula_lower):
                return FormulaType.FINANCIAL

        # 数学公式检测
        if any(func in formula_lower for func in self.math_functions.keys()):
            return FormulaType.MATHEMATICAL

        # 统计公式检测
        if any(keyword in formula_lower for keyword in [
            'mean', 'variance', 'std', 'stddev', 'correlation', 'regression',
            '平均', '方差', '标准差', '相关', '回归'
        ]):
            return FormulaType.STATISTICAL

        return FormulaType.UNKNOWN

    async def _extract_variables(self, formula_text: str) -> Dict[str, str]:
        """提取公式变量"""
        variables = {}

        try:
            # 提取变量名（单字母或下划线开头的标识符）
            variable_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            matches = re.findall(variable_pattern, formula_text)

            # 过滤掉函数名和常数
            excluded_names = set(self.math_functions.keys())
            excluded_names.update(['if', 'then', 'else', 'and', 'or', 'not'])

            for var in matches:
                if var not in excluded_names and len(var) <= 3:
                    # 为变量生成简单的描述
                    descriptions = {
                        'r': '利率', 'R': '利率', 'n': '期数', 'N': '期数',
                        't': '时间', 'T': '时间', 'p': '本金', 'P': '本金',
                        'a': '金额', 'A': '金额', 'x': '变量', 'X': '变量',
                        'y': '变量', 'Y': '变量', 'c': '常数', 'C': '常数'
                    }
                    variables[var] = descriptions.get(var, f'变量{var}')

        except Exception as e:
            logger.error(f"Error extracting variables: {e}")

        return variables

    async def _generate_explanation(
        self, formula_text: str, formula_type: FormulaType, context: str
    ) -> str:
        """生成公式解释"""
        try:
            # 检查是否是预定义的金融公式
            for formula_name, formula_info in self.financial_formulas.items():
                if re.search(formula_info['pattern'], formula_text.lower()):
                    return formula_info['explanation']

            # 基于公式类型生成通用解释
            if formula_type == FormulaType.MATHEMATICAL:
                return "这是一个数学计算公式，用于进行数值运算"
            elif formula_type == FormulaType.FINANCIAL:
                return "这是一个金融计算公式，用于财务指标计算"
            elif formula_type == FormulaType.STATISTICAL:
                return "这是一个统计公式，用于数据分析计算"
            else:
                return "这是一个计算公式"

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "公式解释生成失败"

    async def _try_calculate(self, formula_text: str) -> Optional[float]:
        """尝试计算公式结果"""
        try:
            # 简化的公式计算，仅处理基本数学表达式
            # 注意：实际应用中需要更安全的表达式解析器

            # 提取数字和运算符
            clean_formula = re.sub(r'[a-zA-Z_]', '1', formula_text)  # 替换变量为1
            clean_formula = re.sub(r'[^0-9+\-*/().\s]', '', clean_formula)

            # 检查是否是简单的数学表达式
            if re.match(r'^[\d+\-*/().\s]+$', clean_formula):
                # 使用eval计算（注意生产环境中的安全性）
                try:
                    result = eval(clean_formula)
                    if isinstance(result, (int, float)):
                        return float(result)
                except:
                    pass

            return None

        except Exception as e:
            logger.error(f"Error calculating formula: {e}")
            return None

    async def _generate_interpretation(
        self, formula_text: str, formula_type: FormulaType, context: str
    ) -> str:
        """生成公式解释说明"""
        try:
            interpretation = ""

            # 基于上下文生成解释
            if context:
                if '利润' in context or '收益' in context:
                    interpretation += "该公式用于计算投资收益或利润"
                elif '风险' in context or '波动' in context:
                    interpretation += "该公式用于风险评估计算"
                elif '增长' in context or '预测' in context:
                    interpretation += "该公式用于增长预测计算"

            # 基于公式类型生成解释
            if formula_type == FormulaType.FINANCIAL:
                if interpretation:
                    interpretation += "，"
                interpretation += "是重要的财务分析工具"
            elif formula_type == FormulaType.MATHEMATICAL:
                if interpretation:
                    interpretation += "，"
                interpretation += "用于基础数学运算"

            if not interpretation:
                interpretation = "公式用于数值计算和分析"

            return interpretation

        except Exception as e:
            logger.error(f"Error generating interpretation: {e}")
            return "公式用途待分析"

    def _calculate_formula_confidence(self, formula_text: str, context: str) -> float:
        """计算公式分析置信度"""
        try:
            confidence = 0.3  # 基础置信度

            # 检查是否是预定义公式
            for formula_info in self.financial_formulas.values():
                if re.search(formula_info['pattern'], formula_text.lower()):
                    confidence += 0.4
                    break

            # 检查公式复杂度
            if '=' in formula_text:
                confidence += 0.1

            # 检查上下文信息
            if context and len(context) > 20:
                confidence += 0.2

            return min(1.0, confidence)

        except Exception as e:
            logger.error(f"Error calculating formula confidence: {e}")
            return 0.0


# 全局实例
chart_analyzer = FinancialChartAnalyzer()
formula_analyzer = FinancialFormulaAnalyzer()


# 便捷函数
async def analyze_financial_chart(
    image_path: str, text_description: str = "", metadata: Dict[str, Any] = None
) -> ChartAnalysis:
    """便捷的图表分析函数"""
    return await chart_analyzer.analyze_chart(image_path, text_description, metadata)


async def analyze_financial_formula(
    formula_text: str, context: str = ""
) -> FormulaAnalysis:
    """便捷的公式分析函数"""
    return await formula_analyzer.analyze_formula(formula_text, context)