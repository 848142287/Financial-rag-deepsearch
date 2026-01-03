"""
金融指标抽取器 - 券商研报专用
提取和解析财务指标、市场指标、估值指标等
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

class MetricCategory(str, Enum):
    """指标类别"""
    PROFITABILITY = "profitability"  # 盈利能力
    GROWTH = "growth"  # 成长能力
    VALUATION = "valuation"  # 估值指标
    FINANCIAL_HEALTH = "financial_health"  # 财务健康
    EFFICIENCY = "efficiency"  # 运营效率
    MARKET_PERFORMANCE = "market_performance"  # 市场表现
    TECHNICAL = "technical"  # 技术指标

@dataclass
class FinancialMetric:
    """金融指标"""
    name: str  # 指标名称
    value: float  # 指标值
    unit: str  # 单位 (%, 元, 倍, etc.)
    category: MetricCategory  # 指标类别
    period: Optional[str] = None  # 时间周期 (2023Q1, 过去3年, etc.)
    confidence: float = 0.8  # 置信度
    context: str = ""  # 上下文
    source_text: str = ""  # 原始文本
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricExtractionResult:
    """指标抽取结果"""
    metrics: List[FinancialMetric]
    summary: Dict[str, Any]

class FinancialMetricsExtractor:
    """
    金融指标抽取器

    专门针对券商研报的指标提取：
    1. 盈利能力指标 (ROE, ROA, 毛利率, 净利率)
    2. 成长能力指标 (营收增长, 利润增长, CAGR)
    3. 估值指标 (PE, PB, PS, EV/EBITDA)
    4. 财务健康指标 (负债率, 流动比率)
    5. 运营效率指标 (周转率)
    6. 市场表现指标 (股价, 市值)
    """

    def __init__(self):
        self._compile_patterns()
        self._load_metric_definitions()

    def _compile_patterns(self):
        """预编译正则表达式"""
        self.patterns = {
            # 盈利能力指标
            MetricCategory.PROFITABILITY: [
                # ROE: 净资产收益率
                (
                    r'(?:ROE|净资产收益率|股东权益回报率)\s*[:：]?\s*([0-9.]+)%?',
                    ("ROE", "%", "净资产收益率")
                ),
                # ROA: 总资产收益率
                (
                    r'(?:ROA|总资产收益率)\s*[:：]?\s*([0-9.]+)%?',
                    ("ROA", "%", "总资产收益率")
                ),
                # 毛利率
                (
                    r'(?:毛利率|销售毛利率)\s*[:：]?\s*([0-9.]+)%?',
                    ("GrossMargin", "%", "毛利率")
                ),
                # 净利率
                (
                    r'(?:净利率|销售净利率)\s*[:：]?\s*([0-9.]+)%?',
                    ("NetMargin", "%", "净利率")
                ),
                # EBITDA利润率
                (
                    r'(?:EBITDA利润率|EBITDA\s*Margin)\s*[:：]?\s*([0-9.]+)%?',
                    ("EBITDAMargin", "%", "EBITDA利润率")
                ),
            ],

            # 成长能力指标
            MetricCategory.GROWTH: [
                # 营收增长
                (
                    r'(?:营收增长|收入增长|营业收入增长|YoY)\s*[:：]?\s*([+-]?[0-9.]+)%?',
                    ("RevenueGrowth", "%", "营收增长率")
                ),
                # 利润增长
                (
                    r'(?:净利润增长|归母净利润增长|利润增长)\s*[:：]?\s*([+-]?[0-9.]+)%?',
                    ("ProfitGrowth", "%", "净利润增长率")
                ),
                # CAGR
                (
                    r'(?:CAGR|年均复合增长率|复合增长率)\s*[:：]?\s*([0-9.]+)%?',
                    ("CAGR", "%", "年均复合增长率")
                ),
                # 同比/环比
                (
                    r'(?:同比|YoY)\s*[:：]?\s*([+-]?[0-9.]+)%?',
                    ("YoYGrowth", "%", "同比增长率")
                ),
                (
                    r'(?:环比|QoQ|季度环比)\s*[:：]?\s*([+-]?[0-9.]+)%?',
                    ("QoQGrowth", "%", "环比增长率")
                ),
            ],

            # 估值指标
            MetricCategory.VALUATION: [
                # PE: 市盈率
                (
                    r'(?:PE|市盈率|P/E)\s*[:：]?\s*([0-9.]+)(?:x|倍)?',
                    ("PE", "x", "市盈率")
                ),
                # PB: 市净率
                (
                    r'(?:PB|市净率|P/B)\s*[:：]?\s*([0-9.]+)(?:x|倍)?',
                    ("PB", "x", "市净率")
                ),
                # PS: 市销率
                (
                    r'(?:PS|市销率|P/S)\s*[:：]?\s*([0-9.]+)(?:x|倍)?',
                    ("PS", "x", "市销率")
                ),
                # EV/EBITDA
                (
                    r'(?:EV/EBITDA|企业价值倍数)\s*[:：]?\s*([0-9.]+)(?:x|倍)?',
                    ("EV_EBITDA", "x", "EV/EBITDA")
                ),
                # 股息率
                (
                    r'(?:股息率|分红收益率|Dividend\s*Yield)\s*[:：]?\s*([0-9.]+)%?',
                    ("DividendYield", "%", "股息率")
                ),
            ],

            # 财务健康指标
            MetricCategory.FINANCIAL_HEALTH: [
                # 资产负债率
                (
                    r'(?:资产负债率|负债率|Debt\s*Ratio)\s*[:：]?\s*([0-9.]+)%?',
                    ("DebtRatio", "%", "资产负债率")
                ),
                # 流动比率
                (
                    r'(?:流动比率|Current\s*Ratio)\s*[:：]?\s*([0-9.]+)',
                    ("CurrentRatio", "x", "流动比率")
                ),
                # 速动比率
                (
                    r'(?:速动比率|Quick\s*Ratio)\s*[:：]?\s*([0-9.]+)',
                    ("QuickRatio", "x", "速动比率")
                ),
                # 净负债率
                (
                    r'(?:净负债率|Net\s*Debt\s*Ratio)\s*[:：]?\s*([0-9.]+)%?',
                    ("NetDebtRatio", "%", "净负债率")
                ),
            ],

            # 运营效率指标
            MetricCategory.EFFICIENCY: [
                # 存货周转率
                (
                    r'(?:存货周转率|Inventory\s*Turnover)\s*[:：]?\s*([0-9.]+)(?:次)?',
                    ("InventoryTurnover", "次", "存货周转率")
                ),
                # 应收账款周转率
                (
                    r'(?:应收账款周转率|Receivable\s*Turnover)\s*[:：]?\s*([0-9.]+)(?:次)?',
                    ("ReceivableTurnover", "次", "应收账款周转率")
                ),
                # 总资产周转率
                (
                    r'(?:总资产周转率|Asset\s*Turnover)\s*[:：]?\s*([0-9.]+)(?:次)?',
                    ("AssetTurnover", "次", "总资产周转率")
                ),
            ],

            # 市场表现指标
            MetricCategory.MARKET_PERFORMANCE: [
                # 股价
                (
                    r'(?:股价|当前价|Price)\s*[:：]?\s*([0-9.]+)\s*(?:元|CNY)?',
                    ("StockPrice", "元", "股价")
                ),
                # 市值
                (
                    r'(?:市值|Market\s*Cap)\s*[:：]?\s*([0-9.]+)\s*(?:亿)?(?:元|CNY)?',
                    ("MarketCap", "亿元", "市值")
                ),
                # 目标价
                (
                    r'(?:目标价|TP|Target\s*Price)\s*[:：]?\s*([0-9.]+)\s*(?:元|HKD|USD)?',
                    ("TargetPrice", "元", "目标价")
                ),
                # 涨跌幅
                (
                    r'(?:涨跌幅|Change)\s*[:：]?\s*([+-]?[0-9.]+)%?',
                    ("PriceChange", "%", "涨跌幅")
                ),
            ],

            # 技术指标
            MetricCategory.TECHNICAL: [
                # 最大回撤
                (
                    r'(?:最大回撤|Max\s*Drawdown)\s*[:：]?\s*([0-9.]+)%?',
                    ("MaxDrawdown", "%", "最大回撤")
                ),
                # 夏普比率
                (
                    r'(?:夏普比率|Sharpe\s*Ratio)\s*[:：]?\s*([+-]?[0-9.]+)',
                    ("SharpeRatio", "", "夏普比率")
                ),
                # 波动率
                (
                    r'(?:波动率|Volatility)\s*[:：]?\s*([0-9.]+)%?',
                    ("Volatility", "%", "波动率")
                ),
                # Beta
                (
                    r'(?:Beta|贝塔)\s*[:：]?\s*([+-]?[0-9.]+)',
                    ("Beta", "", "Beta系数")
                ),
            ],
        }

        # 时间周期模式
        self.period_patterns = [
            r'(20\d{2})年',
            r'(20\d{2})[QQ]([1-4])',
            r'([一二三四])季度',
            r'(过去|近)\d+[年季度]',
            r'(上半年|下半年|H1|H2)',
            r'20\d{2}-[0-9]{1,2}',
        ]

    def _load_metric_definitions(self):
        """加载指标定义"""
        self.metric_definitions = {
            # 盈利能力
            "ROE": {
                "full_name": "净资产收益率",
                "formula": "净利润 / 平均净资产",
                "benchmark": {"good": 15, "average": 10, "poor": 5},
            },
            "GrossMargin": {
                "full_name": "毛利率",
                "benchmark": {"good": 40, "average": 25, "poor": 15},
            },
            # 成长能力
            "RevenueGrowth": {
                "full_name": "营收增长率",
                "benchmark": {"high": 30, "medium": 15, "low": 5},
            },
            # 估值
            "PE": {
                "full_name": "市盈率",
                "benchmark": {"low": 10, "medium": 20, "high": 30},
            },
            # 财务健康
            "DebtRatio": {
                "full_name": "资产负债率",
                "benchmark": {"safe": 40, "moderate": 60, "risky": 80},
            },
        }

    def extract_metrics(
        self,
        text: str,
        config: Optional[Dict[str, Any]] = None
    ) -> MetricExtractionResult:
        """
        从文本中提取金融指标

        Args:
            text: 输入文本
            config: 配置参数

        Returns:
            MetricExtractionResult
        """
        config = config or {}
        min_confidence = config.get("min_confidence", 0.6)

        all_metrics = []
        summary = {
            "total_metrics": 0,
            "by_category": {},
        }

        # 遍历所有指标类别
        for category, patterns in self.patterns.items():
            category_metrics = []

            for pattern, (name, unit, full_name) in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    try:
                        value = float(match.group(1))

                        # 提取上下文
                        context_start = max(0, match.start() - 50)
                        context_end = min(len(text), match.end() + 50)
                        context = text[context_start:context_end].strip()

                        # 提取时间周期
                        period = self._extract_period(context)

                        metric = FinancialMetric(
                            name=name,
                            value=value,
                            unit=unit,
                            category=category,
                            period=period,
                            confidence=self._calculate_metric_confidence(match, context),
                            context=context,
                            source_text=match.group(),
                            metadata={
                                "full_name": full_name,
                                "definition": self.metric_definitions.get(name, {}),
                            }
                        )

                        if metric.confidence >= min_confidence:
                            category_metrics.append(metric)

                    except (ValueError, IndexError) as e:
                        logger.debug(f"Failed to parse metric: {match.group()}, error: {e}")
                        continue

            # 去重同一类别的指标
            category_metrics = self._deduplicate_metrics(category_metrics)
            all_metrics.extend(category_metrics)

            summary["by_category"][category.value] = len(category_metrics)

        summary["total_metrics"] = len(all_metrics)

        # 生成洞察
        summary["insights"] = self._generate_insights(all_metrics)

        logger.info(
            f"提取到 {summary['total_metrics']} 个金融指标, "
            f"类别分布: {summary['by_category']}"
        )

        return MetricExtractionResult(metrics=all_metrics, summary=summary)

    def _extract_period(self, text: str) -> Optional[str]:
        """提取时间周期"""
        for pattern in self.period_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return None

    def _calculate_metric_confidence(
        self,
        match: re.Match,
        context: str
    ) -> float:
        """计算指标置信度"""
        base_confidence = 0.85

        # 检查上下文中的验证词
        verification_words = ["约为", "约", "约等于", "达到"]
        if any(word in context for word in verification_words):
            base_confidence -= 0.1

        # 检查明确的指标单位
        if "%" in match.group() or "倍" in match.group() or "x" in match.group().lower():
            base_confidence += 0.05

        return max(0.0, min(1.0, base_confidence))

    def _deduplicate_metrics(
        self,
        metrics: List[FinancialMetric]
    ) -> List[FinancialMetric]:
        """去重指标（保留最高置信度）"""
        metric_map = {}

        for metric in metrics:
            key = (metric.name, metric.period)

            if key not in metric_map or metric.confidence > metric_map[key].confidence:
                metric_map[key] = metric

        return list(metric_map.values())

    def _generate_insights(
        self,
        metrics: List[FinancialMetric]
    ) -> List[str]:
        """生成指标洞察"""
        insights = []

        # 按类别分组
        metrics_by_category = {}
        for metric in metrics:
            if metric.category not in metrics_by_category:
                metrics_by_category[metric.category] = []
            metrics_by_category[metric.category].append(metric)

        # 生成各类别洞察
        for category, category_metrics in metrics_by_category.items():
            if category == MetricCategory.VALUATION:
                pe_metrics = [m for m in category_metrics if m.name == "PE"]
                if pe_metrics:
                    avg_pe = sum(m.value for m in pe_metrics) / len(pe_metrics)
                    if avg_pe < 15:
                        insights.append(f"平均PE({avg_pe:.1f}x)处于较低水平，估值具备吸引力")
                    elif avg_pe > 30:
                        insights.append(f"平均PE({avg_pe:.1f}x)处于较高水平，估值偏贵")

            elif category == MetricCategory.GROWTH:
                revenue_growth = [m for m in category_metrics if m.name == "RevenueGrowth"]
                if revenue_growth:
                    avg_growth = sum(m.value for m in revenue_growth) / len(revenue_growth)
                    if avg_growth > 20:
                        insights.append(f"平均营收增长({avg_growth:.1f}%)表现优异")
                    elif avg_growth < 5:
                        insights.append(f"平均营收增长({avg_growth:.1f}%)放缓，需关注")

            elif category == MetricCategory.PROFITABILITY:
                roe_metrics = [m for m in category_metrics if m.name == "ROE"]
                if roe_metrics:
                    avg_roe = sum(m.value for m in roe_metrics) / len(roe_metrics)
                    if avg_roe > 15:
                        insights.append(f"平均ROE({avg_roe:.1f}%)处于优秀水平")
                    elif avg_roe < 8:
                        insights.append(f"平均ROE({avg_roe:.1f}%)偏低，盈利能力待提升")

        return insights

    def get_metric_summary(
        self,
        metrics: List[FinancialMetric]
    ) -> Dict[str, Any]:
        """
        获取指标汇总

        Args:
            metrics: 指标列表

        Returns:
            汇总信息
        """
        summary = {
            "total_count": len(metrics),
            "by_category": {},
            "key_metrics": {},
        }

        # 按类别统计
        for metric in metrics:
            category = metric.category.value
            if category not in summary["by_category"]:
                summary["by_category"][category] = []

            summary["by_category"][category].append({
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
            })

        # 提取关键指标
        key_metric_names = ["ROE", "PE", "RevenueGrowth", "DebtRatio", "MarketCap"]
        for metric in metrics:
            if metric.name in key_metric_names and metric.name not in summary["key_metrics"]:
                summary["key_metrics"][metric.name] = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "period": metric.period,
                }

        return summary

    async def extract_metrics_batch(
        self,
        texts: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> List[MetricExtractionResult]:
        """
        批量提取指标

        Args:
            texts: 文本列表
            config: 配置参数

        Returns:
            MetricExtractionResult列表
        """
        results = []

        for text in texts:
            result = self.extract_metrics(text, config)
            results.append(result)

        return results

# 全局实例
_financial_metrics_extractor: Optional[FinancialMetricsExtractor] = None

def get_financial_metrics_extractor() -> FinancialMetricsExtractor:
    """获取全局金融指标抽取器"""
    global _financial_metrics_extractor
    if _financial_metrics_extractor is None:
        _financial_metrics_extractor = FinancialMetricsExtractor()
    return _financial_metrics_extractor
