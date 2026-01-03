"""
金融实体抽取器 - 券商研报专用
增强的实体识别，针对金融领域优化
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

class FinancialEntityType(str, Enum):
    """金融实体类型"""
    # 公司相关
    COMPANY = "company"  # 公司名称
    STOCK_CODE = "stock_code"  # 股票代码
    TICKER_SYMBOL = "ticker_symbol"  # 股票简称

    # 财务指标
    FINANCIAL_RATIO = "financial_ratio"  # 财务比率 (PE, PB, ROE等)
    REVENUE_METRIC = "revenue_metric"  # 营收指标
    PROFIT_METRIC = "profit_metric"  # 利润指标
    GROWTH_RATE = "growth_rate"  # 增长率
    VALUATION_METRIC = "valuation_metric"  # 估值指标

    # 投资相关
    INVESTMENT_STRATEGY = "investment_strategy"  # 投资策略
    RISK_FACTOR = "risk_factor"  # 风险因素
    RATING = "rating"  # 评级 (买入/卖出/持有)

    # 市场相关
    MARKET_INDEX = "market_index"  # 市场指数
    SECTOR = "sector"  # 行业板块
    MARKET_CONCEPT = "market_concept"  # 市场概念

    # 分析相关
    ANALYST_VIEW = "analyst_view"  # 分析师观点
    FORECAST = "forecast"  # 预测
    TARGET_PRICE = "target_price"  # 目标价

    # 时间
    TIME_PERIOD = "time_period"  # 时间周期
    REPORT_DATE = "report_date"  # 报告日期

@dataclass
class FinancialEntity:
    """金融实体"""
    text: str  # 实体文本
    entity_type: FinancialEntityType  # 实体类型
    confidence: float  # 置信度
    source: str  # 来源文本片段
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    start_pos: int = 0  # 开始位置
    end_pos: int = 0  # 结束位置

@dataclass
class ExtractionResult:
    """抽取结果"""
    entities: List[FinancialEntity]
    stats: Dict[str, Any]

class FinancialEntityExtractor:
    """
    金融实体抽取器

    专门针对券商研报的实体识别：
    1. 股票代码和公司名称
    2. 财务指标和比率
    3. 估值指标
    4. 投资评级
    5. 目标价格
    6. 时间周期
    """

    def __init__(self):
        self._compile_patterns()
        self._load_company_aliases()
        self._load_financial_terms()

    def _compile_patterns(self):
        """预编译正则表达式"""
        self.patterns = {
            # 股票代码
            FinancialEntityType.STOCK_CODE: [
                re.compile(r'\b[0-9]{6}\.[A-Z]{2}\b'),  # A股: 000001.SZ
                re.compile(r'\b[0-9]{6}\b'),  # 纯数字
                re.compile(r'\b[A-Z]{4}-[0-9]{5}\b'),  # 港股: 00700-HK
                re.compile(r'\b[A-Z]{1,4}\b'),  # 简称: AAPL
            ],

            # 财务比率
            FinancialEntityType.FINANCIAL_RATIO: [
                re.compile(r'\b(PE|PB|ROE|ROA|EPS|BPS)\s*[:：]?\s*[0-9.]+'),
                re.compile(r'(市盈率|市净率|净资产收益率|总资产收益率)\s*[:：]?\s*[0-9.]+'),
                re.compile(r'(毛利率|净利率|负债率|流动比率|速动比率)\s*[:：]?\s*[0-9.]+%?'),
            ],

            # 营收指标
            FinancialEntityType.REVENUE_METRIC: [
                re.compile(r'(营业收入|营收|总收入)\s*[:：]?\s*([0-9.]+(万亿千百)?元)'),
                re.compile(r'(YoY|同比)\s*([0-9.]+)%'),
            ],

            # 利润指标
            FinancialEntityType.PROFIT_METRIC: [
                re.compile(r'(净利润|归母净利润|扣非净利润)\s*[:：]?\s*([0-9.]+(万亿千百)?元)'),
                re.compile(r'(毛利率|净利率)\s*[:：]?\s*([0-9.]+)%'),
            ],

            # 增长率
            FinancialEntityType.GROWTH_RATE: [
                re.compile(r'(同比增长|环比增长|YoY|QoQ)\s*[:：]?\s*([+-]?[0-9.]+)%'),
                re.compile(r'(CAGR|年均复合增长率)\s*[:：]?\s*([0-9.]+)%'),
            ],

            # 估值指标
            FinancialEntityType.VALUATION_METRIC: [
                re.compile(r'(目标价|TP|Target Price)\s*[:：]?\s*([0-9.]+)(元|HKD|USD)?'),
                re.compile(r'(市值|Market Cap)\s*[:：]?\s*([0-9.]+)(亿)?(元|HKD|USD)?'),
                re.compile(r'(PE|PB)\s*[:：]?\s*([0-9.]+)x'),
            ],

            # 投资评级
            FinancialEntityType.RATING: [
                re.compile(r'\b(买入|卖出|持有|增持|减持|中性|跑赢|跑输)\b'),
                re.compile(r'\b(Buy|Sell|Hold|Overweight|Underweight|Neutral)\b', re.IGNORECASE),
                re.compile(r'(评级|Rating)\s*[:：]?\s*(买入|卖出|持有)'),
            ],

            # 市场指数
            FinancialEntityType.MARKET_INDEX: [
                re.compile(r'(上证指数|深证成指|创业板指|科创50|沪深300|中证500)'),
                re.compile(r'(恒生指数|国企指数|红筹指数)'),
                re.compile(r'(纳斯达克|标普500|道琼斯|Russell)\s*\d*'),
            ],

            # 行业板块
            FinancialEntityType.SECTOR: [
                re.compile(r'(新能源|半导体|人工智能|医药|消费|金融|地产|制造)'),
                re.compile(r'(TMT|互联网|软件服务|电子元件|通信设备)'),
            ],

            # 投资策略
            FinancialEntityType.INVESTMENT_STRATEGY: [
                re.compile(r'(价值投资|成长投资|量化投资|指数投资|长期投资)'),
                re.compile(r'(动量策略|均值回归|阿尔法策略|对冲策略|套利策略)'),
            ],

            # 风险因素
            FinancialEntityType.RISK_FACTOR: [
                re.compile(r'(市场风险|政策风险|流动性风险|信用风险|操作风险)'),
                re.compile(r'(竞争加剧|成本上升|需求下滑|监管变化)'),
            ],

            # 分析师观点
            FinancialEntityType.ANALYST_VIEW: [
                re.compile(r'(我们看好|我们看好|我们认为|预计|预测|预期)'),
                re.compile(r'(维持|上调|下调)\s*(评级|目标价)'),
            ],

            # 预测
            FinancialEntityType.FORECAST: [
                re.compile(r'(预计|预期|预测|展望)\s*([0-9.]+)(%|元)?'),
                re.compile(r'(未来\d+年|20\d{2}年)\s*(预计|预期)'),
            ],

            # 目标价
            FinancialEntityType.TARGET_PRICE: [
                re.compile(r'(目标价|TP)\s*[:：]?\s*([0-9.]+)(元|HKD|USD)?'),
                re.compile(r'(给予|维持)\s*([0-9.]+)\s*元.*目标价'),
            ],

            # 时间周期
            FinancialEntityType.TIME_PERIOD: [
                re.compile(r'(20\d{2})[年\-]([1-9]|1[0-2])[月\-]?([1-3]?[0-9])?日?'),
                re.compile(r'([1-4]?[0-9]季度|[一二三四]季度)'),
                re.compile(r'(上半年|下半年|H1|H2)\s*20\d{2}'),
                re.compile(r'(过去|近)\d+[年季度]'),
            ],

            # 报告日期
            FinancialEntityType.REPORT_DATE: [
                re.compile(r'(报告日期|发布日期|Date)\s*[:：]?\s*(20\d{2}[/-]\d{1,2}[/-]?\d{0,2})'),
            ],
        }

    def _load_company_aliases(self):
        """加载公司别名映射"""
        self.company_aliases = {
            "腾讯": ["腾讯控股", "腾讯", "0700.HK", "Tencent"],
            "阿里巴巴": ["阿里巴巴", "阿里", "BABA", "9988.HK"],
            "茅台": ["贵州茅台", "茅台", "600519.SH", "Moutai"],
            # 可以继续添加更多
        }

    def _load_financial_terms(self):
        """加载金融术语词典"""
        self.financial_terms = {
            # 估值术语
            "PE": ["市盈率", "P/E", "Price-to-Earnings"],
            "PB": ["市净率", "P/B", "Price-to-Book"],
            "PS": ["市销率", "P/S", "Price-to-Sales"],
            "EV_EBITDA": ["EV/EBITDA", "企业价值倍数"],

            # 盈利能力
            "ROE": ["净资产收益率", "Return on Equity"],
            "ROA": ["总资产收益率", "Return on Assets"],
            "ROIC": ["投入资本回报率", "Return on Invested Capital"],
            "GROSS_MARGIN": ["毛利率", "Gross Margin"],
            "NET_MARGIN": ["净利率", "Net Margin", "净利率"],

            # 成长能力
            "REVENUE_GROWTH": ["营收增长", "收入增长", "Revenue Growth"],
            "PROFIT_GROWTH": ["利润增长", "净利润增长", "Profit Growth"],
            "CAGR": ["年均复合增长率", "Compound Annual Growth Rate"],

            # 偿债能力
            "DEBT_RATIO": ["资产负债率", "Debt Ratio"],
            "CURRENT_RATIO": ["流动比率", "Current Ratio"],
            "QUICK_RATIO": ["速动比率", "Quick Ratio"],

            # 营运能力
            "INVENTORY_TURNOVER": ["存货周转率", "Inventory Turnover"],
            "RECEIVABLE_TURNOVER": ["应收账款周转率", "Receivable Turnover"],
        }

    def extract_entities(
        self,
        text: str,
        config: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        从文本中提取金融实体

        Args:
            text: 输入文本
            config: 配置参数

        Returns:
            ExtractionResult包含实体列表和统计信息
        """
        config = config or {}
        min_confidence = config.get("min_confidence", 0.6)
        include_low_confidence = config.get("include_low_confidence", False)

        entities = []
        stats = {
            "total_entities": 0,
            "by_type": {},
        }

        # 遍历所有实体类型
        for entity_type, patterns in self.patterns.items():
            type_entities = []

            for pattern in patterns:
                matches = pattern.finditer(text)

                for match in matches:
                    # 计算置信度（基于匹配质量）
                    confidence = self._calculate_confidence(match, entity_type)

                    # 过滤低置信度实体
                    if confidence >= min_confidence or include_low_confidence:
                        entity = FinancialEntity(
                            text=match.group(),
                            entity_type=entity_type,
                            confidence=confidence,
                            source=text[max(0, match.start() - 50):min(len(text), match.end() + 50)],
                            start_pos=match.start(),
                            end_pos=match.end(),
                            metadata=self._extract_entity_metadata(match, entity_type)
                        )

                        type_entities.append(entity)

            # 去重（基于文本内容）
            unique_entities = self._deduplicate_entities(type_entities)
            entities.extend(unique_entities)

            stats["by_type"][entity_type.value] = len(unique_entities)

        stats["total_entities"] = len(entities)

        logger.info(
            f"提取到 {stats['total_entities']} 个金融实体, "
            f"类型分布: {stats['by_type']}"
        )

        return ExtractionResult(entities=entities, stats=stats)

    def _calculate_confidence(
        self,
        match: re.Match,
        entity_type: FinancialEntityType
    ) -> float:
        """
        计算实体置信度

        Args:
            match: 正则匹配
            entity_type: 实体类型

        Returns:
            置信度分数 (0-1)
        """
        base_confidence = 0.8

        # 根据实体类型调整
        type_confidence = {
            FinancialEntityType.STOCK_CODE: 0.95,
            FinancialEntityType.FINANCIAL_RATIO: 0.85,
            FinancialEntityType.TARGET_PRICE: 0.90,
            FinancialEntityType.RATING: 0.75,
            FinancialEntityType.REPORT_DATE: 0.90,
        }

        confidence = type_confidence.get(entity_type, base_confidence)

        # 根据匹配长度调整
        match_text = match.group()
        if len(match_text) < 3:
            confidence -= 0.2
        elif len(match_text) > 20:
            confidence -= 0.1

        # 根据上下文调整
        source = match.group(0)
        context_words = ["预计", "预期", "达到", "约为"]
        if any(word in source for word in context_words):
            confidence += 0.05

        return max(0.0, min(1.0, confidence))

    def _extract_entity_metadata(
        self,
        match: re.Match,
        entity_type: FinancialEntityType
    ) -> Dict[str, Any]:
        """
        提取实体元数据

        Args:
            match: 正则匹配
            entity_type: 实体类型

        Returns:
            元数据字典
        """
        metadata = {
            "entity_type": entity_type.value,
            "matched_text": match.group(),
        }

        # 根据类型提取特定元数据
        if entity_type == FinancialEntityType.STOCK_CODE:
            metadata["market"] = self._detect_stock_market(match.group())
        elif entity_type == FinancialEntityType.FINANCIAL_RATIO:
            metadata["ratio_name"], metadata["ratio_value"] = self._parse_ratio(match.group())
        elif entity_type == FinancialEntityType.TARGET_PRICE:
            metadata["currency"] = self._detect_currency(match.group())
            metadata["price"] = self._extract_price_value(match.group())

        return metadata

    def _detect_stock_market(self, code: str) -> str:
        """检测股票市场"""
        if ".SH" in code or code.startswith("6"):
            return "SH"
        elif ".SZ" in code or code.startswith(("0", "3")):
            return "SZ"
        elif ".HK" in code or "-" in code:
            return "HK"
        else:
            return "UNKNOWN"

    def _parse_ratio(self, text: str) -> tuple:
        """解析财务比率"""
        # 提取比率名称和值
        match = re.search(r'([A-Z]+|[^\d]+)\s*[:：]?\s*([0-9.]+)', text)
        if match:
            return match.group(1), float(match.group(2))
        return None, None

    def _detect_currency(self, text: str) -> str:
        """检测货币类型"""
        if "HKD" in text or "港元" in text:
            return "HKD"
        elif "USD" in text or "美元" in text:
            return "USD"
        else:
            return "CNY"

    def _extract_price_value(self, text: str) -> Optional[float]:
        """提取价格数值"""
        match = re.search(r'([0-9]+\.?[0-9]*)', text)
        if match:
            return float(match.group(1))
        return None

    def _deduplicate_entities(
        self,
        entities: List[FinancialEntity]
    ) -> List[FinancialEntity]:
        """
        去重实体

        Args:
            entities: 实体列表

        Returns:
            去重后的实体列表
        """
        seen = set()
        unique_entities = []

        for entity in entities:
            # 使用文本和类型作为唯一标识
            key = (entity.text, entity.entity_type)

            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    async def extract_entities_batch(
        self,
        texts: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> List[ExtractionResult]:
        """
        批量提取实体

        Args:
            texts: 文本列表
            config: 配置参数

        Returns:
            ExtractionResult列表
        """
        results = []

        for text in texts:
            result = self.extract_entities(text, config)
            results.append(result)

        return results

# 全局实例
_financial_entity_extractor: Optional[FinancialEntityExtractor] = None

def get_financial_entity_extractor() -> FinancialEntityExtractor:
    """获取全局金融实体抽取器"""
    global _financial_entity_extractor
    if _financial_entity_extractor is None:
        _financial_entity_extractor = FinancialEntityExtractor()
    return _financial_entity_extractor
