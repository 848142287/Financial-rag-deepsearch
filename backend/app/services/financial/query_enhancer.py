"""
金融查询增强器
基于Qwen2.5-VL-Embedding方案的查询增强功能
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """查询分析结果"""
    original_query: str
    query_type: str
    extracted_entities: Dict[str, List[str]]
    financial_metrics: List[str]
    time_info: Dict[str, Any]
    company_info: List[str]
    intent: str
    complexity: str


class FinancialQueryEnhancer:
    """金融查询增强器"""

    def __init__(self):
        # 公司数据库（示例）
        self.company_db = {
            "贵州茅台": {"stock_code": "600519.SH", "industry": "白酒"},
            "腾讯控股": {"stock_code": "00700.HK", "industry": "互联网"},
            "阿里巴巴": {"stock_code": "BABA", "industry": "电商"},
            "比亚迪": {"stock_code": "002594.SZ", "industry": "新能源汽车"},
            "宁德时代": {"stock_code": "300750.SZ", "industry": "电池"},
        }

        # 查询类型分类器
        self.query_classifiers = {
            "factual": ["什么是", "哪个", "谁", "哪里", "何时", "定义", "介绍"],
            "analytical": ["分析", "评估", "影响", "趋势", "原因", "为什么"],
            "comparative": ["比较", "对比", "差异", "区别", "优劣", "哪个好"],
            "temporal": ["最近", "过去", "未来", "趋势", "变化", "历史"],
            "forecast": ["预测", "预计", "预期", "展望", "未来"],
            "valuation": ["估值", "价值", "定价", "目标价", "值得买"],
        }

        # 财务指标词典
        self.financial_indicators = {
            "营收": ["营收", "营业收入", "收入", "销售额"],
            "净利润": ["净利润", "净利", "利润", "归母净利润"],
            "毛利率": ["毛利率", "毛利"],
            "净利率": ["净利率"],
            "ROE": ["ROE", "净资产收益率", "净资产回报率"],
            "PE": ["PE", "市盈率"],
            "PB": ["PB", "市净率"],
            "每股收益": ["每股收益", "EPS"],
            "现金流": ["现金流", "现金流量", "经营现金流"],
            "负债率": ["负债率", "资产负债率", "杠杆率"],
        }

        # 时间关键词
        self.time_keywords = {
            "year": ["年", "年度"],
            "quarter": ["季度", "Q1", "Q2", "Q3", "Q4"],
            "month": ["月", "月份"],
            "recent": ["最近", "近期", "近来", "最新"],
            "historical": ["过去", "以往", "历史"],
            "future": ["未来", "将来", "预期", "预测"],
            "yoy": ["同比", "YoY"],
            "qoq": ["环比", "QoQ", "MoM"],
        }

        # 金融术语扩展
        self.financial_synonyms = {
            "营收": ["营业收入", "销售收入", "营业额", "收入规模"],
            "利润": ["净利润", "盈利", "收益", "利润总额"],
            "估值": ["价值评估", "定价", "市值", "企业价值"],
            "风险": ["风险因素", "风险提示", "潜在风险", "挑战"],
        }

    def enhance_query(self, query: str) -> str:
        """增强金融查询"""
        try:
            # 1. 查询分析
            analysis = self.analyze_query(query)

            # 2. 构建增强查询
            enhanced_parts = []

            # 原始查询
            enhanced_parts.append(f"[原始查询] {query}")

            # 实体扩展
            if analysis["extracted_entities"]["companies"]:
                company_expansions = self._expand_companies(
                    analysis["extracted_entities"]["companies"]
                )
                if company_expansions:
                    enhanced_parts.append(f"[公司扩展] {' '.join(company_expansions[:3])}")

            # 指标扩展
            if analysis["financial_metrics"]:
                metric_expansions = self._expand_financial_metrics(
                    analysis["financial_metrics"]
                )
                if metric_expansions:
                    enhanced_parts.append(f"[指标扩展] {' '.join(metric_expansions[:3])}")

            # 时间扩展
            time_expansions = self._add_time_context(analysis["time_info"])
            if time_expansions:
                enhanced_parts.append(f"[时间范围] {time_expansions}")

            # 查询类型标记
            enhanced_parts.append(f"[查询类型] {analysis['query_type']}")
            enhanced_parts.append(f"[意图] {analysis['intent']}")

            return " | ".join(enhanced_parts)

        except Exception as e:
            logger.error(f"查询增强失败: {str(e)}")
            return f"[原始查询] {query} | [增强失败] false"

    def analyze_query(self, query: str) -> QueryAnalysis:
        """分析查询"""
        query_lower = query.lower()

        # 1. 分类查询类型
        query_type = self._classify_query(query_lower)

        # 2. 提取实体
        extracted_entities = {
            "companies": self._extract_companies(query),
            "stock_codes": self._extract_stock_codes(query),
            "industries": self._extract_industries(query),
        }

        # 3. 提取财务指标
        financial_metrics = self._extract_financial_metrics(query)

        # 4. 提取时间信息
        time_info = self._extract_time_info(query)

        # 5. 提取公司信息
        company_info = extracted_entities["companies"]

        # 6. 推断意图
        intent = self._infer_intent(query)

        # 7. 计算复杂度
        complexity = self._calculate_complexity(query, {
            "query_type": query_type,
            "entities_count": sum(len(v) for v in extracted_entities.values()),
            "metrics_count": len(financial_metrics),
            "has_time_info": bool(time_info.get("has_explicit_time")),
        })

        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            extracted_entities=extracted_entities,
            financial_metrics=financial_metrics,
            time_info=time_info,
            company_info=company_info,
            intent=intent,
            complexity=complexity
        )

    def _classify_query(self, query: str) -> str:
        """分类查询类型"""
        for query_type, keywords in self.query_classifiers.items():
            if any(keyword in query for keyword in keywords):
                return query_type
        return "general"

    def _extract_companies(self, query: str) -> List[str]:
        """提取公司名称"""
        companies = []
        for company_name in self.company_db.keys():
            if company_name in query:
                companies.append(company_name)
        return companies

    def _extract_stock_codes(self, query: str) -> List[str]:
        """提取股票代码"""
        stock_codes = []
        # A股代码
        a_stocks = re.findall(r'\b[0-9]{6}\.[SHZ]\b', query)
        stock_codes.extend(a_stocks)

        # 港股代码
        h_stocks = re.findall(r'\b[0-9]{5}\.[HK]\b', query)
        stock_codes.extend(h_stocks)

        # 美股代码
        us_stocks = re.findall(r'\b[A-Z]{1,5}\b', query)
        # 过滤掉常见的英文单词
        us_stocks = [code for code in us_stocks if len(code) <= 5 and code not in ["THE", "AND", "FOR", "WITH"]]
        stock_codes.extend(us_stocks)

        return stock_codes

    def _extract_industries(self, query: str) -> List[str]:
        """提取行业信息"""
        industries = []
        industry_keywords = [
            "白酒", "互联网", "电商", "新能源汽车", "电池", "银行",
            "保险", "证券", "房地产", "医药", "化工", "机械", "电子",
            "通信", "传媒", "教育", "旅游", "航空", "汽车", "钢铁"
        ]

        for industry in industry_keywords:
            if industry in query:
                industries.append(industry)

        return industries

    def _extract_financial_metrics(self, query: str) -> List[str]:
        """提取财务指标"""
        metrics = []
        query_lower = query.lower()

        for metric_name, variations in self.financial_indicators.items():
            for variation in variations:
                if variation in query_lower:
                    metrics.append(metric_name)
                    break

        return list(set(metrics))  # 去重

    def _extract_time_info(self, query: str) -> Dict[str, Any]:
        """提取时间信息"""
        time_info = {
            "has_explicit_time": False,
            "specific_dates": [],
            "time_keywords": [],
            "relative_time": None,
        }

        # 检查时间关键词
        for time_type, keywords in self.time_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    time_info["time_keywords"].append(time_type)
                    break

        # 提取具体年份
        year_matches = re.findall(r'(20\d{2})年', query)
        if year_matches:
            time_info["specific_dates"].extend(year_matches)
            time_info["has_explicit_time"] = True

        # 提取季度
        quarter_matches = re.findall(r'(Q[1-4]|第[一二三四]季度)', query)
        if quarter_matches:
            time_info["specific_dates"].extend(quarter_matches)
            time_info["has_explicit_time"] = True

        # 判断相对时间
        if "最近" in query or "最新" in query:
            time_info["relative_time"] = "recent"
        elif "历史" in query or "过去" in query:
            time_info["relative_time"] = "historical"
        elif "未来" in query or "预测" in query:
            time_info["relative_time"] = "future"

        return time_info

    def _infer_intent(self, query: str) -> str:
        """推断查询意图"""
        query_lower = query.lower()

        intent_patterns = {
            "comparison": ["对比", "比较", "vs", "相对于", "相比"],
            "forecast": ["预测", "预计", "预期", "展望", "未来"],
            "analysis": ["分析", "解析", "研究", "评估", "评价"],
            "trend": ["趋势", "变化", "走势", "发展"],
            "valuation": ["估值", "价值", "定价", "目标价"],
            "risk": ["风险", "问题", "挑战", "不利"],
            "recommendation": ["建议", "推荐", "买入", "卖出", "持有"],
        }

        for intent, keywords in intent_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent

        return "general"

    def _calculate_complexity(self, query: str, features: Dict) -> str:
        """计算查询复杂度"""
        complexity_score = 0

        # 长度因子
        if len(query) > 50:
            complexity_score += 2
        elif len(query) > 20:
            complexity_score += 1

        # 实体数量因子
        if features["entities_count"] > 3:
            complexity_score += 2
        elif features["entities_count"] > 1:
            complexity_score += 1

        # 指标数量因子
        if features["metrics_count"] > 2:
            complexity_score += 2
        elif features["metrics_count"] > 0:
            complexity_score += 1

        # 时间信息因子
        if features["has_time_info"]:
            complexity_score += 1

        # 查询类型因子
        if features["query_type"] in ["analytical", "comparative", "forecast"]:
            complexity_score += 1

        # 分类
        if complexity_score >= 5:
            return "complex"
        elif complexity_score >= 3:
            return "moderate"
        else:
            return "simple"

    def _expand_companies(self, companies: List[str]) -> List[str]:
        """扩展公司信息"""
        expansions = []
        for company in companies:
            if company in self.company_db:
                company_info = self.company_db[company]
                expansions.append(company)
                expansions.append(company_info["stock_code"])
                expansions.append(company_info["industry"])
        return expansions

    def _expand_financial_metrics(self, metrics: List[str]) -> List[str]:
        """扩展财务指标"""
        expansions = []
        for metric in metrics:
            expansions.append(metric)
            if metric in self.financial_synonyms:
                expansions.extend(self.financial_synonyms[metric])
        return expansions

    def _add_time_context(self, time_info: Dict) -> str:
        """添加时间上下文"""
        if time_info.get("relative_time") == "recent":
            return "近期 最新"
        elif time_info.get("relative_time") == "historical":
            return "历史 过去"
        elif time_info.get("relative_time") == "future":
            return "未来 预测"
        return ""


# 全局实例
financial_query_enhancer = FinancialQueryEnhancer()