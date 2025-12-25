"""
Agentic RAG 计划阶段
理解用户意图，制定检索策略
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型"""
    FACT_FINDING = "fact_finding"        # 事实查找
    COMPARISON_ANALYSIS = "comparison_analysis"  # 比较分析
    TREND_PREDICTION = "trend_prediction"  # 趋势预测
    RELATIONSHIP_ANALYSIS = "relationship_analysis"  # 关系分析
    COMPREHENSIVE_RESEARCH = "comprehensive_research"  # 综合研究
    DEFINITION_EXPLANATION = "definition_explanation"  # 定义解释


class QueryComplexity(Enum):
    """查询复杂度"""
    SIMPLE = "simple"      # 简单
    MEDIUM = "medium"      # 中等
    COMPLEX = "complex"    # 复杂


@dataclass
class QueryAnalysis:
    """查询分析结果"""
    original_query: str
    query_type: QueryType
    complexity: QueryComplexity
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    time_constraints: Optional[str] = None
    domain: str = "general"
    confidence: float = 0.0


@dataclass
class RetrievalStrategy:
    """检索策略"""
    primary_method: str  # primary retrieval method
    secondary_methods: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    quality_threshold: float = 0.7
    max_results: int = 10
    expected_time: float = 0.0


@dataclass
class RetrievalPlan:
    """检索计划"""
    plan_id: str
    query_analysis: QueryAnalysis
    main_query: str
    alternative_queries: List[str] = field(default_factory=list)
    strategies: List[RetrievalStrategy] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    estimated_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class AgenticRAGPlanner:
    """Agentic RAG 计划器"""

    def __init__(self):
        # 金融术语词典
        self.financial_terms = {
            "synonyms": {
                "股票": ["股份", "股权", "个股", "证券"],
                "营收": ["营业收入", "收入", "销售额", "业绩"],
                "利润": ["盈利", "净利润", "收益", "收益"],
                "市盈率": ["PE", "本益比", "市盈率比率"],
                "市净率": ["PB", "市净率比率", "股价净值比"],
                "ROE": ["净资产收益率", "股东权益回报率"],
                "GDP": ["国内生产总值", "国民经济生产总值"],
                "CPI": ["居民消费价格指数", "消费物价指数"],
                "PPI": ["工业生产者出厂价格指数", "生产价格指数"]
            },
            "standardization": {
                "公司": ["企业", "股份制公司", "上市公司", "集团"],
                "行业": ["产业", "领域", "板块", "细分市场"],
                "市场": ["股市", "资本市场", "证券市场", "交易所"]
            }
        }

        # 查询类型识别规则
        self.query_type_patterns = {
            QueryType.FACT_FINDING: [
                r"(?:什么|什么是|如何|怎么|多少|哪个|是否).*(?:营收|利润|股价|市值|PE|ROE)",
                r"(?:查询|查找|搜索|获取).*(?:数据|信息|资料|数值)",
                r".*(?:是多少|是什么|在哪里|什么时候).*(?:公司|股票|基金)"
            ],
            QueryType.COMPARISON_ANALYSIS: [
                r"(?:比较|对比|差异|优劣|选择|哪个好).*(?:公司|股票|行业|产品)",
                r".*(?:和|与|vs|VS|对比).*(?:表现|业绩|数据|前景)",
                r"(?:两者|两者之间|三者之间).*(?:对比|比较)"
            ],
            QueryType.TREND_PREDICTION: [
                r"(?:预测|预期|展望|前景|趋势|未来).*(?:走势|发展|变化)",
                r".*(?:将.*?会|预计|大概|可能).*(?:上涨|下跌|增长|下降)",
                r"(?:长期|短期|中期).*(?:趋势|走势|展望)"
            ],
            QueryType.RELATIONSHIP_ANALYSIS: [
                r"(?:关系|影响|关联|联系|作用).*(?:之间|如何).*(?:影响)",
                r".*(?:对.*?的影响|如何影响|因为|所以)",
                r"(?:原因|结果|导致|造成).*(?:关系)"
            ],
            QueryType.COMPREHENSIVE_RESEARCH: [
                r"(?:分析|研究|评估|调查|报告).*(?:全面|综合|详细|深入)",
                r"(?:投资建议|投资分析|行业分析|公司分析).*(?:报告|研究)",
                r".*(?:多方面|全方位|多角度).*(?:分析|研究)"
            ]
        }

        # 复杂度评估规则
        self.complexity_indicators = {
            "simple": [
                r"^.{1,50}$",  # 短查询
                r"(?:什么是|多少|几个)$",  # 简单问题
                r"^(?:查询|查找)\s*\w+$"  # 简单查询
            ],
            "medium": [
                r".{51,150}",  # 中等长度
                r"(?:比较|对比).{1,100}",  # 简单比较
                r"(?:为什么|如何).{1,100}"  # 简单原因/方法
            ],
            "complex": [
                r".{150,}",  # 长查询
                r"(?:分析|研究|评估).+",  # 深度分析
                r".*(?:并且|而且|同时|以及|此外).+",  # 多条件
                r".*(?:因为|所以|导致|影响).+"  # 复杂关系
            ]
        }

    async def create_retrieval_plan(self, query: str, context: Optional[Dict] = None) -> RetrievalPlan:
        """
        创建检索计划

        Args:
            query: 用户查询
            context: 上下文信息

        Returns:
            RetrievalPlan: 检索计划
        """
        try:
            logger.info(f"开始创建检索计划: {query[:100]}...")

            # 1. 查询分析
            query_analysis = await self._analyze_query(query, context)

            # 2. 查询改写
            rewritten_queries = await self._rewrite_query(query_analysis)

            # 3. 策略选择
            strategies = await self._select_strategies(query_analysis)

            # 4. 约束设置
            constraints = await self._set_constraints(query_analysis)

            # 5. 时间估算
            estimated_time = self._estimate_time(strategies)

            # 6. 创建计划
            plan = RetrievalPlan(
                plan_id=self._generate_plan_id(),
                query_analysis=query_analysis,
                main_query=rewritten_queries["main"],
                alternative_queries=rewritten_queries["alternatives"],
                strategies=strategies,
                constraints=constraints,
                estimated_time=estimated_time,
                created_at=datetime.now()
            )

            logger.info(f"检索计划创建完成，计划ID: {plan.plan_id}")
            return plan

        except Exception as e:
            logger.error(f"创建检索计划失败: {str(e)}")
            # 返回默认计划
            return self._create_default_plan(query)

    async def _analyze_query(self, query: str, context: Optional[Dict] = None) -> QueryAnalysis:
        """分析查询"""
        # 提取实体
        entities = await self._extract_entities(query)

        # 提取关键词
        keywords = await self._extract_keywords(query)

        # 识别查询类型
        query_type = self._identify_query_type(query)

        # 评估复杂度
        complexity = self._assess_complexity(query)

        # 提取时间约束
        time_constraints = self._extract_time_constraints(query)

        # 识别领域
        domain = self._identify_domain(query)

        # 计算置信度
        confidence = self._calculate_confidence(query, query_type, complexity)

        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            complexity=complexity,
            entities=entities,
            keywords=keywords,
            time_constraints=time_constraints,
            domain=domain,
            confidence=confidence
        )

    async def _extract_entities(self, query: str) -> List[str]:
        """提取实体"""
        entities = []

        # 金融实体模式
        financial_entity_patterns = [
            r'[A-Z]{1,6}\d{4,6}',  # 股票代码
            r'\d{4}年|\d{4}/\d{1,2}|\d{4}-\d{1,2}',  # 日期
            r'[0-9.]+%|[0-9.,]+万亿?|[0-9.,]+亿',  # 数字和百分比
        ]

        for pattern in financial_entity_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)

        # 公司名称模式（简化）
        company_patterns = [
            r'[\u4e00-\u9fff]+(?:公司|集团|股份|有限|企业|控股)',
            r'[A-Z][a-z]+(?:Corporation|Inc|Ltd|Group|Company)'
        ]

        for pattern in company_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)

        return list(set(entities))

    async def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 移除停用词后的关键词提取
        stop_words = {'的', '了', '和', '在', '是', '我', '有', '要', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}

        # 简单分词
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', query.lower())

        # 过滤停用词和短词
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]

        # 同义词扩展
        expanded_keywords = []
        for keyword in keywords:
            expanded_keywords.append(keyword)
            for term_category in self.financial_terms.values():
                if keyword in term_category:
                    expanded_keywords.extend(term_category[keyword])

        return list(set(expanded_keywords))

    def _identify_query_type(self, query: str) -> QueryType:
        """识别查询类型"""
        query_lower = query.lower()

        # 计算每种类型的匹配分数
        type_scores = {}
        for query_type, patterns in self.query_type_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            type_scores[query_type] = score

        # 选择得分最高的类型
        if not type_scores or max(type_scores.values()) == 0:
            return QueryType.FACT_FINDING  # 默认类型

        best_type = max(type_scores, key=type_scores.get)
        return best_type

    def _assess_complexity(self, query: str) -> QueryComplexity:
        """评估查询复杂度"""
        query_length = len(query)
        pattern_count = 0

        # 计算复杂度指标
        for complexity, patterns in self.complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    pattern_count += 1

        # 综合判断
        if query_length <= 50 and pattern_count == 0:
            return QueryComplexity.SIMPLE
        elif query_length <= 150 and pattern_count <= 1:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.COMPLEX

    def _extract_time_constraints(self, query: str) -> Optional[str]:
        """提取时间约束"""
        time_patterns = [
            r'(\d{4}年)',
            r'(\d{4}/\d{1,2}/\d{1,2})',
            r'(?:最近|近期|近期|今年以来|今年|去年|前年)',
            r'(?:第一季度|第二季度|第三季度|第四季度|Q[1-4])',
            r'(?:上半年|下半年|年度|季度)'
        ]

        for pattern in time_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)

        return None

    def _identify_domain(self, query: str) -> str:
        """识别查询领域"""
        financial_keywords = ['股票', '基金', '债券', '投资', '金融', '银行', '保险', '证券', '市场', '经济']
        keyword_count = sum(1 for keyword in financial_keywords if keyword in query)

        if keyword_count >= 2:
            return "financial"
        elif keyword_count >= 1:
            return "mixed"
        else:
            return "general"

    def _calculate_confidence(self, query: str, query_type: QueryType, complexity: QueryComplexity) -> float:
        """计算分析置信度"""
        base_confidence = 0.7

        # 根据查询类型调整
        type_adjustments = {
            QueryType.FACT_FINDING: 0.1,
            QueryType.COMPARISON_ANALYSIS: 0.05,
            QueryType.TREND_PREDICTION: -0.05,
            QueryType.RELATIONSHIP_ANALYSIS: -0.1,
            QueryType.COMPREHENSIVE_RESEARCH: -0.15
        }

        # 根据复杂度调整
        complexity_adjustments = {
            QueryComplexity.SIMPLE: 0.1,
            QueryComplexity.MEDIUM: 0.0,
            QueryComplexity.COMPLEX: -0.1
        }

        confidence = base_confidence + type_adjustments.get(query_type, 0) + complexity_adjustments.get(complexity, 0)
        return max(0.1, min(1.0, confidence))

    async def _rewrite_query(self, query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """查询改写"""
        original_query = query_analysis.original_query

        # 同义词扩展
        expanded_query = original_query
        for keyword in query_analysis.keywords:
            for term_category in self.financial_terms["synonyms"].values():
                if keyword in term_category:
                    for synonym in term_category[keyword]:
                        if synonym != keyword:
                            expanded_query += f" {synonym}"

        # 意图明确化
        clarified_query = self._clarify_intent(original_query, query_analysis)

        # 备选查询
        alternative_queries = [
            expanded_query,
            clarified_query,
            f"{query_analysis.domain} {original_query}"
        ]

        return {
            "main": clarified_query,
            "expanded": expanded_query,
            "alternatives": alternative_queries[2:]  # 除了主要的两个
        }

    def _clarify_intent(self, query: str, query_analysis: QueryAnalysis) -> str:
        """明确查询意图"""
        # 根据查询类型添加明确的意图词
        intent_prefixes = {
            QueryType.FACT_FINDING: ["查询", "获取", "查找"],
            QueryType.COMPARISON_ANALYSIS: ["比较", "对比", "分析"],
            QueryType.TREND_PREDICTION: ["分析", "预测", "展望"],
            QueryType.RELATIONSHIP_ANALYSIS: ["分析", "研究", "探讨"],
            QueryType.COMPREHENSIVE_RESEARCH: ["全面分析", "深入研究", "综合评估"]
        }

        prefix = intent_prefixes.get(query_analysis.query_type, ["分析"])[0]

        return f"{prefix}：{query}"

    async def _select_strategies(self, query_analysis: QueryAnalysis) -> List[RetrievalStrategy]:
        """选择检索策略"""
        strategies = []

        # 根据查询类型选择主要策略
        if query_analysis.query_type == QueryType.FACT_FINDING:
            primary_strategy = RetrievalStrategy(
                primary_method="vector_search",
                secondary_methods=["keyword_search"],
                parameters={"top_k": 10, "similarity_threshold": 0.7},
                quality_threshold=0.8,
                max_results=10,
                expected_time=2.0
            )
        elif query_analysis.query_type == QueryType.COMPARISON_ANALYSIS:
            primary_strategy = RetrievalStrategy(
                primary_method="hybrid_search",
                secondary_methods=["graph_search", "vector_search"],
                parameters={"graph_depth": 2, "top_k": 15, "similarity_threshold": 0.6},
                quality_threshold=0.75,
                max_results=15,
                expected_time=5.0
            )
        elif query_analysis.query_type == QueryType.TREND_PREDICTION:
            primary_strategy = RetrievalStrategy(
                primary_method="temporal_search",
                secondary_methods=["vector_search", "graph_search"],
                parameters={"time_range": "recent_2_years", "top_k": 20},
                quality_threshold=0.7,
                max_results=20,
                expected_time=8.0
            )
        elif query_analysis.query_type == QueryType.RELATIONSHIP_ANALYSIS:
            primary_strategy = RetrievalStrategy(
                primary_method="graph_search",
                secondary_methods=["vector_search"],
                parameters={"max_depth": 3, "entity_expansion": True},
                quality_threshold=0.75,
                max_results=15,
                expected_time=6.0
            )
        else:  # COMPREHENSIVE_RESEARCH
            primary_strategy = RetrievalStrategy(
                primary_method="multi_modal_search",
                secondary_methods=["vector_search", "graph_search", "keyword_search", "temporal_search"],
                parameters={"comprehensive": True, "max_depth": 4, "top_k": 25},
                quality_threshold=0.7,
                max_results=25,
                expected_time=10.0
            )

        strategies.append(primary_strategy)

        # 根据复杂度添加备选策略
        if query_analysis.complexity == QueryComplexity.COMPLEX:
            backup_strategy = RetrievalStrategy(
                primary_method="fuzzy_search",
                secondary_methods=["semantic_search"],
                parameters={"fuzzy_threshold": 0.8, "semantic_similarity": 0.7},
                quality_threshold=0.6,
                max_results=30,
                expected_time=12.0
            )
            strategies.append(backup_strategy)

        return strategies

    async def _set_constraints(self, query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """设置约束条件"""
        constraints = {
            "max_tokens": 4000,
            "temperature": 0.7,
            "compliance_check": query_analysis.domain == "financial",
            "format_requirements": [],
            "safety_checks": True
        }

        # 根据查询类型添加特定约束
        if query_analysis.query_type in [QueryType.TREND_PREDICTION, QueryType.COMPARISON_ANALYSIS]:
            constraints["format_requirements"].append("data_sources")
            constraints["compliance_check"] = True

        if query_analysis.complexity == QueryComplexity.COMPLEX:
            constraints["max_tokens"] = 6000
            constraints["temperature"] = 0.5  # 降低温度增加准确性

        return constraints

    def _estimate_time(self, strategies: List[RetrievalStrategy]) -> float:
        """估算执行时间"""
        base_time = 2.0  # 基础时间

        # 根据策略调整
        for strategy in strategies:
            base_time += strategy.expected_time

        # 添加处理时间
        processing_time = 3.0  # 生成阶段时间

        total_time = base_time + processing_time

        return total_time

    def _generate_plan_id(self) -> str:
        """生成计划ID"""
        import uuid
        return str(uuid.uuid4())

    def _create_default_plan(self, query: str) -> RetrievalPlan:
        """创建默认计划"""
        query_analysis = QueryAnalysis(
            original_query=query,
            query_type=QueryType.FACT_FINDING,
            complexity=QueryComplexity.SIMPLE,
            confidence=0.5
        )

        default_strategy = RetrievalStrategy(
            primary_method="vector_search",
            secondary_methods=["keyword_search"],
            parameters={"top_k": 10, "similarity_threshold": 0.7},
            quality_threshold=0.7,
            max_results=10,
            expected_time=3.0
        )

        return RetrievalPlan(
            plan_id=self._generate_plan_id(),
            query_analysis=query_analysis,
            main_query=query,
            alternative_queries=[],
            strategies=[default_strategy],
            constraints={"max_tokens": 4000, "temperature": 0.7},
            estimated_time=5.0
        )


# 全局计划器实例
rag_planner = AgenticRAGPlanner()