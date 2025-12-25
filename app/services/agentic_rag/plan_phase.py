"""
Agentic RAG计划阶段
理解用户意图，制定检索策略
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime

from app.services.complexity_analyzer import ComplexityLevel, ComplexityFactors
from app.services.sla_enforcement import RetrievalMode

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型"""
    FACTUAL = "factual"          # 事实查找
    COMPARISON = "comparison"    # 比较分析
    TREND_PREDICTION = "trend_prediction"  # 趋势预测
    ANALYTICAL = "analytical"    # 分析评估
    CAUSAL = "causal"           # 因果关系
    LIST = "list"              # 列表查询
    DEFINITION = "definition"   # 定义解释


class RetrievalStrategy(Enum):
    """检索策略"""
    VECTOR_PRIMARY = "vector_primary"       # 向量检索为主
    GRAPH_PRIMARY = "graph_primary"         # 图谱检索为主
    HYBRID = "hybrid"                       # 混合检索
    KEYWORD_ENHANCED = "keyword_enhanced"   # 关键词增强
    TEMPORAL_FOCUSED = "temporal_focused"   # 时间焦点


@dataclass
class QueryPlan:
    """查询计划"""
    task_id: str
    original_query: str
    processed_query: str
    query_type: QueryType
    complexity_level: ComplexityLevel
    retrieval_strategy: RetrievalStrategy
    main_queries: List[str]
    backup_queries: List[str]
    retrieval_params: Dict[str, Any]
    quality_threshold: float
    estimated_results: int
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['query_type'] = self.query_type.value
        data['complexity_level'] = self.complexity_level.value
        data['retrieval_strategy'] = self.retrieval_strategy.value
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class QueryContext:
    """查询上下文"""
    conversation_id: Optional[int] = None
    previous_queries: List[str] = None
    user_preferences: Dict[str, Any] = None
    session_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.previous_queries is None:
            self.previous_queries = []
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.session_metadata is None:
            self.session_metadata = {}


class PlanPhase:
    """计划阶段处理器"""

    def __init__(self):
        # 金融术语标准化映射
        self.financial_synonyms = {
            '市盈率': ['PE', 'P/E ratio', '本益比'],
            '市净率': ['PB', 'P/B ratio', '市账率'],
            'ROE': ['净资产收益率', '股东权益回报率'],
            'ROA': ['资产收益率', '总资产回报率'],
            '毛利率': ['gross profit margin', 'gross margin'],
            '净利率': ['net profit margin', '净利润率'],
            '营收': ['revenue', '营业收入', '销售收入'],
            '净利润': ['net profit', 'net income'],
            '现金流': ['cash flow', '现金流量'],
            '资产负债率': ['debt to asset ratio', '负债率'],
            '流动比率': ['current ratio'],
            '速动比率': ['quick ratio'],
            'EBITDA': ['息税折旧摊销前利润'],
            'IPO': ['首次公开募股', '上市'],
            'M&A': ['并购', '收购合并'],
            'QFII': ['合格境外机构投资者'],
            'QDII': ['合格境内机构投资者'],
            'A股': ['A股市场'],
            'H股': ['H股市场'],
            '港股': ['香港股市'],
            '美股': ['美国股市']
        }

        # 问题类型关键词映射
        self.query_type_keywords = {
            QueryType.FACTUAL: [
                '什么', '是什么', '哪个', '哪些', 'who', 'what', 'which',
                '定义', '解释', 'describe', 'explain'
            ],
            QueryType.COMPARISON: [
                '对比', '比较', '差异', '区别', '相同点', '不同点',
                'compare', 'comparison', 'difference', 'similar', 'versus', 'vs'
            ],
            QueryType.TREND_PREDICTION: [
                '趋势', '预测', '展望', '未来', '预期', '走势',
                'trend', 'predict', 'forecast', 'outlook', 'projection'
            ],
            QueryType.ANALYTICAL: [
                '分析', '评估', '评价', '研究', '探讨',
                'analyze', 'evaluate', 'assess', 'study', 'examine'
            ],
            QueryType.CAUSAL: [
                '为什么', '原因', '导致', '影响', '结果', '后果',
                'why', 'cause', 'reason', 'lead to', 'impact', 'result'
            ],
            QueryType.LIST: [
                '列出', '有哪些', '包括', '包含', 'list', 'what are', 'name'
            ],
            QueryType.DEFINITION: [
                '定义', '是什么', '含义', '概念', 'define', 'meaning', 'concept'
            ]
        }

    async def process_query(
        self,
        query: str,
        context: QueryContext
    ) -> QueryPlan:
        """
        处理查询，生成检索计划

        Args:
            query: 用户查询
            context: 查询上下文

        Returns:
            查询计划
        """
        try:
            logger.info(f"Processing query in Plan Phase: {query[:100]}...")

            # 1. 问题接收和预处理
            processed_query = await self._preprocess_query(query, context)

            # 2. 问题分类
            query_type, complexity_factors = await self._classify_query(processed_query, context)

            # 3. 查询改写
            expanded_queries = await self._rewrite_query(processed_query, query_type)

            # 4. 策略选择
            retrieval_strategy = await self._select_strategy(query_type, complexity_factors, context)

            # 5. 生成检索计划
            plan = await self._generate_plan(
                query, processed_query, query_type,
                complexity_factors, retrieval_strategy,
                expanded_queries, context
            )

            logger.info(f"Generated plan with strategy: {retrieval_strategy.value}")
            return plan

        except Exception as e:
            logger.error(f"Error in Plan Phase: {str(e)}")
            raise

    async def _preprocess_query(self, query: str, context: QueryContext) -> str:
        """预处理查询"""
        # 移除多余空格
        processed = ' '.join(query.split())

        # 标准化金融术语
        for standard_term, synonyms in self.financial_synonyms.items():
            for synonym in synonyms:
                if synonym.lower() in processed.lower():
                    # 使用更精确的替换逻辑
                    pattern = r'\b' + re.escape(synonym) + r'\b'
                    processed = re.sub(pattern, standard_term, processed, flags=re.IGNORECASE)

        return processed

    async def _classify_query(
        self,
        query: str,
        context: QueryContext
    ) -> Tuple[QueryType, ComplexityFactors]:
        """分类查询"""
        from app.services.complexity_analyzer import complexity_analyzer

        # 使用复杂度分析器
        complexity_level, factors = await complexity_analyzer.analyze_complexity(
            query, context.previous_queries
        )

        # 确定查询类型
        query_lower = query.lower()
        type_scores = {}

        # 计算每种类型的得分
        for qtype, keywords in self.query_type_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            type_scores[qtype] = score

        # 选择得分最高的类型
        if not any(type_scores.values()):
            query_type = QueryType.FACTUAL  # 默认类型
        else:
            query_type = max(type_scores, key=type_scores.get)

        # 特殊规则调整
        if '对比' in query or '比较' in query:
            query_type = QueryType.COMPARISON
        elif '为什么' in query or '原因' in query:
            query_type = QueryType.CAUSAL
        elif '预测' in query or '展望' in query:
            query_type = QueryType.TREND_PREDICTION

        return query_type, factors

    async def _rewrite_query(
        self,
        query: str,
        query_type: QueryType
    ) -> Dict[str, List[str]]:
        """查询改写和扩展"""
        expanded = {
            'main_queries': [query],
            'backup_queries': []
        }

        # 同义词扩展
        main_expanded = [query]
        for standard_term, synonyms in self.financial_synonyms.items():
            if standard_term in query:
                for synonym in synonyms:
                    main_expanded.append(query.replace(standard_term, synonym))

        expanded['main_queries'] = list(set(main_expanded))[:3]  # 最多3个主查询

        # 生成备选查询
        if query_type == QueryType.COMPARISON:
            # 比较查询的备选方案
            if 'vs' in query.lower() or '对比' in query:
                parts = re.split(r' vs |对比|比较', query)
                if len(parts) == 2:
                    expanded['backup_queries'].append(f"{parts[1].strip()} vs {parts[0].strip()}")
                    expanded['backup_queries'].append(f"{parts[0].strip()} 和 {parts[1].strip()} 的区别")
                    expanded['backup_queries'].append(f"比较 {parts[0].strip()} 与 {parts[1].strip()}")

        elif query_type == QueryType.TREND_PREDICTION:
            # 趋势查询的备选方案
            expanded['backup_queries'].append(query.replace('预测', '趋势'))
            expanded['backup_queries'].append(query.replace('展望', '分析'))
            expanded['backup_queries'].append(query + '的影响因素')

        elif query_type == QueryType.ANALYTICAL:
            # 分析查询的备选方案
            expanded['backup_queries'].append(query.replace('分析', '评估'))
            expanded['backup_queries'].append(query.replace('研究', '探讨'))
            expanded['backup_queries'].append(f"{query}的关键因素")

        return expanded

    async def _select_strategy(
        self,
        query_type: QueryType,
        complexity_factors: ComplexityFactors,
        context: QueryContext
    ) -> RetrievalStrategy:
        """选择检索策略"""
        # 基于查询类型的策略选择
        strategy_map = {
            QueryType.FACTUAL: RetrievalStrategy.VECTOR_PRIMARY,
            QueryType.DEFINITION: RetrievalStrategy.VECTOR_PRIMARY,
            QueryType.LIST: RetrievalStrategy.KEYWORD_ENHANCED,
            QueryType.COMPARISON: RetrievalStrategy.GRAPH_PRIMARY,
            QueryType.CAUSAL: RetrievalStrategy.GRAPH_PRIMARY,
            QueryType.TREND_PREDICTION: RetrievalStrategy.TEMPORAL_FOCUSED,
            QueryType.ANALYTICAL: RetrievalStrategy.HYBRID
        }

        base_strategy = strategy_map.get(query_type, RetrievalStrategy.HYBRID)

        # 复杂度调整
        if complexity_factors.complexity_score > 0.7:
            # 高复杂度使用混合策略
            if base_strategy != RetrievalStrategy.HYBRID:
                base_strategy = RetrievalStrategy.HYBRID
        elif complexity_factors.complexity_score < 0.3:
            # 低复杂度可以简化策略
            if base_strategy == RetrievalStrategy.HYBRID:
                base_strategy = RetrievalStrategy.VECTOR_PRIMARY

        # 用户偏好调整
        if context.user_preferences.get('preferred_strategy'):
            try:
                preferred = RetrievalStrategy(context.user_preferences['preferred_strategy'])
                # 如果用户偏好合理，则使用
                if self._is_reasonable_strategy(base_strategy, preferred):
                    base_strategy = preferred
            except ValueError:
                pass

        return base_strategy

    async def _generate_plan(
        self,
        original_query: str,
        processed_query: str,
        query_type: QueryType,
        complexity_factors: ComplexityFactors,
        retrieval_strategy: RetrievalStrategy,
        expanded_queries: Dict[str, List[str]],
        context: QueryContext
    ) -> QueryPlan:
        """生成检索计划"""
        import uuid

        # 确定检索参数
        retrieval_params = await self._determine_retrieval_params(
            retrieval_strategy, complexity_factors, context
        )

        # 设置质量阈值
        quality_threshold = self._determine_quality_threshold(
            query_type, complexity_factors
        )

        # 估算结果数量
        estimated_results = self._estimate_results_count(
            query_type, complexity_factors
        )

        plan = QueryPlan(
            task_id=str(uuid.uuid4()),
            original_query=original_query,
            processed_query=processed_query,
            query_type=query_type,
            complexity_level=complexity_factors.complexity_score,
            retrieval_strategy=retrieval_strategy,
            main_queries=expanded_queries['main_queries'],
            backup_queries=expanded_queries['backup_queries'],
            retrieval_params=retrieval_params,
            quality_threshold=quality_threshold,
            estimated_results=estimated_results,
            created_at=datetime.utcnow()
        )

        return plan

    async def _determine_retrieval_params(
        self,
        strategy: RetrievalStrategy,
        factors: ComplexityFactors,
        context: QueryContext
    ) -> Dict[str, Any]:
        """确定检索参数"""
        params = {
            'vector_top_k': 10,
            'graph_top_k': 10,
            'keyword_top_k': 10,
            'fusion_weights': {'vector': 0.5, 'graph': 0.3, 'keyword': 0.2},
            'rerank': True,
            'diversify': True
        }

        # 根据策略调整
        if strategy == RetrievalStrategy.VECTOR_PRIMARY:
            params['vector_top_k'] = 20
            params['fusion_weights'] = {'vector': 0.7, 'graph': 0.2, 'keyword': 0.1}
        elif strategy == RetrievalStrategy.GRAPH_PRIMARY:
            params['graph_top_k'] = 20
            params['fusion_weights'] = {'vector': 0.2, 'graph': 0.7, 'keyword': 0.1}
        elif strategy == RetrievalStrategy.HYBRID:
            params['vector_top_k'] = 15
            params['graph_top_k'] = 15
            params['fusion_weights'] = {'vector': 0.4, 'graph': 0.4, 'keyword': 0.2}
        elif strategy == RetrievalStrategy.KEYWORD_ENHANCED:
            params['keyword_top_k'] = 20
            params['fusion_weights'] = {'vector': 0.3, 'graph': 0.2, 'keyword': 0.5}

        # 根据复杂度调整
        if factors.complexity_score > 0.7:
            params['vector_top_k'] = int(params['vector_top_k'] * 1.5)
            params['graph_top_k'] = int(params['graph_top_k'] * 1.5)
            params['keyword_top_k'] = int(params['keyword_top_k'] * 1.5)

        # 用户偏好
        if context.user_preferences.get('max_results'):
            max_results = context.user_preferences['max_results']
            params['vector_top_k'] = min(params['vector_top_k'], max_results)
            params['graph_top_k'] = min(params['graph_top_k'], max_results)
            params['keyword_top_k'] = min(params['keyword_top_k'], max_results)

        return params

    def _determine_quality_threshold(
        self,
        query_type: QueryType,
        factors: ComplexityFactors
    ) -> float:
        """确定质量阈值"""
        # 基础阈值
        base_threshold = 0.5

        # 根据查询类型调整
        type_adjustments = {
            QueryType.FACTUAL: 0.1,
            QueryType.DEFINITION: 0.1,
            QueryType.ANALYTICAL: -0.1,
            QueryType.TREND_PREDICTION: -0.2,
            QueryType.CAUSAL: -0.15
        }

        base_threshold += type_adjustments.get(query_type, 0)

        # 根据复杂度调整
        if factors.complexity_score > 0.7:
            base_threshold -= 0.1  # 复杂查询降低阈值
        elif factors.complexity_score < 0.3:
            base_threshold += 0.1  # 简单查询提高阈值

        return max(0.3, min(0.9, base_threshold))

    def _estimate_results_count(
        self,
        query_type: QueryType,
        factors: ComplexityFactors
    ) -> int:
        """估算结果数量"""
        base_count = 10

        # 根据查询类型调整
        if query_type == QueryType.LIST:
            base_count = 20
        elif query_type == QueryType.COMPARISON:
            base_count = 15
        elif query_type == QueryType.ANALYTICAL:
            base_count = 12

        # 根据复杂度调整
        if factors.complexity_score > 0.7:
            base_count = int(base_count * 1.5)

        return min(50, base_count)  # 最大50个结果

    def _is_reasonable_strategy(
        self,
        base_strategy: RetrievalStrategy,
        preferred: RetrievalStrategy
    ) -> bool:
        """判断策略偏好是否合理"""
        # 定义合理的策略组合
        reasonable_combinations = {
            RetrievalStrategy.VECTOR_PRIMARY: [
                RetrievalStrategy.VECTOR_PRIMARY,
                RetrievalStrategy.HYBRID,
                RetrievalStrategy.KEYWORD_ENHANCED
            ],
            RetrievalStrategy.GRAPH_PRIMARY: [
                RetrievalStrategy.GRAPH_PRIMARY,
                RetrievalStrategy.HYBRID
            ],
            RetrievalStrategy.HYBRID: [
                RetrievalStrategy.HYBRID,
                RetrievalStrategy.VECTOR_PRIMARY,
                RetrievalStrategy.GRAPH_PRIMARY
            ],
            RetrievalStrategy.KEYWORD_ENHANCED: [
                RetrievalStrategy.KEYWORD_ENHANCED,
                RetrievalStrategy.VECTOR_PRIMARY,
                RetrievalStrategy.HYBRID
            ],
            RetrievalStrategy.TEMPORAL_FOCUSED: [
                RetrievalStrategy.TEMPORAL_FOCUSED,
                RetrievalStrategy.HYBRID
            ]
        }

        return preferred in reasonable_combinations.get(base_strategy, [RetrievalStrategy.HYBRID])


# 需要导入的模块
import re