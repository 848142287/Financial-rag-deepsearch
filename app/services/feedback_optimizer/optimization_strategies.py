"""
优化策略选择器和执行器
基于反馈类型和问题类型选择并执行具体的优化策略
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio

from .session_state import FeedbackType, OptimizationRecord, SearchSession
from .feedback_analyzer import OptimizationNeed

logger = logging.getLogger(__name__)


@dataclass
class OptimizationParams:
    """优化参数配置"""
    query: str
    top_k: int = 10
    similarity_threshold: float = 0.7
    date_range_days: Optional[int] = None
    source_types: List[str] = None
    weight_config: Dict[str, float] = None
    filters: Dict[str, Any] = None
    sort_by: str = "relevance"
    diversity_threshold: float = 0.8
    authority_boost: float = 1.0


class OptimizationStrategy(ABC):
    """优化策略抽象基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def apply(self, original_params: OptimizationParams, optimization_need: OptimizationNeed) -> OptimizationParams:
        """应用优化策略"""
        pass

    @abstractmethod
    def is_suitable_for(self, feedback_type: FeedbackType, severity_score: float) -> bool:
        """判断策略是否适用于当前反馈类型"""
        pass


class QueryRewriteStrategy(OptimizationStrategy):
    """查询改写策略"""

    def __init__(self):
        super().__init__("query_rewrite", "查询改写：基于用户反馈重新构建查询")

    def is_suitable_for(self, feedback_type: FeedbackType, severity_score: float) -> bool:
        return feedback_type in [FeedbackType.RELEVANCE_LOW, FeedbackType.GENERAL_DISSATISFACTION]

    async def apply(self, original_params: OptimizationParams, optimization_need: OptimizationNeed) -> OptimizationParams:
        """应用查询改写策略"""
        new_params = OptimizationParams(
            query=original_params.query,
            top_k=original_params.top_k,
            similarity_threshold=original_params.similarity_threshold,
            date_range_days=original_params.date_range_days,
            source_types=original_params.source_types,
            weight_config=original_params.weight_config.copy() if original_params.weight_config else {},
            filters=original_params.filters.copy() if original_params.filters else {},
            sort_by=original_params.sort_by,
            diversity_threshold=original_params.diversity_threshold,
            authority_boost=original_params.authority_boost
        )

        # 查询改写逻辑
        rewritten_query = await self._rewrite_query(original_params.query, optimization_need)
        new_params.query = rewritten_query

        # 降低相似度阈值，增加结果多样性
        new_params.similarity_threshold = max(original_params.similarity_threshold - 0.1, 0.5)

        logger.info(f"应用查询改写策略：{original_params.query} -> {rewritten_query}")
        return new_params

    async def _rewrite_query(self, original_query: str, optimization_need: OptimizationNeed) -> str:
        """重写查询"""
        # 使用LLM或规则进行查询改写
        # 这里简化实现，实际应该调用LLM服务
        rewritten_parts = []

        # 添加同义词扩展
        synonyms = {
            "财务": ["财务状况", "财务数据", "财务信息"],
            "分析": ["研究", "评估", "审查"],
            "报告": ["年报", "季报", "财务报告"],
            "投资": ["投资组合", "投资策略", "投资回报"],
            "风险": ["风险控制", "风险管理", "风险评估"]
        }

        query_words = original_query.split()
        for word in query_words:
            rewritten_parts.append(word)
            if word in synonyms:
                rewritten_parts.extend(synonyms[word][:1])  # 添加一个同义词

        # 添加上下文限定词
        if "详细" not in original_query and optimization_need.primary_need == FeedbackType.INCOMPLETE:
            rewritten_parts.append("详细")

        if "准确" not in original_query and optimization_need.primary_need == FeedbackType.ACCURACY_ISSUE:
            rewritten_parts.append("准确")

        rewritten_query = " ".join(rewritten_parts)
        return rewritten_query


class RangeExpansionStrategy(OptimizationStrategy):
    """范围扩展策略"""

    def __init__(self):
        super().__init__("range_expansion", "范围扩展：扩大检索范围和深度")

    def is_suitable_for(self, feedback_type: FeedbackType, severity_score: float) -> bool:
        return feedback_type == FeedbackType.INCOMPLETE

    async def apply(self, original_params: OptimizationParams, optimization_need: OptimizationNeed) -> OptimizationParams:
        """应用范围扩展策略"""
        new_params = OptimizationParams(
            query=original_params.query,
            top_k=original_params.top_k + 10,  # 增加结果数量
            similarity_threshold=max(original_params.similarity_threshold - 0.1, 0.5),  # 降低相似度阈值
            date_range_days=original_params.date_range_days,
            source_types=original_params.source_types,
            weight_config=original_params.weight_config.copy() if original_params.weight_config else {},
            filters=original_params.filters.copy() if original_params.filters else {},
            sort_by=original_params.sort_by,
            diversity_threshold=max(original_params.diversity_threshold - 0.1, 0.5),  # 增加多样性
            authority_boost=original_params.authority_boost
        )

        # 扩展时间范围
        if original_params.date_range_days:
            new_params.date_range_days = original_params.date_range_days * 2

        # 扩展数据源类型
        if original_params.source_types:
            new_params.source_types.extend(["news", "research", "analyst_report"])

        logger.info(f"应用范围扩展策略：top_k {original_params.top_k} -> {new_params.top_k}")
        return new_params


class AuthorityBoostStrategy(OptimizationStrategy):
    """权威性提升策略"""

    def __init__(self):
        super().__init__("authority_boost", "权威性提升：优先显示权威来源")

    def is_suitable_for(self, feedback_type: FeedbackType, severity_score: float) -> bool:
        return feedback_type == FeedbackType.ACCURACY_ISSUE

    async def apply(self, original_params: OptimizationParams, optimization_need: OptimizationNeed) -> OptimizationParams:
        """应用权威性提升策略"""
        new_params = OptimizationParams(
            query=original_params.query,
            top_k=original_params.top_k,
            similarity_threshold=original_params.similarity_threshold,
            date_range_days=original_params.date_range_days,
            source_types=["official", "regulatory", "major_firm"],  # 限制为权威来源
            weight_config=original_params.weight_config.copy() if original_params.weight_config else {},
            filters=original_params.filters.copy() if original_params.filters else {},
            sort_by="authority",  # 按权威性排序
            diversity_threshold=original_params.diversity_threshold,
            authority_boost=2.0  # 大幅提升权威性权重
        )

        # 更新权重配置
        if not new_params.weight_config:
            new_params.weight_config = {}

        new_params.weight_config.update({
            "source_authority": 2.0,
            "date_recency": 1.5,
            "verification_status": 2.0
        })

        # 限制时间范围，确保信息新鲜度
        new_params.date_range_days = min(original_params.date_range_days or 365, 180)

        logger.info("应用权威性提升策略：优先权威来源，限制时间范围")
        return new_params


class SortOptimizationStrategy(OptimizationStrategy):
    """排序优化策略"""

    def __init__(self):
        super().__init__("sort_optimization", "排序优化：调整结果排序算法")

    def is_suitable_for(self, feedback_type: FeedbackType, severity_score: float) -> bool:
        return feedback_type == FeedbackType.SORTING_ISSUE

    async def apply(self, original_params: OptimizationParams, optimization_need: OptimizationNeed) -> OptimizationParams:
        """应用排序优化策略"""
        new_params = OptimizationParams(
            query=original_params.query,
            top_k=original_params.top_k,
            similarity_threshold=original_params.similarity_threshold,
            date_range_days=original_params.date_range_days,
            source_types=original_params.source_types,
            weight_config=original_params.weight_config.copy() if original_params.weight_config else {},
            filters=original_params.filters.copy() if original_params.filters else {},
            sort_by="composite",  # 使用复合排序
            diversity_threshold=0.6,  # 确保多样性
            authority_boost=original_params.authority_boost
        )

        # 更新权重配置，优化排序
        new_params.weight_config.update({
            "relevance_score": 1.0,
            "authority_score": 1.2,
            "freshness_score": 0.8,
            "popularity_score": 0.6,
            "diversity_bonus": 0.4
        })

        # 添加多样性过滤器
        new_params.filters = new_params.filters or {}
        new_params.filters["ensure_diversity"] = True
        new_params.filters["max_similar_per_category"] = 3

        logger.info("应用排序优化策略：使用复合排序，确保多样性")
        return new_params


class ComprehensiveStrategy(OptimizationStrategy):
    """综合优化策略"""

    def __init__(self):
        super().__init__("comprehensive", "综合优化：全面调整检索参数")

    def is_suitable_for(self, feedback_type: FeedbackType, severity_score: float) -> bool:
        return feedback_type == FeedbackType.GENERAL_DISSATISFACTION or severity_score >= 0.8

    async def apply(self, original_params: OptimizationParams, optimization_need: OptimizationNeed) -> OptimizationParams:
        """应用综合优化策略"""
        new_params = OptimizationParams(
            query=original_params.query,
            top_k=original_params.top_k + 5,
            similarity_threshold=max(original_params.similarity_threshold - 0.05, 0.6),
            date_range_days=original_params.date_range_days,
            source_types=original_params.source_types,
            weight_config=original_params.weight_config.copy() if original_params.weight_config else {},
            filters=original_params.filters.copy() if original_params.filters else {},
            sort_by="composite",
            diversity_threshold=0.7,
            authority_boost=1.5
        )

        # 全面优化权重配置
        new_params.weight_config.update({
            "relevance_score": 1.2,
            "authority_score": 1.5,
            "freshness_score": 1.0,
            "completeness_score": 1.3,
            "diversity_bonus": 0.5
        })

        # 添加智能过滤器
        new_params.filters.update({
            "ensure_diversity": True,
            "boost_recent": True,
            "prefer_verified": True,
            "exclude_duplicates": True
        })

        logger.info("应用综合优化策略：全面调整检索参数")
        return new_params


class OptimizationStrategySelector:
    """优化策略选择器"""

    def __init__(self):
        self.strategies = [
            QueryRewriteStrategy(),
            RangeExpansionStrategy(),
            AuthorityBoostStrategy(),
            SortOptimizationStrategy(),
            ComprehensiveStrategy()
        ]

    def select_strategy(self, optimization_need: OptimizationNeed) -> OptimizationStrategy:
        """选择最适合的优化策略"""
        suitable_strategies = []

        for strategy in self.strategies:
            if strategy.is_suitable_for(optimization_need.primary_need, optimization_need.severity_score):
                suitable_strategies.append(strategy)

        # 如果有多个适用策略，优先选择最具体的
        if suitable_strategies:
            # 优先级：具体策略 > 综合策略
            if ComprehensiveStrategy() in suitable_strategies and len(suitable_strategies) > 1:
                suitable_strategies.remove(ComprehensiveStrategy())

            return suitable_strategies[0]

        # 如果没有特定策略适用，使用综合策略
        return ComprehensiveStrategy()

    def get_strategy_ranking(self, optimization_need: OptimizationNeed) -> List[Tuple[OptimizationStrategy, float]]:
        """获取策略排名和适用度分数"""
        rankings = []

        for strategy in self.strategies:
            if strategy.is_suitable_for(optimization_need.primary_need, optimization_need.severity_score):
                score = self._calculate_strategy_score(strategy, optimization_need)
                rankings.append((strategy, score))

        # 按分数排序
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def _calculate_strategy_score(self, strategy: OptimizationStrategy, optimization_need: OptimizationNeed) -> float:
        """计算策略适用度分数"""
        base_score = 0.5

        # 基于反馈类型的匹配度
        type_scores = {
            FeedbackType.RELEVANCE_LOW: {
                QueryRewriteStrategy: 0.9,
                ComprehensiveStrategy: 0.7
            },
            FeedbackType.INCOMPLETE: {
                RangeExpansionStrategy: 0.9,
                QueryRewriteStrategy: 0.6,
                ComprehensiveStrategy: 0.7
            },
            FeedbackType.ACCURACY_ISSUE: {
                AuthorityBoostStrategy: 0.9,
                ComprehensiveStrategy: 0.8
            },
            FeedbackType.SORTING_ISSUE: {
                SortOptimizationStrategy: 0.9,
                ComprehensiveStrategy: 0.6
            },
            FeedbackType.GENERAL_DISSATISFACTION: {
                ComprehensiveStrategy: 0.8,
                QueryRewriteStrategy: 0.7
            }
        }

        feedback_type_scores = type_scores.get(optimization_need.primary_need, {})
        base_score = feedback_type_scores.get(type(strategy), 0.5)

        # 基于严重程度调整
        if optimization_need.severity_score >= 0.8:
            base_score += 0.1
        elif optimization_need.severity_score >= 0.6:
            base_score += 0.05

        # 基于置信度调整
        base_score *= optimization_need.confidence

        return round(base_score, 2)

    async def apply_strategy(self, strategy: OptimizationStrategy, original_params: OptimizationParams,
                           optimization_need: OptimizationNeed) -> OptimizationParams:
        """应用选定的优化策略"""
        try:
            optimized_params = await strategy.apply(original_params, optimization_need)
            logger.info(f"成功应用优化策略：{strategy.name}")
            return optimized_params
        except Exception as e:
            logger.error(f"应用优化策略失败：{strategy.name}, 错误：{str(e)}")
            # 返回原始参数，避免完全失败
            return original_params