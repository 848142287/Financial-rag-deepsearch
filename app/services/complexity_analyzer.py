"""
查询复杂度分析器
用于自动选择最适合的检索模式
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.services.sla_enforcement import RetrievalMode
from app.core.config import settings

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """复杂度级别"""
    SIMPLE = "simple"        # 简单
    MODERATE = "moderate"    # 中等
    COMPLEX = "complex"      # 复杂
    VERY_COMPLEX = "very_complex"  # 非常复杂


@dataclass
class ComplexityFactors:
    """复杂度因子"""
    query_length: int
    word_count: int
    has_entities: bool
    entity_count: int
    has_comparisons: bool
    has_temporal_queries: bool
    has_causal_questions: bool
    has_analytical_keywords: bool
    requires_deep_analysis: bool
    question_type: str
    complexity_score: float


class ComplexityAnalyzer:
    """查询复杂度分析器"""

    def __init__(self):
        # 简单查询关键词
        self.simple_keywords = {
            '什么', '是什么', '定义', '解释', '列出', '简单', '基本',
            'what is', 'define', 'list', 'basic', 'simple'
        }

        # 复杂查询关键词
        self.complex_keywords = {
            '分析', '比较', '对比', '评估', '预测', '影响', '原因',
            '趋势', '关系', '优缺点', '利弊', '战略', '策略',
            'analyze', 'compare', 'evaluate', 'predict', 'impact',
            'relationship', 'trend', 'advantages', 'disadvantages'
        }

        # 深度分析关键词
        self.deep_keywords = {
            '深入研究', '深度分析', '全面分析', '综合评估', '系统分析',
            '多角度', '多维度', '详细分析', '深入探讨',
            'deep analysis', 'comprehensive', 'systematic', 'multi-faceted'
        }

        # 比较关键词
        self.comparison_keywords = {
            '对比', '比较', '差异', '区别', '相同点', '不同点',
            'vs', 'versus', 'compared to', 'difference', 'similar'
        }

        # 时间相关关键词
        self.temporal_keywords = {
            '最近', '过去', '未来', '趋势', '变化', '历史',
            '预测', '展望', '时间线', '发展',
            'recent', 'past', 'future', 'trend', 'history', 'timeline'
        }

        # 因果关系关键词
        self.causal_keywords = {
            '为什么', '原因', '导致', '影响', '结果', '后果',
            '因为', '由于', '所以', '因此',
            'why', 'cause', 'reason', 'lead to', 'result', 'impact'
        }

    async def analyze_complexity(self, query: str, history: Optional[List[str]] = None) -> Tuple[ComplexityLevel, ComplexityFactors]:
        """
        分析查询复杂度

        Args:
            query: 查询文本
            history: 历史对话

        Returns:
            (复杂度级别, 复杂度因子)
        """
        try:
            # 基础分析
            factors = self._analyze_basic_factors(query)

            # 分析问题类型
            factors.question_type = self._classify_question_type(query)

            # 分析实体
            factors.has_entities, factors.entity_count = self._detect_entities(query)

            # 分析比较查询
            factors.has_comparisons = self._detect_comparisons(query)

            # 分析时间查询
            factors.has_temporal_queries = self._detect_temporal_queries(query)

            # 分析因果问题
            factors.has_causal_questions = self._detect_causal_questions(query)

            # 分析分析性关键词
            factors.has_analytical_keywords = self._detect_analytical_keywords(query)

            # 分析是否需要深度分析
            factors.requires_deep_analysis = self._requires_deep_analysis(query)

            # 计算复杂度分数
            factors.complexity_score = self._calculate_complexity_score(factors)

            # 确定复杂度级别
            complexity_level = self._determine_complexity_level(factors)

            # 考虑历史对话的影响
            if history:
                complexity_level = self._adjust_complexity_by_history(complexity_level, history)

            logger.info(f"Query complexity analyzed: {complexity_level.value} (score: {factors.complexity_score:.2f})")
            return complexity_level, factors

        except Exception as e:
            logger.error(f"Error analyzing query complexity: {str(e)}")
            # 返回默认值
            return ComplexityLevel.MODERATE, ComplexityFactors(
                query_length=len(query),
                word_count=len(query.split()),
                has_entities=False,
                entity_count=0,
                has_comparisons=False,
                has_temporal_queries=False,
                has_causal_questions=False,
                has_analytical_keywords=False,
                requires_deep_analysis=False,
                question_type="unknown",
                complexity_score=0.5
            )

    def get_recommended_mode(self, complexity_level: ComplexityLevel, user_preference: Optional[RetrievalMode] = None) -> RetrievalMode:
        """
        根据复杂度级别推荐检索模式

        Args:
            complexity_level: 复杂度级别
            user_preference: 用户偏好模式

        Returns:
            推荐的检索模式
        """
        # 如果用户有明确偏好且在合理范围内，尊重用户选择
        if user_preference:
            if self._is_reasonable_preference(complexity_level, user_preference):
                return user_preference

        # 根据复杂度级别自动推荐
        if complexity_level == ComplexityLevel.SIMPLE:
            return RetrievalMode.SIMPLE
        elif complexity_level == ComplexityLevel.MODERATE:
            return RetrievalMode.ENHANCED
        elif complexity_level == ComplexityLevel.COMPLEX:
            return RetrievalMode.DEEP_SEARCH
        else:  # VERY_COMPLEX
            return RetrievalMode.AGENTIC

    def _analyze_basic_factors(self, query: str) -> ComplexityFactors:
        """基础分析因子"""
        return ComplexityFactors(
            query_length=len(query),
            word_count=len(query.split()),
            has_entities=False,
            entity_count=0,
            has_comparisons=False,
            has_temporal_queries=False,
            has_causal_questions=False,
            has_analytical_keywords=False,
            requires_deep_analysis=False,
            question_type="unknown",
            complexity_score=0.0
        )

    def _classify_question_type(self, query: str) -> str:
        """分类问题类型"""
        query_lower = query.lower()

        # 事实性问题
        if any(word in query_lower for word in ['什么', '是什么', 'what is', 'define']):
            return 'factual'

        # 列表性问题
        if any(word in query_lower for word in ['列出', '有哪些', 'list', 'what are']):
            return 'list'

        # 比较性问题
        if any(word in query_lower for word in self.comparison_keywords):
            return 'comparison'

        # 分析性问题
        if any(word in query_lower for word in self.complex_keywords):
            return 'analytical'

        # 预测性问题
        if any(word in query_lower for word in ['预测', '展望', 'predict', 'forecast']):
            return 'predictive'

        # 因果问题
        if any(word in query_lower for word in self.causal_keywords):
            return 'causal'

        # 评估性问题
        if any(word in query_lower for word in ['评估', '评价', 'evaluate', 'assess']):
            return 'evaluative'

        return 'general'

    def _detect_entities(self, query: str) -> Tuple[bool, int]:
        """检测实体（简化版）"""
        # 使用简单的正则表达式检测可能的实体
        patterns = [
            r'\b[A-Z]{2,}\b',  # 大写缩写（如IPO, CEO）
            r'\b\d{4}\b',      # 年份
            r'\b\d+%\b',       # 百分比
            r'[\u4e00-\u9fff]+(?:公司|集团|企业|机构)',  # 中文名词
        ]

        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)

        return len(entities) > 0, len(entities)

    def _detect_comparisons(self, query: str) -> bool:
        """检测比较查询"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.comparison_keywords)

    def _detect_temporal_queries(self, query: str) -> bool:
        """检测时间查询"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.temporal_keywords)

    def _detect_causal_questions(self, query: str) -> bool:
        """检测因果问题"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.causal_keywords)

    def _detect_analytical_keywords(self, query: str) -> bool:
        """检测分析性关键词"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.complex_keywords)

    def _requires_deep_analysis(self, query: str) -> bool:
        """判断是否需要深度分析"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.deep_keywords)

    def _calculate_complexity_score(self, factors: ComplexityFactors) -> float:
        """计算复杂度分数 (0-1)"""
        score = 0.0

        # 查询长度因子 (0-0.2)
        length_score = min(factors.query_length / 500, 1.0) * 0.2
        score += length_score

        # 实体数量因子 (0-0.15)
        entity_score = min(factors.entity_count / 10, 1.0) * 0.15
        score += entity_score

        # 比较查询因子 (0-0.15)
        if factors.has_comparisons:
            score += 0.15

        # 时间查询因子 (0-0.1)
        if factors.has_temporal_queries:
            score += 0.1

        # 因果问题因子 (0-0.15)
        if factors.has_causal_questions:
            score += 0.15

        # 分析性关键词因子 (0-0.15)
        if factors.has_analytical_keywords:
            score += 0.15

        # 深度分析需求因子 (0-0.1)
        if factors.requires_deep_analysis:
            score += 0.1

        return min(score, 1.0)

    def _determine_complexity_level(self, factors: ComplexityFactors) -> ComplexityLevel:
        """根据分数确定复杂度级别"""
        score = factors.complexity_score

        if score < 0.3:
            return ComplexityLevel.SIMPLE
        elif score < 0.6:
            return ComplexityLevel.MODERATE
        elif score < 0.8:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX

    def _adjust_complexity_by_history(self, complexity_level: ComplexityLevel, history: List[str]) -> ComplexityLevel:
        """根据历史对话调整复杂度"""
        if not history:
            return complexity_level

        # 如果历史对话较长，可能需要更深入的分析
        if len(history) > 5:
            # 提升一个复杂度级别
            if complexity_level == ComplexityLevel.SIMPLE:
                return ComplexityLevel.MODERATE
            elif complexity_level == ComplexityLevel.MODERATE:
                return ComplexityLevel.COMPLEX
            elif complexity_level == ComplexityLevel.COMPLEX:
                return ComplexityLevel.VERY_COMPLEX

        return complexity_level

    def _is_reasonable_preference(self, complexity_level: ComplexityLevel, user_preference: RetrievalMode) -> bool:
        """判断用户偏好是否合理"""
        # 定义合理的复杂度-模式映射
        reasonable_mappings = {
            ComplexityLevel.SIMPLE: [RetrievalMode.SIMPLE, RetrievalMode.ENHANCED],
            ComplexityLevel.MODERATE: [RetrievalMode.SIMPLE, RetrievalMode.ENHANCED, RetrievalMode.DEEP_SEARCH],
            ComplexityLevel.COMPLEX: [RetrievalMode.ENHANCED, RetrievalMode.DEEP_SEARCH, RetrievalMode.AGENTIC],
            ComplexityLevel.VERY_COMPLEX: [RetrievalMode.DEEP_SEARCH, RetrievalMode.AGENTIC]
        }

        return user_preference in reasonable_mappings.get(complexity_level, [])


# 全局实例
complexity_analyzer = ComplexityAnalyzer()