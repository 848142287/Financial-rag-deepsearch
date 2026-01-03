"""
查询分类器和复杂度评估器

为自适应检索提供查询特征分析
"""

import re
from typing import Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class QueryType(Enum):
    """查询类型"""
    FACTUAL = "factual"           # 事实查询 ("XX公司的营收是多少？")
    ANALYTICAL = "analytical"     # 分析查询 ("分析XX公司的盈利能力")
    COMPARATIVE = "comparative"   # 比较查询 ("A公司和B公司哪个更好？")
    RELATIONAL = "relational"     # 关系查询 ("XX公司有哪些子公司？")
    DEFINITIONAL = "definitional" # 定义查询 ("什么是ROE？")
    PROCEDURAL = "procedural"     # 流程查询 ("如何计算现金流？")


class QueryComplexity(Enum):
    """查询复杂度"""
    SIMPLE = "simple"           # 简单 (单一实体，明确目标)
    MEDIUM = "medium"           # 中等 (多实体，需要推理)
    COMPLEX = "complex"         # 复杂 (多步骤，综合分析)


@dataclass
class QueryFeatures:
    """查询特征"""
    query_type: QueryType
    complexity: QueryComplexity
    entity_count: int
    has_numbers: bool
    has_quotes: bool
    query_length: int
    keyword_density: float
    financial_term_count: int
    estimated_difficulty: float  # 0-1之间


class QueryClassifier:
    """查询分类器"""

    def __init__(self):
        # 金融术语库
        self.financial_terms = {
            # 财务指标
            "净利润", "营业收入", "毛利率", "净利率", "ROE", "ROA", "EPS",
            "资产总额", "负债率", "现金流", "EBITDA",

            # 财务比率
            "市盈率", "市净率", "资产负债率", "流动比率", "速动比率",

            # 业务类型
            "营业收入", "营业成本", "营业利润", "净利润",

            # 其他
            "同比增长", "环比增长", "年度", "季度"
        }

        # 实体识别模式
        self.entity_patterns = [
            r"[A-Z][a-z]+(?:公司|股份有限公司|集团)",  # 公司名
            r"\d{4}年",  # 年份
            r"Q[1-4]",  # 季度
        ]

        # 关系词
        self.relation_words = [
            "子公司", "母公司", "控股", "参股",
            "供应商", "客户", "合作伙伴", "竞争对手"
        ]

        # 比较词
        self.comparison_words = [
            "对比", "相比", "vs", " versus", "和", "与",
            "更高", "更低", "更好", "更差"
        ]

        # 分析词
        self.analysis_words = [
            "分析", "评估", "评价", "如何", "为什么",
            "原因", "影响", "趋势", "变化"
        ]

        # 定义词
        self.definition_words = [
            "是什么", "什么是", "定义", "含义",
            "解释", "说明"
        ]

        # 流程词
        self.procedural_words = [
            "如何", "怎么", "怎样", "方法", "步骤",
            "流程", "计算", "公式"
        ]

        logger.info("查询分类器初始化完成")

    def classify(self, query: str) -> QueryFeatures:
        """
        分类查询

        Args:
            query: 用户查询

        Returns:
            QueryFeatures: 查询特征
        """
        # 1. 基础特征提取
        query_type = self._classify_type(query)
        complexity = self._classify_complexity(query, query_type)

        # 2. 提取特征
        entity_count = self._count_entities(query)
        has_numbers = bool(re.search(r'\d', query))
        has_quotes = '"' in query or '"' in query
        query_length = len(query)

        # 3. 计算关键词密度
        keywords = self._extract_keywords(query)
        keyword_density = len(keywords) / max(query_length, 1)

        # 4. 统计金融术语
        financial_term_count = sum(
            1 for term in self.financial_terms
            if term in query
        )

        # 5. 估算难度
        difficulty = self._estimate_difficulty(
            query_type, complexity, entity_count, financial_term_count
        )

        return QueryFeatures(
            query_type=query_type,
            complexity=complexity,
            entity_count=entity_count,
            has_numbers=has_numbers,
            has_quotes=has_quotes,
            query_length=query_length,
            keyword_density=keyword_density,
            financial_term_count=financial_term_count,
            estimated_difficulty=difficulty
        )

    def _classify_type(self, query: str) -> QueryType:
        """分类查询类型"""
        # 检查关系词
        if any(word in query for word in self.relation_words):
            return QueryType.RELATIONAL

        # 检查比较词
        if any(word in query for word in self.comparison_words):
            return QueryType.COMPARATIVE

        # 检查定义词
        if any(word in query for word in self.definition_words):
            return QueryType.DEFINITIONAL

        # 检查流程词
        if any(word in query for word in self.procedural_words):
            return QueryType.PROCEDURAL

        # 检查分析词
        if any(word in query for word in self.analysis_words):
            return QueryType.ANALYTICAL

        # 默认为事实查询
        return QueryType.FACTUAL

    def _classify_complexity(
        self,
        query: str,
        query_type: QueryType
    ) -> QueryComplexity:
        """分类查询复杂度"""
        complexity_score = 0

        # 实体数量
        entity_count = self._count_entities(query)
        complexity_score += min(entity_count * 0.3, 0.4)

        # 查询长度
        if len(query) > 20:
            complexity_score += 0.2

        # 金融术语数量
        financial_count = sum(
            1 for term in self.financial_terms
            if term in query
        )
        complexity_score += min(financial_count * 0.1, 0.3)

        # 查询类型权重
        type_weights = {
            QueryType.DEFINITIONAL: 0.0,
            QueryType.FACTUAL: 0.1,
            QueryType.COMPARATIVE: 0.3,
            QueryType.RELATIONAL: 0.3,
            QueryType.PROCEDURAL: 0.4,
            QueryType.ANALYTICAL: 0.5
        }
        complexity_score += type_weights.get(query_type, 0.2)

        # 分类
        if complexity_score < 0.3:
            return QueryComplexity.SIMPLE
        elif complexity_score < 0.6:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.COMPLEX

    def _count_entities(self, query: str) -> int:
        """计算实体数量"""
        count = 0

        # 查找公司名模式
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query)
            count += len(matches)

        return count

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        keywords = []

        # 提取金融术语
        for term in self.financial_terms:
            if term in query:
                keywords.append(term)

        # 提取数字
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        keywords.extend(numbers)

        return keywords

    def _estimate_difficulty(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        entity_count: int,
        financial_term_count: int
    ) -> float:
        """
        估算查询难度 (0-1之间)

        考虑因素:
        1. 查询类型
        2. 复杂度
        3. 实体数量
        4. 金融术语数量
        """
        difficulty = 0.0

        # 查询类型难度
        type_difficulty = {
            QueryType.DEFINITIONAL: 0.2,
            QueryType.FACTUAL: 0.3,
            QueryType.COMPARATIVE: 0.5,
            QueryType.RELATIONAL: 0.6,
            QueryType.PROCEDURAL: 0.5,
            QueryType.ANALYTICAL: 0.7
        }
        difficulty += type_difficulty.get(query_type, 0.4)

        # 复杂度难度
        complexity_difficulty = {
            QueryComplexity.SIMPLE: 0.0,
            QueryComplexity.MEDIUM: 0.2,
            QueryComplexity.COMPLEX: 0.4
        }
        difficulty += complexity_difficulty.get(complexity, 0.0)

        # 实体数量难度
        difficulty += min(entity_count * 0.1, 0.2)

        # 金融术语数量难度
        difficulty += min(financial_term_count * 0.05, 0.15)

        return min(difficulty, 1.0)


# 全局实例
_query_classifier = None


def get_query_classifier() -> QueryClassifier:
    """获取查询分类器实例"""
    global _query_classifier
    if _query_classifier is None:
        _query_classifier = QueryClassifier()
    return _query_classifier
