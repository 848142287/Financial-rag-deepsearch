"""
反馈意图解析器
分析用户反馈，识别优化需求和策略
"""

from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass

from .session_state import FeedbackType, FeedbackData

logger = logging.getLogger(__name__)


@dataclass
class OptimizationNeed:
    """优化需求分析结果"""
    primary_need: FeedbackType
    confidence: float
    secondary_needs: List[Tuple[FeedbackType, float]]
    specific_issues: List[str]
    suggested_actions: List[str]
    severity_score: float  # 严重程度 0-1


class FeedbackAnalyzer:
    """反馈意图解析器"""

    def __init__(self):
        # 反馈关键词映射
        self.feedback_keywords = {
            FeedbackType.RELEVANCE_LOW: [
                '不相关', '无关', '离题', '偏题', '不是这个', '错误方向',
                '不匹配', '不合适', '不太对', '跑题了', '文不对题'
            ],
            FeedbackType.INCOMPLETE: [
                '不够详细', '不完整', '缺少', '没找到', '信息不足',
                '需要更多', '不全面', '太简单', '内容太少', '不够深入'
            ],
            FeedbackType.ACCURACY_ISSUE: [
                '不准确', '错误', '过时', '不对', '有误',
                '数据错误', '事实错误', '信息过时', '不真实', '有问题'
            ],
            FeedbackType.SORTING_ISSUE: [
                '顺序不对', '排序问题', '重要', '优先', '前面',
                '应该排前面', '太靠后', '顺序', '重要性'
            ],
            FeedbackType.GENERAL_DISSATISFACTION: [
                '不满意', '不好', '不行', '重新', '再来',
                '不理想', '差', '不好用', '有问题', '改进'
            ],
            FeedbackType.SPECIFIC_REQUIREMENT: [
                '我需要', '要求', '必须', '希望', '最好是',
                '特定', '具体', '专门', '针对', '特定要求'
            ]
        }

        # 严重程度关键词
        self.severity_keywords = {
            'high': ['非常', '特别', '极其', '严重', '完全', '根本'],
            'medium': ['比较', '有些', '有点', '还算', '稍微'],
            'low': ['有点', '略微', '稍微', '可能', '似乎']
        }

    def analyze_feedback(self, feedback_data: FeedbackData) -> OptimizationNeed:
        """分析用户反馈"""
        # 1. 基于明确的反馈类型
        if feedback_data.feedback_type != FeedbackType.UNCLEAR:
            primary_need = feedback_data.feedback_type
            confidence = 0.8
        else:
            # 2. 基于评分析
            if feedback_data.rating:
                if feedback_data.rating <= 2:
                    primary_need = FeedbackType.GENERAL_DISSATISFACTION
                    confidence = 0.7
                elif feedback_data.rating == 3:
                    primary_need = FeedbackType.INCOMPLETE
                    confidence = 0.6
                else:
                    primary_need = FeedbackType.SORTING_ISSUE  # 小问题
                    confidence = 0.4
            else:
                # 3. 基于文字内容分析
                primary_need, confidence = self._analyze_text_feedback(feedback_data.comments or "")

        # 4. 分析次要需求
        secondary_needs = self._analyze_secondary_needs(feedback_data)

        # 5. 识别具体问题
        specific_issues = self._extract_specific_issues(feedback_data)

        # 6. 生成建议行动
        suggested_actions = self._generate_suggested_actions(primary_need, specific_issues)

        # 7. 计算严重程度
        severity_score = self._calculate_severity_score(feedback_data, primary_need)

        return OptimizationNeed(
            primary_need=primary_need,
            confidence=confidence,
            secondary_needs=secondary_needs,
            specific_issues=specific_issues,
            suggested_actions=suggested_actions,
            severity_score=severity_score
        )

    def _analyze_text_feedback(self, text: str) -> Tuple[FeedbackType, float]:
        """分析文字反馈"""
        if not text:
            return FeedbackType.UNCLEAR, 0.0

        text = text.lower()
        scores = {}

        # 计算每种反馈类型的匹配分数
        for feedback_type, keywords in self.feedback_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            scores[feedback_type] = score

        # 找到最高分数的类型
        if not scores or max(scores.values()) == 0:
            return FeedbackType.UNCLEAR, 0.0

        primary_type = max(scores, key=scores.get)
        confidence = min(scores[primary_type] / 3.0, 1.0)  # 最多3个关键词匹配

        return primary_type, confidence

    def _analyze_secondary_needs(self, feedback_data: FeedbackData) -> List[Tuple[FeedbackType, float]]:
        """分析次要需求"""
        secondary_needs = []

        # 基于评分分析次要需求
        if feedback_data.rating:
            if feedback_data.rating <= 2:
                secondary_needs.append((FeedbackType.GENERAL_DISSATISFACTION, 0.5))
            if feedback_data.rating <= 3:
                secondary_needs.append((FeedbackType.INCOMPLETE, 0.3))

        # 基于改写查询分析
        if feedback_data.rewritten_query and feedback_data.rewritten_query != feedback_data.rewritten_query:
            secondary_needs.append((FeedbackType.RELEVANCE_LOW, 0.4))

        # 基于高亮项分析
        if feedback_data.highlighted_items:
            secondary_needs.append((FeedbackType.SORTING_ISSUE, 0.3))

        return secondary_needs

    def _extract_specific_issues(self, feedback_data: FeedbackData) -> List[str]:
        """提取具体问题"""
        issues = []

        # 从评论中提取问题
        if feedback_data.comments:
            issues.extend(self._extract_issues_from_text(feedback_data.comments))

        # 从改写查询中提取问题
        if feedback_data.rewritten_query:
            issues.append("用户改写了查询，说明原始查询不够准确")

        # 从特定需求中提取
        if feedback_data.specific_requirements:
            for req in feedback_data.specific_requirements:
                issues.append(f"特定需求：{req}")

        return issues

    def _extract_issues_from_text(self, text: str) -> List[str]:
        """从文本中提取问题"""
        issues = []

        # 使用正则表达式提取常见问题模式
        patterns = [
            r'(需要|想要|希望|要).*?(更|多|少|新|旧|快|慢)',
            r'(不是|没有|缺少|不够).*?(详细|准确|相关|具体)',
            r'(应该|最好|可以).*?(优先|重点|主要)',
            r'(太|很|非常).*?(长|短、旧、新、多、少)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                issues.extend([f"表达的需求：{' '.join(match)}" for match in matches])

        return issues

    def _generate_suggested_actions(self, primary_need: FeedbackType, specific_issues: List[str]) -> List[str]:
        """生成建议行动"""
        action_map = {
            FeedbackType.RELEVANCE_LOW: [
                "查询改写：重新构建检索查询",
                "关键词调整：调整关键词权重和匹配策略",
                "语义扩展：使用同义词和相关概念",
                "过滤优化：调整结果过滤条件"
            ],
            FeedbackType.INCOMPLETE: [
                "范围扩展：扩大检索范围",
                "深度增加：增加检索深度",
                "多源检索：从更多数据源检索",
                "时间范围：扩大时间窗口"
            ],
            FeedbackType.ACCURACY_ISSUE: [
                "权威性提升：优先显示权威来源",
                "时间过滤：限制到更近的时间范围",
                "事实核查：增加事实验证机制",
                "可信度评分：提高可信度权重"
            ],
            FeedbackType.SORTING_ISSUE: [
                "排序算法调整：优化结果排序",
                "多样性保证：确保结果多样性",
                "个性化调整：考虑用户偏好",
                "重新排序：基于新的排序规则"
            ],
            FeedbackType.GENERAL_DISSATISFACTION: [
                "全面优化：综合调整所有参数",
                "重新开始：使用全新的检索策略",
                "多策略尝试：并行使用多种策略",
                "人工介入：请求更多指导"
            ],
            FeedbackType.SPECIFIC_REQUIREMENT: [
                "定制化检索：基于特定需求定制",
                "参数调整：调整检索参数",
                "策略切换：切换到更适合的策略",
                "专业检索：使用专业领域的检索方法"
            ]
        }

        actions = action_map.get(primary_need, ["通用优化：调整检索参数"])

        # 基于具体问题添加额外行动
        if "改写" in " ".join(specific_issues):
            actions.append("查询重构：基于用户意图重构查询")

        if "权威" in " ".join(specific_issues):
            actions.append("权威源优先：优先显示权威来源")

        return actions

    def _calculate_severity_score(self, feedback_data: FeedbackData, primary_need: FeedbackType) -> float:
        """计算严重程度分数"""
        base_score = 0.5

        # 基于评分调整
        if feedback_data.rating:
            base_score = (5 - feedback_data.rating) / 4.0

        # 基于反馈类型调整
        type_severity = {
            FeedbackType.ACCURACY_ISSUE: 0.9,
            FeedbackType.RELEVANCE_LOW: 0.8,
            FeedbackType.INCOMPLETE: 0.6,
            FeedbackType.SORTING_ISSUE: 0.4,
            FeedbackType.GENERAL_DISSATISFACTION: 0.7,
            FeedbackType.SPECIFIC_REQUIREMENT: 0.5
        }

        base_score = max(base_score, type_severity.get(primary_need, 0.5))

        # 基于文字强度调整
        if feedback_data.comments:
            text = feedback_data.comments.lower()
            for level, keywords in self.severity_keywords.items():
                if any(keyword in text for keyword in keywords):
                    if level == 'high':
                        base_score = min(base_score + 0.2, 1.0)
                    elif level == 'medium':
                        base_score = min(base_score + 0.1, 0.8)
                    else:  # low
                        base_score = max(base_score - 0.1, 0.2)
                    break

        return round(base_score, 2)

    def get_optimization_priority(self, optimization_need: OptimizationNeed) -> int:
        """获取优化优先级 (1-5, 5为最高)"""
        priority = 3

        # 基于严重程度
        if optimization_need.severity_score >= 0.8:
            priority = 5
        elif optimization_need.severity_score >= 0.6:
            priority = 4
        elif optimization_need.severity_score >= 0.4:
            priority = 3
        elif optimization_need.severity_score >= 0.2:
            priority = 2
        else:
            priority = 1

        # 基于置信度调整
        if optimization_need.confidence >= 0.8:
            priority = min(priority + 1, 5)
        elif optimization_need.confidence < 0.5:
            priority = max(priority - 1, 1)

        return priority

    def should_optimize(self, optimization_need: OptimizationNeed) -> bool:
        """判断是否需要进行优化"""
        # 置信度太低不确定优化方向
        if optimization_need.confidence < 0.3:
            return False

        # 严重程度太低可能不需要优化
        if optimization_need.severity_score < 0.2:
            return False

        # 评分较高（4-5分）可能不需要优化
        # 这个检查在analyze_feedback中已经考虑

        return True