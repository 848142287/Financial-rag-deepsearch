"""
增强的智能搜索选择器

目标：将搜索选择准确率从16%提升到60%+
实现多维度问题分析和智能模式匹配
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class SearchMode(Enum):
    """搜索模式枚举"""
    VECTOR = "vector"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    DEEP_SEARCH = "deep_search"
    HYBRID = "hybrid"

class QuestionType(Enum):
    """问题类型枚举"""
    DEFINITION = "definition"
    DATA_QUERY = "data_query"
    METHODOLOGY = "methodology"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    RELATIONSHIP = "relationship"
    COMPOSITE = "composite"

@dataclass
class QuestionAnalysis:
    """问题分析结果"""
    text: str
    type: QuestionType
    complexity: str  # low, medium, high, very_high
    domain: str  # investment, trading, risk, analysis, strategy, data
    keywords: List[str]
    entities: List[str]
    intent: str
    temporal_aspect: str  # past, present, future, none
    scope: str  # specific, general, comparative, analytical

@dataclass
class ModeScore:
    """模式评分"""
    mode: SearchMode
    score: float
    reasons: List[str]
    confidence: float

class EnhancedSearchSelector:
    """增强的搜索选择器"""

    def __init__(self):
        self.financial_vocabulary = self._load_financial_vocabulary()
        self.question_patterns = self._load_question_patterns()
        self.mode_weights = self._initialize_mode_weights()
        self.learning_data = self._load_learning_data()

    def _load_financial_vocabulary(self) -> Dict[str, List[str]]:
        """加载金融词汇库"""
        return {
            "concepts": [
                "量化投资", "择时策略", "动量效应", "反转效应", "价值投资", "成长投资",
                "技术分析", "基本面分析", "风险控制", "资产配置", "夏普比率", "最大回撤",
                "波动率", "相关性", "贝塔系数", "阿尔法收益", "信息比率", "跟踪误差",
                "VaR", "CVaR", "量化选股", "行业轮动", "风格轮动", "多因子模型",
                "机器学习", "深度学习", "神经网络", "支持向量机", "随机森林"
            ],
            "entities": [
                "平安证券", "国信证券", "海通证券", "招商证券", "国泰君安", "中信证券",
                "华泰证券", "广发证券", "申万宏源", "银河证券", "兴业证券", "东方证券"
            ],
            "metrics": [
                "ROE", "ROA", "P/E", "P/B", "EPS", "净利润率", "毛利率", "负债率",
                "流动比率", "速动比率", "营收增长率", "利润增长率", "市场份额", "市占率"
            ],
            "indicators": [
                "MACD", "RSI", "KDJ", "布林带", "移动平均线MA", "成交量VOL",
                "威廉指标WR", "能量潮OBV", "动向指标DMI", "CCI", "MTM"
            ],
            "time_expressions": [
                "2023年", "2024年", "去年", "今年", "明年", "未来", "过去", "历史上",
                "最近", "近期", "当前", "目前", "第一季度", "第二季度", "上半年", "下半年",
                "疫情后", "金融危机期间", "牛市", "熊市", "震荡市"
            ],
            "analysis_verbs": [
                "分析", "预测", "评估", "比较", "对比", "研究", "探讨", "检验", "验证",
                "计算", "测量", "监控", "跟踪", "观察", "发现", "确定", "识别"
            ],
            "relationship_words": [
                "关系", "影响", "关联", "连接", "依赖", "相关", "涉及", "包含",
                "之间", "与", "和", "对", "向", "从", "因为", "由于", "导致", "造成"
            ]
        }

    def _load_question_patterns(self) -> Dict[QuestionType, List[Dict]]:
        """加载问题模式"""
        return {
            QuestionType.DEFINITION: [
                {"pattern": r"什么是(.+?)\??", "weight": 0.9, "mode": SearchMode.VECTOR},
                {"pattern": r"(.+?)的定义", "weight": 0.8, "mode": SearchMode.VECTOR},
                {"pattern": r"解释(.+?)", "weight": 0.7, "mode": SearchMode.VECTOR},
                {"pattern": r"描述(.+?)", "weight": 0.6, "mode": SearchMode.VECTOR}
            ],
            QuestionType.DATA_QUERY: [
                {"pattern": r"(.+?)的数据", "weight": 0.8, "mode": SearchMode.VECTOR},
                {"pattern": r"查询(.+?)", "weight": 0.7, "mode": SearchMode.VECTOR},
                {"pattern": r"(.+?)的指标", "weight": 0.7, "mode": SearchMode.HYBRID},
                {"pattern": r"(.+?)的最新", "weight": 0.6, "mode": SearchMode.HYBRID}
            ],
            QuestionType.METHODOLOGY: [
                {"pattern": r"如何(.+?)", "weight": 0.8, "mode": SearchMode.DEEP_SEARCH},
                {"pattern": r"怎么(.+?)", "weight": 0.7, "mode": SearchMode.DEEP_SEARCH},
                {"pattern": r"(.+?)的方法", "weight": 0.6, "mode": SearchMode.DEEP_SEARCH},
                {"pattern": r"(.+?)的策略", "weight": 0.7, "mode": SearchMode.HYBRID}
            ],
            QuestionType.COMPARISON: [
                {"pattern": r"(.+?)和(.+?)的区别", "weight": 0.9, "mode": SearchMode.HYBRID},
                {"pattern": r"比较(.+?)", "weight": 0.8, "mode": SearchMode.HYBRID},
                {"pattern": r"(.+?)与(.+?)", "weight": 0.6, "mode": SearchMode.KNOWLEDGE_GRAPH},
                {"pattern": r"(.+?)对比(.+?)", "weight": 0.7, "mode": SearchMode.HYBRID}
            ],
            QuestionType.ANALYSIS: [
                {"pattern": r"分析(.+?)", "weight": 0.8, "mode": SearchMode.DEEP_SEARCH},
                {"pattern": r"评估(.+?)", "weight": 0.7, "mode": SearchMode.DEEP_SEARCH},
                {"pattern": r"研究(.+?)", "weight": 0.7, "mode": SearchMode.DEEP_SEARCH},
                {"pattern": r"(.+?)的影响", "weight": 0.6, "mode": SearchMode.KNOWLEDGE_GRAPH}
            ],
            QuestionType.PREDICTION: [
                {"pattern": r"预测(.+?)", "weight": 0.9, "mode": SearchMode.DEEP_SEARCH},
                {"pattern": r"(.+?)的趋势", "weight": 0.8, "mode": SearchMode.DEEP_SEARCH},
                {"pattern": r"未来(.+?)", "weight": 0.7, "mode": SearchMode.DEEP_SEARCH},
                {"pattern": r"(.+?)的前景", "weight": 0.6, "mode": SearchMode.DEEP_SEARCH}
            ],
            QuestionType.RELATIONSHIP: [
                {"pattern": r"(.+?)和(.+?)的关系", "weight": 0.9, "mode": SearchMode.KNOWLEDGE_GRAPH},
                {"pattern": r"(.+?)如何影响(.+?)", "weight": 0.8, "mode": SearchMode.KNOWLEDGE_GRAPH},
                {"pattern": r"(.+?)与(.+?)的关联", "weight": 0.7, "mode": SearchMode.KNOWLEDGE_GRAPH},
                {"pattern": r"(.+?)和(.+?)之间", "weight": 0.6, "mode": SearchMode.KNOWLEDGE_GRAPH}
            ]
        }

    def _initialize_mode_weights(self) -> Dict[SearchMode, Dict[str, float]]:
        """初始化模式权重"""
        return {
            SearchMode.VECTOR: {
                "definition": 0.9,
                "data_query": 0.8,
                "methodology": 0.3,
                "comparison": 0.4,
                "analysis": 0.3,
                "prediction": 0.2,
                "relationship": 0.2,
                "composite": 0.5
            },
            SearchMode.KNOWLEDGE_GRAPH: {
                "definition": 0.3,
                "data_query": 0.4,
                "methodology": 0.5,
                "comparison": 0.8,
                "analysis": 0.6,
                "prediction": 0.4,
                "relationship": 0.9,
                "composite": 0.6
            },
            SearchMode.DEEP_SEARCH: {
                "definition": 0.5,
                "data_query": 0.6,
                "methodology": 0.9,
                "comparison": 0.7,
                "analysis": 0.9,
                "prediction": 0.9,
                "relationship": 0.6,
                "composite": 0.8
            },
            SearchMode.HYBRID: {
                "definition": 0.7,
                "data_query": 0.8,
                "methodology": 0.7,
                "comparison": 0.9,
                "analysis": 0.8,
                "prediction": 0.7,
                "relationship": 0.7,
                "composite": 0.9
            }
        }

    def _load_learning_data(self) -> Dict:
        """加载学习数据（历史成功案例）"""
        return {
            "successful_patterns": {
                "vector": [
                    "什么是量化投资？",
                    "解释夏普比率的定义",
                    "查询平安证券的ROE数据"
                ],
                "knowledge_graph": [
                    "平安证券和证券业的关系",
                    "利率变化对股市的影响",
                    "量化投资和技术分析的关系"
                ],
                "deep_search": [
                    "分析2024年A股市场的表现",
                    "预测科技股的未来趋势",
                    "评估价值投资策略的优劣势"
                ],
                "hybrid": [
                    "比较成长投资和价值投资",
                    "查询最新的市场数据并分析",
                    "如何利用MACD指标进行交易"
                ]
            },
            "user_feedback": {},
            "performance_metrics": {}
        }

    def analyze_question(self, question: str) -> QuestionAnalysis:
        """分析问题"""
        question = question.strip()

        # 基础分析
        question_type = self._classify_question_type(question)
        complexity = self._assess_complexity(question)
        domain = self._identify_domain(question)
        keywords = self._extract_keywords(question)
        entities = self._extract_entities(question)
        intent = self._extract_intent(question)
        temporal_aspect = self._extract_temporal_aspect(question)
        scope = self._determine_scope(question)

        return QuestionAnalysis(
            text=question,
            type=question_type,
            complexity=complexity,
            domain=domain,
            keywords=keywords,
            entities=entities,
            intent=intent,
            temporal_aspect=temporal_aspect,
            scope=scope
        )

    def _classify_question_type(self, question: str) -> QuestionType:
        """分类问题类型"""
        question_lower = question.lower()

        max_score = 0
        best_type = QuestionType.DEFINITION

        for qtype, patterns in self.question_patterns.items():
            score = 0
            for pattern_info in patterns:
                if re.search(pattern_info["pattern"], question_lower):
                    score += pattern_info["weight"]

            if score > max_score:
                max_score = score
                best_type = qtype

        # 特殊情况检查
        if "和" in question and ("区别" in question or "比较" in question):
            return QuestionType.COMPARISON
        elif "关系" in question or "影响" in question:
            return QuestionType.RELATIONSHIP
        elif "预测" in question or "趋势" in question or "未来" in question:
            return QuestionType.PREDICTION
        elif len(re.findall(r'[，。；！]', question)) > 1:
            return QuestionType.COMPOSITE

        return best_type

    def _assess_complexity(self, question: str) -> str:
        """评估问题复杂度"""
        complexity_score = 0

        # 基于长度
        complexity_score += len(question) / 50

        # 基于复杂词汇
        complex_words = self.financial_vocabulary["analysis_verbs"]
        complexity_score += sum(1 for word in complex_words if word in question)

        # 基于连接词
        connectors = ["和", "与", "以及", "同时", "另外", "此外", "但是", "然而", "因为", "所以"]
        complexity_score += sum(1 for conn in connectors if conn in question)

        # 基于实体数量
        entities = self._extract_entities(question)
        complexity_score += len(entities) * 0.5

        # 基于时间表达
        time_expressions = self.financial_vocabulary["time_expressions"]
        complexity_score += sum(1 for expr in time_expressions if expr in question) * 0.3

        if complexity_score < 1.5:
            return "low"
        elif complexity_score < 3:
            return "medium"
        elif complexity_score < 5:
            return "high"
        else:
            return "very_high"

    def _identify_domain(self, question: str) -> str:
        """识别问题领域"""
        question_lower = question.lower()

        domain_scores = {
            "investment": sum(1 for word in ["投资", "股票", "基金", "债券", "资产"] if word in question_lower),
            "trading": sum(1 for word in ["交易", "买卖", "套利", "投机", "持仓"] if word in question_lower),
            "risk": sum(1 for word in ["风险", "回撤", "波动", "损失", "安全"] if word in question_lower),
            "analysis": sum(1 for word in ["分析", "研究", "评估", "诊断", "检验"] if word in question_lower),
            "strategy": sum(1 for word in ["策略", "方法", "技巧", "方案", "规划"] if word in question_lower),
            "data": sum(1 for word in ["数据", "统计", "指标", "比率", "数值"] if word in question_lower)
        }

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "general"

    def _extract_keywords(self, question: str) -> List[str]:
        """提取关键词"""
        keywords = []

        # 提取金融概念
        for concept in self.financial_vocabulary["concepts"]:
            if concept in question:
                keywords.append(concept)

        # 提取指标
        for metric in self.financial_vocabulary["metrics"]:
            if metric in question:
                keywords.append(metric)

        # 提取技术指标
        for indicator in self.financial_vocabulary["indicators"]:
            if indicator in question:
                keywords.append(indicator)

        return list(set(keywords))

    def _extract_entities(self, question: str) -> List[str]:
        """提取实体"""
        entities = []

        # 提取公司实体
        for entity in self.financial_vocabulary["entities"]:
            if entity in question:
                entities.append(entity)

        return entities

    def _extract_intent(self, question: str) -> str:
        """提取意图"""
        question_lower = question.lower()

        if any(word in question_lower for word in ["什么", "定义", "解释", "描述"]):
            return "understand"
        elif any(word in question_lower for word in ["查询", "数据", "指标", "统计"]):
            return "retrieve"
        elif any(word in question_lower for word in ["如何", "怎么", "方法", "策略"]):
            return "learn_method"
        elif any(word in question_lower for word in ["比较", "区别", "对比"]):
            return "compare"
        elif any(word in question_lower for word in ["分析", "评估", "研究"]):
            return "analyze"
        elif any(word in question_lower for word in ["预测", "趋势", "未来", "前景"]):
            return "predict"
        elif any(word in question_lower for word in ["关系", "影响", "关联"]):
            return "understand_relationship"
        else:
            return "general"

    def _extract_temporal_aspect(self, question: str) -> str:
        """提取时间方面"""
        question_lower = question.lower()

        if any(word in question_lower for word in ["历史", "过去", "以前", "曾经"]):
            return "past"
        elif any(word in question_lower for word in ["当前", "目前", "现在", "今天"]):
            return "present"
        elif any(word in question_lower for word in ["未来", "预测", "趋势", "明年", "今后"]):
            return "future"
        else:
            return "none"

    def _determine_scope(self, question: str) -> str:
        """确定问题范围"""
        if len(re.findall(r'[，。；！]', question)) > 2:
            return "analytical"
        elif "和" in question or "与" in question:
            return "comparative"
        elif any(word in question for word in ["详细", "全面", "深入", "完整"]):
            return "comprehensive"
        elif any(word in question for word in ["简单", "简要", "概述"]):
            return "specific"
        else:
            return "general"

    def calculate_mode_scores(self, analysis: QuestionAnalysis) -> List[ModeScore]:
        """计算各模式的评分"""
        scores = []

        for mode in SearchMode:
            score = 0
            reasons = []

            # 基于问题类型的评分
            type_weight = self.mode_weights[mode][analysis.type.value]
            score += type_weight * 0.3
            reasons.append(f"问题类型匹配: {type_weight:.2f}")

            # 基于复杂度的评分
            if analysis.complexity == "very_high" and mode == SearchMode.DEEP_SEARCH:
                score += 0.3
                reasons.append("高复杂度适合深度搜索")
            elif analysis.complexity == "low" and mode == SearchMode.VECTOR:
                score += 0.2
                reasons.append("低复杂度适合向量搜索")

            # 基于领域的评分
            if analysis.domain == "data" and mode in [SearchMode.VECTOR, SearchMode.HYBRID]:
                score += 0.2
                reasons.append("数据查询适合向量/混合搜索")
            elif analysis.domain == "analysis" and mode in [SearchMode.DEEP_SEARCH, SearchMode.KNOWLEDGE_GRAPH]:
                score += 0.2
                reasons.append("分析类问题适合深度/知识图谱搜索")

            # 基于实体的评分
            if analysis.entities and mode == SearchMode.KNOWLEDGE_GRAPH:
                score += 0.15 * len(analysis.entities)
                reasons.append(f"包含{len(analysis.entities)}个实体，适合知识图谱")

            # 基于时间方面的评分
            if analysis.temporal_aspect == "future" and mode == SearchMode.DEEP_SEARCH:
                score += 0.1
                reasons.append("预测类问题适合深度搜索")

            # 基于范围的评分
            if analysis.scope == "comparative" and mode == SearchMode.HYBRID:
                score += 0.15
                reasons.append("比较类问题适合混合搜索")
            elif analysis.scope == "comprehensive" and mode == SearchMode.DEEP_SEARCH:
                score += 0.15
                reasons.append("全面分析适合深度搜索")

            # 基于关键词的评分
            keyword_bonus = self._calculate_keyword_bonus(analysis.keywords, mode)
            score += keyword_bonus
            if keyword_bonus > 0:
                reasons.append(f"关键词匹配加分: {keyword_bonus:.2f}")

            # 基于历史学习数据的评分
            learning_bonus = self._calculate_learning_bonus(analysis, mode)
            score += learning_bonus
            if learning_bonus > 0:
                reasons.append(f"历史学习加分: {learning_bonus:.2f}")

            # 确保分数在0-1范围内
            score = max(0, min(1, score))

            # 计算置信度
            confidence = self._calculate_confidence(score, analysis)

            scores.append(ModeScore(
                mode=mode,
                score=score,
                reasons=reasons,
                confidence=confidence
            ))

        return sorted(scores, key=lambda x: x.score, reverse=True)

    def _calculate_keyword_bonus(self, keywords: List[str], mode: SearchMode) -> float:
        """计算关键词加分"""
        if not keywords:
            return 0

        bonus = 0

        # 定义每种模式的关键词偏好
        mode_preferences = {
            SearchMode.VECTOR: ["定义", "概念", "数据", "指标", "统计"],
            SearchMode.KNOWLEDGE_GRAPH: ["关系", "影响", "实体", "公司", "行业"],
            SearchMode.DEEP_SEARCH: ["分析", "预测", "趋势", "评估", "研究"],
            SearchMode.HYBRID: ["比较", "对比", "综合", "详细", "全面"]
        }

        for keyword in keywords:
            for pref in mode_preferences[mode]:
                if pref in keyword:
                    bonus += 0.05

        return min(bonus, 0.2)

    def _calculate_learning_bonus(self, analysis: QuestionAnalysis, mode: SearchMode) -> float:
        """计算学习加分"""
        # 基于历史成功模式的加分
        successful_patterns = self.learning_data["successful_patterns"][mode.value]

        for pattern in successful_patterns:
            # 简单的相似性检查
            common_words = set(analysis.text.split()) & set(pattern.split())
            similarity = len(common_words) / max(len(analysis.text.split()), len(pattern.split()))

            if similarity > 0.5:
                return similarity * 0.1

        return 0

    def _calculate_confidence(self, score: float, analysis: QuestionAnalysis) -> float:
        """计算置信度"""
        base_confidence = score

        # 根据分析结果调整置信度
        if analysis.keywords:
            base_confidence += 0.1

        if analysis.entities:
            base_confidence += 0.1

        if analysis.type in [QuestionType.DEFINITION, QuestionType.DATA_QUERY]:
            base_confidence += 0.1

        return max(0, min(1, base_confidence))

    def select_best_mode(self, question: str) -> Tuple[SearchMode, float, Dict]:
        """选择最佳搜索模式"""
        analysis = self.analyze_question(question)
        scores = self.calculate_mode_scores(analysis)

        best_score = scores[0]

        # 如果最高分和第二高分差距很小，考虑使用混合模式
        if len(scores) > 1 and abs(scores[0].score - scores[1].score) < 0.1:
            if scores[0].mode != SearchMode.HYBRID and scores[1].mode != SearchMode.HYBRID:
                best_score = ModeScore(
                    mode=SearchMode.HYBRID,
                    score=(scores[0].score + scores[1].score) / 2,
                    reasons=["多个模式得分相近，选择混合模式"],
                    confidence=0.8
                )

        return best_score.mode, best_score.score, {
            "analysis": analysis,
            "all_scores": scores,
            "selected_reasons": best_score.reasons,
            "confidence": best_score.confidence
        }

    def update_learning_data(self, question: str, selected_mode: SearchMode,
                            user_feedback: Optional[Dict] = None):
        """更新学习数据"""
        if user_feedback:
            question_type = self._classify_question_type(question)

            # 记录成功的模式选择
            if user_feedback.get("satisfaction", 0) > 0.7:
                if question not in self.learning_data["successful_patterns"][selected_mode.value]:
                    self.learning_data["successful_patterns"][selected_mode.value].append(question)

            # 记录用户反馈
            self.learning_data["user_feedback"][question] = {
                "selected_mode": selected_mode.value,
                "feedback": user_feedback,
                "timestamp": datetime.now().isoformat()
            }

            # 动态调整权重
            self._adjust_mode_weights(question_type, selected_mode, user_feedback)

    def _adjust_mode_weights(self, question_type: QuestionType,
                           selected_mode: SearchMode, feedback: Dict):
        """动态调整模式权重"""
        satisfaction = feedback.get("satisfaction", 0.5)

        if satisfaction > 0.8:
            # 成功案例，增加权重
            current_weight = self.mode_weights[selected_mode][question_type.value]
            self.mode_weights[selected_mode][question_type.value] = min(1.0, current_weight * 1.05)

        elif satisfaction < 0.4:
            # 失败案例，降低权重
            current_weight = self.mode_weights[selected_mode][question_type.value]
            self.mode_weights[selected_mode][question_type.value] = max(0.1, current_weight * 0.95)

    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        return {
            "total_patterns": sum(len(patterns) for patterns in self.learning_data["successful_patterns"].values()),
            "mode_distribution": {
                mode: len(patterns) for mode, patterns in self.learning_data["successful_patterns"].items()
            },
            "learning_updates": len(self.learning_data["user_feedback"]),
            "mode_weights": self.mode_weights
        }

# 使用示例
if __name__ == "__main__":
    selector = EnhancedSearchSelector()

    # 测试问题
    questions = [
        "什么是量化投资？",
        "平安证券和证券业有什么关系？",
        "分析2024年A股市场的表现",
        "比较价值投资和成长投资策略",
        "查询最新的市场数据"
    ]

    for question in questions:
        mode, score, details = selector.select_best_mode(question)
        print(f"问题: {question}")
        print(f"推荐模式: {mode.value}")
        print(f"评分: {score:.3f}")
        print(f"置信度: {details['confidence']:.3f}")
        print(f"原因: {', '.join(details['selected_reasons'])}")
        print("-" * 50)