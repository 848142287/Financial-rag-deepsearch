"""
优化的模式匹配算法

实现高效的搜索模式匹配和选择
支持动态权重调整和性能优化
"""

import re
import json
import math
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class SearchMode(Enum):
    """搜索模式枚举"""
    VECTOR = "vector"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    DEEP_SEARCH = "deep_search"
    HYBRID = "hybrid"

class MatchStrategy(Enum):
    """匹配策略"""
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

@dataclass
class PatternMatch:
    """模式匹配结果"""
    pattern_id: str
    match_type: str
    confidence: float
    matched_text: str
    start_pos: int
    end_pos: int
    features: Dict[str, Any]

@dataclass
class ModeRecommendation:
    """模式推荐结果"""
    mode: SearchMode
    confidence: float
    primary_reason: str
    supporting_reasons: List[str]
    risk_factors: List[str]
    alternative_modes: List[Tuple[SearchMode, float]]
    metadata: Dict[str, Any]

class OptimizedPatternMatcher:
    """优化的模式匹配器"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.patterns = self._initialize_patterns()
        self.performance_tracker = PerformanceTracker()
        self.learning_engine = LearningEngine()
        self.semantic_index = SemanticIndex()
        self.fuzzy_matcher = FuzzyMatcher()

        # 动态权重
        self.dynamic_weights = self.config.get("dynamic_weights", {})
        self.weight_history = []

        # 缓存
        self.match_cache = {}
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1小时

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        default_config = {
            "dynamic_weights": True,
            "semantic_similarity_threshold": 0.7,
            "fuzzy_match_threshold": 0.8,
            "cache_enabled": True,
            "cache_ttl": 3600,
            "performance_tracking": True,
            "learning_enabled": True
        }

        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except FileNotFoundError:
                logger.warning(f"Config file {config_path} not found, using defaults")
            except Exception as e:
                logger.error(f"Error loading config: {e}")

        return default_config

    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """初始化模式库"""
        return {
            SearchMode.VECTOR.value: [
                {
                    "id": "vector_definition_1",
                    "type": "regex",
                    "pattern": r"什么是(.+?)\？?",
                    "weight": 0.95,
                    "features": ["definition", "concept"],
                    "examples": ["什么是量化投资？", "什么是夏普比率？"]
                },
                {
                    "id": "vector_definition_2",
                    "type": "keyword",
                    "keywords": ["定义", "解释", "描述", "概念", "含义"],
                    "weight": 0.8,
                    "features": ["definition", "explanation"],
                    "examples": ["解释量化投资的定义", "描述价值投资的概念"]
                },
                {
                    "id": "vector_data_1",
                    "type": "regex",
                    "pattern": r"(.+?)的(.+?)是多少|(.+?)的数据|(.+?)的指标|(.+?)的统计",
                    "weight": 0.9,
                    "features": ["data", "metrics", "query"],
                    "examples": ["查询ROE数据", "获取市场指标", "平安证券的市盈率是多少"]
                },
                {
                    "id": "vector_simple_1",
                    "type": "length_based",
                    "max_length": 20,
                    "max_entities": 1,
                    "weight": 0.7,
                    "features": ["simple", "concise"],
                    "examples": ["什么是MACD？", "查询股价"]
                }
            ],
            SearchMode.KNOWLEDGE_GRAPH.value: [
                {
                    "id": "kg_relationship_1",
                    "type": "regex",
                    "pattern": r"(.+?)和(.+?)有什么关系\？|(.+?)和(.+?)的关系|(.+?)与(.+?)的关联|(.+?)如何影响(.+?)",
                    "weight": 0.95,
                    "features": ["relationship", "influence"],
                    "examples": ["利率变化对股市的影响", "平安证券和证券业的关系", "平安证券和证券业有什么关系？"]
                },
                {
                    "id": "kg_entity_1",
                    "type": "entity_based",
                    "min_entities": 2,
                    "weight": 0.8,
                    "features": ["multi_entity", "connection"],
                    "examples": ["平安证券和招商证券", "银行业和证券业"]
                },
                {
                    "id": "kg_network_1",
                    "type": "keyword",
                    "keywords": ["关系", "影响", "关联", "连接", "网络", "图谱"],
                    "weight": 0.7,
                    "features": ["network", "graph"],
                    "examples": ["企业关系网络", "行业关联分析"]
                }
            ],
            SearchMode.DEEP_SEARCH.value: [
                {
                    "id": "deep_analysis_1",
                    "type": "regex",
                    "pattern": r"分析(.+?)的(.+?)表现|分析(.+?)|评估(.+?)|研究(.+?)|探讨(.+?)",
                    "weight": 0.9,
                    "features": ["analysis", "evaluation"],
                    "examples": ["分析市场趋势", "评估投资风险", "分析2024年的A股表现"]
                },
                {
                    "id": "deep_prediction_1",
                    "type": "regex",
                    "pattern": r"预测(.+?)未来(.+?)的发展趋势|预测(.+?)|(.+?)的趋势|(.+?)的前景|未来(.+?)",
                    "weight": 0.95,
                    "features": ["prediction", "forecasting"],
                    "examples": ["预测股市走势", "分析未来趋势", "预测科技股未来一年的发展趋势"]
                },
                {
                    "id": "deep_complex_1",
                    "type": "complexity_based",
                    "min_complexity": "high",
                    "weight": 0.7,
                    "features": ["complex", "comprehensive"],
                    "examples": ["深入分析投资策略", "全面评估市场风险"]
                },
                {
                    "id": "deep_multi_step_1",
                    "type": "multi_step",
                    "min_steps": 2,
                    "weight": 0.6,
                    "features": ["multi_step", "reasoning"],
                    "examples": ["分析原因并提出建议", "评估现状并预测未来"]
                }
            ],
            SearchMode.HYBRID.value: [
                {
                    "id": "hybrid_comparison_1",
                    "type": "regex",
                    "pattern": r"比较(.+?)|(.+?)和(.+?)的区别|(.+?)对比(.+?)",
                    "weight": 0.9,
                    "features": ["comparison", "contrast"],
                    "examples": ["比较价值投资和成长投资", "分析两种策略的区别"]
                },
                {
                    "id": "hybrid_comprehensive_1",
                    "type": "comprehensive",
                    "requires_multiple_aspects": True,
                    "weight": 0.8,
                    "features": ["comprehensive", "multi_aspect"],
                    "examples": ["全面分析投资机会", "综合评估市场状况"]
                },
                {
                    "id": "hybrid_balanced_1",
                    "type": "default_fallback",
                    "weight": 0.5,
                    "features": ["balanced", "versatile"],
                    "examples": ["查询投资建议", "获取市场信息"]
                }
            ]
        }

    def match_patterns(self, text: str, mode: SearchMode) -> List[PatternMatch]:
        """匹配特定模式的文本"""
        # 确保text是字符串类型
        if not isinstance(text, str):
            logger.warning(f"Expected string for text, got {type(text)}: {text}")
            text = str(text)

        # 检查缓存
        try:
            cache_key = f"{mode.value}:{hash(text)}"
        except TypeError as e:
            logger.error(f"Cannot create cache key for text: {e}")
            # 使用不基于hash的缓存键
            cache_key = f"{mode.value}:{len(text)}_{text[:50] if len(text) > 50 else text}"

        if self.config.get("cache_enabled", False) and cache_key in self.match_cache:
            cached_result = self.match_cache[cache_key]
            if datetime.now().timestamp() - cached_result["timestamp"] < self.cache_ttl:
                return cached_result["matches"]

        matches = []
        patterns = self.patterns.get(mode.value, [])

        for pattern in patterns:
            pattern_matches = self._match_single_pattern(text, pattern)
            matches.extend(pattern_matches)

        # 缓存结果
        if self.config.get("cache_enabled", False):
            self.match_cache[cache_key] = {
                "matches": matches,
                "timestamp": datetime.now().timestamp()
            }

        return matches

    def _match_single_pattern(self, text: str, pattern: Dict) -> List[PatternMatch]:
        """匹配单个模式"""
        matches = []
        pattern_type = pattern.get("type", "unknown")

        try:
            if pattern_type == "regex":
                matches = self._match_regex_pattern(text, pattern)
            elif pattern_type == "keyword":
                matches = self._match_keyword_pattern(text, pattern)
            elif pattern_type == "entity_based":
                matches = self._match_entity_pattern(text, pattern)
            elif pattern_type == "length_based":
                matches = self._match_length_pattern(text, pattern)
            elif pattern_type == "complexity_based":
                matches = self._match_complexity_pattern(text, pattern)
            elif pattern_type == "semantic":
                matches = self._match_semantic_pattern(text, pattern)
            elif pattern_type in ["multi_step", "comprehensive", "default_fallback"]:
                # 暂时跳过这些未知的模式类型，专注于核心模式
                logger.debug(f"Skipping pattern type: {pattern_type}")
                matches = []
            else:
                logger.warning(f"Unknown pattern type: {pattern_type}")
                matches = []

        except Exception as e:
            logger.error(f"Error matching pattern {pattern.get('id', 'unknown')}: {e}")

        return matches

    def _match_regex_pattern(self, text: str, pattern: Dict) -> List[PatternMatch]:
        """匹配正则表达式模式"""
        matches = []
        regex_pattern = pattern.get("pattern", "")

        try:
            for match in re.finditer(regex_pattern, text, re.IGNORECASE):
                confidence = pattern.get("weight", 0.5)

                # 调整置信度基于匹配质量
                match_text = match.group(0)
                if len(match_text) / len(text) > 0.5:  # 匹配了文本的大部分
                    confidence *= 1.2

                matches.append(PatternMatch(
                    pattern_id=pattern["id"],
                    match_type="regex",
                    confidence=min(confidence, 1.0),
                    matched_text=match_text,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    features=pattern.get("features", [])
                ))

        except re.error as e:
            logger.error(f"Regex error in pattern {pattern.get('id', 'unknown')}: {e}")

        return matches

    def _match_keyword_pattern(self, text: str, pattern: Dict) -> List[PatternMatch]:
        """匹配关键词模式"""
        matches = []
        keywords = pattern.get("keywords", [])
        text_lower = text.lower()

        for keyword in keywords:
            keyword_lower = keyword.lower()
            start = 0
            while True:
                pos = text_lower.find(keyword_lower, start)
                if pos == -1:
                    break

                confidence = pattern.get("weight", 0.5)

                # 基于关键词重要性调整置信度
                if keyword in text:  # 原始匹配（区分大小写）
                    confidence *= 1.1

                matches.append(PatternMatch(
                    pattern_id=pattern["id"],
                    match_type="keyword",
                    confidence=min(confidence, 1.0),
                    matched_text=text[pos:pos+len(keyword)],
                    start_pos=pos,
                    end_pos=pos+len(keyword),
                    features=pattern.get("features", [])
                ))

                start = pos + 1

        return matches

    def _match_entity_pattern(self, text: str, pattern: Dict) -> List[PatternMatch]:
        """匹配实体模式"""
        # 简化的实体提取
        entities = ["平安证券", "国信证券", "海通证券", "招商证券", "国泰君安", "中信证券",
                   "华泰证券", "广发证券", "申万宏源", "银河证券"]

        found_entities = [entity for entity in entities if entity in text]

        min_entities = pattern.get("min_entities", 2)
        if len(found_entities) >= min_entities:
            confidence = pattern.get("weight", 0.5)
            # 实体越多，置信度越高
            confidence *= min(1.0, len(found_entities) / min_entities)

            return [PatternMatch(
                pattern_id=pattern["id"],
                match_type="entity",
                confidence=confidence,
                matched_text=", ".join(found_entities),
                start_pos=0,
                end_pos=len(text),
                features=pattern.get("features", [])
            )]

        return []

    def _match_length_pattern(self, text: str, pattern: Dict) -> List[PatternMatch]:
        """匹配长度模式"""
        text_length = len(text)
        max_length = pattern.get("max_length", float('inf'))

        if text_length <= max_length:
            confidence = pattern.get("weight", 0.5)
            # 越短，置信度越高
            confidence *= (1.0 - (text_length / max_length))

            return [PatternMatch(
                pattern_id=pattern["id"],
                match_type="length",
                confidence=max(confidence, 0),
                matched_text=text,
                start_pos=0,
                end_pos=len(text),
                features=pattern.get("features", [])
            )]

        return []

    def _match_complexity_pattern(self, text: str, pattern: Dict) -> List[PatternMatch]:
        """匹配复杂度模式"""
        # 简化的复杂度计算
        complexity_indicators = ["分析", "评估", "预测", "研究", "比较", "影响"]
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in text)

        min_complexity = pattern.get("min_complexity", "medium")
        complexity_levels = {"low": 1, "medium": 3, "high": 5, "very_high": 8}

        required_score = complexity_levels.get(min_complexity, 3)

        if complexity_score >= required_score:
            confidence = pattern.get("weight", 0.5)
            # 复杂度越高，置信度越高
            confidence *= min(1.0, complexity_score / required_score)

            return [PatternMatch(
                pattern_id=pattern["id"],
                match_type="complexity",
                confidence=min(confidence, 1.0),
                matched_text=text,
                start_pos=0,
                end_pos=len(text),
                features=pattern.get("features", [])
            )]

        return []

    def _match_semantic_pattern(self, text: str, pattern: Dict) -> List[PatternMatch]:
        """匹配语义模式"""
        # 使用语义索引进行匹配
        if hasattr(self, 'semantic_index'):
            similarity = self.semantic_index.get_similarity(text, pattern.get("semantic_vector"))
            threshold = pattern.get("similarity_threshold", 0.7)

            if similarity >= threshold:
                return [PatternMatch(
                    pattern_id=pattern["id"],
                    match_type="semantic",
                    confidence=similarity,
                    matched_text=text,
                    start_pos=0,
                    end_pos=len(text),
                    features=pattern.get("features", [])
                )]

        return []

    def calculate_mode_scores(self, text: str) -> Dict[SearchMode, float]:
        """计算各模式的评分"""
        scores = {}
        total_matches = 0

        # 为每个模式计算基础分数
        for mode in SearchMode:
            matches = self.match_patterns(text, mode)

            if not matches:
                scores[mode] = 0.0
                continue

            # 加权平均置信度
            total_confidence = sum(match.confidence for match in matches)
            avg_confidence = total_confidence / len(matches)

            # 匹配数量因子
            match_count_factor = min(1.0, len(matches) / 3)  # 最多3个匹配为满分

            # 特征多样性因子
            unique_features = len(set(feature for match in matches for feature in match.features))
            diversity_factor = min(1.0, unique_features / 5)  # 最多5个不同特征为满分

            # 位置权重（匹配文本开始位置的权重更高）
            position_factor = 1.0
            if matches:
                avg_position = sum(match.start_pos for match in matches) / len(matches)
                position_factor = max(0.3, 1.0 - (avg_position / len(text)))

            # 综合评分
            base_score = avg_confidence * 0.4 + match_count_factor * 0.3 + diversity_factor * 0.2 + position_factor * 0.1

            # 应用动态权重调整
            if self.config.get("dynamic_weights", False):
                base_score = self._apply_dynamic_weights(mode, base_score, text)

            scores[mode] = base_score
            total_matches += len(matches)

        # 归一化分数
        if total_matches == 0:
            return scores

        # 确保分数在0-1范围内
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {mode: score / max_score for mode, score in scores.items()}

        return scores

    def _apply_dynamic_weights(self, mode: SearchMode, base_score: float, text: str) -> float:
        """应用动态权重调整"""
        # 基于历史性能调整
        performance_factor = self.performance_tracker.get_performance_factor(mode)
        base_score *= performance_factor

        # 基于学习引擎调整
        learning_factor = self.learning_engine.get_learning_factor(mode, text)
        base_score *= learning_factor

        # 基于当前上下文调整
        context_factor = self._calculate_context_factor(mode, text)
        base_score *= context_factor

        return min(base_score, 1.0)

    def _calculate_context_factor(self, mode: SearchMode, text: str) -> float:
        """计算上下文因子"""
        # 基于文本长度调整
        if len(text) < 20 and mode == SearchMode.VECTOR:
            return 1.2
        elif len(text) > 100 and mode == SearchMode.DEEP_SEARCH:
            return 1.2

        # 基于时间表达调整
        if any(word in text for word in ["预测", "未来", "趋势"]) and mode == SearchMode.DEEP_SEARCH:
            return 1.1

        # 基于实体数量调整
        entity_count = text.count("证券") + text.count("银行") + text.count("保险")
        if entity_count >= 2 and mode == SearchMode.KNOWLEDGE_GRAPH:
            return 1.1

        return 1.0

    def recommend_mode(self, text: str) -> ModeRecommendation:
        """推荐最佳搜索模式"""
        scores = self.calculate_mode_scores(text)

        if not scores or max(scores.values()) == 0:
            # 如果没有匹配，返回默认模式
            return ModeRecommendation(
                mode=SearchMode.HYBRID,
                confidence=0.5,
                primary_reason="无明确模式匹配，使用默认混合模式",
                supporting_reasons=["提供平衡的检索结果"],
                risk_factors=["可能不是最优选择"],
                alternative_modes=[],
                metadata={"scores": scores}
            )

        # 获取最佳模式
        best_mode = max(scores, key=scores.get)
        best_score = scores[best_mode]

        # 生成推荐理由
        matches = self.match_patterns(text, best_mode)
        primary_reason, supporting_reasons = self._generate_recommendation_reasons(best_mode, matches)

        # 获取替代模式
        alternative_modes = [(mode, score) for mode, score in scores.items() if mode != best_mode and score > 0.3]
        alternative_modes.sort(key=lambda x: x[1], reverse=True)

        # 风险因子
        risk_factors = self._identify_risk_factors(best_mode, scores)

        # 元数据
        metadata = {
            "scores": scores,
            "matches_count": len(matches),
            "text_length": len(text),
            "timestamp": datetime.now().isoformat()
        }

        return ModeRecommendation(
            mode=best_mode,
            confidence=best_score,
            primary_reason=primary_reason,
            supporting_reasons=supporting_reasons,
            risk_factors=risk_factors,
            alternative_modes=alternative_modes[:3],  # 最多3个替代选项
            metadata=metadata
        )

    def _generate_recommendation_reasons(self, mode: SearchMode, matches: List[PatternMatch]) -> Tuple[str, List[str]]:
        """生成推荐理由"""
        if not matches:
            return "基于历史性能选择", ["该模式在类似查询中表现良好"]

        # 获取最高置信度的匹配
        best_match = max(matches, key=lambda m: m.confidence)

        primary_reason = f"匹配到{best_match.match_type}模式，置信度{best_match.confidence:.2f}"

        supporting_reasons = []
        features = set()
        for match in matches:
            if isinstance(match.features, list):
                features.update(match.features)
            else:
                features.add(match.features)
        for feature in features:
            if feature == "definition":
                supporting_reasons.append("问题涉及概念定义")
            elif feature == "data":
                supporting_reasons.append("问题需要数据查询")
            elif feature == "relationship":
                supporting_reasons.append("问题涉及实体关系")
            elif feature == "analysis":
                supporting_reasons.append("问题需要深度分析")
            elif feature == "comparison":
                supporting_reasons.append("问题需要进行比较")
            elif feature == "prediction":
                supporting_reasons.append("问题涉及预测分析")

        return primary_reason, supporting_reasons

    def _identify_risk_factors(self, selected_mode: SearchMode, scores: Dict[SearchMode, float]) -> List[str]:
        """识别风险因子"""
        risk_factors = []

        # 如果最佳模式分数较低
        best_score = scores[selected_mode]
        if best_score < 0.6:
            risk_factors.append("匹配置信度较低，可能不是最优选择")

        # 如果有多个模式得分相近
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0][1] - sorted_scores[1][1] < 0.2:
            second_best = sorted_scores[1][0]
            risk_factors.append(f"与{second_best.value}模式得分相近，可考虑混合模式")

        # 基于模式特定风险
        if selected_mode == SearchMode.VECTOR:
            risk_factors.append("向量检索可能无法处理复杂关系查询")
        elif selected_mode == SearchMode.KNOWLEDGE_GRAPH:
            risk_factors.append("知识图谱检索需要明确的实体关系")
        elif selected_mode == SearchMode.DEEP_SEARCH:
            risk_factors.append("深度检索可能需要更长的响应时间")
        elif selected_mode == SearchMode.HYBRID:
            risk_factors.append("混合模式可能在某些专业领域不够深入")

        return risk_factors

    def update_performance(self, mode: SearchMode, feedback_score: float, text: str):
        """更新性能数据"""
        if self.config.get("performance_tracking", False):
            self.performance_tracker.update_performance(mode, feedback_score, text)

        if self.config.get("learning_enabled", False):
            self.learning_engine.update_learning(mode, feedback_score, text)

    def get_optimization_suggestions(self) -> List[str]:
        """获取优化建议"""
        suggestions = []

        # 性能分析
        if self.config.get("performance_tracking", False):
            performance_analysis = self.performance_tracker.get_analysis()
            if performance_analysis:
                suggestions.extend(performance_analysis.get("suggestions", []))

        # 学习分析
        if self.config.get("learning_enabled", False):
            learning_analysis = self.learning_engine.get_analysis()
            if learning_analysis:
                suggestions.extend(learning_analysis.get("suggestions", []))

        # 模式覆盖分析
        pattern_coverage = self._analyze_pattern_coverage()
        suggestions.extend(pattern_coverage.get("suggestions", []))

        return suggestions

    def _analyze_pattern_coverage(self) -> Dict[str, Any]:
        """分析模式覆盖度"""
        analysis = {"suggestions": []}

        # 检查每种模式的模式数量
        for mode in SearchMode:
            patterns = self.patterns.get(mode.value, [])
            if len(patterns) < 3:
                analysis["suggestions"].append(f"{mode.value}模式只有{len(patterns)}个模式，建议增加更多模式")

        return analysis

    def save_state(self):
        """保存状态"""
        state = {
            "config": self.config,
            "dynamic_weights": self.dynamic_weights,
            "weight_history": self.weight_history[-100:],  # 保留最近100条记录
            "timestamp": datetime.now().isoformat()
        }

        if self.config.get("performance_tracking", False):
            state["performance_tracker"] = self.performance_tracker.get_state()

        if self.config.get("learning_enabled", False):
            state["learning_engine"] = self.learning_engine.get_state()

        # 保存到文件
        try:
            with open("pattern_matcher_state.json", 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def load_state(self):
        """加载状态"""
        try:
            with open("pattern_matcher_state.json", 'r', encoding='utf-8') as f:
                state = json.load(f)

            self.config.update(state.get("config", {}))
            self.dynamic_weights = state.get("dynamic_weights", {})
            self.weight_history = state.get("weight_history", [])

            if self.config.get("performance_tracking", False):
                self.performance_tracker.load_state(state.get("performance_tracker", {}))

            if self.config.get("learning_enabled", False):
                self.learning_engine.load_state(state.get("learning_engine", {}))

        except FileNotFoundError:
            logger.info("No saved state found, using defaults")
        except Exception as e:
            logger.error(f"Error loading state: {e}")


class PerformanceTracker:
    """性能跟踪器"""

    def __init__(self):
        self.performance_data = defaultdict(list)
        self.mode_stats = {}

    def update_performance(self, mode: SearchMode, feedback_score: float, text: str):
        """更新性能数据"""
        self.performance_data[mode.value].append({
            "score": feedback_score,
            "text_length": len(text),
            "timestamp": datetime.now().isoformat()
        })

        # 更新统计信息
        if len(self.performance_data[mode.value]) > 0:
            scores = [item["score"] for item in self.performance_data[mode.value]]
            self.mode_stats[mode.value] = {
                "avg_score": np.mean(scores),
                "std_score": np.std(scores),
                "count": len(scores),
                "recent_avg": np.mean(scores[-10:])  # 最近10次的平均分
            }

    def get_performance_factor(self, mode: SearchMode) -> float:
        """获取性能因子"""
        stats = self.mode_stats.get(mode.value, {})
        recent_avg = stats.get("recent_avg", 0.7)

        # 将平均分转换为因子 (0.5-1.5的范围)
        factor = 0.5 + recent_avg
        return max(0.5, min(1.5, factor))

    def get_analysis(self) -> Dict[str, Any]:
        """获取性能分析"""
        analysis = {"suggestions": []}

        for mode, stats in self.mode_stats.items():
            if stats["avg_score"] < 0.6:
                analysis["suggestions"].append(f"{mode}模式平均得分较低({stats['avg_score']:.2f})，需要优化")

            if stats["std_score"] > 0.3:
                analysis["suggestions"].append(f"{mode}模式性能不稳定(std={stats['std_score']:.2f})，需要改进一致性")

        return analysis

    def get_state(self) -> Dict:
        """获取状态"""
        return {
            "performance_data": dict(self.performance_data),
            "mode_stats": self.mode_stats
        }

    def load_state(self, state: Dict):
        """加载状态"""
        self.performance_data = defaultdict(list, state.get("performance_data", {}))
        self.mode_stats = state.get("mode_stats", {})


class LearningEngine:
    """学习引擎"""

    def __init__(self):
        self.learning_data = defaultdict(list)
        self.pattern_success = defaultdict(lambda: defaultdict(int))
        self.weights = {}

    def update_learning(self, mode: SearchMode, feedback_score: float, text: str):
        """更新学习数据"""
        self.learning_data[mode.value].append({
            "score": feedback_score,
            "text": text,
            "timestamp": datetime.now().isoformat()
        })

        # 简化的模式学习
        if feedback_score > 0.7:
            self.weights[mode.value] = self.weights.get(mode.value, 1.0) * 1.05
        elif feedback_score < 0.4:
            self.weights[mode.value] = self.weights.get(mode.value, 1.0) * 0.95

    def get_learning_factor(self, mode: SearchMode, text: str) -> float:
        """获取学习因子"""
        base_factor = self.weights.get(mode.value, 1.0)
        return max(0.5, min(1.5, base_factor))

    def get_analysis(self) -> Dict[str, Any]:
        """获取学习分析"""
        return {"suggestions": []}

    def get_state(self) -> Dict:
        """获取状态"""
        return {
            "learning_data": dict(self.learning_data),
            "pattern_success": dict(self.pattern_success),
            "weights": self.weights
        }

    def load_state(self, state: Dict):
        """加载状态"""
        self.learning_data = defaultdict(list, state.get("learning_data", {}))
        self.pattern_success = defaultdict(lambda: defaultdict(int), state.get("pattern_success", {}))
        self.weights = state.get("weights", {})

    def _match_multi_step_pattern(self, text: str, pattern: Dict) -> List[PatternMatch]:
        """匹配多步骤模式"""
        matches = []
        try:
            # 简化实现：基于文本长度和复杂度判断
            min_steps = pattern.get("min_steps", 2)
            complexity_indicators = ["分析", "评估", "比较", "预测", "研究"]

            step_count = sum(1 for indicator in complexity_indicators if indicator in text)

            if step_count >= min_steps:
                match = PatternMatch(
                    pattern_id=pattern.get("id", "multi_step_unknown"),
                    match_type="multi_step",
                    confidence=min(step_count / min_steps, 1.0) * 0.8,
                    span=(0, len(text)),
                    extracted_data={"steps": step_count, "indicators": complexity_indicators}
                )
                matches.append(match)
        except Exception as e:
            logger.error(f"Error in multi_step pattern matching: {e}")

        return matches

    def _match_comprehensive_pattern(self, text: str, pattern: Dict) -> List[PatternMatch]:
        """匹配综合模式"""
        matches = []
        try:
            # 简化实现：基于关键词密度判断
            keywords = pattern.get("keywords", [])
            keyword_count = sum(1 for keyword in keywords if keyword in text)

            if keyword_count >= 2:
                match = PatternMatch(
                    pattern_id=pattern.get("id", "comprehensive_unknown"),
                    match_type="comprehensive",
                    confidence=min(keyword_count / len(keywords), 1.0) * 0.7,
                    span=(0, len(text)),
                    extracted_data={"keywords_found": keyword_count, "total_keywords": len(keywords)}
                )
                matches.append(match)
        except Exception as e:
            logger.error(f"Error in comprehensive pattern matching: {e}")

        return matches

    def _match_default_fallback_pattern(self, text: str, pattern: Dict) -> List[PatternMatch]:
        """匹配默认回退模式"""
        matches = []
        try:
            # 简化实现：总是匹配，但置信度较低
            match = PatternMatch(
                pattern_id=pattern.get("id", "default_fallback"),
                match_type="default_fallback",
                confidence=0.3,
                span=(0, len(text)),
                extracted_data={"fallback": True, "text_length": len(text)}
            )
            matches.append(match)
        except Exception as e:
            logger.error(f"Error in default_fallback pattern matching: {e}")

        return matches


class SemanticIndex:
    """语义索引"""

    def __init__(self):
        self.index = {}

    def get_similarity(self, text: str, vector: Optional[List[float]]) -> float:
        """获取语义相似度"""
        # 简化实现，实际应该使用向量数据库
        return 0.0


  

class FuzzyMatcher:
    """模糊匹配器"""

    def __init__(self):
        self.threshold = 0.8

    def match(self, text: str, pattern: str) -> float:
        """模糊匹配"""
        # 简化实现，可以使用Levenshtein距离等
        return 0.0


# 使用示例
if __name__ == "__main__":
    matcher = OptimizedPatternMatcher()

    test_questions = [
        "什么是量化投资？",
        "平安证券和证券业的关系",
        "分析2024年股市表现",
        "比较价值投资和成长投资",
        "查询最新的市场数据"
    ]

    for question in test_questions:
        recommendation = matcher.recommend_mode(question)
        print(f"问题: {question}")
        print(f"推荐模式: {recommendation.mode.value}")
        print(f"置信度: {recommendation.confidence:.3f}")
        print(f"主要原因: {recommendation.primary_reason}")
        print(f"支持理由: {', '.join(recommendation.supporting_reasons)}")
        if recommendation.risk_factors:
            print(f"风险因子: {', '.join(recommendation.risk_factors)}")
        print("-" * 50)