"""
自适应参数优化器

基于查询特征和反馈历史，自动调整检索参数
"""

import math
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from app.core.structured_logging import get_structured_logger
from .query_classifier import get_query_classifier, QueryFeatures, QueryComplexity

logger = get_structured_logger(__name__)


class AdaptiveParameterOptimizer:
    """
    自适应参数优化器

    功能:
    1. 基于查询特征调整参数
    2. 基于反馈历史优化参数
    3. 动态平衡准确性和效率
    """

    def __init__(self):
        # 查询分类器
        self.classifier = get_query_classifier()

        # 参数边界
        self.param_bounds = {
            "top_k": (3, 20),
            "similarity_threshold": (0.3, 0.8),
            "compression_rate": (0.3, 0.8),
            "rerank_enabled": (False, True),
            "graph_search_enabled": (False, True)
        }

        # 历史性能统计 (用于学习最优参数)
        self.param_performance = defaultdict(lambda: {
            "usage_count": 0,
            "total_rating": 0.0,
            "avg_rating": 0.0,
            "success_count": 0
        })

        # 不同查询类型的最优参数 (从学习中获得)
        self.learned_optimal_params = {
            "factual_simple": {"top_k": 5, "similarity_threshold": 0.7, "compression_rate": 0.4},
            "factual_medium": {"top_k": 10, "similarity_threshold": 0.6, "compression_rate": 0.5},
            "factual_complex": {"top_k": 15, "similarity_threshold": 0.5, "compression_rate": 0.6},
            "analytical_simple": {"top_k": 8, "similarity_threshold": 0.65, "compression_rate": 0.5, "rerank_enabled": True},
            "analytical_medium": {"top_k": 12, "similarity_threshold": 0.55, "compression_rate": 0.6, "rerank_enabled": True},
            "analytical_complex": {"top_k": 15, "similarity_threshold": 0.5, "compression_rate": 0.7, "rerank_enabled": True, "graph_search_enabled": True},
        }

        logger.info("自适应参数优化器初始化完成")

    def optimize_parameters(
        self,
        query: str,
        base_params: Dict[str, Any],
        feedback_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        优化检索参数

        Args:
            query: 用户查询
            base_params: 基础参数
            feedback_context: 反馈上下文

        Returns:
            优化后的参数
        """
        try:
            # 1. 分析查询特征
            features = self.classifier.classify(query)

            logger.debug(
                f"查询特征: type={features.query_type.value}, "
                f"complexity={features.complexity.value}, "
                f"difficulty={features.estimated_difficulty:.2f}"
            )

            # 2. 获取基础参数
            params = base_params.copy()

            # 3. 基于查询特征调整参数
            params = self._adjust_by_features(features, params)

            # 4. 基于反馈历史调整参数
            if feedback_context:
                params = self._adjust_by_feedback(features, feedback_context, params)

            # 5. 应用参数边界约束
            params = self._apply_bounds(params)

            # 6. 记录参数使用
            self._record_param_usage(features, params)

            logger.info(
                f"参数优化完成: query={query[:30]}, "
                f"top_k={params.get('top_k')}, "
                f"threshold={params.get('similarity_threshold'):.2f}, "
                f"compression={params.get('compression_rate'):.2f}"
            )

            return params

        except Exception as e:
            logger.error(f"参数优化失败: {e}")
            return base_params

    def _adjust_by_features(
        self,
        features: QueryFeatures,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基于查询特征调整参数"""

        # 获取学习的最优参数
        lookup_key = f"{features.query_type.value}_{features.complexity.value}"

        if lookup_key in self.learned_optimal_params:
            # 使用学习的最优参数
            learned_params = self.learned_optimal_params[lookup_key]

            # 合并到当前参数
            for key, value in learned_params.items():
                if key not in params:  # 不覆盖用户显式设置的参数
                    params[key] = value

        else:
            # 基于规则生成参数
            params = self._rule_based_adjustment(features, params)

        return params

    def _rule_based_adjustment(
        self,
        features: QueryFeatures,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基于规则调整参数"""

        # 规则1: 根据复杂度调整top_k
        if features.complexity == QueryComplexity.SIMPLE:
            params["top_k"] = params.get("top_k", 10)
            params["similarity_threshold"] = params.get("similarity_threshold", 0.7)
            params["compression_rate"] = params.get("compression_rate", 0.4)

        elif features.complexity == QueryComplexity.MEDIUM:
            params["top_k"] = params.get("top_k", 10) + 2
            params["similarity_threshold"] = params.get("similarity_threshold", 0.6)
            params["compression_rate"] = params.get("compression_rate", 0.5)

        else:  # COMPLEX
            params["top_k"] = params.get("top_k", 10) + 5
            params["similarity_threshold"] = params.get("similarity_threshold", 0.5)
            params["compression_rate"] = params.get("compression_rate", 0.6)
            params["rerank_enabled"] = True

        # 规则2: 根据查询类型调整
        if features.query_type.value in ["relational", "analytical"]:
            # 关系和分析查询，启用图谱搜索
            params["graph_search_enabled"] = True

        # 规则3: 根据难度调整
        if features.estimated_difficulty > 0.7:
            # 高难度查询，降低阈值增加召回
            params["similarity_threshold"] -= 0.1
            params["top_k"] += 3

        return params

    def _adjust_by_feedback(
        self,
        features: QueryFeatures,
        feedback_context: Dict,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基于反馈历史调整参数"""

        # 1. 根据平均点击位置调整
        if feedback_context.get("session_history"):
            session_stats = feedback_context["session_history"]
            avg_click_pos = session_stats.get("avg_click_position", 5)

            if avg_click_pos < 3:
                # 用户点击前面，减少结果，增加压缩
                params["top_k"] = max(5, params.get("top_k", 10) - 3)
                params["compression_rate"] = min(0.8, params.get("compression_rate", 0.5) + 0.1)

            elif avg_click_pos > 7:
                # 用户点击后面，增加结果，减少压缩
                params["top_k"] = min(20, params.get("top_k", 10) + 5)
                params["compression_rate"] = max(0.3, params.get("compression_rate", 0.5) - 0.1)

        # 2. 根据历史评分调整
        if feedback_context.get("query_stats"):
            query_stats = feedback_context["query_stats"]
            avg_rating = query_stats.get("avg_rating", 3.5)

            if avg_rating < 3.0:
                # 评分低，降低阈值，增加召回
                params["similarity_threshold"] = max(
                    self.param_bounds["similarity_threshold"][0],
                    params.get("similarity_threshold", 0.6) - 0.1
                )

        # 3. 根据用户偏好调整
        if feedback_context.get("user_preferences"):
            user_prefs = feedback_context["user_preferences"]

            if "preferred_compression" in user_prefs:
                params["compression_rate"] = user_prefs["preferred_compression"]

        return params

    def _apply_bounds(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """应用参数边界约束"""
        for key, (min_val, max_val) in self.param_bounds.items():
            if key in params:
                if isinstance(params[key], (int, float)):
                    params[key] = max(min_val, min(max_val, params[key]))

        return params

    def _record_param_usage(
        self,
        features: QueryFeatures,
        params: Dict[str, Any]
    ):
        """记录参数使用情况"""
        # 创建参数签名
        param_signature = self._create_param_signature(params)

        # 更新统计
        stats = self.param_performance[param_signature]
        stats["usage_count"] += 1

    def _create_param_signature(self, params: Dict[str, Any]) -> str:
        """创建参数签名"""
        key_parts = [
            f"top_k={params.get('top_k', 10)}",
            f"threshold={params.get('similarity_threshold', 0.6):.2f}",
            f"compression={params.get('compression_rate', 0.5):.2f}"
        ]
        return "&".join(key_parts)

    def update_param_performance(
        self,
        query: str,
        params: Dict[str, Any],
        rating: float,
        success: bool
    ):
        """
        更新参数性能统计

        用于学习最优参数组合
        """
        param_signature = self._create_param_signature(params)
        stats = self.param_performance[param_signature]

        stats["total_rating"] += rating
        stats["avg_rating"] = stats["total_rating"] / stats["usage_count"]

        if success:
            stats["success_count"] += 1

        # 如果这个参数组合被使用足够多次，可以更新learned_optimal_params
        if stats["usage_count"] >= 10:
            # TODO: 实现参数更新逻辑
            pass

        logger.debug(
            f"参数性能更新: {param_signature}, "
            f"avg_rating={stats['avg_rating']:.2f}, "
            f"success_rate={stats['success_count']/stats['usage_count']:.2f}"
        )

    def get_parameter_insights(self) -> Dict[str, Any]:
        """获取参数使用洞察"""
        total_usage = sum(
            stats["usage_count"]
            for stats in self.param_performance.values()
        )

        # 找出表现最好的参数组合
        best_params = max(
            self.param_performance.items(),
            key=lambda x: x[1]["avg_rating"]
        ) if self.param_performance else (None, {"avg_rating": 0})

        return {
            "total_param_combinations": len(self.param_performance),
            "total_usage": total_usage,
            "best_param_signature": best_params[0],
            "best_avg_rating": best_params[1]["avg_rating"]
        }

    def get_insights(self) -> Dict[str, Any]:
        """获取洞察（别名）"""
        return self.get_parameter_insights()


# 全局实例
_adaptive_param_optimizer = None


def get_adaptive_parameter_optimizer() -> AdaptiveParameterOptimizer:
    """获取自适应参数优化器实例"""
    global _adaptive_param_optimizer
    if _adaptive_param_optimizer is None:
        _adaptive_param_optimizer = AdaptiveParameterOptimizer()
    return _adaptive_param_optimizer
