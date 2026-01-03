"""
自适应反馈处理器

整合查询分类、参数优化、方法选择的智能反馈系统
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict

from app.core.structured_logging import get_structured_logger
from .query_classifier import get_query_classifier, QueryFeatures
from .adaptive_optimizer import get_adaptive_parameter_optimizer
from .bandit_selector import get_bandit_selector
from ..feedback_loop.realtime_feedback import RealTimeFeedbackProcessor

# 别名
def get_adaptive_optimizer():
    return get_adaptive_parameter_optimizer()

logger = get_structured_logger(__name__)


class AdaptiveFeedbackProcessor:
    """
    自适应反馈处理器

    整合:
    1. 查询分类 (QueryClassifier)
    2. 参数优化 (AdaptiveParameterOptimizer)
    3. 方法选择 (BanditRetrievalSelector)
    4. 反馈学习 (RealTimeFeedbackProcessor)
    """

    def __init__(
        self,
        enable_classification: bool = True,
        enable_optimization: bool = True,
        enable_bandit: bool = True,
        enable_feedback: bool = True
    ):
        self.enable_classification = enable_classification
        self.enable_optimization = enable_optimization
        self.enable_bandit = enable_bandit
        self.enable_feedback = enable_feedback

        # 初始化组件
        if enable_classification:
            self.classifier = get_query_classifier()

        if enable_optimization:
            self.optimizer = get_adaptive_optimizer()

        if enable_bandit:
            self.bandit = get_bandit_selector()

        if enable_feedback:
            self.feedback = RealTimeFeedbackProcessor()

        # 统计信息
        self.query_stats = defaultdict(lambda: {
            "count": 0,
            "avg_reward": 0.0,
            "success_count": 0
        })

        logger.info(
            f"自适应反馈处理器初始化: "
            f"classification={enable_classification}, "
            f"optimization={enable_optimization}, "
            f"bandit={enable_bandit}, "
            f"feedback={enable_feedback}"
        )

    async def process_query_with_adaptive_feedback(
        self,
        query: str,
        user_id: Optional[str],
        session_id: str,
        retrieval_level: str = "enhanced",
        base_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        使用自适应反馈处理查询

        Args:
            query: 用户查询
            user_id: 用户ID
            session_id: 会话ID
            retrieval_level: 检索级别
            base_params: 基础参数

        Returns:
            增强的查询配置，包含:
            - query: 优化后的查询
            - params: 优化后的参数
            - method: 选择的检索方法
            - features: 查询特征
            - selection_info: 选择信息
        """
        try:
            # 0. 初始化基础参数
            if base_params is None:
                base_params = {}

            # 1. 查询分类
            features = None
            if self.enable_classification:
                features = self.classifier.classify(query)
                logger.info(
                    f"查询分类: type={features.query_type.value}, "
                    f"complexity={features.complexity.value}, "
                    f"difficulty={features.estimated_difficulty:.2f}"
                )
            else:
                # 默认特征
                from .query_classifier import QueryFeatures, QueryType, QueryComplexity
                features = QueryFeatures(
                    query_type=QueryType.FACTUAL,
                    complexity=QueryComplexity.MEDIUM,
                    estimated_difficulty=0.5
                )

            # 2. 获取反馈上下文
            feedback_context = None
            if self.enable_feedback:
                feedback_context = await self.feedback._get_feedback_context(
                    query, user_id, session_id
                )

            # 3. 方法选择 (多臂老虎机)
            selected_method = "vector"  # 默认
            selection_info = {}
            if self.enable_bandit:
                selected_method, selection_info = self.bandit.select_method(
                    query, feedback_context
                )
                logger.info(f"选择检索方法: {selected_method}")

            # 4. 参数优化
            optimized_params = base_params.copy()
            if self.enable_optimization:
                optimized_params = self.optimizer.optimize_parameters(
                    query, base_params, feedback_context
                )
                logger.info(f"参数优化: {optimized_params}")

            # 5. 查询重写 (基于反馈)
            optimized_query = query
            if self.enable_feedback and feedback_context:
                optimized_query = self.feedback._rewrite_query(query, feedback_context)
                if optimized_query != query:
                    logger.info(f"查询重写: {query} → {optimized_query}")

            # 6. 组装结果
            result = {
                "query": optimized_query,
                "params": optimized_params,
                "method": selected_method,
                "retrieval_level": retrieval_level,
                "features": {
                    "query_type": features.query_type.value,
                    "complexity": features.complexity.value,
                    "difficulty": features.estimated_difficulty,
                    "entity_count": features.entity_count,
                    "financial_term_count": features.financial_term_count
                },
                "selection_info": selection_info,
                "feedback_context": feedback_context or {}
            }

            return result

        except Exception as e:
            logger.error(f"自适应反馈处理失败: {e}", exc_info=True)
            # 降级到基础配置
            return {
                "query": query,
                "params": base_params or {},
                "method": "vector",
                "retrieval_level": retrieval_level,
                "error": str(e)
            }

    async def collect_result_feedback(
        self,
        query: str,
        user_id: Optional[str],
        session_id: str,
        method: str,
        results: List[Dict],
        user_interactions: Dict[str, Any]
    ):
        """
        收集结果反馈并更新所有组件

        Args:
            query: 查询
            user_id: 用户ID
            session_id: 会话ID
            method: 使用的检索方法
            results: 检索结果
            user_interactions: 用户交互数据
        """
        try:
            # 1. 解析交互并计算奖励
            reward, success = self._calculate_reward(user_interactions, results)

            logger.info(
                f"收集反馈: query={query[:30]}, "
                f"method={method}, "
                f"reward={reward:.2f}, "
                f"success={success}"
            )

            # 2. 更新多臂老虎机
            if self.enable_bandit:
                self.bandit.update_performance(
                    query=query,
                    method=method,
                    reward=reward,
                    success=success
                )

            # 3. 更新反馈处理器
            if self.enable_feedback:
                await self.feedback.collect_result_feedback(
                    query=query,
                    user_id=user_id,
                    session_id=session_id,
                    results=results,
                    user_interactions=user_interactions
                )

            # 4. 更新统计信息
            self._update_query_stats(query, reward, success)

        except Exception as e:
            logger.error(f"反馈收集失败: {e}", exc_info=True)

    def _calculate_reward(
        self,
        interactions: Dict[str, Any],
        results: List[Dict]
    ) -> Tuple[float, bool]:
        """
        计算奖励值

        Args:
            interactions: 用户交互
            results: 检索结果

        Returns:
            (奖励值, 是否成功)
        """
        reward = 0.0
        success = False

        # 1. 点击奖励 (0-0.4)
        clicks = interactions.get("clicks", [])
        if clicks:
            # 点击位置越靠前，奖励越高
            positions = [c.get("position", 10) for c in clicks]
            avg_position = sum(positions) / len(positions)
            click_reward = 0.4 * (1 - (avg_position - 1) / 10)
            reward += max(0, click_reward)

            # 有点击即视为部分成功
            if len(clicks) > 0:
                success = True

        # 2. 停留时间奖励 (0-0.3)
        dwell_times = interactions.get("dwell_times", {})
        if dwell_times:
            avg_dwell = sum(dwell_times.values()) / len(dwell_times)
            # 30秒以上为高分
            dwell_reward = min(0.3, avg_dwell / 100.0)
            reward += dwell_reward

        # 3. 评分奖励 (0-0.3)
        rating = interactions.get("rating")
        if rating is not None:
            # 5分制转换为0-0.3
            rating_reward = 0.3 * (rating / 5.0)
            reward += rating_reward

            # 评分>=4视为成功
            if rating >= 4:
                success = True

        return min(1.0, reward), success

    def _update_query_stats(
        self,
        query: str,
        reward: float,
        success: bool
    ):
        """更新查询统计"""
        stats = self.query_stats[query]
        stats["count"] += 1

        # 更新平均奖励
        old_avg = stats["avg_reward"]
        count = stats["count"]
        stats["avg_reward"] = (old_avg * (count - 1) + reward) / count

        if success:
            stats["success_count"] += 1

    def get_adaptive_insights(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取自适应系统洞察

        Args:
            user_id: 用户ID (可选)
            session_id: 会话ID (可选)

        Returns:
            洞察数据
        """
        insights = {
            "components_enabled": {
                "classification": self.enable_classification,
                "optimization": self.enable_optimization,
                "bandit": self.enable_bandit,
                "feedback": self.enable_feedback
            },
            "query_stats": dict(self.query_stats)
        }

        # 添加分类器洞察
        if self.enable_classification:
            insights["classifier"] = {
                "type": "QueryClassifier",
                "initialized": True
            }

        # 添加优化器洞察
        if self.enable_optimization:
            insights["optimizer"] = self.optimizer.get_insights()

        # 添加多臂老虎机洞察
        if self.enable_bandit:
            insights["bandit"] = self.bandit.get_insights()

        # 添加反馈洞察
        if self.enable_feedback:
            insights["feedback"] = self.feedback.get_insights(
                user_id, session_id
            )

        return insights

    def reset_stats(self):
        """重置统计信息"""
        self.query_stats.clear()

        if self.enable_bandit:
            self.bandit.reset_stats()

        logger.info("自适应反馈统计已重置")


# 全局实例
_adaptive_feedback_processor: Optional[AdaptiveFeedbackProcessor] = None


def get_adaptive_feedback_processor(
    enable_classification: bool = True,
    enable_optimization: bool = True,
    enable_bandit: bool = True,
    enable_feedback: bool = True
) -> AdaptiveFeedbackProcessor:
    """
    获取自适应反馈处理器实例

    Args:
        enable_classification: 启用查询分类
        enable_optimization: 启用参数优化
        enable_bandit: 启用多臂老虎机
        enable_feedback: 启用反馈学习

    Returns:
        自适应反馈处理器实例
    """
    global _adaptive_feedback_processor
    if _adaptive_feedback_processor is None:
        _adaptive_feedback_processor = AdaptiveFeedbackProcessor(
            enable_classification=enable_classification,
            enable_optimization=enable_optimization,
            enable_bandit=enable_bandit,
            enable_feedback=enable_feedback
        )
    return _adaptive_feedback_processor
