"""
多臂老虎机检索策略选择器

自动选择最优检索方法（向量/图谱/混合）
"""

import random
import math
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict

from app.core.structured_logging import get_structured_logger
from .query_classifier import get_query_classifier, QueryFeatures, QueryType, QueryComplexity

logger = get_structured_logger(__name__)


class RetrievalArm:
    """检索臂（一个检索方法）"""

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str
    ):
        self.name = name
        self.display_name = display_name
        self.description = description

        # 统计信息
        self.pulls = 0  # 拉取次数
        self.rewards = []  # 奖励列表
        self.total_reward = 0.0
        self.avg_reward = 0.0
        self.success_count = 0
        self.last_pull_time = None

    def pull(self) -> float:
        """拉取一次（使用这个方法）"""
        self.pulls += 1
        self.last_pull_time = datetime.now()
        return self.avg_reward

    def update(self, reward: float, success: bool):
        """
        更新奖励

        Args:
            reward: 奖励值 (0-1之间)
            success: 是否成功
        """
        self.rewards.append(reward)
        self.total_reward += reward
        # 防止除以零
        if self.pulls > 0:
            self.avg_reward = self.total_reward / self.pulls
        else:
            self.avg_reward = reward

        if success:
            self.success_count += 1

    def get_stats(self) -> Dict:
        """获取统计信息"""
        success_rate = self.success_count / self.pulls if self.pulls > 0 else 0

        return {
            "name": self.name,
            "display_name": self.display_name,
            "pulls": self.pulls,
            "avg_reward": self.avg_reward,
            "success_rate": success_rate,
            "confidence": math.sqrt(self.pulls) / (self.pulls + 1)
        }


class BanditRetrievalSelector:
    """
    多臂老虎机检索选择器

    使用UCB (Upper Confidence Bound) 算法自动选择最优检索方法
    """

    def __init__(
        self,
        exploration_rate: float = 0.1,  # 探索率
        confidence_level: float = 2.0,    # UCB置信度
        min_pulls_for_ucb: int = 3        # UCB最小拉取次数
    ):
        self.exploration_rate = exploration_rate
        self.confidence_level = confidence_level
        self.min_pulls_for_ucb = min_pulls_for_ucb

        # 初始化检索臂
        self.arms = {
            "vector": RetrievalArm(
                name="vector",
                display_name="向量检索",
                description="基于语义相似度的检索，适合概念性查询"
            ),
            "graph": RetrievalArm(
                name="graph",
                display_name="图谱检索",
                description="基于知识图谱的检索，适合关系查询"
            ),
            "hybrid": RetrievalArm(
                name="hybrid",
                display_name="混合检索",
                description="结合多种方法的检索，适合复杂查询"
            )
        }

        # 查询分类器
        self.classifier = get_query_classifier()

        # 不同查询类型的历史最佳方法
        self.query_type_best_methods = defaultdict(lambda: {
            "vector": 0,
            "graph": 0,
            "hybrid": 0
        })

        logger.info(
            f"多臂老虎机选择器初始化: "
            f"arms={list(self.arms.keys())}, "
            f"exploration={exploration_rate}"
        )

    def select_method(
        self,
        query: str,
        feedback_context: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        """
        选择检索方法

        Args:
            query: 用户查询
            feedback_context: 反馈上下文

        Returns:
            (方法名称, 选择信息)
        """
        try:
            # 1. 分析查询特征
            features = self.classifier.classify(query)

            # 2. 基于查询类型预筛选
            candidate_methods = self._filter_methods_by_features(features)

            # 3. 选择策略：UCB或Epsilon-Greedy
            selected_arm = self._select_arm_ucb(candidate_methods)

            # 拉取一次（记录使用）
            selected_arm.pull()

            # 4. 考虑用户历史偏好
            if feedback_context and feedback_context.get("user_preferences"):
                # 如果用户有明确偏好，优先考虑
                user_pref_method = self._consider_user_preferences(
                    features, feedback_context
                )
                if user_pref_method:
                    selected_arm = self.arms[user_pref_method]

            selection_info = {
                "selected_method": selected_arm.name,
                "query_type": features.query_type.value,
                "complexity": features.complexity.value,
                "difficulty": features.estimated_difficulty,
                "algorithm": "UCB",
                "candidate_methods": candidate_methods,
                "all_arm_stats": {
                    name: arm.get_stats()
                    for name, arm in self.arms.items()
                }
            }

            logger.info(
                f"选择检索方法: {selected_arm.name}, "
                f"query_type={features.query_type.value}, "
                f"complexity={features.complexity.value}"
            )

            return selected_arm.name, selection_info

        except Exception as e:
            logger.error(f"方法选择失败: {e}")
            # 降级到默认方法
            return "vector", {"error": str(e)}

    def _filter_methods_by_features(
        self,
        features: QueryFeatures
    ) -> List[str]:
        """基于查询特征筛选候选方法"""

        candidates = list(self.arms.keys())

        # 规则1: 关系查询优先使用图谱
        if features.query_type.value == "relational":
            if "graph" in candidates:
                candidates = ["graph", "hybrid", "vector"]

        # 规则2: 简单查询优先使用向量
        elif features.complexity == QueryComplexity.SIMPLE:
            candidates = ["vector", "hybrid", "graph"]

        # 规则3: 复杂查询优先使用混合
        elif features.complexity == QueryComplexity.COMPLEX:
            candidates = ["hybrid", "graph", "vector"]

        # 规则4: 高难度查询使用混合
        if features.estimated_difficulty > 0.7:
            if "hybrid" in candidates:
                candidates = ["hybrid", "graph", "vector"]

        return candidates

    def _select_arm_ucb(self, candidate_methods: List[str]) -> RetrievalArm:
        """
        使用UCB算法选择arm

        UCB = avg_reward + confidence_bonus

        Args:
            candidate_methods: 候选方法列表

        Returns:
            选择的arm
        """
        best_arm = None
        best_ucb = -float('inf')

        for arm_name in candidate_methods:
            arm = self.arms[arm_name]

            # 如果这个arm还没被拉取过，先尝试
            if arm.pulls < self.min_pulls_for_ucb:
                return arm

            # 计算UCB值
            avg_reward = arm.avg_reward
            pulls = arm.pulls
            total_pulls = sum(a.pulls for a in self.arms.values())

            # UCB公式
            exploration_bonus = self.confidence_level * math.sqrt(
                math.log(total_pulls) / pulls
            )
            ucb = avg_reward + exploration_bonus

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm

        return best_arm

    def _consider_user_preferences(
        self,
        features: QueryFeatures,
        feedback_context: Dict
    ) -> Optional[str]:
        """考虑用户历史偏好"""

        user_prefs = feedback_context["user_preferences"]
        query_type = features.query_type.value

        # 找出该用户在该查询类型下的最优方法
        if query_type in self.query_type_best_methods:
            stats = self.query_type_best_methods[query_type]

            # 找成功率最高的方法
            best_method = max(stats.items(), key=lambda x: x[1])
            if best_method[1] > 3:  # 至少使用过3次
                return best_method[0]

        return None

    def update_performance(
        self,
        query: str,
        method: str,
        reward: float,
        success: bool
    ):
        """
        更新检索方法性能

        Args:
            query: 查询
            method: 使用的方法
            reward: 奖励 (0-1)
            success: 是否成功
        """
        # 更新arm统计
        if method in self.arms:
            self.arms[method].update(reward, success)

        # 更新查询类型统计
        features = self.classifier.classify(query)
        query_type = features.query_type.value

        if success:
            self.query_type_best_methods[query_type][method] += 1

        logger.debug(
            f"性能更新: method={method}, "
            f"query_type={query_type}, "
            f"reward={reward:.2f}, "
            f"success={success}"
        )

    def get_insights(self) -> Dict[str, Any]:
        """获取选择器洞察"""
        total_pulls = sum(arm.pulls for arm in self.arms.values())

        # 找出每个查询类型的最优方法
        best_methods_by_type = {}
        for query_type, stats in self.query_type_best_methods.items():
            if stats:
                best = max(stats.items(), key=lambda x: x[1])
                best_methods_by_type[query_type] = {
                    "method": best[0],
                    "usage_count": best[1]
                }

        return {
            "total_selections": total_pulls,
            "arm_stats": {
                name: arm.get_stats()
                for name, arm in self.arms.items()
            },
            "best_methods_by_query_type": best_methods_by_type,
            "exploration_rate": self.exploration_rate
        }

    def reset_stats(self):
        """重置统计（用于测试或重新学习）"""
        for arm in self.arms.values():
            arm.pulls = 0
            arm.rewards = []
            arm.total_reward = 0.0
            arm.avg_reward = 0.0
            arm.success_count = 0

        self.query_type_best_methods.clear()

        logger.info("多臂老虎机统计已重置")


# 全局实例
_bandit_selector = None


def get_bandit_selector(
    exploration_rate: float = 0.1,
    confidence_level: float = 2.0
) -> BanditRetrievalSelector:
    """获取多臂老虎机选择器实例"""
    global _bandit_selector
    if _bandit_selector is None:
        _bandit_selector = BanditRetrievalSelector(
            exploration_rate=exploration_rate,
            confidence_level=confidence_level
        )
    return _bandit_selector
