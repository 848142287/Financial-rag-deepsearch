"""
降级/升级条件处理机制
根据系统性能和用户反馈动态调整RAG策略执行
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time

from app.services.rag_unified.strategies import RAGStrategy, QueryFeatures
from app.services.rag_unified.config import config_manager, RAGRoutingConfig
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


class AdjustmentType(Enum):
    """调整类型"""
    DOWNGRADE = "downgrade"     # 降级：使用更简单/快速的策略
    UPGRADE = "upgrade"         # 升级：使用更复杂/强大的策略
    MAINTAIN = "maintain"       # 维持：保持当前策略


@dataclass
class ConditionMetrics:
    """条件指标"""
    query_id: str
    strategy: RAGStrategy
    execution_time: float
    success: bool
    user_feedback: Optional[float] = None
    error_count: int = 0
    result_count: int = 0
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AdjustmentDecision:
    """调整决策"""
    adjustment_type: AdjustmentType
    original_strategy: RAGStrategy
    recommended_strategy: RAGStrategy
    reason: str
    confidence: float
    metrics: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ConditionEvaluator:
    """条件评估器"""

    def __init__(self):
        self.config = config_manager.get_config()
        self.metrics_history: List[ConditionMetrics] = []
        self.adjustment_history: List[AdjustmentDecision] = []

    def record_execution_metrics(self, metrics: ConditionMetrics) -> None:
        """记录执行指标"""
        self.metrics_history.append(metrics)

        # 保持历史记录在合理范围内
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-800:]

        # 异步保存到Redis
        asyncio.create_task(self._save_metrics_to_redis(metrics))

    async def evaluate_downgrade_conditions(
        self,
        current_strategy: RAGStrategy,
        query_features: QueryFeatures,
        recent_metrics: List[ConditionMetrics] = None
    ) -> Optional[AdjustmentDecision]:
        """评估降级条件"""

        conditions = self.config.conditions.downgrade_conditions
        reasons = []
        total_score = 0.0

        # 1. 响应时间超限检查
        if recent_metrics:
            avg_response_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
            if avg_response_time > conditions["response_time_exceeds"] / 1000.0:  # 转换为秒
                reasons.append(f"平均响应时间 {avg_response_time:.2f}s 超过阈值 {conditions['response_time_exceeds']/1000.0}s")
                total_score += 0.3

        # 2. 用户反馈分数低检查
        if recent_metrics:
            feedback_scores = [m.user_feedback for m in recent_metrics if m.user_feedback is not None]
            if feedback_scores:
                avg_feedback = sum(feedback_scores) / len(feedback_scores)
                if avg_feedback < conditions["user_feedback_score"]:
                    reasons.append(f"用户反馈分数 {avg_feedback:.1f} 低于阈值 {conditions['user_feedback_score']}")
                    total_score += 0.4

        # 3. 错误率高检查
        if recent_metrics:
            error_rate = sum(m.error_count for m in recent_metrics) / len(recent_metrics)
            if error_rate > conditions["error_rate_threshold"]:
                reasons.append(f"错误率 {error_rate:.2%} 超过阈值 {conditions['error_rate_threshold']:.2%}")
                total_score += 0.5

        # 4. 置信度下降检查
        if recent_metrics:
            avg_confidence = sum(m.confidence for m in recent_metrics) / len(recent_metrics)
            if avg_confidence < (1.0 - conditions["confidence_drop_threshold"]):
                reasons.append(f"平均置信度 {avg_confidence:.2f} 显著下降")
                total_score += 0.3

        # 5. 结果数量不足检查
        if recent_metrics:
            avg_result_count = sum(m.result_count for m in recent_metrics) / len(recent_metrics)
            if avg_result_count < conditions["result_count_insufficient"]:
                reasons.append(f"平均结果数量 {avg_result_count} 不足")
                total_score += 0.2

        # 如果满足降级条件
        if total_score >= 0.3:  # 降低阈值
            recommended_strategy = self._get_downgrade_strategy(current_strategy, query_features)

            return AdjustmentDecision(
                adjustment_type=AdjustmentType.DOWNGRADE,
                original_strategy=current_strategy,
                recommended_strategy=recommended_strategy,
                reason="; ".join(reasons),
                confidence=min(1.0, total_score),
                metrics={
                    "trigger_reasons": reasons,
                    "total_score": total_score,
                    "avg_response_time": avg_response_time if recent_metrics else 0,
                    "avg_feedback": avg_feedback if recent_metrics and feedback_scores else 0,
                    "error_rate": error_rate if recent_metrics else 0
                }
            )

        return None

    async def evaluate_upgrade_conditions(
        self,
        current_strategy: RAGStrategy,
        query_features: QueryFeatures,
        conversation_context: Dict[str, Any] = None
    ) -> Optional[AdjustmentDecision]:
        """评估升级条件"""

        conditions = self.config.conditions.upgrade_conditions
        reasons = []
        total_score = 0.0

        # 1. 连续追问次数检查
        if conversation_context:
            follow_up_count = conversation_context.get("follow_up_queries", 0)
            if follow_up_count >= conditions["follow_up_queries"]:
                reasons.append(f"连续追问次数 {follow_up_count} 达到阈值 {conditions['follow_up_queries']}")
                total_score += 0.4

            # 2. 查询修改检查
            if conversation_context.get("query_modification", False):
                reasons.append("用户修改了原查询")
                total_score += 0.3

        # 3. 复杂度增加检查
        if query_features.complexity.value > 0.7:  # 高复杂度
            reasons.append(f"查询复杂度 {query_features.complexity.value:.2f} 较高")
            total_score += 0.3

        # 4. 用户满意度低检查
        recent_metrics = self._get_recent_metrics(current_strategy, hours=1)
        if recent_metrics:
            satisfaction_scores = [m.user_feedback for m in recent_metrics if m.user_feedback is not None]
            if satisfaction_scores:
                avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)
                if avg_satisfaction < conditions["user_satisfaction_low"]:
                    reasons.append(f"用户满意度 {avg_satisfaction:.1f} 偏低")
                    total_score += 0.3

        # 5. 结果不足检查（基于历史数据）
        if recent_metrics:
            insufficient_results = sum(
                1 for m in recent_metrics
                if m.result_count < self.config.thresholds.min_results
            )
            if insufficient_results > len(recent_metrics) * 0.5:
                reasons.append("历史结果经常不足")
                total_score += 0.2

        # 如果满足升级条件
        if total_score >= 0.3:  # 降低阈值
            recommended_strategy = self._get_upgrade_strategy(current_strategy, query_features)

            return AdjustmentDecision(
                adjustment_type=AdjustmentType.UPGRADE,
                original_strategy=current_strategy,
                recommended_strategy=recommended_strategy,
                reason="; ".join(reasons),
                confidence=min(1.0, total_score),
                metrics={
                    "trigger_reasons": reasons,
                    "total_score": total_score,
                    "follow_up_count": conversation_context.get("follow_up_queries", 0) if conversation_context else 0,
                    "query_complexity": query_features.complexity.value
                }
            )

        return None

    def _get_downgrade_strategy(
        self,
        current_strategy: RAGStrategy,
        query_features: QueryFeatures
    ) -> RAGStrategy:
        """获取降级后的策略"""

        # 策略降级优先级
        downgrade_priority = {
            RAGStrategy.AGENTIC_RAG: [RAGStrategy.GRAPH_RAG, RAGStrategy.HYBRID_RAG, RAGStrategy.LIGHT_RAG],
            RAGStrategy.GRAPH_RAG: [RAGStrategy.HYBRID_RAG, RAGStrategy.LIGHT_RAG],
            RAGStrategy.HYBRID_RAG: [RAGStrategy.LIGHT_RAG]
        }

        candidates = downgrade_priority.get(current_strategy, [RAGStrategy.LIGHT_RAG])

        # 根据查询特征选择最合适的降级策略
        for candidate in candidates:
            if self._is_strategy_suitable(candidate, query_features):
                return candidate

        # 默认返回最简单的策略
        return RAGStrategy.LIGHT_RAG

    def _get_upgrade_strategy(
        self,
        current_strategy: RAGStrategy,
        query_features: QueryFeatures
    ) -> RAGStrategy:
        """获取升级后的策略"""

        # 策略升级优先级
        upgrade_priority = {
            RAGStrategy.LIGHT_RAG: [RAGStrategy.HYBRID_RAG, RAGStrategy.GRAPH_RAG, RAGStrategy.AGENTIC_RAG],
            RAGStrategy.GRAPH_RAG: [RAGStrategy.AGENTIC_RAG, RAGStrategy.HYBRID_RAG],
            RAGStrategy.HYBRID_RAG: [RAGStrategy.AGENTIC_RAG]
        }

        candidates = upgrade_priority.get(current_strategy, [RAGStrategy.AGENTIC_RAG])

        # 根据查询特征选择最合适的升级策略
        for candidate in candidates:
            if self._is_strategy_suitable(candidate, query_features):
                return candidate

        # 默认返回最强大的策略
        return RAGStrategy.AGENTIC_RAG

    def _is_strategy_suitable(self, strategy: RAGStrategy, query_features: QueryFeatures) -> bool:
        """判断策略是否适合当前查询特征"""

        if strategy == RAGStrategy.LIGHT_RAG:
            # 适合简单、快速的查询
            return query_features.complexity.value < 0.5 and query_features.entity_count <= 2

        elif strategy == RAGStrategy.GRAPH_RAG:
            # 适合关系复杂的查询
            return query_features.entity_count >= 2 or query_features.relation_complexity.value >= 2

        elif strategy == RAGStrategy.AGENTIC_RAG:
            # 适合复杂、分析型查询
            return query_features.complexity.value >= 0.7 or query_features.intent.value in ["analytical", "predictive"]

        elif strategy == RAGStrategy.HYBRID_RAG:
            # 适合大多数查询
            return True

        return False

    def _get_recent_metrics(
        self,
        strategy: RAGStrategy,
        hours: int = 1
    ) -> List[ConditionMetrics]:
        """获取最近的指标"""

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            m for m in self.metrics_history
            if m.strategy == strategy and m.timestamp > cutoff_time
        ]

    async def _save_metrics_to_redis(self, metrics: ConditionMetrics) -> None:
        """保存指标到Redis"""
        try:
            key = f"rag_routing:metrics:{metrics.strategy.value}:{datetime.utcnow().strftime('%Y%m%d')}"
            data = {
                "query_id": metrics.query_id,
                "execution_time": metrics.execution_time,
                "success": metrics.success,
                "user_feedback": metrics.user_feedback,
                "error_count": metrics.error_count,
                "result_count": metrics.result_count,
                "confidence": metrics.confidence,
                "timestamp": metrics.timestamp.isoformat()
            }
            await redis_client.lpush(key, json.dumps(data, ensure_ascii=False))
            await redis_client.expire(key, 86400 * 7)  # 保存7天
        except Exception as e:
            logger.error(f"Failed to save metrics to Redis: {e}")

    async def load_metrics_from_redis(self) -> None:
        """从Redis加载指标"""
        try:
            for strategy in RAGStrategy:
                key_pattern = f"rag_routing:metrics:{strategy.value}:*"
                keys = await redis_client.keys(key_pattern)

                for key in keys:
                    data_list = await redis_client.lrange(key, 0, -1)
                    for data in data_list:
                        try:
                            metrics_dict = json.loads(data)
                            metrics = ConditionMetrics(
                                query_id=metrics_dict["query_id"],
                                strategy=RAGStrategy(strategy.value),
                                execution_time=metrics_dict["execution_time"],
                                success=metrics_dict["success"],
                                user_feedback=metrics_dict.get("user_feedback"),
                                error_count=metrics_dict.get("error_count", 0),
                                result_count=metrics_dict.get("result_count", 0),
                                confidence=metrics_dict.get("confidence", 0.0),
                                timestamp=datetime.fromisoformat(metrics_dict["timestamp"])
                            )
                            self.metrics_history.append(metrics)
                        except Exception as e:
                            logger.error(f"Failed to parse metrics data: {e}")

        except Exception as e:
            logger.error(f"Failed to load metrics from Redis: {e}")


class StrategyAdjustmentEngine:
    """策略调整引擎"""

    def __init__(self):
        self.evaluator = ConditionEvaluator()
        self.config = config_manager.get_config()
        self.adjustment_cache: Dict[str, AdjustmentDecision] = {}
        self.cache_ttl = 300  # 5分钟缓存

    async def evaluate_adjustment(
        self,
        query_id: str,
        current_strategy: RAGStrategy,
        query_features: QueryFeatures,
        conversation_context: Dict[str, Any] = None
    ) -> Optional[AdjustmentDecision]:
        """评估是否需要调整策略"""

        # 检查缓存
        cached_decision = self._get_cached_decision(query_id)
        if cached_decision and self._is_cache_valid(cached_decision):
            return cached_decision

        # 获取最近的指标
        recent_metrics = self.evaluator._get_recent_metrics(current_strategy, hours=1)

        # 评估降级条件
        downgrade_decision = await self.evaluator.evaluate_downgrade_conditions(
            current_strategy, query_features, recent_metrics
        )

        if downgrade_decision:
            self._cache_decision(query_id, downgrade_decision)
            self.evaluator.adjustment_history.append(downgrade_decision)
            return downgrade_decision

        # 评估升级条件
        upgrade_decision = await self.evaluator.evaluate_upgrade_conditions(
            current_strategy, query_features, conversation_context
        )

        if upgrade_decision:
            self._cache_decision(query_id, upgrade_decision)
            self.evaluator.adjustment_history.append(upgrade_decision)
            return upgrade_decision

        return None

    async def apply_adjustment(
        self,
        decision: AdjustmentDecision,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """应用调整决策"""

        try:
            # 记录调整应用
            adjustment_record = {
                "timestamp": decision.timestamp.isoformat(),
                "adjustment_type": decision.adjustment_type.value,
                "original_strategy": decision.original_strategy.value,
                "recommended_strategy": decision.recommended_strategy.value,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "execution_context": execution_context
            }

            # 保存到Redis
            await self._save_adjustment_to_redis(adjustment_record)

            logger.info(
                f"Applied {decision.adjustment_type.value} adjustment: "
                f"{decision.original_strategy.value} -> {decision.recommended_strategy.value} "
                f"(confidence: {decision.confidence:.2f})"
            )

            return {
                "success": True,
                "adjustment_applied": True,
                "new_strategy": decision.recommended_strategy.value,
                "reason": decision.reason,
                "confidence": decision.confidence
            }

        except Exception as e:
            logger.error(f"Failed to apply adjustment: {e}")
            return {
                "success": False,
                "adjustment_applied": False,
                "error": str(e)
            }

    def record_execution_result(
        self,
        query_id: str,
        strategy: RAGStrategy,
        execution_time: float,
        success: bool,
        result_count: int,
        confidence: float,
        user_feedback: Optional[float] = None,
        error_count: int = 0
    ) -> None:
        """记录执行结果"""

        metrics = ConditionMetrics(
            query_id=query_id,
            strategy=strategy,
            execution_time=execution_time,
            success=success,
            user_feedback=user_feedback,
            error_count=error_count,
            result_count=result_count,
            confidence=confidence
        )

        self.evaluator.record_execution_metrics(metrics)

    def _get_cached_decision(self, query_id: str) -> Optional[AdjustmentDecision]:
        """获取缓存的决策"""
        return self.adjustment_cache.get(query_id)

    def _cache_decision(self, query_id: str, decision: AdjustmentDecision) -> None:
        """缓存决策"""
        self.adjustment_cache[query_id] = decision

        # 清理过期缓存
        self._cleanup_expired_cache()

    def _is_cache_valid(self, decision: AdjustmentDecision) -> bool:
        """检查缓存是否有效"""
        age = (datetime.utcnow() - decision.timestamp).total_seconds()
        return age < self.cache_ttl

    def _cleanup_expired_cache(self) -> None:
        """清理过期缓存"""
        expired_keys = [
            query_id for query_id, decision in self.adjustment_cache.items()
            if not self._is_cache_valid(decision)
        ]
        for key in expired_keys:
            del self.adjustment_cache[key]

    async def _save_adjustment_to_redis(self, adjustment_record: Dict[str, Any]) -> None:
        """保存调整记录到Redis"""
        try:
            key = f"rag_routing:adjustments:{datetime.utcnow().strftime('%Y%m%d')}"
            await redis_client.lpush(key, json.dumps(adjustment_record, ensure_ascii=False))
            await redis_client.expire(key, 86400 * 30)  # 保存30天
        except Exception as e:
            logger.error(f"Failed to save adjustment to Redis: {e}")

    async def get_adjustment_statistics(self) -> Dict[str, Any]:
        """获取调整统计信息"""

        if not self.evaluator.adjustment_history:
            return {"total_adjustments": 0}

        total_adjustments = len(self.evaluator.adjustment_history)

        # 按类型统计
        downgrade_count = len([
            a for a in self.evaluator.adjustment_history
            if a.adjustment_type == AdjustmentType.DOWNGRADE
        ])
        upgrade_count = len([
            a for a in self.evaluator.adjustment_history
            if a.adjustment_type == AdjustmentType.UPGRADE
        ])

        # 按策略统计
        strategy_adjustments = {}
        for decision in self.evaluator.adjustment_history:
            original = decision.original_strategy.value
            if original not in strategy_adjustments:
                strategy_adjustments[original] = {"downgrade": 0, "upgrade": 0}

            if decision.adjustment_type == AdjustmentType.DOWNGRADE:
                strategy_adjustments[original]["downgrade"] += 1
            elif decision.adjustment_type == AdjustmentType.UPGRADE:
                strategy_adjustments[original]["upgrade"] += 1

        # 最近24小时的调整
        recent_adjustments = [
            a for a in self.evaluator.adjustment_history
            if a.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]

        return {
            "total_adjustments": total_adjustments,
            "downgrade_count": downgrade_count,
            "upgrade_count": upgrade_count,
            "recent_adjustments_24h": len(recent_adjustments),
            "strategy_adjustments": strategy_adjustments,
            "cache_size": len(self.adjustment_cache)
        }

    async def load_adjustments_from_redis(self) -> None:
        """从Redis加载调整历史"""
        try:
            key_pattern = "rag_routing:adjustments:*"
            keys = await redis_client.keys(key_pattern)

            for key in keys:
                data_list = await redis_client.lrange(key, 0, -1)
                for data in data_list:
                    try:
                        adjustment_dict = json.loads(data)
                        decision = AdjustmentDecision(
                            adjustment_type=AdjustmentType(adjustment_dict["adjustment_type"]),
                            original_strategy=RAGStrategy(adjustment_dict["original_strategy"]),
                            recommended_strategy=RAGStrategy(adjustment_dict["recommended_strategy"]),
                            reason=adjustment_dict["reason"],
                            confidence=adjustment_dict["confidence"],
                            metrics=adjustment_dict.get("metrics", {}),
                            timestamp=datetime.fromisoformat(adjustment_dict["timestamp"])
                        )
                        self.evaluator.adjustment_history.append(decision)
                    except Exception as e:
                        logger.error(f"Failed to parse adjustment data: {e}")

        except Exception as e:
            logger.error(f"Failed to load adjustments from Redis: {e}")


# 全局策略调整引擎实例
strategy_adjustment_engine = StrategyAdjustmentEngine()