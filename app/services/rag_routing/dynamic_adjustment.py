"""
动态路由调整机制
基于用户反馈和性能监控动态优化RAG策略路由
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np

from app.services.rag_routing.strategies import RAGStrategy
from app.services.rag_routing.router import routing_engine, RoutingDecision
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """反馈类型"""
    SATISFACTION = "satisfaction"  # 满意度评分
    RELEVANCE = "relevance"  # 相关性反馈
    COMPLETENESS = "completeness"  # 完整性反馈
    RESPONSE_TIME = "response_time"  # 响应时间反馈
    ERROR_REPORT = "error_report"  # 错误报告


class AdjustmentTrigger(Enum):
    """调整触发条件"""
    LOW_SUCCESS_RATE = "low_success_rate"  # 成功率低
    HIGH_RESPONSE_TIME = "high_response_time"  # 响应时间长
    POOR_SATISFACTION = "poor_satisfaction"  # 用户满意度差
    STRATEGY_IMBALANCE = "strategy_imbalance"  # 策略使用不均衡
    PERFORMANCE_DEGRADATION = "performance_degradation"  # 性能下降


@dataclass
class FeedbackRecord:
    """反馈记录"""
    query_id: str
    strategy: RAGStrategy
    feedback_type: FeedbackType
    value: float  # 反馈值
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdjustmentRule:
    """调整规则"""
    trigger: AdjustmentTrigger
    condition: Dict[str, Any]  # 触发条件
    adjustment: Dict[str, Any]  # 调整方案
    priority: int = 1  # 优先级，数值越高优先级越高
    enabled: bool = True


@dataclass
class StrategyAdjustment:
    """策略调整方案"""
    strategy: RAGStrategy
    adjustment_type: str  # 权重调整、阈值调整等
    adjustment_value: Any
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DynamicRouterAdjuster:
    """动态路由调整器"""

    def __init__(self):
        self.feedback_history: List[FeedbackRecord] = []
        self.adjustment_rules: List[AdjustmentRule] = []
        self.strategy_weights: Dict[RAGStrategy, float] = {
            RAGStrategy.LIGHT_RAG: 0.3,
            RAGStrategy.GRAPH_RAG: 0.3,
            RAGStrategy.AGENTIC_RAG: 0.3,
            RAGStrategy.HYBRID_RAG: 0.1
        }
        self.strategy_thresholds: Dict[str, float] = {}
        self.adjustment_history: List[StrategyAdjustment] = []

        # 初始化默认调整规则
        self._initialize_default_rules()

        # 启动后台监控任务
        self.monitoring_task = None
        self.start_monitoring()

    def _initialize_default_rules(self) -> None:
        """初始化默认调整规则"""

        # 成功率过低规则
        self.adjustment_rules.append(AdjustmentRule(
            trigger=AdjustmentTrigger.LOW_SUCCESS_RATE,
            condition={
                "success_rate_threshold": 0.7,
                "min_samples": 10,
                "time_window_hours": 24
            },
            adjustment={
                "type": "weight_decrease",
                "factor": 0.8,
                "redistribution": True
            },
            priority=3
        ))

        # 响应时间过长规则
        self.adjustment_rules.append(AdjustmentRule(
            trigger=AdjustmentTrigger.HIGH_RESPONSE_TIME,
            condition={
                "response_time_threshold": 10.0,  # 秒
                "p95_threshold": 15.0,
                "time_window_hours": 6
            },
            adjustment={
                "type": "weight_decrease",
                "factor": 0.7,
                "redistribution": True
            },
            priority=2
        ))

        # 用户满意度差规则
        self.adjustment_rules.append(AdjustmentRule(
            trigger=AdjustmentTrigger.POOR_SATISFACTION,
            condition={
                "satisfaction_threshold": 3.0,  # 5分制
                "min_samples": 5,
                "time_window_hours": 12
            },
            adjustment={
                "type": "weight_decrease",
                "factor": 0.6,
                "redistribution": True
            },
            priority=3
        ))

        # 策略使用不均衡规则
        self.adjustment_rules.append(AdjustmentRule(
            trigger=AdjustmentTrigger.STRATEGY_IMBALANCE,
            condition={
                "usage_threshold": 0.6,  # 单个策略使用率超过60%
                "time_window_hours": 24
            },
            adjustment={
                "type": "weight_rebalance",
                "target_distribution": "balanced"  # 均衡分布
            },
            priority=1
        ))

    async def record_feedback(
        self,
        query_id: str,
        strategy: RAGStrategy,
        feedback_type: FeedbackType,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录用户反馈"""

        feedback = FeedbackRecord(
            query_id=query_id,
            strategy=strategy,
            feedback_type=feedback_type,
            value=value,
            timestamp=datetime.utcnow(),
            context=context or {}
        )

        self.feedback_history.append(feedback)

        # 保持历史记录在合理范围内
        if len(self.feedback_history) > 10000:
            self.feedback_history = self.feedback_history[-8000:]

        # 异步保存到Redis
        await self._save_feedback_to_redis(feedback)

        # 触发即时调整检查
        if feedback_type in [FeedbackType.SATISFACTION, FeedbackType.ERROR_REPORT]:
            asyncio.create_task(self._check_immediate_adjustment(feedback))

    async def _save_feedback_to_redis(self, feedback: FeedbackRecord) -> None:
        """保存反馈到Redis"""
        try:
            key = f"rag_routing:feedback:{feedback.strategy.value}"
            data = {
                "query_id": feedback.query_id,
                "feedback_type": feedback.feedback_type.value,
                "value": feedback.value,
                "timestamp": feedback.timestamp.isoformat(),
                "context": feedback.context
            }
            await redis_client.lpush(key, json.dumps(data, ensure_ascii=False))
            await redis_client.expire(key, 86400 * 7)  # 保存7天
        except Exception as e:
            logger.error(f"Failed to save feedback to Redis: {e}")

    async def _check_immediate_adjustment(self, feedback: FeedbackRecord) -> None:
        """检查是否需要立即调整"""

        # 检查是否为严重错误报告
        if feedback.feedback_type == FeedbackType.ERROR_REPORT and feedback.value < 2.0:
            await self._apply_emergency_adjustment(feedback.strategy)

        # 检查是否为极差满意度
        elif feedback.feedback_type == FeedbackType.SATISFACTION and feedback.value < 2.0:
            await self._apply_satisfaction_adjustment(feedback.strategy, feedback.value)

    async def _apply_emergency_adjustment(self, strategy: RAGStrategy) -> None:
        """应用紧急调整（针对严重错误）"""

        # 临时大幅降低问题策略的权重
        current_weight = self.strategy_weights.get(strategy, 0.25)
        new_weight = current_weight * 0.3  # 降低70%

        # 重新分配权重给其他策略
        other_strategies = [s for s in RAGStrategy if s != strategy]
        weight_increase = (current_weight - new_weight) / len(other_strategies)

        for other_strategy in other_strategies:
            self.strategy_weights[other_strategy] = min(
                0.8,  # 最高权重限制
                self.strategy_weights.get(other_strategy, 0.25) + weight_increase
            )

        self.strategy_weights[strategy] = new_weight

        # 记录调整
        adjustment = StrategyAdjustment(
            strategy=strategy,
            adjustment_type="emergency_weight_decrease",
            adjustment_value={"old_weight": current_weight, "new_weight": new_weight},
            reason="Emergency adjustment due to error report"
        )
        self.adjustment_history.append(adjustment)

        logger.warning(f"Emergency adjustment applied to {strategy.value}: weight {current_weight:.3f} -> {new_weight:.3f}")

        # 保存调整记录
        await self._save_adjustment_to_redis(adjustment)

    async def _apply_satisfaction_adjustment(
        self,
        strategy: RAGStrategy,
        satisfaction_score: float
    ) -> None:
        """应用满意度调整"""

        current_weight = self.strategy_weights.get(strategy, 0.25)
        # 根据满意度分数调整权重
        adjustment_factor = satisfaction_score / 5.0  # 归一化到[0,1]
        new_weight = current_weight * (0.5 + adjustment_factor * 0.5)  # 保持在[0.5, 1.0]倍

        self.strategy_weights[strategy] = max(0.05, new_weight)  # 最低权重限制

        # 记录调整
        adjustment = StrategyAdjustment(
            strategy=strategy,
            adjustment_type="satisfaction_weight_adjustment",
            adjustment_value={
                "old_weight": current_weight,
                "new_weight": new_weight,
                "satisfaction_score": satisfaction_score
            },
            reason=f"Adjustment based on satisfaction score: {satisfaction_score}"
        )
        self.adjustment_history.append(adjustment)

        logger.info(f"Satisfaction adjustment applied to {strategy.value}: weight {current_weight:.3f} -> {new_weight:.3f}")

        await self._save_adjustment_to_redis(adjustment)

    async def periodic_monitoring(self) -> None:
        """定期监控和调整"""

        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟检查一次

                # 获取当前性能指标
                performance_data = await self._collect_performance_data()

                # 检查所有调整规则
                adjustments = await self._evaluate_adjustment_rules(performance_data)

                # 应用调整
                for adjustment in adjustments:
                    await self._apply_strategy_adjustment(adjustment)

            except Exception as e:
                logger.error(f"Error in periodic monitoring: {e}")

    async def _collect_performance_data(self) -> Dict[str, Any]:
        """收集性能数据"""

        # 从路由引擎获取统计数据
        routing_stats = await routing_engine.get_routing_statistics()

        # 计算时间窗口内的反馈数据
        now = datetime.utcnow()
        time_windows = [1, 6, 12, 24]  # 小时

        performance_data = {
            "routing_stats": routing_stats,
            "feedback_analysis": {}
        }

        for hours in time_windows:
            cutoff_time = now - timedelta(hours=hours)
            recent_feedback = [
                f for f in self.feedback_history
                if f.timestamp > cutoff_time
            ]

            if recent_feedback:
                performance_data["feedback_analysis"][f"{hours}h"] = {
                    "total_feedback": len(recent_feedback),
                    "avg_satisfaction": np.mean([
                        f.value for f in recent_feedback
                        if f.feedback_type == FeedbackType.SATISFACTION
                    ]) if any(f.feedback_type == FeedbackType.SATISFACTION for f in recent_feedback) else None,
                    "strategy_feedback": {}
                }

                # 按策略分组反馈
                for strategy in RAGStrategy:
                    strategy_feedback = [
                        f for f in recent_feedback
                        if f.strategy == strategy
                    ]

                    if strategy_feedback:
                        performance_data["feedback_analysis"][f"{hours}h"]["strategy_feedback"][strategy.value] = {
                            "count": len(strategy_feedback),
                            "avg_satisfaction": np.mean([
                                f.value for f in strategy_feedback
                                if f.feedback_type == FeedbackType.SATISFACTION
                            ]) if any(f.feedback_type == FeedbackType.SATISFACTION for f in strategy_feedback) else None,
                            "error_rate": len([
                                f for f in strategy_feedback
                                if f.feedback_type == FeedbackType.ERROR_REPORT and f.value < 3.0
                            ]) / len(strategy_feedback)
                        }

        return performance_data

    async def _evaluate_adjustment_rules(
        self,
        performance_data: Dict[str, Any]
    ) -> List[StrategyAdjustment]:
        """评估调整规则"""

        adjustments = []

        for rule in self.adjustment_rules:
            if not rule.enabled:
                continue

            try:
                if await self._check_rule_condition(rule, performance_data):
                    adjustment = await self._create_adjustment_from_rule(rule, performance_data)
                    if adjustment:
                        adjustments.append(adjustment)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.trigger.value}: {e}")

        # 按优先级排序
        adjustments.sort(key=lambda x: self._get_rule_priority(x.reason), reverse=True)

        return adjustments

    async def _check_rule_condition(
        self,
        rule: AdjustmentRule,
        performance_data: Dict[str, Any]
    ) -> bool:
        """检查规则条件是否满足"""

        if rule.trigger == AdjustmentTrigger.LOW_SUCCESS_RATE:
            return await self._check_low_success_rate(rule, performance_data)

        elif rule.trigger == AdjustmentTrigger.HIGH_RESPONSE_TIME:
            return await self._check_high_response_time(rule, performance_data)

        elif rule.trigger == AdjustmentTrigger.POOR_SATISFACTION:
            return await self._check_poor_satisfaction(rule, performance_data)

        elif rule.trigger == AdjustmentTrigger.STRATEGY_IMBALANCE:
            return await self._check_strategy_imbalance(rule, performance_data)

        elif rule.trigger == AdjustmentTrigger.PERFORMANCE_DEGRADATION:
            return await self._check_performance_degradation(rule, performance_data)

        return False

    async def _check_low_success_rate(
        self,
        rule: AdjustmentRule,
        performance_data: Dict[str, Any]
    ) -> bool:
        """检查低成功率条件"""

        threshold = rule.condition["success_rate_threshold"]
        min_samples = rule.condition["min_samples"]
        time_window = rule.condition["time_window_hours"]

        routing_stats = performance_data.get("routing_stats", {}).get("routing_metrics", {})

        for strategy_name, metrics in routing_stats.items():
            if metrics.get("success_rate", 1.0) < threshold and metrics.get("total_count", 0) >= min_samples:
                return True

        return False

    async def _check_high_response_time(
        self,
        rule: AdjustmentRule,
        performance_data: Dict[str, Any]
    ) -> bool:
        """检查高响应时间条件"""

        threshold = rule.condition["response_time_threshold"]
        time_window = rule.condition["time_window_hours"]

        routing_stats = performance_data.get("routing_stats", {}).get("routing_metrics", {})

        for strategy_name, metrics in routing_stats.items():
            if metrics.get("avg_response_time", 0) > threshold:
                return True

        return False

    async def _check_poor_satisfaction(
        self,
        rule: AdjustmentRule,
        performance_data: Dict[str, Any]
    ) -> bool:
        """检查差满意度条件"""

        threshold = rule.condition["satisfaction_threshold"]
        min_samples = rule.condition["min_samples"]
        time_window = rule.condition["time_window_hours"]

        feedback_analysis = performance_data.get("feedback_analysis", {}).get(f"{time_window}h", {})

        if not feedback_analysis:
            return False

        strategy_feedback = feedback_analysis.get("strategy_feedback", {})

        for strategy_name, feedback_data in strategy_feedback.items():
            if (
                feedback_data.get("count", 0) >= min_samples and
                feedback_data.get("avg_satisfaction", 5.0) < threshold
            ):
                return True

        return False

    async def _check_strategy_imbalance(
        self,
        rule: AdjustmentRule,
        performance_data: Dict[str, Any]
    ) -> bool:
        """检查策略使用不均衡条件"""

        threshold = rule.condition["usage_threshold"]

        routing_stats = performance_data.get("routing_stats", {})
        strategy_usage = routing_stats.get("strategy_usage", {})

        total_decisions = routing_stats.get("total_decisions", 0)
        if total_decisions < 50:  # 样本太少不调整
            return False

        for strategy_name, usage_data in strategy_usage.items():
            if usage_data.get("percentage", 0) > threshold * 100:
                return True

        return False

    async def _check_performance_degradation(
        self,
        rule: AdjustmentRule,
        performance_data: Dict[str, Any]
    ) -> bool:
        """检查性能下降条件"""

        # 与历史数据比较，检测性能下降趋势
        # 这里简化实现，实际应该与更长时间窗口的数据比较
        return False

    async def _create_adjustment_from_rule(
        self,
        rule: AdjustmentRule,
        performance_data: Dict[str, Any]
    ) -> Optional[StrategyAdjustment]:
        """根据规则创建调整方案"""

        if rule.trigger == AdjustmentTrigger.LOW_SUCCESS_RATE:
            # 找到成功率最低的策略
            routing_stats = performance_data.get("routing_stats", {}).get("routing_metrics", {})
            worst_strategy = min(
                routing_stats.items(),
                key=lambda x: x[1].get("success_rate", 1.0)
            )

            strategy = RAGStrategy(worst_strategy[0])
            current_weight = self.strategy_weights.get(strategy, 0.25)
            new_weight = current_weight * rule.adjustment["factor"]

            return StrategyAdjustment(
                strategy=strategy,
                adjustment_type="rule_based_weight_decrease",
                adjustment_value={
                    "rule": rule.trigger.value,
                    "old_weight": current_weight,
                    "new_weight": new_weight
                },
                reason=f"Applied rule: {rule.trigger.value}"
            )

        elif rule.trigger == AdjustmentTrigger.STRATEGY_IMBALANCE:
            # 重新均衡权重
            total_weight = sum(self.strategy_weights.values())
            balanced_weight = 1.0 / len(self.strategy_weights)

            adjustments = []
            for strategy, current_weight in self.strategy_weights.items():
                if abs(current_weight - balanced_weight) > 0.1:  # 只调整差异较大的
                    adjustments.append(StrategyAdjustment(
                        strategy=strategy,
                        adjustment_type="rebalance_weight",
                        adjustment_value={
                            "old_weight": current_weight,
                            "new_weight": balanced_weight
                        },
                        reason="Strategy usage rebalancing"
                    ))

            # 返回第一个调整（其他将在后续迭代中处理）
            return adjustments[0] if adjustments else None

        return None

    async def _apply_strategy_adjustment(self, adjustment: StrategyAdjustment) -> None:
        """应用策略调整"""

        if adjustment.adjustment_type in [
            "weight_decrease", "emergency_weight_decrease",
            "satisfaction_weight_adjustment", "rule_based_weight_decrease"
        ]:
            if isinstance(adjustment.adjustment_value, dict) and "new_weight" in adjustment.adjustment_value:
                self.strategy_weights[adjustment.strategy] = adjustment.adjustment_value["new_weight"]

        elif adjustment.adjustment_type == "rebalance_weight":
            if isinstance(adjustment.adjustment_value, dict) and "new_weight" in adjustment.adjustment_value:
                self.strategy_weights[adjustment.strategy] = adjustment.adjustment_value["new_weight"]

        # 记录调整历史
        self.adjustment_history.append(adjustment)

        # 保存调整记录
        await self._save_adjustment_to_redis(adjustment)

        logger.info(f"Applied adjustment: {adjustment.adjustment_type} to {adjustment.strategy.value}")

        # 归一化权重
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """归一化策略权重"""

        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total_weight

    def _get_rule_priority(self, reason: str) -> int:
        """获取调整原因对应的优先级"""

        if "emergency" in reason:
            return 10
        elif "error" in reason:
            return 8
        elif "satisfaction" in reason:
            return 6
        elif "rule_based" in reason:
            return 4
        elif "rebalance" in reason:
            return 2
        else:
            return 1

    async def _save_adjustment_to_redis(self, adjustment: StrategyAdjustment) -> None:
        """保存调整记录到Redis"""
        try:
            key = f"rag_routing:adjustments:{datetime.utcnow().strftime('%Y%m%d')}"
            data = {
                "strategy": adjustment.strategy.value,
                "adjustment_type": adjustment.adjustment_type,
                "adjustment_value": adjustment.adjustment_value,
                "reason": adjustment.reason,
                "timestamp": adjustment.timestamp.isoformat()
            }
            await redis_client.lpush(key, json.dumps(data, ensure_ascii=False))
            await redis_client.expire(key, 86400 * 30)  # 保存30天
        except Exception as e:
            logger.error(f"Failed to save adjustment to Redis: {e}")

    def start_monitoring(self) -> None:
        """启动监控任务"""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self.periodic_monitoring())
            logger.info("Dynamic router adjustment monitoring started")

    def stop_monitoring(self) -> None:
        """停止监控任务"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
            logger.info("Dynamic router adjustment monitoring stopped")

    async def get_adjustment_summary(self) -> Dict[str, Any]:
        """获取调整摘要"""

        recent_adjustments = [
            adj for adj in self.adjustment_history
            if adj.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]

        # 按调整类型分组
        adjustment_types = {}
        for adj in recent_adjustments:
            adj_type = adj.adjustment_type
            if adj_type not in adjustment_types:
                adjustment_types[adj_type] = 0
            adjustment_types[adj_type] += 1

        return {
            "current_weights": {
                strategy.value: weight for strategy, weight in self.strategy_weights.items()
            },
            "recent_adjustments_24h": len(recent_adjustments),
            "adjustment_types": adjustment_types,
            "total_feedback_records": len(self.feedback_history),
            "monitoring_active": self.monitoring_task is not None
        }

    async def load_feedback_from_redis(self) -> None:
        """从Redis加载反馈历史"""
        try:
            for strategy in RAGStrategy:
                key = f"rag_routing:feedback:{strategy.value}"
                feedback_data = await redis_client.lrange(key, 0, -1)

                for data in feedback_data:
                    try:
                        feedback_dict = json.loads(data)
                        feedback = FeedbackRecord(
                            query_id=feedback_dict["query_id"],
                            strategy=RAGStrategy(strategy.value),
                            feedback_type=FeedbackType(feedback_dict["feedback_type"]),
                            value=feedback_dict["value"],
                            timestamp=datetime.fromisoformat(feedback_dict["timestamp"]),
                            context=feedback_dict.get("context", {})
                        )
                        self.feedback_history.append(feedback)
                    except Exception as e:
                        logger.error(f"Failed to parse feedback data: {e}")

        except Exception as e:
            logger.error(f"Failed to load feedback from Redis: {e}")


# 全局动态调整器实例
dynamic_adjuster = DynamicRouterAdjuster()