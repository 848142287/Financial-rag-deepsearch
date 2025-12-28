"""
RAG策略路由决策引擎
实现智能RAG策略选择和动态调整机制
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import math

from app.services.rag_unified.strategies import (
    RAGStrategy,
    StrategyConfig,
    RAGStrategyExecutor,
    QueryFeatures,
    rag_executor_registry
)
from app.services.rag_unified.query_analyzer import QueryAnalyzer, QueryIntent, QueryComplexity
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


@dataclass
class DecisionFactor:
    """决策因子"""
    name: str
    weight: float
    score: float
    description: str = ""
    threshold: Optional[float] = None


@dataclass
class RoutingDecision:
    """路由决策结果"""
    selected_strategy: RAGStrategy
    confidence: float
    decision_factors: List[DecisionFactor]
    alternative_strategies: List[Tuple[RAGStrategy, float]]
    reasoning: str
    estimated_response_time: float
    estimated_cost: float


@dataclass
class RoutingMetrics:
    """路由性能指标"""
    strategy_name: str
    success_count: int = 0
    total_count: int = 0
    avg_response_time: float = 0.0
    avg_satisfaction_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class RoutingDecisionEngine:
    """RAG策略路由决策引擎"""

    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.routing_metrics: Dict[str, RoutingMetrics] = {}
        self.decision_history: List[Dict[str, Any]] = []
        self.performance_window = timedelta(hours=24)

        # 初始化策略指标
        for strategy in RAGStrategy:
            self.routing_metrics[strategy.value] = RoutingMetrics(
                strategy_name=strategy.value
            )

    async def analyze_query_and_route(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """分析查询并做出路由决策"""

        # 1. 提取查询特征
        query_features = await self.query_analyzer.extract_features(
            query, conversation_history, context
        )

        # 2. 计算决策因子得分
        decision_factors = await self._calculate_decision_factors(query_features)

        # 3. 评估每个策略的适宜性
        strategy_scores = await self._evaluate_strategy_fitness(
            query_features, decision_factors
        )

        # 4. 选择最佳策略
        selected_strategy, confidence = self._select_best_strategy(strategy_scores)

        # 5. 生成替代策略
        alternative_strategies = self._get_alternative_strategies(
            strategy_scores, selected_strategy, count=2
        )

        # 6. 估算性能指标
        estimated_time, estimated_cost = self._estimate_performance(
            selected_strategy, query_features
        )

        # 7. 生成决策解释
        reasoning = self._generate_reasoning(
            selected_strategy, decision_factors, query_features
        )

        # 8. 创建决策结果
        decision = RoutingDecision(
            selected_strategy=selected_strategy,
            confidence=confidence,
            decision_factors=decision_factors,
            alternative_strategies=alternative_strategies,
            reasoning=reasoning,
            estimated_response_time=estimated_time,
            estimated_cost=estimated_cost
        )

        # 9. 记录决策历史
        self._record_decision(query, query_features, decision)

        return decision

    async def _calculate_decision_factors(
        self,
        features: QueryFeatures
    ) -> List[DecisionFactor]:
        """计算决策因子得分"""

        factors = []

        # 1. 实体复杂度因子
        entity_score = min(features.entity_count / 5.0, 1.0)  # 归一化到[0,1]
        factors.append(DecisionFactor(
            name="entity_complexity",
            weight=0.2,
            score=entity_score,
            description="查询中实体的数量和复杂度",
            threshold=0.6
        ))

        # 2. 关系复杂度因子
        relation_score = min(features.relation_complexity.value / 3.0, 1.0)
        factors.append(DecisionFactor(
            name="relation_complexity",
            weight=0.15,
            score=relation_score,
            description="实体间关系的复杂程度",
            threshold=0.5
        ))

        # 3. 时间敏感度因子
        time_score = features.time_sensitivity.value / 3.0
        factors.append(DecisionFactor(
            name="time_sensitivity",
            weight=0.15,
            score=time_score,
            description="查询对时间信息的要求程度",
            threshold=0.6
        ))

        # 4. 答案粒度因子
        granularity_score = features.answer_granularity.value / 3.0
        factors.append(DecisionFactor(
            name="answer_granularity",
            weight=0.2,
            score=granularity_score,
            description="所需答案的详细程度",
            threshold=0.7
        ))

        # 5. 查询长度因子（简单性指标）
        length_score = min(len(features.original_query) / 200.0, 1.0)
        simplicity_score = 1.0 - length_score  # 查询越简单越好
        factors.append(DecisionFactor(
            name="query_simplicity",
            weight=0.1,
            score=simplicity_score,
            description="查询的简洁程度（反向指标）",
            threshold=0.3
        ))

        # 6. 历史成功率因子（基于历史数据）
        success_score = await self._calculate_historical_success_factor(features)
        factors.append(DecisionFactor(
            name="historical_success",
            weight=0.2,
            score=success_score,
            description="基于历史数据策略的成功率",
            threshold=0.7
        ))

        return factors

    async def _evaluate_strategy_fitness(
        self,
        features: QueryFeatures,
        decision_factors: List[DecisionFactor]
    ) -> Dict[RAGStrategy, float]:
        """评估每个策略对查询的适宜性"""

        strategy_scores = {}

        for strategy in RAGStrategy:
            score = 0.0

            # 基础适应性评分
            base_score = await self._calculate_base_fitness_score(strategy, features)
            score += base_score * 0.4  # 基础分占40%

            # 决策因子加权评分
            factor_score = await self._calculate_factor_fitness_score(
                strategy, decision_factors, features
            )
            score += factor_score * 0.4  # 因子分占40%

            # 性能历史评分
            performance_score = await self._calculate_performance_fitness_score(strategy)
            score += performance_score * 0.2  # 性能分占20%

            strategy_scores[strategy] = score

        return strategy_scores

    async def _calculate_base_fitness_score(
        self,
        strategy: RAGStrategy,
        features: QueryFeatures
    ) -> float:
        """计算策略基础适宜性得分"""

        score = 0.0

        # 根据策略特点评分
        if strategy == RAGStrategy.LIGHT_RAG:
            # LightRAG适合简单、快速的查询
            if features.complexity == QueryComplexity.SIMPLE:
                score += 0.8
            elif features.complexity == QueryComplexity.MODERATE:
                score += 0.5
            else:
                score += 0.2

            # 时间敏感查询的加分
            if features.time_sensitivity.value >= 2:
                score += 0.2

            # 实体少的加分
            if features.entity_count <= 2:
                score += 0.2

        elif strategy == RAGStrategy.GRAPH_RAG:
            # GraphRAG适合关系复杂的查询
            if features.relation_complexity.value >= 2:
                score += 0.7

            # 实体多的加分
            if features.entity_count >= 3:
                score += 0.3

            # 关系型意图的加分
            if features.intent in [QueryIntent.RELATIONAL, QueryIntent.COMPARATIVE]:
                score += 0.2

        elif strategy == RAGStrategy.AGENTIC_RAG:
            # AgenticRAG适合复杂、研究型查询
            if features.complexity == QueryComplexity.COMPLEX:
                score += 0.8
            elif features.complexity == QueryComplexity.MODERATE:
                score += 0.5
            else:
                score += 0.3

            # 分析性和预测性意图的加分
            if features.intent in [QueryIntent.ANALYTICAL, QueryIntent.PREDICTIVE, QueryIntent.CAUSAL]:
                score += 0.3

            # 细粒度答案的加分
            if features.answer_granularity.value >= 2:
                score += 0.2

        elif strategy == RAGStrategy.HYBRID_RAG:
            # HybridRAG作为通用策略，适合所有查询
            score = 0.6  # 基础分

            # 在其他策略都不明确时加分
            if features.complexity == QueryComplexity.MODERATE:
                score += 0.2

            # 多实体查询的加分
            if features.entity_count >= 2:
                score += 0.2

        return min(score, 1.0)

    async def _calculate_factor_fitness_score(
        self,
        strategy: RAGStrategy,
        decision_factors: List[DecisionFactor],
        features: QueryFeatures
    ) -> float:
        """基于决策因子计算策略适宜性得分"""

        total_score = 0.0
        total_weight = 0.0

        for factor in decision_factors:
            factor_score = 0.0

            # 根据策略类型调整因子得分
            if factor.name == "entity_complexity":
                if strategy == RAGStrategy.GRAPH_RAG:
                    factor_score = factor.score  # GraphRAG喜欢复杂实体
                elif strategy == RAGStrategy.LIGHT_RAG:
                    factor_score = 1.0 - factor.score  # LightRAG喜欢简单实体
                else:
                    factor_score = 0.5  # 中性

            elif factor.name == "relation_complexity":
                if strategy == RAGStrategy.GRAPH_RAG:
                    factor_score = factor.score  # GraphRAG适合复杂关系
                elif strategy == RAGStrategy.AGENTIC_RAG:
                    factor_score = factor.score * 0.7  # AgenticRAG也适合
                else:
                    factor_score = factor.score * 0.3

            elif factor.name == "time_sensitivity":
                if strategy == RAGStrategy.LIGHT_RAG:
                    factor_score = factor.score  # LightRAG快速响应
                elif strategy == RAGStrategy.AGENTIC_RAG:
                    factor_score = max(0.2, 1.0 - factor.score)  # 复杂查询相对较慢
                else:
                    factor_score = 0.6

            elif factor.name == "answer_granularity":
                if strategy == RAGStrategy.AGENTIC_RAG:
                    factor_score = factor.score  # AgenticRAG适合细粒度
                elif strategy == RAGStrategy.GRAPH_RAG:
                    factor_score = factor.score * 0.8
                else:
                    factor_score = factor.score * 0.5

            elif factor.name == "query_simplicity":
                if strategy == RAGStrategy.LIGHT_RAG:
                    factor_score = factor.score  # LightRAG适合简单查询
                else:
                    factor_score = max(0.3, 1.0 - factor.score * 0.5)

            elif factor.name == "historical_success":
                factor_score = factor.score  # 直接使用历史成功率

            total_score += factor_score * factor.weight
            total_weight += factor.weight

        return total_score / total_weight if total_weight > 0 else 0.0

    async def _calculate_performance_fitness_score(self, strategy: RAGStrategy) -> float:
        """基于性能历史计算策略适宜性得分"""

        metrics = self.routing_metrics.get(strategy.value)
        if not metrics or metrics.total_count == 0:
            return 0.5  # 默认中性分数

        # 成功率
        success_rate = metrics.success_count / metrics.total_count

        # 响应时间评分（响应时间越短越好）
        time_score = max(0.1, 1.0 - metrics.avg_response_time / 10.0)  # 10秒为基准

        # 用户满意度评分
        satisfaction_score = metrics.avg_satisfaction_score / 5.0  # 假设满分为5分

        # 综合评分
        performance_score = (
            success_rate * 0.4 +
            time_score * 0.3 +
            satisfaction_score * 0.3
        )

        return performance_score

    def _select_best_strategy(
        self,
        strategy_scores: Dict[RAGStrategy, float]
    ) -> Tuple[RAGStrategy, float]:
        """选择最佳策略"""

        if not strategy_scores:
            return RAGStrategy.HYBRID_RAG, 0.5

        # 找到得分最高的策略
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])

        # 计算置信度
        scores = list(strategy_scores.values())
        max_score = max(scores)
        second_max_score = sorted(scores)[-2] if len(scores) > 1 else 0.0

        # 置信度 = (最高分 - 第二高分) / 最高分
        confidence = (max_score - second_max_score) / max_score if max_score > 0 else 0.5
        confidence = max(0.1, min(1.0, confidence))

        return best_strategy[0], confidence

    def _get_alternative_strategies(
        self,
        strategy_scores: Dict[RAGStrategy, float],
        selected_strategy: RAGStrategy,
        count: int = 2
    ) -> List[Tuple[RAGStrategy, float]]:
        """获取替代策略"""

        # 过滤掉已选择的策略
        alternatives = [
            (strategy, score) for strategy, score in strategy_scores.items()
            if strategy != selected_strategy
        ]

        # 按得分排序并返回前count个
        alternatives.sort(key=lambda x: x[1], reverse=True)
        return alternatives[:count]

    def _estimate_performance(
        self,
        strategy: RAGStrategy,
        features: QueryFeatures
    ) -> Tuple[float, float]:
        """估算响应时间和成本"""

        # 基础性能参数（基于策略特点）
        base_params = {
            RAGStrategy.LIGHT_RAG: {"time": 2.0, "cost": 1.0},
            RAGStrategy.GRAPH_RAG: {"time": 5.0, "cost": 2.0},
            RAGStrategy.AGENTIC_RAG: {"time": 15.0, "cost": 5.0},
            RAGStrategy.HYBRID_RAG: {"time": 8.0, "cost": 3.0}
        }

        base = base_params[strategy]

        # 复杂度调整系数
        complexity_multiplier = {
            QueryComplexity.SIMPLE: 0.8,
            QueryComplexity.MODERATE: 1.2,
            QueryComplexity.COMPLEX: 1.5
        }

        multiplier = complexity_multiplier.get(features.complexity, 1.0)

        estimated_time = base["time"] * multiplier
        estimated_cost = base["cost"] * multiplier

        return estimated_time, estimated_cost

    def _generate_reasoning(
        self,
        strategy: RAGStrategy,
        factors: List[DecisionFactor],
        features: QueryFeatures
    ) -> str:
        """生成决策解释"""

        reasoning_parts = []

        # 策略选择说明
        strategy_reasons = {
            RAGStrategy.LIGHT_RAG: "选择LightRAG是因为查询相对简单，需要快速响应",
            RAGStrategy.GRAPH_RAG: "选择GraphRAG是因为查询涉及多个实体间的复杂关系",
            RAGStrategy.AGENTIC_RAG: "选择AgenticRAG是因为查询复杂度高，需要深度分析",
            RAGStrategy.HYBRID_RAG: "选择HybridRAG是因为查询需要综合多种检索策略"
        }

        reasoning_parts.append(strategy_reasons.get(strategy, "基于综合分析选择该策略"))

        # 关键决策因子
        key_factors = [f for f in factors if f.score > (f.threshold or 0.5)]
        if key_factors:
            factor_descs = [f"{f.name}({f.score:.2f})" for f in key_factors]
            reasoning_parts.append(f"关键因子: {', '.join(factor_descs)}")

        # 查询特征说明
        reasoning_parts.append(
            f"查询特征: 复杂度={features.complexity.value}, "
            f"意图={features.intent.value}, "
            f"实体数={features.entity_count}"
        )

        return "; ".join(reasoning_parts)

    def _record_decision(
        self,
        query: str,
        features: QueryFeatures,
        decision: RoutingDecision
    ) -> None:
        """记录决策历史"""

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "query_length": len(query),
            "entity_count": features.entity_count,
            "complexity": features.complexity.value,
            "intent": features.intent.value,
            "selected_strategy": decision.selected_strategy.value,
            "confidence": decision.confidence,
            "estimated_time": decision.estimated_response_time,
            "decision_factors": [
                {"name": f.name, "score": f.score, "weight": f.weight}
                for f in decision.decision_factors
            ]
        }

        self.decision_history.append(record)

        # 保持历史记录在合理范围内
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-800:]

        # 异步保存到Redis
        asyncio.create_task(self._save_decision_to_redis(record))

    async def _save_decision_to_redis(self, record: Dict[str, Any]) -> None:
        """保存决策记录到Redis"""
        try:
            key = f"rag_routing:decision:{datetime.utcnow().strftime('%Y%m%d')}"
            await redis_client.lpush(key, json.dumps(record, ensure_ascii=False))
            await redis_client.expire(key, 86400 * 7)  # 保存7天
        except Exception as e:
            logger.error(f"Failed to save decision to Redis: {e}")

    async def _calculate_historical_success_factor(self, features: QueryFeatures) -> float:
        """计算历史成功率因子"""

        # 从Redis获取历史决策数据
        try:
            key = f"rag_routing:success_pattern:{features.complexity.value}:{features.intent.value}"
            success_data = await redis_client.get(key)

            if success_data:
                pattern = json.loads(success_data)
                return pattern.get("success_rate", 0.5)
        except Exception as e:
            logger.error(f"Failed to get historical success factor: {e}")

        return 0.5  # 默认中性分数

    async def update_strategy_performance(
        self,
        strategy: RAGStrategy,
        response_time: float,
        user_satisfaction: Optional[float] = None,
        success: bool = True
    ) -> None:
        """更新策略性能指标"""

        metrics = self.routing_metrics.get(strategy.value)
        if not metrics:
            metrics = RoutingMetrics(strategy_name=strategy.value)
            self.routing_metrics[strategy.value] = metrics

        # 更新指标
        metrics.total_count += 1
        if success:
            metrics.success_count += 1

        # 更新平均响应时间（使用指数移动平均）
        alpha = 0.3  # 平滑因子
        metrics.avg_response_time = (
            alpha * response_time +
            (1 - alpha) * metrics.avg_response_time
        )

        # 更新用户满意度
        if user_satisfaction is not None:
            if metrics.avg_satisfaction_score == 0:
                metrics.avg_satisfaction_score = user_satisfaction
            else:
                metrics.avg_satisfaction_score = (
                    alpha * user_satisfaction +
                    (1 - alpha) * metrics.avg_satisfaction_score
                )

        metrics.last_updated = datetime.utcnow()

        # 异步保存到Redis
        asyncio.create_task(self._save_metrics_to_redis(strategy.value, metrics))

    async def _save_metrics_to_redis(self, strategy_name: str, metrics: RoutingMetrics) -> None:
        """保存性能指标到Redis"""
        try:
            key = f"rag_routing:metrics:{strategy_name}"
            data = {
                "success_count": metrics.success_count,
                "total_count": metrics.total_count,
                "avg_response_time": metrics.avg_response_time,
                "avg_satisfaction_score": metrics.avg_satisfaction_score,
                "last_updated": metrics.last_updated.isoformat()
            }
            await redis_client.setex(
                key,
                86400 * 30,  # 保存30天
                json.dumps(data, ensure_ascii=False)
            )
        except Exception as e:
            logger.error(f"Failed to save metrics to Redis: {e}")

    async def load_metrics_from_redis(self) -> None:
        """从Redis加载性能指标"""
        try:
            for strategy in RAGStrategy:
                key = f"rag_routing:metrics:{strategy.value}"
                data = await redis_client.get(key)

                if data:
                    metrics_dict = json.loads(data)
                    metrics = RoutingMetrics(
                        strategy_name=strategy.value,
                        success_count=metrics_dict.get("success_count", 0),
                        total_count=metrics_dict.get("total_count", 0),
                        avg_response_time=metrics_dict.get("avg_response_time", 0.0),
                        avg_satisfaction_score=metrics_dict.get("avg_satisfaction_score", 0.0),
                        last_updated=datetime.fromisoformat(
                            metrics_dict.get("last_updated", datetime.utcnow().isoformat())
                        )
                    )
                    self.routing_metrics[strategy.value] = metrics
        except Exception as e:
            logger.error(f"Failed to load metrics from Redis: {e}")

    async def get_routing_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""

        total_decisions = len(self.decision_history)
        if total_decisions == 0:
            return {"total_decisions": 0}

        # 策略使用统计
        strategy_usage = {}
        for strategy in RAGStrategy:
            count = sum(
                1 for d in self.decision_history
                if d.get("selected_strategy") == strategy.value
            )
            strategy_usage[strategy.value] = {
                "count": count,
                "percentage": (count / total_decisions) * 100
            }

        # 平均置信度
        avg_confidence = sum(
            d.get("confidence", 0) for d in self.decision_history
        ) / total_decisions

        # 最近24小时的决策
        recent_decisions = [
            d for d in self.decision_history
            if datetime.fromisoformat(d["timestamp"]) > datetime.utcnow() - timedelta(hours=24)
        ]

        return {
            "total_decisions": total_decisions,
            "recent_decisions_24h": len(recent_decisions),
            "strategy_usage": strategy_usage,
            "average_confidence": avg_confidence,
            "routing_metrics": {
                name: {
                    "success_rate": metrics.success_count / metrics.total_count if metrics.total_count > 0 else 0,
                    "avg_response_time": metrics.avg_response_time,
                    "avg_satisfaction": metrics.avg_satisfaction_score
                }
                for name, metrics in self.routing_metrics.items()
            }
        }


# 全局路由决策引擎实例
routing_engine = RoutingDecisionEngine()