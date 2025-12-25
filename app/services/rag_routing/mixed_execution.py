"""
混合执行模式
实现并行-融合模式和流水线模式的RAG策略执行
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import time

from app.services.rag_routing.strategies import (
    RAGStrategy,
    QueryFeatures,
    RAGStrategyExecutor,
    strategy_manager
)
from app.services.rag_routing.router import routing_engine

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """执行模式"""
    PARALLEL_FUSION = "parallel_fusion"    # 并行-融合模式
    PIPELINE = "pipeline"                  # 流水线模式
    ADAPTIVE = "adaptive"                  # 自适应模式


@dataclass
class ExecutionContext:
    """执行上下文"""
    query: str
    features: QueryFeatures
    mode: ExecutionMode
    strategies: List[RAGStrategy]
    context: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0  # 超时时间（秒）
    max_concurrent: int = 3  # 最大并发数


@dataclass
class StrategyResult:
    """策略执行结果"""
    strategy: RAGStrategy
    results: List[Dict[str, Any]]
    execution_time: float
    confidence: float
    metadata: Dict[str, Any]
    status: str  # success, timeout, error
    error_message: Optional[str] = None


@dataclass
class FusionConfig:
    """融合配置"""
    deduplication: bool = True
    conflict_resolution: str = "evidence_weighted"  # evidence_weighted, confidence_based, majority_vote
    evidence_weighting: str = "source_reliability"  # source_reliability, recency, citation_count
    min_evidence_count: int = 2
    confidence_threshold: float = 0.6


class StrategyDecomposer:
    """策略分解器"""

    def __init__(self):
        self.decomposition_rules = self._initialize_decomposition_rules()

    def _initialize_decomposition_rules(self) -> Dict[str, Any]:
        """初始化分解规则"""
        return {
            "parallel_fusion": {
                "simple_query": {
                    "complexity_threshold": 0.4,
                    "strategies": [RAGStrategy.LIGHT_RAG, RAGStrategy.HYBRID_RAG]
                },
                "relational_query": {
                    "min_entities": 2,
                    "strategies": [RAGStrategy.LIGHT_RAG, RAGStrategy.GRAPH_RAG, RAGStrategy.HYBRID_RAG]
                },
                "complex_query": {
                    "complexity_threshold": 0.7,
                    "strategies": [RAGStrategy.LIGHT_RAG, RAGStrategy.GRAPH_RAG, RAGStrategy.AGENTIC_RAG]
                },
                "research_query": {
                    "complexity_threshold": 0.8,
                    "intent_types": ["analytical", "predictive", "causal"],
                    "strategies": [RAGStrategy.GRAPH_RAG, RAGStrategy.AGENTIC_RAG, RAGStrategy.HYBRID_RAG]
                }
            },
            "pipeline": {
                "standard_sequence": [
                    RAGStrategy.LIGHT_RAG,
                    RAGStrategy.GRAPH_RAG,
                    RAGStrategy.AGENTIC_RAG
                ],
                "lightweight_sequence": [
                    RAGStrategy.LIGHT_RAG,
                    RAGStrategy.HYBRID_RAG
                ],
                "research_sequence": [
                    RAGStrategy.GRAPH_RAG,
                    RAGStrategy.AGENTIC_RAG
                ]
            }
        }

    async def decompose_query(
        self,
        query: str,
        features: QueryFeatures,
        mode: ExecutionMode
    ) -> ExecutionContext:
        """分解查询并创建执行上下文"""

        strategies = []

        if mode == ExecutionMode.PARALLEL_FUSION:
            strategies = await self._decompose_for_parallel_fusion(features)
        elif mode == ExecutionMode.PIPELINE:
            strategies = await self._decompose_for_pipeline(features)
        elif mode == ExecutionMode.ADAPTIVE:
            strategies = await self._decompose_adaptive(features)
        else:
            strategies = [RAGStrategy.HYBRID_RAG]

        return ExecutionContext(
            query=query,
            features=features,
            mode=mode,
            strategies=strategies,
            context={"decomposition_time": datetime.utcnow().isoformat()}
        )

    async def _decompose_for_parallel_fusion(self, features: QueryFeatures) -> List[RAGStrategy]:
        """为并行-融合模式分解策略"""

        complexity = features.complexity.value
        entity_count = features.entity_count
        intent = features.intent.value

        # 复杂查询：全策略并行
        if complexity >= 0.8 or intent in ["analytical", "predictive", "causal"]:
            return [
                RAGStrategy.LIGHT_RAG,
                RAGStrategy.GRAPH_RAG,
                RAGStrategy.AGENTIC_RAG
            ]

        # 关系查询：向量+图谱
        elif entity_count >= 2 or features.relation_complexity.value >= 2:
            return [
                RAGStrategy.LIGHT_RAG,
                RAGStrategy.GRAPH_RAG,
                RAGStrategy.HYBRID_RAG
            ]

        # 简单查询：轻量级+混合
        elif complexity < 0.4:
            return [
                RAGStrategy.LIGHT_RAG,
                RAGStrategy.HYBRID_RAG
            ]

        # 默认：混合策略
        return [RAGStrategy.HYBRID_RAG]

    async def _decompose_for_pipeline(self, features: QueryFeatures) -> List[RAGStrategy]:
        """为流水线模式分解策略"""

        # 基于复杂度和意图确定流水线序列
        if features.complexity.value >= 0.8:
            # 深度研究流水线
            return [
                RAGStrategy.LIGHT_RAG,    # 快速筛选
                RAGStrategy.GRAPH_RAG,    # 关系挖掘
                RAGStrategy.AGENTIC_RAG   # 深度推理
            ]
        elif features.entity_count >= 3:
            # 关系密集流水线
            return [
                RAGStrategy.LIGHT_RAG,    # 基础检索
                RAGStrategy.GRAPH_RAG,    # 实体关系
                RAGStrategy.HYBRID_RAG    # 综合分析
            ]
        else:
            # 轻量流水线
            return [
                RAGStrategy.LIGHT_RAG,
                RAGStrategy.HYBRID_RAG
            ]

    async def _decompose_adaptive(self, features: QueryFeatures) -> List[RAGStrategy]:
        """自适应分解策略"""

        # 综合考虑多种因素
        factors = {
            "complexity": features.complexity.value,
            "entities": min(features.entity_count / 3.0, 1.0),
            "time_sensitivity": features.time_sensitivity.value / 3.0,
            "granularity": features.answer_granularity.value / 3.0
        }

        # 计算综合得分
        adaptive_score = sum(factors.values()) / len(factors)

        if adaptive_score >= 0.7:
            return await self._decompose_for_parallel_fusion(features)
        elif adaptive_score >= 0.4:
            return await self._decompose_for_pipeline(features)
        else:
            return [RAGStrategy.LIGHT_RAG]


class ResultFusionEngine:
    """结果融合引擎"""

    def __init__(self, config: FusionConfig = None):
        self.config = config or FusionConfig()
        self.fusion_methods = {
            "parallel_fusion": self._parallel_fusion,
            "pipeline": self._pipeline_fusion,
            "evidence_weighted": self._evidence_weighted_fusion
        }

    async def fuse_results(
        self,
        strategy_results: List[StrategyResult],
        execution_context: ExecutionContext
    ) -> Dict[str, Any]:
        """融合多个策略的执行结果"""

        # 过滤成功的策略结果
        successful_results = [r for r in strategy_results if r.status == "success"]

        if not successful_results:
            return {
                "fused_results": [],
                "fusion_confidence": 0.0,
                "fusion_metadata": {
                    "error": "No successful strategy results",
                    "total_attempts": len(strategy_results)
                }
            }

        # 选择融合方法
        fusion_method = self._select_fusion_method(execution_context.mode, successful_results)

        # 执行融合
        fused_results = await fusion_method(successful_results, execution_context)

        # 计算融合置信度
        fusion_confidence = self._calculate_fusion_confidence(successful_results, fused_results)

        return {
            "fused_results": fused_results,
            "fusion_confidence": fusion_confidence,
            "fusion_metadata": {
                "method": fusion_method.__name__,
                "input_strategies": [r.strategy.value for r in successful_results],
                "total_input_results": sum(len(r.results) for r in successful_results),
                "fusion_config": self.config.__dict__
            }
        }

    def _select_fusion_method(
        self,
        mode: ExecutionMode,
        results: List[StrategyResult]
    ):
        """选择融合方法"""

        if mode == ExecutionMode.PARALLEL_FUSION:
            return self.fusion_methods["parallel_fusion"]
        elif mode == ExecutionMode.PIPELINE:
            return self.fusion_methods["pipeline"]
        else:
            return self.fusion_methods["evidence_weighted"]

    async def _parallel_fusion(
        self,
        results: List[StrategyResult],
        context: ExecutionContext
    ) -> List[Dict[str, Any]]:
        """并行融合模式"""

        # 收集所有结果
        all_results = []
        for result in results:
            for item in result.results:
                item["source_strategy"] = result.strategy.value
                item["strategy_confidence"] = result.confidence
                all_results.append(item)

        # 去重
        if self.config.deduplication:
            all_results = self._deduplicate_results(all_results)

        # 冲突解决
        if self.config.conflict_resolution != "none":
            all_results = self._resolve_conflicts(all_results)

        # 证据加权
        if self.config.evidence_weighting != "none":
            all_results = self._apply_evidence_weighting(all_results, results)

        # 排序和限制
        all_results.sort(key=lambda x: x.get("fused_score", x.get("score", 0)), reverse=True)

        return all_results[:20]  # 返回前20个结果

    async def _pipeline_fusion(
        self,
        results: List[StrategyResult],
        context: ExecutionContext
    ) -> List[Dict[str, Any]]:
        """流水线融合模式"""

        # 按流水线顺序处理结果
        fused_results = []
        processed_ids = set()

        for result in results:
            # 处理当前策略的结果
            for item in result.results:
                item_id = item.get("id", "")

                # 如果未被处理，则添加
                if item_id not in processed_ids:
                    item["source_strategy"] = result.strategy.value
                    item["pipeline_stage"] = len(fused_results) + 1
                    fused_results.append(item)
                    processed_ids.add(item_id)

        # 应用流水线特定的增强
        fused_results = self._enhance_pipeline_results(fused_results, context)

        return fused_results

    async def _evidence_weighted_fusion(
        self,
        results: List[StrategyResult],
        context: ExecutionContext
    ) -> List[Dict[str, Any]]:
        """证据加权融合"""

        evidence_map = {}

        # 构建证据图
        for result in results:
            for item in result.results:
                item_id = item.get("id", "")
                content = item.get("content", "")

                if item_id not in evidence_map:
                    evidence_map[item_id] = {
                        "id": item_id,
                        "content": content,
                        "evidence": [],
                        "base_score": item.get("score", 0)
                    }

                evidence_map[item_id]["evidence"].append({
                    "strategy": result.strategy.value,
                    "confidence": result.confidence,
                    "score": item.get("score", 0),
                    "metadata": item.get("metadata", {})
                })

        # 计算证据加权分数
        fused_results = []
        for item_id, evidence_data in evidence_map.items():
            # 计算加权分数
            weighted_score = self._calculate_evidence_weighted_score(evidence_data)

            # 检查最小证据数量
            if len(evidence_data["evidence"]) >= self.config.min_evidence_count:
                fused_results.append({
                    "id": item_id,
                    "content": evidence_data["content"],
                    "score": weighted_score,
                    "evidence_count": len(evidence_data["evidence"]),
                    "evidence_sources": [e["strategy"] for e in evidence_data["evidence"]],
                    "fusion_method": "evidence_weighted"
                })

        # 排序
        fused_results.sort(key=lambda x: x["score"], reverse=True)

        return fused_results

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重结果"""
        seen_content = set()
        unique_results = []

        for result in results:
            content = result.get("content", "")
            content_hash = hash(content[:200])  # 使用前200字符去重

            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)

        return unique_results

    def _resolve_conflicts(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """解决冲突"""
        # 简化的冲突解决：基于置信度和分数
        conflict_groups = {}

        for result in results:
            content_key = result.get("content", "")[:100]  # 使用前100字符分组

            if content_key not in conflict_groups:
                conflict_groups[content_key] = []
            conflict_groups[content_key].append(result)

        resolved_results = []
        for content_key, conflicts in conflict_groups.items():
            if len(conflicts) == 1:
                resolved_results.extend(conflicts)
            else:
                # 根据冲突解决策略选择最佳结果
                best_result = max(conflicts, key=lambda x: x.get("score", 0))
                best_result["conflict_resolved"] = True
                resolved_results.append(best_result)

        return resolved_results

    def _apply_evidence_weighting(
        self,
        results: List[Dict[str, Any]],
        strategy_results: List[StrategyResult]
    ) -> List[Dict[str, Any]]:
        """应用证据加权"""

        # 策略可靠性权重
        strategy_weights = {
            RAGStrategy.LIGHT_RAG.value: 0.7,
            RAGStrategy.GRAPH_RAG.value: 0.85,
            RAGStrategy.AGENTIC_RAG.value: 0.9,
            RAGStrategy.HYBRID_RAG.value: 0.8
        }

        for result in results:
            strategy = result.get("source_strategy", "")
            strategy_weight = strategy_weights.get(strategy, 0.5)

            # 应用权重
            original_score = result.get("score", 0)
            weighted_score = original_score * strategy_weight

            # 结合策略置信度
            strategy_confidence = result.get("strategy_confidence", 0.5)
            final_score = weighted_score * (0.7 + 0.3 * strategy_confidence)

            result["fused_score"] = final_score
            result["strategy_weight"] = strategy_weight

        return results

    def _enhance_pipeline_results(
        self,
        results: List[Dict[str, Any]],
        context: ExecutionContext
    ) -> List[Dict[str, Any]]:
        """增强流水线结果"""

        # 为不同阶段的结果添加增强信息
        for i, result in enumerate(results):
            stage = result.get("pipeline_stage", 1)

            if stage == 1:
                # 第一阶段：基础检索增强
                result["stage_type"] = "fast_screening"
                result["stage_description"] = "快速筛选基础信息"
            elif stage == 2:
                # 第二阶段：关系增强
                result["stage_type"] = "relationship_mining"
                result["stage_description"] = "挖掘实体关系"
            elif stage == 3:
                # 第三阶段：推理增强
                result["stage_type"] = "deep_reasoning"
                result["stage_description"] = "深度推理分析"

            # 调整分数：后续阶段分数略有提升
            stage_bonus = (stage - 1) * 0.1
            original_score = result.get("score", 0)
            result["fused_score"] = min(1.0, original_score * (1 + stage_bonus))

        return results

    def _calculate_evidence_weighted_score(self, evidence_data: Dict[str, Any]) -> float:
        """计算证据加权分数"""

        if not evidence_data["evidence"]:
            return evidence_data["base_score"]

        # 策略权重
        strategy_weights = {
            RAGStrategy.LIGHT_RAG.value: 0.6,
            RAGStrategy.GRAPH_RAG.value: 0.8,
            RAGStrategy.AGENTIC_RAG.value: 0.9,
            RAGStrategy.HYBRID_RAG.value: 0.7
        }

        # 计算加权平均分数
        total_weight = 0
        weighted_sum = 0

        for evidence in evidence_data["evidence"]:
            strategy = evidence["strategy"]
            weight = strategy_weights.get(strategy, 0.5)
            score = evidence["score"]
            confidence = evidence.get("confidence", 0.5)

            # 综合权重 = 策略权重 × 置信度
            final_weight = weight * confidence
            total_weight += final_weight
            weighted_sum += score * final_weight

        if total_weight == 0:
            return evidence_data["base_score"]

        return weighted_sum / total_weight

    def _calculate_fusion_confidence(
        self,
        strategy_results: List[StrategyResult],
        fused_results: List[Dict[str, Any]]
    ) -> float:
        """计算融合置信度"""

        if not strategy_results or not fused_results:
            return 0.0

        # 基于策略置信度的平均值
        avg_strategy_confidence = sum(r.confidence for r in strategy_results) / len(strategy_results)

        # 基于融合结果的质量
        if fused_results:
            avg_result_score = sum(r.get("fused_score", r.get("score", 0)) for r in fused_results) / len(fused_results)
        else:
            avg_result_score = 0.0

        # 基于策略多样性
        strategy_diversity = len(set(r.strategy for r in strategy_results)) / 4.0  # 最多4种策略

        # 综合置信度
        fusion_confidence = (
            avg_strategy_confidence * 0.4 +
            avg_result_score * 0.4 +
            strategy_diversity * 0.2
        )

        return max(0.0, min(1.0, fusion_confidence))


class MixedExecutionEngine:
    """混合执行引擎"""

    def __init__(self, config: FusionConfig = None):
        self.decomposer = StrategyDecomposer()
        self.fusion_engine = ResultFusionEngine(config)
        self.execution_history: List[Dict[str, Any]] = []

    async def execute_mixed(
        self,
        query: str,
        features: QueryFeatures,
        mode: ExecutionMode = ExecutionMode.ADAPTIVE,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """执行混合模式"""

        start_time = time.time()

        try:
            # 1. 策略分解
            execution_context = await self.decomposer.decompose_query(query, features, mode)

            # 2. 策略执行
            if mode == ExecutionMode.PARALLEL_FUSION:
                strategy_results = await self._execute_parallel(execution_context)
            elif mode == ExecutionMode.PIPELINE:
                strategy_results = await self._execute_pipeline(execution_context)
            else:
                strategy_results = await self._execute_adaptive(execution_context)

            # 3. 结果融合
            fusion_result = await self.fusion_engine.fuse_results(strategy_results, execution_context)

            # 4. 记录执行历史
            execution_time = time.time() - start_time
            self._record_execution(query, mode, execution_context, strategy_results, fusion_result, execution_time)

            return {
                "query": query,
                "execution_mode": mode.value,
                "execution_context": {
                    "strategies": [s.value for s in execution_context.strategies],
                    "decomposition_time": execution_context.context.get("decomposition_time")
                },
                "strategy_results": [
                    {
                        "strategy": r.strategy.value,
                        "status": r.status,
                        "result_count": len(r.results),
                        "execution_time": r.execution_time,
                        "confidence": r.confidence
                    }
                    for r in strategy_results
                ],
                "fused_results": fusion_result["fused_results"],
                "fusion_confidence": fusion_result["fusion_confidence"],
                "fusion_metadata": fusion_result["fusion_metadata"],
                "total_execution_time": execution_time,
                "success": True
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Mixed execution failed: {e}")

            return {
                "query": query,
                "execution_mode": mode.value,
                "error": str(e),
                "total_execution_time": execution_time,
                "success": False
            }

    async def _execute_parallel(self, context: ExecutionContext) -> List[StrategyResult]:
        """并行执行策略"""

        tasks = []
        for strategy in context.strategies:
            task = self._execute_single_strategy(strategy, context)
            tasks.append(task)

        # 并行执行，带超时控制
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=context.timeout
        )

        # 处理结果
        strategy_results = []
        for i, result in enumerate(results):
            strategy = context.strategies[i]

            if isinstance(result, Exception):
                strategy_results.append(StrategyResult(
                    strategy=strategy,
                    results=[],
                    execution_time=0.0,
                    confidence=0.0,
                    metadata={"error": str(result)},
                    status="error",
                    error_message=str(result)
                ))
            else:
                strategy_results.append(result)

        return strategy_results

    async def _execute_pipeline(self, context: ExecutionContext) -> List[StrategyResult]:
        """流水线执行策略"""

        strategy_results = []
        accumulated_context = context.context.copy()

        for strategy in context.strategies:
            # 更新上下文，包含前面策略的结果
            current_context = {
                **accumulated_context,
                "previous_results": [r.results for r in strategy_results],
                "stage": len(strategy_results) + 1
            }

            result = await self._execute_single_strategy(strategy, context, current_context)
            strategy_results.append(result)

            # 如果策略失败，决定是否继续
            if result.status == "error" and len(strategy_results) > 1:
                logger.warning(f"Pipeline strategy {strategy.value} failed, continuing with next stage")
                continue

            # 累积上下文
            accumulated_context.update({
                f"{strategy.value}_results": result.results,
                f"{strategy.value}_confidence": result.confidence
            })

        return strategy_results

    async def _execute_adaptive(self, context: ExecutionContext) -> List[StrategyResult]:
        """自适应执行策略"""

        # 先执行第一个策略
        if not context.strategies:
            return []

        first_strategy = context.strategies[0]
        first_result = await self._execute_single_strategy(first_strategy, context)

        strategy_results = [first_result]

        # 基于第一个策略的结果决定后续策略
        if first_result.status == "success" and first_result.confidence >= 0.8:
            # 第一个策略效果很好，不需要更多策略
            return strategy_results

        # 需要更多策略补充
        for strategy in context.strategies[1:]:
            # 检查是否已经达到预期效果
            if self._check_sufficient_results(strategy_results):
                break

            result = await self._execute_single_strategy(strategy, context)
            strategy_results.append(result)

            # 如果这个策略成功且置信度高，可以停止
            if result.status == "success" and result.confidence >= 0.85:
                break

        return strategy_results

    async def _execute_single_strategy(
        self,
        strategy: RAGStrategy,
        context: ExecutionContext,
        additional_context: Dict[str, Any] = None
    ) -> StrategyResult:
        """执行单个策略"""

        start_time = time.time()

        try:
            # 获取策略执行器
            executor = strategy_manager.get_executor(strategy)
            if not executor:
                return StrategyResult(
                    strategy=strategy,
                    results=[],
                    execution_time=0.0,
                    confidence=0.0,
                    metadata={"error": "Executor not found"},
                    status="error",
                    error_message="Executor not found"
                )

            # 合并上下文
            merged_context = {
                **context.context,
                **(additional_context or {})
            }

            # 执行策略
            result = await executor.execute(
                context.query,
                context.features,
                merged_context
            )

            execution_time = time.time() - start_time

            return StrategyResult(
                strategy=strategy,
                results=result.get("results", []),
                execution_time=execution_time,
                confidence=result.get("confidence", 0.0),
                metadata=result.get("metadata", {}),
                status="success"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Strategy {strategy.value} execution failed: {e}")

            return StrategyResult(
                strategy=strategy,
                results=[],
                execution_time=execution_time,
                confidence=0.0,
                metadata={"error": str(e)},
                status="error",
                error_message=str(e)
            )

    def _check_sufficient_results(self, strategy_results: List[StrategyResult]) -> bool:
        """检查是否已经有足够的结果"""

        # 计算总结果数量
        total_results = sum(len(r.results) for r in strategy_results if r.status == "success")

        # 计算平均置信度
        successful_results = [r for r in strategy_results if r.status == "success"]
        if successful_results:
            avg_confidence = sum(r.confidence for r in successful_results) / len(successful_results)
        else:
            avg_confidence = 0.0

        # 判断是否足够
        return (
            total_results >= 10 or  # 至少10个结果
            avg_confidence >= 0.85 or  # 平均置信度很高
            len(successful_results) >= 2  # 至少2个成功策略
        )

    def _record_execution(
        self,
        query: str,
        mode: ExecutionMode,
        context: ExecutionContext,
        strategy_results: List[StrategyResult],
        fusion_result: Dict[str, Any],
        execution_time: float
    ) -> None:
        """记录执行历史"""

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "mode": mode.value,
            "strategies": [s.value for s in context.strategies],
            "strategy_count": len(context.strategies),
            "successful_strategies": len([r for r in strategy_results if r.status == "success"]),
            "total_results": fusion_result.get("fused_results", []),
            "fusion_confidence": fusion_result.get("fusion_confidence", 0.0),
            "execution_time": execution_time,
            "success": True
        }

        self.execution_history.append(record)

        # 保持历史记录在合理范围内
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-800:]

    async def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""

        if not self.execution_history:
            return {"total_executions": 0}

        total_executions = len(self.execution_history)
        successful_executions = len([r for r in self.execution_history if r["success"]])

        # 按模式统计
        mode_stats = {}
        for record in self.execution_history:
            mode = record["mode"]
            if mode not in mode_stats:
                mode_stats[mode] = {"count": 0, "success": 0, "avg_time": 0}

            mode_stats[mode]["count"] += 1
            if record["success"]:
                mode_stats[mode]["success"] += 1
                mode_stats[mode]["avg_time"] += record["execution_time"]

        # 计算平均值
        for mode_stats_item in mode_stats.values():
            if mode_stats_item["count"] > 0:
                mode_stats_item["success_rate"] = mode_stats_item["success"] / mode_stats_item["count"]
                mode_stats_item["avg_time"] /= mode_stats_item["success"] if mode_stats_item["success"] > 0 else 1

        return {
            "total_executions": total_executions,
            "success_rate": successful_executions / total_executions,
            "mode_statistics": mode_stats,
            "avg_fusion_confidence": sum(r["fusion_confidence"] for r in self.execution_history) / total_executions
        }


# 全局混合执行引擎实例
mixed_execution_engine = MixedExecutionEngine()