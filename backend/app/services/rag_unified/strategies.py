"""
RAG策略定义与管理
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class RAGStrategy(Enum):
    """RAG策略类型"""
    LIGHT_RAG = "light_rag"           # LightRAG: 轻量级向量检索
    GRAPH_RAG = "graph_rag"           # Graph RAG: 知识图谱检索
    AGENTIC_RAG = "agentic_rag"       # Agentic RAG: 智能体驱动检索
    HYBRID_RAG = "hybrid_rag"          # Hybrid RAG: 混合策略


@dataclass
class StrategyConfig:
    """策略配置"""
    strategy: RAGStrategy
    name: str
    description: str
    weight: float = 1.0
    enabled: bool = True
    priority: int = 1  # 优先级，数字越大优先级越高
    max_response_time: float = 5000.0  # 最大响应时间（毫秒）
    min_confidence: float = 0.6  # 最小置信度
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)


@dataclass
class QueryFeatures:
    """查询特征"""
    query: str
    entity_count: int = 0
    relation_complexity: int = 0
    time_sensitivity: int = 0
    answer_granularity: int = 0
    query_length: int = 0
    question_type: str = ""
    domain: str = ""
    complexity_score: float = 0.0
    extracted_entities: List[str] = field(default_factory=list)
    relation_words: List[str] = field(default_factory=list)
    time_indicators: List[str] = field(default_factory=list)
    complexity_indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """路由决策"""
    selected_strategy: RAGStrategy
    confidence: float
    reasoning: str
    alternative_strategies: List[Tuple[RAGStrategy, float]]
    fallback_strategy: Optional[RAGStrategy] = None
    expected_response_time: float
    resource_estimate: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGStrategyExecutor(ABC):
    """RAG策略执行器抽象基类"""

    def __init__(self, config: StrategyConfig):
        self.config = config

    @abstractmethod
    async def execute(self, query: str, features: QueryFeatures, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行检索策略"""
        pass

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        pass


class LightRAGExecutor(RAGStrategyExecutor):
    """LightRAG执行器"""

    async def execute(self, query: str, features: QueryFeatures, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行轻量级RAG检索
        主要基于向量相似度快速匹配
        """
        logger.info(f"Executing LightRAG for query: {query[:50]}...")

        start_time = asyncio.get_event_loop().time()

        try:
            # 1. 向量检索
            from app.services.milvus_service import MilvusService
            milvus_service = MilvusService()

            # 简化的向量检索，返回top-k结果
            vector_results = await milvus_service.search_vectors(
                query_text=query,
                top_k=5,  # LightRAG使用较小的k值
                filter_expression=None
            )

            # 2. 轻量级结果处理
            processed_results = []
            for result in vector_results:
                processed_results.append({
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "metadata": {
                        "source": "vector_search",
                        "strategy": "light",
                        "entity_matches": len(set(features.extracted_entities) & set(result.get("entities", [])))
                    }
                })

            # 3. 快速排序和过滤
            processed_results.sort(key=lambda x: x["score"], reverse=True)

            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return {
                "strategy": "light",
                "results": processed_results,
                "execution_time_ms": execution_time,
                "result_count": len(processed_results),
                "confidence": self._calculate_confidence(processed_results),
                "metadata": {
                    "vector_search_time": execution_time,
                    "top_k": 5
                }
            }

        except Exception as e:
            logger.error(f"LightRAG execution failed: {str(e)}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "avg_response_time": 2500.0,  # ms
            "accuracy": 0.75,
            "coverage": 0.85,
            "resource_usage": "low"
        }

    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """计算置信度"""
        if not results:
            return 0.0

        # 基于得分分布计算置信度
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores)
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)

        # 高平均分且低方差表示高置信度
        confidence = avg_score * (1 - min(score_variance, 1.0))
        return max(0.0, min(1.0, confidence))


class GraphRAGExecutor(RAGStrategyExecutor):
    """Graph RAG执行器"""

    async def execute(self, query: str, features: QueryFeatures, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行图谱RAG检索
        基于知识图谱的关联检索
        """
        logger.info(f"Executing GraphRAG for query: {query[:50]}...")

        start_time = asyncio.get_event_loop().time()

        try:
            # 1. 实体识别和关系提取
            entities = features.extracted_entities
            relation_words = features.relation_words

            # 2. 图谱检索
            from app.services.neo4j_service import Neo4jService
            neo4j_service = Neo4jService()

            # 构建Cypher查询
            graph_results = await neo4j_service.search_graph(
                entities=entities,
                relation_words=relation_words,
                max_depth=2,
                top_k=8
            )

            # 3. 补充向量检索
            if len(graph_results) < 5:
                from app.services.milvus_service import MilvusService
                milvus_service = MilvusService()
                vector_results = await milvus_service.search_vectors(
                    query_text=query,
                    top_k=10 - len(graph_results),
                    filter_expression=None
                )
                graph_results.extend(vector_results)

            # 4. 结果处理和融合
            processed_results = []
            for result in graph_results:
                processed_results.append({
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "metadata": {
                        "source": "graph_search",
                        "strategy": "graph",
                        "entities": result.get("entities", []),
                        "relationships": result.get("relationships", []),
                        "graph_depth": result.get("depth", 0)
                    }
                })

            # 5. 基于关系相关性排序
            processed_results = self._rank_by_relevance(processed_results, entities, relation_words)

            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return {
                "strategy": "graph",
                "results": processed_results,
                "execution_time_ms": execution_time,
                "result_count": len(processed_results),
                "confidence": self._calculate_confidence(processed_results, entities),
                "metadata": {
                    "graph_search_time": execution_time * 0.6,
                    "vector_search_time": execution_time * 0.4,
                    "entity_count": len(entities),
                    "relation_count": len(relation_words)
                }
            }

        except Exception as e:
            logger.error(f"GraphRAG execution failed: {str(e)}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "avg_response_time": 5000.0,  # ms
            "accuracy": 0.85,
            "coverage": 0.90,
            "resource_usage": "medium"
        }

    def _rank_by_relevance(self, results: List[Dict[str, Any]],
                         entities: List[str], relation_words: List[str]) -> List[Dict[str, Any]]:
        """基于关系相关性排序"""
        for result in results:
            relevance_score = result["score"]

            # 实体匹配加分
            result_entities = result["metadata"].get("entities", [])
            entity_match = len(set(entities) & set(result_entities))
            relevance_score += entity_match * 0.2

            # 关系词匹配加分
            content = result.get("content", "").lower()
            relation_match = sum(1 for word in relation_words if word in content)
            relevance_score += relation_match * 0.1

            # 图深度加分
            depth = result["metadata"].get("graph_depth", 0)
            relevance_score += min(depth, 3) * 0.05

            result["adjusted_score"] = relevance_score

        return sorted(results, key=lambda x: x["adjusted_score"], reverse=True)

    def _calculate_confidence(self, results: List[Dict[str, Any]], entities: List[str]) -> float:
        """计算置信度"""
        if not results:
            return 0.0

        # 图谱匹配度
        graph_matches = sum(1 for r in results if r["metadata"].get("entities"))
        entity_coverage = graph_matches / len(entities) if entities else 0

        # 关系覆盖度
        relation_coverage = len([r for r in results if r["metadata"].get("relationships")]) / max(len(results), 1)

        # 综合置信度
        confidence = (entity_coverage * 0.6 + relation_coverage * 0.4)
        return max(0.0, min(1.0, confidence))


class AgenticRAGExecutor(RAGStrategyExecutor):
    """Agentic RAG执行器"""

    async def execute(self, query: str, features: QueryFeatures, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行智能体驱动RAG检索
        多轮迭代优化，复杂推理
        """
        logger.info(f"Executing AgenticRAG for query: {query[:50]}...")

        start_time = asyncio.get_event_loop().time()

        try:
            # 1. 查询分析和规划
            from app.services.agentic_rag.plan_phase import PlanPhase
            from app.services.agentic_rag.execute_phase import ExecutePhase
            from app.services.agentic_rag.generation_phase import GenerationPhase

            plan_phase = PlanPhase()
            execute_phase = ExecutePhase()
            generation_phase = GenerationPhase()

            # 2. 执行完整的Agentic RAG流程
            query_plan = await plan_phase.process_query(query, {})
            search_results = await execute_phase.execute(query_plan, "deep")
            answer = await generation_phase.generate_answer(query, search_results, query_plan)

            # 3. 结果处理
            processed_results = []
            for result in search_results.fused_results:
                processed_results.append({
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "metadata": {
                        "source": result.get("source", "agentic"),
                        "strategy": "agentic",
                        "optimization_rounds": len(search_results.optimization_history or []),
                        "confidence": result.get("confidence", 0.0)
                    }
                })

            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return {
                "strategy": "agentic",
                "results": processed_results,
                "execution_time_ms": execution_time,
                "result_count": len(processed_results),
                "confidence": answer.confidence if hasattr(answer, 'confidence') else 0.8,
                "answer": answer.content if hasattr(answer, 'content') else "",
                "metadata": {
                    "query_plan": query_plan.to_dict() if hasattr(query_plan, 'to_dict') else {},
                    "optimization_history": search_results.optimization_history or [],
                    "generation_time": execution_time * 0.3
                }
            }

        except Exception as e:
            logger.error(f"AgenticRAG execution failed: {str(e)}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "avg_response_time": 10000.0,  # ms
            "accuracy": 0.92,
            "coverage": 0.95,
            "resource_usage": "high"
        }


class HybridRAGExecutor(RAGStrategyExecutor):
    """混合RAG执行器"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # 初始化各种策略执行器
        self.light_executor = LightRAGExecutor(
            StrategyConfig(RAGStrategy.LIGHT_RAG, "LightRAG Component", "Light component")
        )
        self.graph_executor = GraphRAGExecutor(
            StrategyConfig(RAGStrategy.GRAPH_RAG, "GraphRAG Component", "Graph component")
        )
        self.agentic_executor = AgenticRAGExecutor(
            StrategyConfig(RAGStrategy.AGENTIC_RAG, "AgenticRAG Component", "Agentic component")
        )

    async def execute(self, query: str, features: QueryFeatures, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行混合RAG策略
        并行执行多种策略并智能融合结果
        """
        logger.info(f"Executing HybridRAG for query: {query[:50]}...")

        start_time = asyncio.get_event_loop().time()

        try:
            # 1. 并行执行多种策略
            tasks = []

            # 根据查询特征决定启用哪些策略
            if features.complexity_score < 0.3:
                tasks.append(self.light_executor.execute(query, features, context))

            if features.entity_count > 0 or features.relation_complexity > 0:
                tasks.append(self.graph_executor.execute(query, features, context))

            if features.complexity_score > 0.6:
                tasks.append(self.agentic_executor.execute(query, features, context))

            # 如果没有启用任何策略，默认使用LightRAG
            if not tasks:
                tasks.append(self.light_executor.execute(query, features, context))

            # 2. 并行执行
            strategy_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 3. 结果融合
            fused_results = self._fuse_results(strategy_results, features)

            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return {
                "strategy": "hybrid",
                "results": fused_results,
                "execution_time_ms": execution_time,
                "result_count": len(fused_results),
                "confidence": self._calculate_fused_confidence(fused_results, strategy_results),
                "metadata": {
                    "strategy_results": [r for r in strategy_results if isinstance(r, dict)],
                    "fusion_method": "weighted_average",
                    "active_strategies": len([r for r in strategy_results if isinstance(r, dict)])
                }
            }

        except Exception as e:
            logger.error(f"HybridRAG execution failed: {str(e)}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "avg_response_time": 7500.0,  # ms
            "accuracy": 0.88,
            "coverage": 0.93,
            "resource_usage": "medium"
        }

    def _fuse_results(self, strategy_results: List[Any], features: QueryFeatures) -> List[Dict[str, Any]]:
        """融合多种策略的结果"""
        all_results = []
        strategy_weights = {}

        # 收集所有有效结果
        for i, result in enumerate(strategy_results):
            if isinstance(result, dict) and "results" in result:
                strategy_name = result["strategy"]
                all_results.extend(result["results"])

                # 计算策略权重
                strategy_weights[strategy_name] = self._calculate_strategy_weight(
                    strategy_name, features, result
                )

        # 去重和重新评分
        seen_ids = set()
        unique_results = []

        for result in all_results:
            result_id = result.get("id", "")
            if result_id not in seen_ids:
                seen_ids.add(result_id)

                # 调整得分
                adjusted_score = self._adjust_score(result, strategy_weights)
                result["fused_score"] = adjusted_score
                unique_results.append(result)

        # 按融合得分排序
        unique_results.sort(key=lambda x: x["fused_score"], reverse=True)

        return unique_results[:15]  # 限制返回数量

    def _calculate_strategy_weight(self, strategy: str, features: QueryFeatures,
                                   result: Dict[str, Any]) -> float:
        """计算策略权重"""
        base_weights = {
            "light": 0.3,
            "graph": 0.4,
            "agentic": 0.5
        }

        weight = base_weights.get(strategy, 0.3)

        # 根据特征调整权重
        if strategy == "light" and features.complexity_score < 0.3:
            weight *= 1.5
        elif strategy == "graph" and features.entity_count > 0:
            weight *= 1.3
        elif strategy == "agentic" and features.complexity_score > 0.6:
            weight *= 1.4

        # 根据结果质量调整权重
        confidence = result.get("confidence", 0.5)
        weight *= (0.5 + confidence)

        return max(0.1, min(1.0, weight))

    def _adjust_score(self, result: Dict[str, Any], strategy_weights: Dict[str, float]) -> float:
        """调整融合得分"""
        base_score = result.get("score", 0.0)
        strategy = result.get("metadata", {}).get("strategy", "")

        # 应用策略权重
        strategy_weight = strategy_weights.get(strategy, 1.0)

        # 应用额外因子
        entity_bonus = result["metadata"].get("entity_matches", 0) * 0.1

        adjusted_score = base_score * strategy_weight + entity_bonus
        return max(0.0, min(1.0, adjusted_score))

    def _calculate_fused_confidence(self, fused_results: List[Dict[str, Any]],
                                   strategy_results: List[Any]) -> float:
        """计算融合结果的置信度"""
        if not fused_results:
            return 0.0

        # 基于融合得分计算置信度
        fused_scores = [r.get("fused_score", 0.0) for r in fused_results]
        avg_score = sum(fused_scores) / len(fused_scores)

        # 策略多样性加分
        strategy_count = len([r for r in strategy_results if isinstance(r, dict)])
        diversity_bonus = min(0.2, strategy_count * 0.05)

        return max(0.0, min(1.0, avg_score + diversity_bonus))


# 策略配置
STRATEGY_CONFIGS = {
    RAGStrategy.LIGHT_RAG: StrategyConfig(
        strategy=RAGStrategy.LIGHT_RAG,
        name="LightRAG",
        description="轻量级向量检索，适合简单事实查询",
        priority=3,
        max_response_time=3000.0,
        min_confidence=0.6,
        performance_targets={
            "p95_response_time": 3000.0,
            "accuracy": 0.75,
            "coverage": 0.85
        }
    ),
    RAGStrategy.GRAPH_RAG: StrategyConfig(
        strategy=RAGStrategy.GRAPH_RAG,
        name="Graph RAG",
        description="知识图谱检索，适合关系分析查询",
        priority=2,
        max_response_time=8000.0,
        min_confidence=0.7,
        performance_targets={
            "p95_response_time": 8000.0,
            "accuracy": 0.85,
            "coverage": 0.90
        }
    ),
    RAGStrategy.AGENTIC_RAG: StrategyConfig(
        strategy=RAGStrategy.AGENTIC_RAG,
        name="Agentic RAG",
        description="智能体驱动检索，适合复杂分析查询",
        priority=1,
        max_response_time=15000.0,
        min_confidence=0.8,
        performance_targets={
            "p95_response_time": 15000.0,
            "accuracy": 0.92,
            "coverage": 0.95
        }
    ),
    RAGStrategy.HYBRID_RAG: StrategyConfig(
        strategy=RAGStrategy.HYBRID_RAG,
        name="Hybrid RAG",
        description="混合策略，智能组合多种方法",
        priority=0,
        max_response_time=12000.0,
        min_confidence=0.75,
        performance_targets={
            "p95_response_time": 7500.0,
            "accuracy": 0.88,
            "coverage": 0.93
        }
    )
}


class RAGStrategyManager:
    """RAG策略管理器"""

    def __init__(self):
        self.executors = {}
        self._initialize_executors()

    def _initialize_executors(self):
        """初始化策略执行器"""
        for strategy, config in STRATEGY_CONFIGS.items():
            if strategy == RAGStrategy.LIGHT_RAG:
                self.executors[strategy] = LightRAGExecutor(config)
            elif strategy == RAGStrategy.GRAPH_RAG:
                self.executors[strategy] = GraphRAGExecutor(config)
            elif strategy == RAGStrategy.AGENTIC_RAG:
                self.executors[strategy] = AgenticRAGExecutor(config)
            elif strategy == RAGStrategy.HYBRID_RAG:
                self.executors[strategy] = HybridRAGExecutor(config)

    def get_executor(self, strategy: RAGStrategy) -> RAGStrategyExecutor:
        """获取策略执行器"""
        return self.executors.get(strategy)

    def get_strategy_config(self, strategy: RAGStrategy) -> StrategyConfig:
        """获取策略配置"""
        return STRATEGY_CONFIGS.get(strategy)

    def get_all_configs(self) -> Dict[RAGStrategy, StrategyConfig]:
        """获取所有策略配置"""
        return STRATEGY_CONFIGS

    def is_strategy_available(self, strategy: RAGStrategy) -> bool:
        """检查策略是否可用"""
        return strategy in self.executors and STRATEGY_CONFIGS[strategy].enabled


# 全局策略管理器实例
strategy_manager = RAGStrategyManager()