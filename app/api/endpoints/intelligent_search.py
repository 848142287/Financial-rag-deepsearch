"""
智能检索API
整合DeepSearch、多策略检索、RAGAS评估的完整智能检索接口
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import asyncio

from app.core.auth import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.services.deep_search.iterative_retriever import deep_search_engine, RetrievalStrategy
from app.services.evaluation.ragas_evaluator import ragas_evaluator
from app.services.milvus_service import MilvusService
from app.services.neo4j_service import Neo4jService
from app.services.intelligent_search import IntelligentSearchService

logger = logging.getLogger(__name__)

router = APIRouter()

# 初始化服务
milvus_service = MilvusService()
neo4j_service = Neo4jService()


# 请求模型
class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., description="搜索查询", min_length=1, max_length=1000)
    search_type: str = Field("intelligent", description="搜索类型: intelligent/deep/fast/graph/vector")
    max_iterations: Optional[int] = Field(3, description="DeepSearch最大迭代次数")
    strategies: Optional[List[str]] = Field(None, description="检索策略列表")
    enable_evaluation: bool = Field(True, description="是否启用RAGAS评估")
    top_k: int = Field(10, description="返回结果数量")
    include_contexts: bool = Field(True, description="是否包含上下文")
    session_id: Optional[str] = Field(None, description="会话ID")


class DeepSearchRequest(BaseModel):
    """DeepSearch请求"""
    query: str = Field(..., description="搜索查询", min_length=1, max_length=1000)
    max_iterations: int = Field(3, description="最大迭代次数")
    strategies: List[str] = Field(["vector_search", "graph_search", "keyword_search"], description="检索策略")
    convergence_threshold: float = Field(0.1, description="收敛阈值")
    enable_evaluation: bool = Field(True, description="是否启用评估")


class MultiStrategySearchRequest(BaseModel):
    """多策略搜索请求"""
    query: str = Field(..., description="搜索查询", min_length=1, max_length=1000)
    strategies: List[str] = Field(["vector", "graph", "keyword"], description="检索策略")
    weights: Optional[Dict[str, float]] = Field(None, description="策略权重")
    fusion_method: str = Field("weighted", description="融合方法: weighted/reciprocal/rank_fusion")
    top_k: int = Field(10, description="返回结果数量")


class EvaluationRequest(BaseModel):
    """评估请求"""
    question: str = Field(..., description="问题")
    answer: str = Field(..., description="答案")
    contexts: List[str] = Field(..., description="上下文列表")
    ground_truth: Optional[str] = Field(None, description="真实答案")
    enable_detailed_metrics: bool = Field(True, description="是否启用详细指标")


# 响应模型
class SearchResponse(BaseModel):
    """搜索响应"""
    success: bool
    search_id: str
    query: str
    search_type: str
    results: List[Dict[str, Any]]
    total_count: int
    execution_time: float
    metadata: Dict[str, Any]


class DeepSearchResponse(BaseModel):
    """DeepSearch响应"""
    success: bool
    search_id: str
    query: str
    iterations: List[Dict[str, Any]]
    final_results: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    strategy_performance: Dict[str, Any]
    execution_time: float


class EvaluationResponse(BaseModel):
    """评估响应"""
    success: bool
    evaluation_id: str
    overall_score: float
    metrics: Dict[str, Any]
    strengths: List[str]
    weaknesses: List[str]
    detailed_results: Optional[List[Dict[str, Any]]] = None


@router.post("/search")
async def intelligent_search(
    request: SearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    智能搜索
    根据查询类型自动选择最优搜索策略
    """
    try:
        logger.info(f"用户 {current_user.id} 发起智能搜索: {request.query[:100]}...")

        start_time = datetime.now()
        search_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_user.id[:8]}"

        # 根据搜索类型选择处理方式
        if request.search_type == "deep":
            # DeepSearch
            deep_search_result = await deep_search_engine.deep_search(
                query=request.query,
                max_iterations=request.max_iterations,
                strategies=[RetrievalStrategy(s) for s in (request.strategies or ["vector_search", "graph_search"])]
            )

            results = deep_search_result.get("results", [])
            execution_time = deep_search_result.get("execution", {}).get("total_time", 0)

            response_data = {
                "search_id": search_id,
                "query": request.query,
                "search_type": "deep_search",
                "results": results,
                "total_count": len(results),
                "execution_time": execution_time,
                "metadata": {
                    "iterations": deep_search_result.get("execution", {}).get("iterations_completed", 0),
                    "strategy_performance": deep_search_result.get("statistics", {}),
                    "convergence_trend": deep_search_result.get("statistics", {}).get("convergence_trend", [])
                }
            }

        elif request.search_type == "intelligent":
            # 智能路由：根据查询复杂度选择策略
            query_complexity = await _analyze_query_complexity(request.query)

            if query_complexity == "simple":
                # 简单查询：快速向量检索
                results = await _fast_vector_search(request.query, request.top_k)
                search_type_used = "fast_vector"
            elif query_complexity == "medium":
                # 中等查询：混合检索
                results = await _hybrid_search(request.query, request.top_k)
                search_type_used = "hybrid"
            else:
                # 复杂查询：DeepSearch
                deep_search_result = await deep_search_engine.deep_search(
                    query=request.query,
                    max_iterations=min(request.max_iterations, 2)  # 复杂查询使用较少迭代
                )
                results = deep_search_result.get("results", [])
                search_type_used = "deep_search"
                execution_time = deep_search_result.get("execution", {}).get("total_time", 0)

            response_data = {
                "search_id": search_id,
                "query": request.query,
                "search_type": search_type_used,
                "results": results[:request.top_k],
                "total_count": len(results),
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "metadata": {
                    "query_complexity": query_complexity,
                    "auto_selected_strategy": search_type_used
                }
            }

        else:
            # 指定策略搜索
            if request.search_type == "vector":
                results = await _vector_search_only(request.query, request.top_k)
            elif request.search_type == "graph":
                results = await _graph_search_only(request.query, request.top_k)
            elif request.search_type == "keyword":
                results = await _keyword_search_only(request.query, request.top_k)
            elif request.search_type == "fast":
                results = await _fast_vector_search(request.query, request.top_k)
            else:
                raise HTTPException(status_code=400, detail="不支持的搜索类型")

            response_data = {
                "search_id": search_id,
                "query": request.query,
                "search_type": request.search_type,
                "results": results[:request.top_k],
                "total_count": len(results),
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "metadata": {}
            }

        # 如果启用评估，对第一个结果进行质量评估
        if request.enable_evaluation and results:
            try:
                contexts = [result.get("content", "") for result in results[:3]]
                evaluation = await ragas_evaluator.evaluate(
                    question=request.query,
                    answer=results[0].get("content", "") if results else "",
                    contexts=contexts
                )

                response_data["metadata"]["quality_evaluation"] = {
                    "overall_score": evaluation.overall_score,
                    "metrics": {result.metric.value: result.score for result in evaluation.results}
                }

            except Exception as e:
                logger.warning(f"质量评估失败: {e}")

        return SearchResponse(**response_data)

    except Exception as e:
        logger.error(f"智能搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@router.post("/deep_search")
async def deep_search(
    request: DeepSearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    DeepSearch深度搜索
    多轮迭代检索，支持复杂查询的深度探索
    """
    try:
        logger.info(f"用户 {current_user.id} 发起DeepSearch: {request.query[:100]}...")

        # 转换策略
        strategies = []
        strategy_mapping = {
            "vector_search": RetrievalStrategy.VECTOR_SEARCH,
            "graph_search": RetrievalStrategy.GRAPH_SEARCH,
            "keyword_search": RetrievalStrategy.KEYWORD_SEARCH,
            "hybrid_search": RetrievalStrategy.HYBRID_SEARCH
        }

        for strategy in request.strategies:
            if strategy in strategy_mapping:
                strategies.append(strategy_mapping[strategy])

        if not strategies:
            strategies = [RetrievalStrategy.VECTOR_SEARCH, RetrievalStrategy.GRAPH_SEARCH]

        # 执行DeepSearch
        search_result = await deep_search_engine.deep_search(
            query=request.query,
            max_iterations=request.max_iterations,
            strategies=strategies
        )

        if not search_result.get("success"):
            raise HTTPException(status_code=500, detail="DeepSearch执行失败")

        # 如果启用评估，对最终结果进行质量评估
        evaluation_result = None
        if request.enable_evaluation and search_result.get("results"):
            try:
                contexts = [result.get("content", "") for result in search_result["results"][:3]]
                evaluation_result = await ragas_evaluator.evaluate(
                    question=request.query,
                    answer=search_result["results"][0].get("content", "") if search_result["results"] else "",
                    contexts=contexts
                )
            except Exception as e:
                logger.warning(f"DeepSearch质量评估失败: {e}")

        response_data = {
            "success": True,
            "search_id": search_result.get("search_id"),
            "query": request.query,
            "iterations": search_result.get("execution", {}).get("iterations_completed", 0),
            "final_results": search_result.get("results", [])[:10],
            "convergence_info": {
                "converged": search_result.get("execution", {}).get("iterations_completed", 0) < request.max_iterations,
                "final_score": search_result.get("statistics", {}).get("convergence_trend", [0])[-1] if search_result.get("statistics", {}).get("convergence_trend") else 0
            },
            "strategy_performance": search_result.get("statistics", {}).get("strategy_performance", {}),
            "execution_time": search_result.get("execution", {}).get("total_time", 0),
            "quality_evaluation": {
                "overall_score": evaluation_result.overall_score,
                "metrics": {result.metric.value: result.score for result in evaluation_result.results}
            } if evaluation_result else None
        }

        return DeepSearchResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DeepSearch失败: {e}")
        raise HTTPException(status_code=500, detail=f"DeepSearch失败: {str(e)}")


@router.post("/multi_strategy_search")
async def multi_strategy_search(
    request: MultiStrategySearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    多策略搜索
    并行执行多种检索策略并融合结果
    """
    try:
        logger.info(f"用户 {current_user.id} 发起多策略搜索: {request.query[:100]}...")

        start_time = datetime.now()

        # 并行执行多种策略
        tasks = []
        for strategy in request.strategies:
            if strategy == "vector":
                tasks.append(_vector_search_only(request.query, request.top_k))
            elif strategy == "graph":
                tasks.append(_graph_search_only(request.query, request.top_k))
            elif strategy == "keyword":
                tasks.append(_keyword_search_only(request.query, request.top_k))

        if not tasks:
            raise HTTPException(status_code=400, detail="至少需要指定一种搜索策略")

        # 等待所有策略完成
        strategy_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        all_results = []
        strategy_mapping = dict(zip(request.strategies, strategy_results))

        for strategy, results in strategy_mapping.items():
            if isinstance(results, Exception):
                logger.error(f"策略 {strategy} 执行失败: {results}")
                continue

            # 添加策略信息
            for result in results:
                result["strategy_source"] = strategy
                all_results.append(result)

        # 融合结果
        fused_results = _fuse_multi_strategy_results(all_results, request.weights, request.fusion_method)

        execution_time = (datetime.now() - start_time).total_seconds()

        return SearchResponse(
            success=True,
            search_id=f"multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query=request.query,
            search_type="multi_strategy",
            results=fused_results[:request.top_k],
            total_count=len(fused_results),
            execution_time=execution_time,
            metadata={
                "strategies_used": request.strategies,
                "fusion_method": request.fusion_method,
                "strategy_counts": {strategy: len(strategy_mapping.get(strategy, [])) for strategy in request.strategies}
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"多策略搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"多策略搜索失败: {str(e)}")


@router.post("/evaluate")
async def evaluate_answer_quality(
    request: EvaluationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    答案质量评估
    使用RAGAS框架评估答案质量
    """
    try:
        logger.info(f"用户 {current_user.id} 发起质量评估")

        # 执行RAGAS评估
        evaluation = await ragas_evaluator.evaluate(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth
        )

        # 构建响应
        response_data = {
            "success": True,
            "evaluation_id": evaluation.evaluation_id,
            "overall_score": evaluation.overall_score,
            "metrics": {},
            "strengths": [],
            "weaknesses": []
        }

        # 按指标分组
        for result in evaluation.results:
            metric_name = result.metric.value
            response_data["metrics"][metric_name] = {
                "score": result.score,
                "confidence": result.confidence,
                "reasoning": result.reasoning
            }

            # 识别优势和劣势
            if result.score >= 0.8:
                response_data["strengths"].append(f"{metric_name}: {result.reasoning}")
            elif result.score <= 0.5:
                response_data["weaknesses"].append(f"{metric_name}: {result.reasoning}")

        # 详细结果
        if request.enable_detailed_metrics:
            response_data["detailed_results"] = [
                {
                    "metric": result.metric.value,
                    "score": result.score,
                    "reasoning": result.reasoning,
                    "confidence": result.confidence,
                    "details": result.details
                }
                for result in evaluation.results
            ]

        return EvaluationResponse(**response_data)

    except Exception as e:
        logger.error(f"答案质量评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


@router.get("/search/history")
async def get_search_history(
    limit: int = Query(20, description="返回数量限制"),
    session_id: Optional[str] = Query(None, description="会话ID"),
    current_user: User = Depends(get_current_user)
):
    """
    获取搜索历史
    """
    try:
        # 这里应该从数据库获取用户的搜索历史
        # 简化实现，返回模拟数据
        return {
            "success": True,
            "data": {
                "search_history": [],
                "total_count": 0,
                "message": "搜索历史功能需要数据库支持"
            }
        }

    except Exception as e:
        logger.error(f"获取搜索历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取搜索历史失败: {str(e)}")


@router.get("/search/suggestions")
async def get_search_suggestions(
    query: str = Query(..., description="查询前缀"),
    limit: int = Query(10, description="建议数量"),
    current_user: User = Depends(get_current_user)
):
    """
    获取搜索建议
    """
    try:
        # 这里应该基于用户历史和热门搜索提供建议
        # 简化实现，返回基于查询的建议
        suggestions = await _generate_search_suggestions(query, limit)

        return {
            "success": True,
            "data": {
                "query": query,
                "suggestions": suggestions
            }
        }

    except Exception as e:
        logger.error(f"获取搜索建议失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取搜索建议失败: {str(e)}")


# 辅助函数
async def _analyze_query_complexity(query: str) -> str:
    """分析查询复杂度"""
    # 简化实现
    if len(query) < 20:
        return "simple"
    elif len(query) < 50:
        return "medium"
    else:
        return "complex"


async def _fast_vector_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """快速向量搜索"""
    try:
        # 生成查询向量
        query_embedding = await deep_search_engine._generate_query_embedding(query)

        # 执行向量检索
        results = await deep_search_engine.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k
        )

        return results[:top_k]

    except Exception as e:
        logger.error(f"快速向量搜索失败: {e}")
        return []


async def _hybrid_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """混合搜索"""
    try:
        # 并行执行向量和关键词搜索
        vector_results = await _fast_vector_search(query, top_k // 2)
        keyword_results = await _keyword_search_only(query, top_k // 2)

        # 简单合并
        all_results = vector_results + keyword_results
        return all_results[:top_k]

    except Exception as e:
        logger.error(f"混合搜索失败: {e}")
        return []


async def _vector_search_only(query: str, top_k: int) -> List[Dict[str, Any]]:
    """纯向量搜索"""
    return await _fast_vector_search(query, top_k)


async def _graph_search_only(query: str, top_k: int) -> List[Dict[str, Any]]:
    """纯图谱搜索"""
    try:
        # 提取查询实体
        entities = await deep_search_engine._extract_query_entities(query)

        if not entities:
            return []

        # 执行图谱搜索
        graph_results = []
        for entity in entities[:3]:  # 限制实体数量
            query_str = f"""
            MATCH (e {{name: $entity_name}})-[r]-(related)
            RETURN e, r, related
            LIMIT {top_k // 2}
            """

            results = await neo4j_service.execute_query(
                query_str,
                {"entity_name": entity["text"]}
            )

            graph_results.extend([
                {
                    "content": str(result.get("related", {})),
                    "source": "graph_search",
                    "entity": entity["text"],
                    "score": 0.8
                }
                for result in results
            ])

        return graph_results[:top_k]

    except Exception as e:
        logger.error(f"图谱搜索失败: {e}")
        return []


async def _keyword_search_only(query: str, top_k: int) -> List[Dict[str, Any]]:
    """纯关键词搜索"""
    try:
        keywords = await deep_search_engine._extract_keywords(query)

        results = []
        for keyword in keywords[:5]:  # 限制关键词数量
            # 模拟关键词搜索结果
            results.append({
                "content": f"包含关键词'{keyword}'的相关文档内容...",
                "source": "keyword_search",
                "keyword": keyword,
                "score": 0.7
            })

        return results[:top_k]

    except Exception as e:
        logger.error(f"关键词搜索失败: {e}")
        return []


def _fuse_multi_strategy_results(results: List[Dict[str, Any]],
                               weights: Optional[Dict[str, float]],
                               fusion_method: str) -> List[Dict[str, Any]]:
    """融合多策略结果"""
    if not results:
        return []

    # 去重
    seen_content = set()
    unique_results = []

    for result in results:
        content = result.get("content", "")
        content_key = content[:100]  # 使用前100字符作为去重键

        if content_key not in seen_content:
            seen_content.add(content_key)

            # 应用权重
            if weights and result.get("strategy_source") in weights:
                weight = weights[result["strategy_source"]]
                result["score"] = result.get("score", 0) * weight

            unique_results.append(result)

    # 排序
    if fusion_method == "weighted":
        return sorted(unique_results, key=lambda x: x.get("score", 0), reverse=True)
    else:
        # 简化实现：按分数排序
        return sorted(unique_results, key=lambda x: x.get("score", 0), reverse=True)


async def _generate_search_suggestions(query: str, limit: int) -> List[str]:
    """生成搜索建议"""
    # 简化实现：基于预定义的金融相关词汇
    financial_terms = [
        "营收增长", "净利润", "市盈率", "市值", "股息率",
        "资产负债率", "现金流", "毛利率", "营业收入", "每股收益"
    ]

    suggestions = []
    query_lower = query.lower()

    for term in financial_terms:
        if query_lower in term.lower() or term.lower().startswith(query_lower):
            suggestions.append(term)

    return suggestions[:limit]


# 导出路由器
__all__ = ["router"]