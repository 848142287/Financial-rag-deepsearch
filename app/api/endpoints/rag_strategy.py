"""
RAG策略选择与路由API端点
提供策略选择、路由决策和策略执行的API接口
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from app.core.auth import get_current_user
from app.models.user import User
from app.services.rag.strategy_router import (
    RAGStrategyRouter, RAGStrategy, QueryFeatures, RoutingDecision
)
from app.services.rag.strategy_executor import (
    RAGStrategyExecutor, ExecutionContext, ExecutionResult
)

logger = logging.getLogger(__name__)

router = APIRouter()

# 初始化路由器和执行器
strategy_router = RAGStrategyRouter()
strategy_executor = RAGStrategyExecutor()


# 请求模型
class StrategyRoutingRequest(BaseModel):
    """策略路由请求"""
    query: str = Field(..., description="用户查询")
    context: Optional[Dict[str, Any]] = Field(None, description="额外上下文")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")


class StrategyExecutionRequest(BaseModel):
    """策略执行请求"""
    query: str = Field(..., description="用户查询")
    strategy: Optional[str] = Field(None, description="指定策略，不指定则自动选择")
    execution_mode: Optional[str] = Field("single", description="执行模式: single, parallel, pipeline")
    max_tokens: Optional[int] = Field(4000, description="最大生成token数")
    temperature: Optional[float] = Field(0.7, description="生成温度")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    context: Optional[Dict[str, Any]] = Field(None, description="额外上下文")


class FeedbackRequest(BaseModel):
    """反馈请求"""
    query: str = Field(..., description="原始查询")
    strategy_used: str = Field(..., description="使用的策略")
    execution_time: float = Field(..., description="执行时间")
    accuracy: float = Field(..., description="答案准确度 0-1")
    satisfaction: float = Field(..., description="用户满意度 1-5")
    comment: Optional[str] = Field(None, description="反馈评论")
    requires_deeper_analysis: bool = Field(False, description="是否需要更深入分析")


class WeightUpdateRequest(BaseModel):
    """权重更新请求"""
    weights: Dict[str, float] = Field(..., description="路由权重配置")


class ThresholdUpdateRequest(BaseModel):
    """阈值更新请求"""
    thresholds: Dict[str, float] = Field(..., description="策略阈值配置")


@router.post("/routing/analyze")
async def analyze_routing(
    request: StrategyRoutingRequest,
    current_user: User = Depends(get_current_user)
):
    """
    分析查询的路由决策（不执行）
    """
    try:
        logger.info(f"用户 {current_user.id} 请求路由分析: {request.query[:100]}...")

        # 执行路由分析
        routing_decision = await strategy_router.route_query(
            query=request.query,
            context=request.context
        )

        # 格式化返回结果
        response_data = {
            "query": request.query,
            "primary_strategy": routing_decision.primary_strategy.value,
            "backup_strategy": routing_decision.backup_strategy.value if routing_decision.backup_strategy else None,
            "confidence": routing_decision.confidence,
            "execution_mode": routing_decision.execution_mode,
            "reasoning": routing_decision.reasoning,
            "features": {
                "entity_count": routing_decision.features.entity_count,
                "entity_list": routing_decision.features.entity_list,
                "relation_complexity": routing_decision.features.relation_complexity,
                "time_sensitivity": routing_decision.features.time_sensitivity,
                "answer_granularity": routing_decision.features.answer_granularity,
                "query_length": routing_decision.features.query_length,
                "complex_keywords": routing_decision.features.complex_keywords,
                "has_comparison": routing_decision.features.has_comparison,
                "has_trend_analysis": routing_decision.features.has_trend_analysis,
                "has_causality": routing_decision.features.has_causality,
                "has_sentiment": routing_decision.features.has_sentiment
            },
            "strategy_scores": [
                {
                    "strategy": score.strategy.value,
                    "score": score.score,
                    "reasoning": score.reasoning,
                    "estimated_time": score.estimated_time,
                    "resource_level": score.resource_level
                }
                for score in routing_decision.strategy_scores
            ]
        }

        return {
            "success": True,
            "data": response_data
        }

    except Exception as e:
        logger.error(f"路由分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"路由分析失败: {str(e)}")


@router.post("/execute")
async def execute_query(
    request: StrategyExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    执行RAG查询
    """
    try:
        logger.info(f"用户 {current_user.id} 执行查询: {request.query[:100]}...")

        # 路由决策
        if request.strategy:
            # 用户指定策略
            try:
                strategy = RAGStrategy(request.strategy)
                routing_decision = RoutingDecision(
                    primary_strategy=strategy,
                    execution_mode=request.execution_mode
                )
            except ValueError:
                raise HTTPException(status_code=400, detail=f"不支持的策略: {request.strategy}")
        else:
            # 自动路由
            routing_decision = await strategy_router.route_query(
                query=request.query,
                context=request.context
            )
            routing_decision.execution_mode = request.execution_mode

        # 构建执行上下文
        execution_context = ExecutionContext(
            query=request.query,
            strategy=routing_decision.primary_strategy,
            features=routing_decision.features,
            config=request.context or {},
            user_id=request.user_id or str(current_user.id),
            session_id=request.session_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # 执行策略
        execution_result = await strategy_executor.execute(routing_decision, execution_context)

        # 记录执行日志
        background_tasks.add_task(
            log_execution_result,
            current_user.id,
            request.query,
            routing_decision,
            execution_result
        )

        # 格式化返回结果
        response_data = {
            "query": request.query,
            "answer": execution_result.answer,
            "strategy_used": execution_result.strategy_used.value,
            "execution_time": execution_result.execution_time,
            "confidence": execution_result.confidence,
            "execution_mode": routing_decision.execution_mode,
            "reasoning_trace": execution_result.reasoning_trace,
            "retrieval_count": len(execution_result.retrieval_results),
            "metadata": execution_result.metadata
        }

        # 包含检索结果概要
        if execution_result.retrieval_results:
            response_data["retrieval_summary"] = {
                "total_count": len(execution_result.retrieval_results),
                "avg_score": sum(r.score for r in execution_result.retrieval_results) / len(execution_result.retrieval_results),
                "top_sources": list(set(r.source for r in execution_result.retrieval_results[:5]))
            }

        return {
            "success": True,
            "data": response_data
        }

    except Exception as e:
        logger.error(f"查询执行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询执行失败: {str(e)}")


@router.get("/strategies")
async def get_strategies(current_user: User = Depends(get_current_user)):
    """
    获取所有可用的RAG策略信息
    """
    try:
        strategies_info = {}

        for strategy in RAGStrategy:
            config = await strategy_router.get_strategy_config(strategy)
            strategies_info[strategy.value] = config

        return {
            "success": True,
            "data": {
                "strategies": strategies_info,
                "routing_weights": strategy_router.weights,
                "strategy_thresholds": {
                    strategy.value: threshold
                    for strategy, threshold in strategy_router.thresholds.items()
                }
            }
        }

    except Exception as e:
        logger.error(f"获取策略信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取策略信息失败: {str(e)}")


@router.get("/strategy/{strategy_name}")
async def get_strategy_config(
    strategy_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    获取特定策略的详细配置
    """
    try:
        try:
            strategy = RAGStrategy(strategy_name)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"策略不存在: {strategy_name}")

        config = await strategy_router.get_strategy_config(strategy)

        return {
            "success": True,
            "data": {
                "strategy": strategy_name,
                "config": config
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取策略配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取策略配置失败: {str(e)}")


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    current_user: User = Depends(get_current_user)
):
    """
    提交策略执行反馈
    """
    try:
        logger.info(f"用户 {current_user.id} 提交反馈: 策略={request.strategy_used}, 满意度={request.satisfaction}")

        # 解析策略
        try:
            current_strategy = RAGStrategy(request.strategy_used)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"不支持的策略: {request.strategy_used}")

        # 构建性能指标
        performance_metrics = {
            "response_time": request.execution_time * 1000,  # 转换为毫秒
            "accuracy": request.accuracy
        }

        # 构建用户反馈
        user_feedback = {
            "score": request.satisfaction,
            "comment": request.comment,
            "requires_deeper_analysis": request.requires_deeper_analysis
        }

        # 动态调整策略
        adjusted_strategy = await strategy_router.dynamic_adjustment(
            current_strategy=current_strategy,
            performance_metrics=performance_metrics,
            user_feedback=user_feedback
        )

        # 保存反馈数据
        feedback_data = {
            "user_id": current_user.id,
            "query": request.query,
            "strategy_used": request.strategy_used,
            "execution_time": request.execution_time,
            "accuracy": request.accuracy,
            "satisfaction": request.satisfaction,
            "comment": request.comment,
            "timestamp": datetime.now().isoformat(),
            "adjusted_strategy": adjusted_strategy.value if adjusted_strategy else None
        }

        # 这里可以保存到数据库或Redis
        # await save_feedback_to_db(feedback_data)

        return {
            "success": True,
            "data": {
                "message": "反馈已提交",
                "adjusted_strategy": adjusted_strategy.value if adjusted_strategy else None,
                "recommendation": _generate_feedback_recommendation(adjusted_strategy, request.satisfaction)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交反馈失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")


@router.put("/config/weights")
async def update_routing_weights(
    request: WeightUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    更新路由权重配置
    """
    try:
        # 验证权重总和
        total_weight = sum(request.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail="权重总和必须为1.0")

        # 验证权重键
        valid_keys = set(strategy_router.weights.keys())
        provided_keys = set(request.weights.keys())
        if not provided_keys.issubset(valid_keys):
            raise HTTPException(status_code=400, detail=f"无效的权重键: {provided_keys - valid_keys}")

        # 更新权重
        strategy_router.update_routing_weights(request.weights)

        logger.info(f"用户 {current_user.id} 更新路由权重: {request.weights}")

        return {
            "success": True,
            "data": {
                "message": "路由权重已更新",
                "new_weights": request.weights
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新路由权重失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新路由权重失败: {str(e)}")


@router.put("/config/thresholds")
async def update_strategy_thresholds(
    request: ThresholdUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    更新策略阈值配置
    """
    try:
        # 验证策略名称
        valid_strategies = set(strategy.value for strategy in RAGStrategy)
        provided_strategies = set(request.thresholds.keys())
        if not provided_strategies.issubset(valid_strategies):
            raise HTTPException(status_code=400, detail=f"无效的策略名称: {provided_strategies - valid_strategies}")

        # 验证阈值范围
        for strategy, threshold in request.thresholds.items():
            if not (0 <= threshold <= 30):
                raise HTTPException(status_code=400, detail=f"策略 {strategy} 的阈值必须在0-30之间")

        # 更新阈值
        strategy_router.update_strategy_thresholds(request.thresholds)

        logger.info(f"用户 {current_user.id} 更新策略阈值: {request.thresholds}")

        return {
            "success": True,
            "data": {
                "message": "策略阈值已更新",
                "new_thresholds": request.thresholds
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新策略阈值失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新策略阈值失败: {str(e)}")


@router.get("/performance/stats")
async def get_performance_stats(
    days: int = Query(7, description="统计天数"),
    strategy: Optional[str] = Query(None, description="策略过滤"),
    current_user: User = Depends(get_current_user)
):
    """
    获取策略执行性能统计
    """
    try:
        # 这里应该从数据库或日志中统计性能数据
        # 目前返回模拟数据

        stats = {
            "total_queries": 1250,
            "avg_response_time": 4.2,
            "avg_accuracy": 0.87,
            "avg_satisfaction": 4.1,
            "strategy_distribution": {
                "light_rag": {"count": 625, "percentage": 50.0, "avg_time": 2.1},
                "graph_rag": {"count": 375, "percentage": 30.0, "avg_time": 5.8},
                "agentic_rag": {"count": 187, "percentage": 15.0, "avg_time": 11.2},
                "hybrid_rag": {"count": 63, "percentage": 5.0, "avg_time": 8.9}
            },
            "daily_stats": [
                {"date": "2024-01-15", "queries": 180, "avg_time": 4.1, "accuracy": 0.88},
                {"date": "2024-01-14", "queries": 165, "avg_time": 4.3, "accuracy": 0.86},
                {"date": "2024-01-13", "queries": 172, "avg_time": 4.0, "accuracy": 0.89}
            ],
            "top_queries": [
                {"query": "腾讯2023年财报", "count": 25, "strategy": "light_rag"},
                {"query": "美联储加息对A股影响", "count": 18, "strategy": "graph_rag"},
                {"query": "新能源汽车投资分析", "count": 15, "strategy": "agentic_rag"}
            ]
        }

        # 如果指定了策略过滤
        if strategy:
            if strategy not in stats["strategy_distribution"]:
                raise HTTPException(status_code=404, detail=f"策略不存在: {strategy}")

            strategy_stats = stats["strategy_distribution"][strategy]
            return {
                "success": True,
                "data": {
                    "strategy": strategy,
                    "stats": strategy_stats,
                    "overall_stats": {
                        "total_queries": strategy_stats["count"],
                        "avg_response_time": strategy_stats["avg_time"],
                        "accuracy_percentage": strategy_stats["avg_time"] * 10  # 简化计算
                    }
                }
            }

        return {
            "success": True,
            "data": stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取性能统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取性能统计失败: {str(e)}")


@router.get("/analytics/trends")
async def get_strategy_trends(
    days: int = Query(30, description="分析天数"),
    current_user: User = Depends(get_current_user)
):
    """
    获取策略使用趋势分析
    """
    try:
        # 这里应该从数据库分析趋势数据
        # 目前返回模拟数据

        trends = {
            "time_range": f"最近{days}天",
            "strategy_usage_trend": {
                "light_rag": {
                    "trend": "stable",
                    "change_percentage": 2.3,
                    "daily_usage": [45, 48, 42, 50, 47, 52, 49]
                },
                "graph_rag": {
                    "trend": "increasing",
                    "change_percentage": 15.7,
                    "daily_usage": [25, 28, 32, 30, 35, 38, 36]
                },
                "agentic_rag": {
                    "trend": "increasing",
                    "change_percentage": 23.1,
                    "daily_usage": [15, 17, 16, 19, 22, 20, 24]
                },
                "hybrid_rag": {
                    "trend": "stable",
                    "change_percentage": -1.2,
                    "daily_usage": [8, 7, 9, 6, 8, 7, 9]
                }
            },
            "performance_trend": {
                "avg_response_time": {
                    "trend": "improving",
                    "current": 4.2,
                    "change": -0.3,
                    "unit": "seconds"
                },
                "accuracy": {
                    "trend": "improving",
                    "current": 0.87,
                    "change": 0.05,
                    "unit": "percentage"
                },
                "user_satisfaction": {
                    "trend": "stable",
                    "current": 4.1,
                    "change": 0.1,
                    "unit": "score"
                }
            },
            "routing_effectiveness": {
                "routing_accuracy": 0.92,
                "strategy_switching_rate": 0.08,
                "fallback_usage_rate": 0.03
            },
            "recommendations": [
                "Graph RAG使用率持续上升，建议增加图谱数据投入",
                "Agentic RAG用户满意度最高，可考虑推广到更多场景",
                "LightRAG响应时间最优，适合高频简单查询场景"
            ]
        }

        return {
            "success": True,
            "data": trends
        }

    except Exception as e:
        logger.error(f"获取趋势分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取趋势分析失败: {str(e)}")


# 辅助函数
async def log_execution_result(user_id: int, query: str, routing_decision: RoutingDecision,
                              execution_result: ExecutionResult):
    """记录执行结果日志"""
    try:
        log_data = {
            "user_id": user_id,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "routing_decision": {
                "primary_strategy": routing_decision.primary_strategy.value,
                "confidence": routing_decision.confidence,
                "execution_mode": routing_decision.execution_mode
            },
            "execution_result": {
                "strategy_used": execution_result.strategy_used.value,
                "execution_time": execution_result.execution_time,
                "confidence": execution_result.confidence,
                "retrieval_count": len(execution_result.retrieval_results)
            }
        }

        # 这里可以保存到数据库或日志系统
        logger.info(f"执行日志: {log_data}")

    except Exception as e:
        logger.error(f"记录执行日志失败: {str(e)}")


def _generate_feedback_recommendation(adjusted_strategy: Optional[RAGStrategy],
                                    satisfaction: float) -> str:
    """根据反馈生成建议"""
    if adjusted_strategy:
        return f"系统建议下次使用 {adjusted_strategy.value} 策略以获得更好效果"

    if satisfaction >= 4.0:
        return "当前策略效果良好，建议继续使用"
    elif satisfaction >= 3.0:
        return "当前策略效果一般，可尝试其他策略进行对比"
    else:
        return "当前策略效果不佳，建议尝试更高级的策略或提供更具体的查询"


# 导出路由器
__all__ = ["router"]