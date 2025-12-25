"""
增强评估API端点
提供RAG系统评估、优化建议和监控功能
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from app.services.evaluation_service import enhanced_rag_evaluator, OptimizationLevel
from app.services.evaluation.ragas_evaluator import automated_evaluator
from app.core.database import get_db
from app.core.redis_client import redis_client
from app.core.auth import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


class EvaluationRequest(BaseModel):
    """评估请求模型"""
    sample_size: int = Field(default=100, ge=10, le=1000, description="测试样本数量")
    query_types: Optional[List[str]] = Field(default=None, description="要评估的查询类型")
    run_immediately: bool = Field(default=False, description="是否立即运行评估")


class OptimizationPlanRequest(BaseModel):
    """优化计划请求模型"""
    evaluation_id: str = Field(..., description="评估ID")
    apply_strategies: List[str] = Field(default=[], description="要应用的优化策略")


class BenchmarkRequest(BaseModel):
    """基准测试请求模型"""
    test_suite: str = Field(default="comprehensive", description="测试套件类型")
    compare_with_previous: bool = Field(default=True, description="是否与之前结果对比")


@router.post("/comprehensive")
async def run_comprehensive_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    运行综合评估

    评估RAG系统在各个维度的表现，包括：
    - 事实一致性
    - 答案相关性
    - 上下文相关性
    - 检索精度
    - 响应性能
    """
    try:
        logger.info(f"User {current_user.id} requested comprehensive evaluation")

        # 创建评估任务ID
        evaluation_id = f"eval_{current_user.id}_{int(datetime.now().timestamp())}"

        # 检查是否有正在运行的评估
        cache_key = f"evaluation_status:{current_user.id}"
        existing_status = redis_client.get(cache_key)

        if existing_status:
            status = json.loads(existing_status)
            if status.get("status") == "running":
                raise HTTPException(
                    status_code=409,
                    detail="已有评估任务在运行中"
                )

        # 更新状态
        redis_client.setex(
            cache_key,
            3600,  # 1小时过期
            json.dumps({
                "status": "running",
                "evaluation_id": evaluation_id,
                "started_at": datetime.now().isoformat(),
                "progress": 0.0
            })
        )

        if request.run_immediately:
            # 立即运行评估
            try:
                result = await enhanced_rag_evaluator.comprehensive_evaluation(
                    sample_size=request.sample_size,
                    query_types=request.query_types
                )

                # 更新状态为完成
                redis_client.setex(
                    cache_key,
                    86400 * 7,  # 缓存7天
                    json.dumps({
                        "status": "completed",
                        "evaluation_id": result["evaluation_id"],
                        "completed_at": datetime.now().isoformat(),
                        "meets_target": result["meets_target"],
                        "overall_score": result["overall_metrics"]["overall_score"]
                    })
                )

                return {
                    "success": True,
                    "evaluation_id": result["evaluation_id"],
                    "message": "评估完成",
                    "results": result
                }

            except Exception as e:
                # 更新状态为失败
                redis_client.setex(
                    cache_key,
                    3600,
                    json.dumps({
                        "status": "failed",
                        "error": str(e),
                        "failed_at": datetime.now().isoformat()
                    })
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"评估失败: {str(e)}"
                )
        else:
            # 后台运行评估
            background_tasks.add_task(
                run_evaluation_background,
                evaluation_id,
                current_user.id,
                request.sample_size,
                request.query_types
            )

            return {
                "success": True,
                "evaluation_id": evaluation_id,
                "message": "评估任务已启动，正在后台运行",
                "estimated_duration": f"{request.sample_size * 2}秒"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"评估请求失败: {str(e)}"
        )


@router.get("/status/{evaluation_id}")
async def get_evaluation_status(
    evaluation_id: str,
    current_user: User = Depends(get_current_user)
):
    """获取评估状态"""
    try:
        cache_key = f"evaluation_status:{current_user.id}"
        status_data = redis_client.get(cache_key)

        if not status_data:
            raise HTTPException(
                status_code=404,
                detail="未找到评估任务"
            )

        status = json.loads(status_data)

        if status.get("evaluation_id") != evaluation_id:
            raise HTTPException(
                status_code=404,
                detail="评估ID不匹配"
            )

        return {
            "evaluation_id": evaluation_id,
            "status": status.get("status"),
            "progress": status.get("progress", 0.0),
            "started_at": status.get("started_at"),
            "completed_at": status.get("completed_at"),
            "overall_score": status.get("overall_score"),
            "meets_target": status.get("meets_target", False),
            "error": status.get("error")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get evaluation status failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取评估状态失败: {str(e)}"
        )


@router.get("/results/{evaluation_id}")
async def get_evaluation_results(
    evaluation_id: str,
    current_user: User = Depends(get_current_user)
):
    """获取评估结果"""
    try:
        cache_key = f"comprehensive_evaluation:{evaluation_id}"
        results_data = redis_client.get(cache_key)

        if not results_data:
            raise HTTPException(
                status_code=404,
                detail="未找到评估结果"
            )

        results = json.loads(results_data)

        # 添加分析报告
        analysis_report = generate_analysis_report(results)
        results["analysis_report"] = analysis_report

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get evaluation results failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取评估结果失败: {str(e)}"
        )


@router.post("/optimization-plan")
async def generate_optimization_plan(
    request: OptimizationPlanRequest,
    current_user: User = Depends(get_current_user)
):
    """生成优化计划"""
    try:
        # 获取评估结果
        cache_key = f"comprehensive_evaluation:{request.evaluation_id}"
        results_data = redis_client.get(cache_key)

        if not results_data:
            raise HTTPException(
                status_code=404,
                detail="未找到评估结果"
            )

        results = json.loads(results_data)

        # 分析并生成优化计划
        optimization_plan = analyze_and_generate_optimization_plan(
            results,
            request.apply_strategies
        )

        return {
            "evaluation_id": request.evaluation_id,
            "optimization_plan": optimization_plan,
            "estimated_improvement": calculate_estimated_improvement(optimization_plan),
            "implementation_roadmap": create_implementation_roadmap(optimization_plan)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generate optimization plan failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"生成优化计划失败: {str(e)}"
        )


@router.get("/dashboard")
async def get_evaluation_dashboard(
    current_user: User = Depends(get_current_user)
):
    """获取评估仪表板数据"""
    try:
        # 获取最近的评估结果
        user_cache_key = f"user_evaluations:{current_user.id}"
        user_evaluations = redis_client.get(user_cache_key)

        evaluations = []
        if user_evaluations:
            evaluations = json.loads(user_evaluations)

        # 获取系统整体状态
        system_status = await get_system_evaluation_status()

        # 获取性能趋势
        performance_trends = await get_performance_trends(current_user.id)

        # 获取优化建议
        optimization_suggestions = await get_optimization_suggestions(current_user.id)

        return {
            "current_user": {
                "id": current_user.id,
                "username": current_user.username
            },
            "recent_evaluations": evaluations[-5:],  # 最近5次评估
            "system_status": system_status,
            "performance_trends": performance_trends,
            "optimization_suggestions": optimization_suggestions,
            "target_scores": enhanced_rag_evaluator.target_scores,
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Get evaluation dashboard failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取仪表板数据失败: {str(e)}"
        )


@router.post("/benchmark")
async def run_benchmark(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """运行基准测试"""
    try:
        benchmark_id = f"benchmark_{current_user.id}_{int(datetime.now().timestamp())}"

        # 运行基准测试
        if request.test_suite == "comprehensive":
            # 运行综合基准测试
            background_tasks.add_task(
                run_comprehensive_benchmark,
                benchmark_id,
                current_user.id,
                request.compare_with_previous
            )
        else:
            # 运行特定基准测试
            background_tasks.add_task(
                run_specific_benchmark,
                benchmark_id,
                current_user.id,
                request.test_suite
            )

        return {
            "benchmark_id": benchmark_id,
            "message": "基准测试已启动",
            "test_suite": request.test_suite,
            "compare_with_previous": request.compare_with_previous
        }

    except Exception as e:
        logger.error(f"Run benchmark failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"基准测试失败: {str(e)}"
        )


@router.get("/benchmark/{benchmark_id}")
async def get_benchmark_results(
    benchmark_id: str,
    current_user: User = Depends(get_current_user)
):
    """获取基准测试结果"""
    try:
        cache_key = f"benchmark_results:{benchmark_id}"
        results_data = redis_client.get(cache_key)

        if not results_data:
            raise HTTPException(
                status_code=404,
                detail="未找到基准测试结果"
            )

        results = json.loads(results_data)

        return {
            "benchmark_id": benchmark_id,
            "results": results,
            "summary": generate_benchmark_summary(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get benchmark results failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取基准测试结果失败: {str(e)}"
        )


# 后台任务函数
async def run_evaluation_background(
    evaluation_id: str,
    user_id: int,
    sample_size: int,
    query_types: Optional[List[str]]
):
    """后台运行评估任务"""
    try:
        # 更新进度
        cache_key = f"evaluation_status:{user_id}"
        redis_client.setex(
            cache_key,
            3600,
            json.dumps({
                "status": "running",
                "evaluation_id": evaluation_id,
                "progress": 0.1
            })
        )

        # 运行评估
        result = await enhanced_rag_evaluator.comprehensive_evaluation(
            sample_size=sample_size,
            query_types=query_types
        )

        # 更新状态为完成
        redis_client.setex(
            cache_key,
            86400 * 7,
            json.dumps({
                "status": "completed",
                "evaluation_id": evaluation_id,
                "completed_at": datetime.now().isoformat(),
                "meets_target": result["meets_target"],
                "overall_score": result["overall_metrics"]["overall_score"],
                "progress": 1.0
            })
        )

        # 添加到用户评估历史
        user_evaluations_key = f"user_evaluations:{user_id}"
        user_evaluations = redis_client.get(user_evaluations_key)

        evaluations = json.loads(user_evaluations) if user_evaluations else []
        evaluations.append({
            "evaluation_id": evaluation_id,
            "timestamp": datetime.now().isoformat(),
            "overall_score": result["overall_metrics"]["overall_score"],
            "meets_target": result["meets_target"],
            "sample_size": sample_size
        })

        # 只保留最近20次评估
        evaluations = evaluations[-20:]

        redis_client.setex(
            user_evaluations_key,
            86400 * 30,  # 30天过期
            json.dumps(evaluations)
        )

        logger.info(f"Background evaluation {evaluation_id} completed successfully")

    except Exception as e:
        logger.error(f"Background evaluation {evaluation_id} failed: {str(e)}")

        # 更新状态为失败
        redis_client.setex(
            cache_key,
            3600,
            json.dumps({
                "status": "failed",
                "evaluation_id": evaluation_id,
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            })
        )


async def run_comprehensive_benchmark(
    benchmark_id: str,
    user_id: int,
    compare_with_previous: bool
):
    """运行综合基准测试"""
    try:
        # 运行每日评估作为基准测试
        report = await automated_evaluator.run_daily_evaluation()

        # 如果需要与之前结果对比
        comparison_results = None
        if compare_with_previous:
            comparison_results = await get_previous_benchmark_results(user_id)

        results = {
            "report": report.to_dict() if hasattr(report, 'to_dict') else str(report),
            "comparison": comparison_results,
            "timestamp": datetime.now().isoformat()
        }

        # 缓存结果
        cache_key = f"benchmark_results:{benchmark_id}"
        redis_client.setex(
            cache_key,
            86400 * 7,
            json.dumps(results, default=str)
        )

        logger.info(f"Benchmark {benchmark_id} completed successfully")

    except Exception as e:
        logger.error(f"Benchmark {benchmark_id} failed: {str(e)}")


# 辅助函数
def generate_analysis_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """生成分析报告"""
    overall_metrics = results.get("overall_metrics", {})
    overall_score = overall_metrics.get("overall_score", 0)
    target_score = 0.80

    analysis = {
        "summary": "",
        "key_insights": [],
        "strengths": [],
        "weaknesses": [],
        "recommendations": []
    }

    # 生成总结
    if overall_score >= target_score:
        analysis["summary"] = f"系统表现优秀，综合评分达到 {overall_score:.3f}，超过目标分数 {target_score}"
    else:
        gap = target_score - overall_score
        analysis["summary"] = f"系统需要改进，当前评分 {overall_score:.3f}，距离目标 {target_score} 还差 {gap:.3f}"

    # 分析关键指标
    ragas_metrics = overall_metrics.get("ragas_metrics", {})
    for metric, score in ragas_metrics.items():
        if score >= 0.8:
            analysis["strengths"].append(f"{metric}: {score:.3f} (优秀)")
        elif score < 0.6:
            analysis["weaknesses"].append(f"{metric}: {score:.3f} (需要改进)")

    # 生成关键洞察
    if overall_score < target_score:
        analysis["key_insights"].append("系统整体性能未达到目标，需要针对性优化")

    faithfulness = ragas_metrics.get("faithfulness", 0)
    if faithfulness < 0.7:
        analysis["key_insights"].append("答案忠实度偏低，建议加强上下文验证")

    answer_relevancy = ragas_metrics.get("answer_relevancy", 0)
    if answer_relevancy < 0.7:
        analysis["key_insights"].append("答案相关性不足，建议改进查询理解")

    return analysis


def analyze_and_generate_optimization_plan(
    results: Dict[str, Any],
    apply_strategies: List[str]
) -> List[Dict[str, Any]]:
    """分析并生成优化计划"""
    optimization_plan = results.get("optimization_plan", [])

    if not apply_strategies:
        return optimization_plan

    # 过滤用户选择的策略
    filtered_plan = []
    for strategy in optimization_plan:
        for optimization in strategy.optimizations:
            if optimization.get("action") in apply_strategies:
                filtered_plan.append({
                    "level": strategy.level.value,
                    "target_score": strategy.target_score,
                    "optimization": optimization,
                    "estimated_improvement": strategy.estimated_improvement,
                    "implementation_effort": strategy.implementation_effort,
                    "priority": strategy.priority
                })

    return filtered_plan


def calculate_estimated_improvement(optimization_plan: List[Dict[str, Any]]) -> float:
    """计算预估改进幅度"""
    if not optimization_plan:
        return 0.0

    total_improvement = sum(strategy.get("estimated_improvement", 0) for strategy in optimization_plan)

    # 考虑策略间的重叠效应
    adjusted_improvement = total_improvement * 0.8

    return min(adjusted_improvement, 0.5)  # 最大改进幅度限制为50%


def create_implementation_roadmap(optimization_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    """创建实施路线图"""
    if not optimization_plan:
        return {"phases": [], "timeline": "无优化计划"}

    # 按优先级和实施难度分组
    phases = []

    # 第一阶段：高优先级、低难度
    phase1 = [s for s in optimization_plan if s.get("priority", 0) >= 7 and s.get("implementation_effort") == "low"]
    if phase1:
        phases.append({
            "phase": 1,
            "name": "快速改进阶段",
            "duration": "1-2周",
            "strategies": phase1
        })

    # 第二阶段：中优先级、中等难度
    phase2 = [s for s in optimization_plan if 5 <= s.get("priority", 0) < 7 and s.get("implementation_effort") == "medium"]
    if phase2:
        phases.append({
            "phase": 2,
            "name": "系统优化阶段",
            "duration": "3-4周",
            "strategies": phase2
        })

    # 第三阶段：高难度优化
    phase3 = [s for s in optimization_plan if s.get("implementation_effort") == "high"]
    if phase3:
        phases.append({
            "phase": 3,
            "name": "深度优化阶段",
            "duration": "4-6周",
            "strategies": phase3
        })

    total_duration = f"{len(phases) * 2}-{len(phases) * 6}周"

    return {
        "phases": phases,
        "timeline": total_duration,
        "total_phases": len(phases)
    }


async def get_system_evaluation_status() -> Dict[str, Any]:
    """获取系统评估状态"""
    try:
        # 获取最近的系统评估结果
        cache_key = "system_evaluation_status"
        status_data = redis_client.get(cache_key)

        if status_data:
            return json.loads(status_data)

        return {
            "overall_health": "unknown",
            "last_evaluation": None,
            "active_issues": []
        }

    except Exception as e:
        logger.error(f"Get system evaluation status failed: {str(e)}")
        return {
            "overall_health": "error",
            "error": str(e)
        }


async def get_performance_trends(user_id: int) -> List[Dict[str, Any]]:
    """获取性能趋势"""
    try:
        user_evaluations_key = f"user_evaluations:{user_id}"
        evaluations_data = redis_client.get(user_evaluations_key)

        if not evaluations_data:
            return []

        evaluations = json.loads(evaluations_data)

        # 生成趋势数据
        trends = []
        for eval_data in evaluations[-10:]:  # 最近10次评估
            trends.append({
                "date": eval_data["timestamp"],
                "overall_score": eval_data["overall_score"],
                "meets_target": eval_data["meets_target"]
            })

        return trends

    except Exception as e:
        logger.error(f"Get performance trends failed: {str(e)}")
        return []


async def get_optimization_suggestions(user_id: int) -> List[Dict[str, Any]]:
    """获取优化建议"""
    try:
        # 基于用户历史评估结果生成建议
        suggestions = []

        user_evaluations_key = f"user_evaluations:{user_id}"
        evaluations_data = redis_client.get(user_evaluations_key)

        if evaluations_data:
            evaluations = json.loads(evaluations_data)

            if evaluations:
                latest_eval = evaluations[-1]

                if latest_eval["overall_score"] < 0.80:
                    suggestions.append({
                        "type": "performance",
                        "priority": "high",
                        "title": "系统性能需要改进",
                        "description": f"当前评分 {latest_eval['overall_score']:.3f} 低于目标 0.80",
                        "action": "运行详细评估以获取优化建议"
                    })

                # 检查性能趋势
                if len(evaluations) >= 3:
                    recent_scores = [e["overall_score"] for e in evaluations[-3:]]
                    if recent_scores[-1] < recent_scores[0]:
                        suggestions.append({
                            "type": "trend",
                            "priority": "medium",
                            "title": "性能呈下降趋势",
                            "description": "最近几次评估显示性能下降",
                            "action": "检查系统配置和数据质量"
                        })

        # 添加通用建议
        suggestions.append({
            "type": "maintenance",
            "priority": "low",
            "title": "定期维护建议",
            "description": "建议定期运行评估以监控系统性能",
            "action": "设置定期评估任务"
        })

        return suggestions

    except Exception as e:
        logger.error(f"Get optimization suggestions failed: {str(e)}")
        return []


def generate_benchmark_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """生成基准测试总结"""
    return {
        "executed_at": results.get("timestamp"),
        "overall_score": results.get("report", {}).get("overall_score", 0),
        "key_metrics": {
            "faithfulness": results.get("report", {}).get("faithfulness", 0),
            "answer_relevancy": results.get("report", {}).get("answer_relevancy", 0),
            "context_relevancy": results.get("report", {}).get("context_relevancy", 0)
        },
        "performance_trend": "stable"  # 这里可以添加趋势分析
    }