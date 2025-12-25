"""
评估与运维API端点
提供评估任务、指标查询、告警管理等功能
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from app.core.auth import get_current_user
from app.models.user import User
from app.services.evaluation.auto_evaluator import AutoEvaluator, QuestionType, DifficultyLevel
from app.services.evaluation.metrics_calculator import MetricsCalculator, MetricCategory
from app.services.monitoring.monitoring_system import monitoring_system, AlertLevel
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)

router = APIRouter()

# 请求模型
class EvaluationRequest(BaseModel):
    """评估请求"""
    question_ids: Optional[List[str]] = Field(None, description="指定问题ID列表")
    question_types: Optional[List[str]] = Field(None, description="问题类型列表")
    difficulty_levels: Optional[List[str]] = Field(None, description="难度级别列表")
    limit: Optional[int] = Field(50, description="限制评估问题数量")
    save_results: bool = Field(True, description="是否保存评估结果")


class CustomQuestionRequest(BaseModel):
    """自定义问题请求"""
    question: str = Field(..., description="问题内容")
    question_type: str = Field(..., description="问题类型")
    difficulty: str = Field(..., description="难度级别")
    domain: str = Field(..., description="领域")
    expected_keywords: List[str] = Field(default_factory=list, description="期望关键词")
    reference_answer: Optional[str] = Field(None, description="参考答案")


class MetricQueryRequest(BaseModel):
    """指标查询请求"""
    metric_names: Optional[List[str]] = Field(None, description="指标名称列表")
    categories: Optional[List[str]] = Field(None, description="指标类别列表")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    limit: Optional[int] = Field(100, description="限制返回数量")


class AlertManagementRequest(BaseModel):
    """告警管理请求"""
    action: str = Field(..., description="操作类型: resolve, ignore")
    alert_ids: List[str] = Field(..., description="告警ID列表")


# 初始化评估器
evaluator = AutoEvaluator()
metrics_calculator = MetricsCalculator()


@router.post("/evaluation/run")
async def run_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    运行评估任务
    """
    try:
        logger.info(f"用户 {current_user.id} 启动评估任务")

        # 转换参数类型
        question_types = None
        if request.question_types:
            question_types = [QuestionType(t) for t in request.question_types]

        difficulty_levels = None
        if request.difficulty_levels:
            difficulty_levels = [DifficultyLevel(d) for d in request.difficulty_levels]

        # 运行评估
        results = await evaluator.run_evaluation(
            question_ids=request.question_ids,
            question_types=question_types,
            difficulty_levels=difficulty_levels,
            limit=request.limit
        )

        # 生成评估报告
        report = await evaluator.generate_evaluation_report(results)

        # 保存结果
        if request.save_results:
            # 在后台任务中保存到数据库
            background_tasks.add_task(
                save_evaluation_results_to_db,
                current_user.id,
                results,
                report
            )

        return {
            "success": True,
            "data": {
                "evaluation_summary": {
                    "total_questions": len(results),
                    "average_score": report.get('evaluation_summary', {}).get('average_score', 0),
                    "evaluation_date": datetime.now().isoformat()
                },
                "report": report,
                "results_count": len(results)
            }
        }

    except Exception as e:
        logger.error(f"运行评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"运行评估失败: {str(e)}")


@router.post("/evaluation/schedule")
async def schedule_daily_evaluation(
    current_user: User = Depends(get_current_user)
):
    """
    调度每日评估
    """
    try:
        logger.info(f"用户 {current_user.id} 启动每日评估")

        # 运行每日评估
        report = await evaluator.schedule_daily_evaluation()

        return {
            "success": True,
            "data": {
                "message": "每日评估已启动",
                "report": report,
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"调度每日评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"调度每日评估失败: {str(e)}")


@router.post("/evaluation/questions")
async def add_custom_question(
    request: CustomQuestionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    添加自定义问题到测试集
    """
    try:
        # 创建问题项
        question_item = AutoEvaluator.QuestionItem(
            id=f"custom_{int(datetime.now().timestamp())}",
            question=request.question,
            type=QuestionType(request.question_type),
            difficulty=DifficultyLevel(request.difficulty),
            domain=request.domain,
            expected_keywords=request.expected_keywords,
            expected_entities=[],
            context_requirements=[],
            reference_answer=request.reference_answer
        )

        # 添加到测试集
        evaluator.test_dataset.add_question(question_item)

        logger.info(f"用户 {current_user.id} 添加自定义问题: {question_item.id}")

        return {
            "success": True,
            "data": {
                "question_id": question_item.id,
                "message": "自定义问题添加成功"
            }
        }

    except Exception as e:
        logger.error(f"添加自定义问题失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加自定义问题失败: {str(e)}")


@router.get("/evaluation/questions")
async def get_test_questions(
    question_type: Optional[str] = Query(None, description="问题类型过滤"),
    difficulty: Optional[str] = Query(None, description="难度过滤"),
    domain: Optional[str] = Query(None, description="领域过滤"),
    current_user: User = Depends(get_current_user)
):
    """
    获取测试问题列表
    """
    try:
        questions = evaluator.test_dataset.questions

        # 应用过滤条件
        if question_type:
            questions = [q for q in questions if q.type.value == question_type]

        if difficulty:
            questions = [q for q in questions if q.difficulty.value == difficulty]

        if domain:
            questions = [q for q in questions if q.domain == domain]

        return {
            "success": True,
            "data": {
                "questions": [
                    {
                        "id": q.id,
                        "question": q.question,
                        "type": q.type.value,
                        "difficulty": q.difficulty.value,
                        "domain": q.domain,
                        "expected_keywords": q.expected_keywords
                    }
                    for q in questions
                ],
                "total_count": len(questions)
            }
        }

    except Exception as e:
        logger.error(f"获取测试问题失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取测试问题失败: {str(e)}")


@router.get("/evaluation/history")
async def get_evaluation_history(
    days: int = Query(30, description="查询天数"),
    current_user: User = Depends(get_current_user)
):
    """
    获取评估历史统计
    """
    try:
        stats = evaluator.get_evaluation_statistics(days)

        return {
            "success": True,
            "data": stats
        }

    except Exception as e:
        logger.error(f"获取评估历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取评估历史失败: {str(e)}")


@router.get("/evaluation/reports")
async def get_evaluation_reports(
    limit: int = Query(10, description="返回报告数量"),
    current_user: User = Depends(get_current_user)
):
    """
    获取评估报告列表
    """
    try:
        # 从Redis获取报告历史
        report_keys = redis_client.keys("daily_evaluation_report:*")
        report_keys.sort(reverse=True)

        reports = []
        for key in report_keys[:limit]:
            try:
                report_data = redis_client.get(key)
                if report_data:
                    data = json.loads(report_data.decode('utf-8'))
                    reports.append(data)
            except Exception as e:
                logger.error(f"解析报告失败: {e}")
                continue

        return {
            "success": True,
            "data": {
                "reports": reports,
                "total_count": len(report_keys)
            }
        }

    except Exception as e:
        logger.error(f"获取评估报告失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取评估报告失败: {str(e)}")


@router.get("/metrics")
async def get_metrics(
    request: MetricQueryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    获取监控指标
    """
    try:
        # 获取指标摘要
        summary = await monitoring_system.get_metrics_summary(minutes=60)

        # 如果指定了具体指标名称，进行筛选
        if request.metric_names:
            filtered_metrics = {
                name: data for name, data in summary['metrics'].items()
                if name in request.metric_names
            }
            summary['metrics'] = filtered_metrics

        # 如果指定了类别，进行筛选
        if request.categories:
            filtered_metrics = {}
            for name, data in summary['metrics'].items():
                # 简化实现，基于指标名称判断类别
                if any(cat in name for cat in request.categories):
                    filtered_metrics[name] = data
            summary['metrics'] = filtered_metrics

        return {
            "success": True,
            "data": summary
        }

    except Exception as e:
        logger.error(f"获取监控指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取监控指标失败: {str(e)}")


@router.get("/metrics/categories")
async def get_metric_categories(current_user: User = Depends(get_current_user)):
    """
    获取指标类别
    """
    try:
        # 获取所有类别的指标
        all_metrics = {
            'accuracy': await metrics_calculator.calculate_accuracy_metrics([]),
            'efficiency': await metrics_calculator.calculate_efficiency_metrics([]),
            'availability': await metrics_calculator.calculate_availability_metrics([])
        }

        categories = {}
        for category, metrics in all_metrics.items():
            categories[category] = [
                {
                    "name": metric.name,
                    "category": metric.category.value,
                    "value": metric.value,
                    "unit": metric.unit,
                    "description": metric.description,
                    "status": metric.status
                }
                for metric in metrics
            ]

        return {
            "success": True,
            "data": {
                "categories": categories,
                "summary": metrics_calculator.get_metrics_summary([])
            }
        }

    except Exception as e:
        logger.error(f"获取指标类别失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取指标类别失败: {str(e)}")


@router.get("/monitoring/health")
async def get_system_health(current_user: User = Depends(get_current_user)):
    """
    获取系统健康状态
    """
    try:
        health_status = monitoring_system.get_system_health()

        return {
            "success": True,
            "data": health_status
        }

    except Exception as e:
        logger.error(f"获取系统健康状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统健康状态失败: {str(e)}")


@router.get("/monitoring/alerts")
async def get_alerts(
    active_only: bool = Query(True, description="只获取活跃告警"),
    level: Optional[str] = Query(None, description="告警级别过滤"),
    current_user: User = Depends(get_current_user)
):
    """
    获取告警信息
    """
    try:
        if active_only:
            alerts = await monitoring_system.get_active_alerts()
        else:
            # 获取所有告警（包括已解决的）
            alerts = [
                {
                    'id': alert.id,
                    'level': alert.level.value,
                    'source': alert.source,
                    'message': alert.message,
                    'details': alert.details,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved,
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
                }
                for alert in monitoring_system.alerts
            ]

        # 按级别过滤
        if level:
            alerts = [a for a in alerts if a['level'] == level]

        return {
            "success": True,
            "data": {
                "alerts": alerts,
                "total_count": len(alerts)
            }
        }

    except Exception as e:
        logger.error(f"获取告警信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取告警信息失败: {str(e)}")


@router.post("/monitoring/alerts/manage")
async def manage_alerts(
    request: AlertManagementRequest,
    current_user: User = Depends(get_current_user)
):
    """
    管理告警
    """
    try:
        results = []

        if request.action == "resolve":
            for alert_id in request.alert_ids:
                success = await monitoring_system.resolve_alert(alert_id)
                results.append({
                    "alert_id": alert_id,
                    "resolved": success
                })

        elif request.action == "ignore":
            # 实现忽略告警的逻辑
            for alert_id in request.alert_ids:
                results.append({
                    "alert_id": alert_id,
                    "ignored": True
                })

        else:
            raise HTTPException(status_code=400, detail="不支持的操作类型")

        return {
            "success": True,
            "data": {
                "action": request.action,
                "results": results,
                "processed_count": len(results)
            }
        }

    except Exception as e:
        logger.error(f"管理告警失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"管理告警失败: {str(e)}")


@router.post("/monitoring/metrics/custom")
async def add_custom_metric(
    metric_name: str = Query(..., description="指标名称"),
    value: float = Query(..., description="指标值"),
    metric_type: str = Query("gauge", description="指标类型"),
    current_user: User = Depends(get_current_user)
):
    """
    添加自定义指标
    """
    try:
        from app.services.monitoring.monitoring_system import MetricType

        metric_type_enum = MetricType(metric_type)
        monitoring_system.add_custom_metric(
            name=metric_name,
            value=value,
            metric_type=metric_type_enum
        )

        logger.info(f"用户 {current_user.id} 添加自定义指标: {metric_name}={value}")

        return {
            "success": True,
            "data": {
                "message": f"自定义指标 {metric_name} 添加成功",
                "value": value,
                "type": metric_type
            }
        }

    except Exception as e:
        logger.error(f"添加自定义指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加自定义指标失败: {str(e)}")


@router.get("/dashboard/overview")
async def get_dashboard_overview(current_user: User = Depends(get_current_user)):
    """
    获取仪表板概览数据
    """
    try:
        # 获取系统健康状态
        health = monitoring_system.get_system_health()

        # 获取指标摘要
        metrics_summary = await monitoring_system.get_metrics_summary(minutes=60)

        # 获取活跃告警
        active_alerts = await monitoring_system.get_active_alerts()

        # 获取评估统计
        evaluation_stats = evaluator.get_evaluation_statistics(days=7)

        # 获取RAG指标
        rag_metrics = {}
        if 'rag_overall_score' in metrics_summary.get('metrics', {}):
            rag_metrics['overall_score'] = metrics_summary['metrics']['rag_overall_score']

        dashboard_data = {
            "health": health,
            "metrics": metrics_summary,
            "alerts": {
                "active_count": len(active_alerts),
                "critical_count": len([a for a in active_alerts if a['level'] == 'critical']),
                "warning_count": len([a for a in active_alerts if a['level'] == 'warning'])
            },
            "evaluation": evaluation_stats,
            "rag_metrics": rag_metrics,
            "timestamp": datetime.now().isoformat()
        }

        return {
            "success": True,
            "data": dashboard_data
        }

    except Exception as e:
        logger.error(f"获取仪表板概览失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取仪表板概览失败: {str(e)}")


@router.get("/logs/analysis")
async def get_log_analysis(
    log_type: str = Query("error", description="日志类型"),
    hours: int = Query(24, description="查询时间范围（小时）"),
    current_user: User = Depends(get_current_user)
):
    """
    获取日志分析结果
    """
    try:
        # 从Redis获取日志统计
        log_stats = redis_client.hgetall(f"log_statistics:{log_type}")

        if not log_stats:
            return {
                "success": True,
                "data": {
                    "message": f"没有找到 {log_type} 日志统计数据"
                }
            }

        # 解析统计信息
        analysis = {
            "log_type": log_type,
            "total_count": int(log_stats.get('total_count', 0)),
            "error_count": int(log_stats.get('error_count', 0)),
            "warning_count": int(log_stats.get('warning_count', 0)),
            "top_errors": json.loads(log_stats.get('top_errors', '[]')),
            "time_range_hours": hours,
            "timestamp": datetime.now().isoformat()
        }

        return {
            "success": True,
            "data": analysis
        }

    except Exception as e:
        logger.error(f"获取日志分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取日志分析失败: {str(e)}")


@router.get("/metrics/trends")
async def get_metrics_trends(
    metric_name: str = Query(..., description="指标名称"),
    days: int = Query(7, description="查询天数"),
    current_user: User = Depends(get_current_user)
):
    """
    获取指标趋势
    """
    try:
        from app.services.evaluation.metrics_calculator import cache_manager

        # 从缓存管理器获取指标趋势
        trend_data = metrics_calculator.get_metrics_trend(metric_name, days)

        if not trend_data:
            return {
                "success": True,
                "data": {
                    "message": f"没有找到指标 {metric_name} 的趋势数据"
                }
            }

        return {
            "success": True,
            "data": trend_data
        }

    except Exception as e:
        logger.error(f"获取指标趋势失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取指标趋势失败: {str(e)}")


# 后台任务函数
async def save_evaluation_results_to_db(user_id: int, results: List, report: Dict):
    """保存评估结果到数据库的后台任务"""
    try:
        # 这里应该保存到数据库
        # 简化实现，保存到Redis
        save_data = {
            'user_id': user_id,
            'results_count': len(results),
            'report': report,
            'saved_at': datetime.now().isoformat()
        }

        redis_client.setex(
            f"evaluation_save:{user_id}:{int(datetime.now().timestamp())}",
            86400 * 7,  # 保存7天
            json.dumps(save_data, ensure_ascii=False, default=str)
        )

        logger.info(f"评估结果已保存: 用户={user_id}, 结果数量={len(results)}")

    except Exception as e:
        logger.error(f"保存评估结果失败: {e}")