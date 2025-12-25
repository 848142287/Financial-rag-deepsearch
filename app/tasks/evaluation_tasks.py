"""
评估相关异步任务
"""

from celery import current_task
from app.core.async_tasks.celery_app import celery_app
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from app.services.evaluation.ragas_evaluator import automated_evaluator, EvaluationTestCase
from app.services.evaluation.metrics import metrics_registry, MetricResult, EvaluationReport
from app.services.monitoring.monitor import monitoring_service
from app.services.websocket_service import websocket_service
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.evaluation_tasks.run_daily_evaluation")
def run_daily_evaluation(self) -> Dict[str, Any]:
    """
    运行每日评估任务
    """
    task_id = self.request.id
    logger.info(f"Starting daily evaluation task {task_id}")

    try:
        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "初始化评估..."}
        )

        # 发送开始通知
        websocket_service.send_message(
            "evaluation_progress",
            {
                "task_id": task_id,
                "status": "started",
                "message": "开始每日评估...",
                "progress": 0
            }
        )

        # 执行评估
        report = automated_evaluator.run_daily_evaluation()

        # 生成评估摘要
        summary = {
            "evaluation_id": report.evaluation_id,
            "overall_score": report.overall_score,
            "test_cases_count": len(report.metrics),
            "dimension_scores": report.dimension_scores,
            "timestamp": report.timestamp.isoformat()
        }

        # 发送完成通知
        websocket_service.send_message(
            "evaluation_progress",
            {
                "task_id": task_id,
                "status": "completed",
                "message": "每日评估完成",
                "progress": 100,
                "result": summary
            }
        )

        # 保存到历史记录
        redis_key = f"evaluation_history:daily:{datetime.utcnow().strftime('%Y-%m-%d')}"
        redis_client.setex(
            redis_key,
            86400 * 30,  # 保存30天
            json.dumps(report.to_dict(), ensure_ascii=False)
        )

        logger.info(f"Daily evaluation completed: {report.evaluation_id}")
        return summary

    except Exception as e:
        logger.error(f"Daily evaluation failed: {str(e)}")

        # 发送错误通知
        websocket_service.send_message(
            "evaluation_progress",
            {
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            }
        )

        raise


@celery_app.task(bind=True, name="app.tasks.evaluation_tasks.run_custom_evaluation")
def run_custom_evaluation(self, test_cases: List[Dict[str, Any]], evaluation_name: str = "custom") -> Dict[str, Any]:
    """
    运行自定义评估任务
    """
    task_id = self.request.id
    logger.info(f"Starting custom evaluation task {task_id}: {evaluation_name}")

    try:
        # 转换测试用例
        evaluation_cases = []
        for case_data in test_cases:
            test_case = EvaluationTestCase(
                id=case_data.get("id", f"custom_{len(evaluation_cases)}"),
                query=case_data["query"],
                ground_truth=case_data.get("ground_truth"),
                query_type=case_data.get("query_type", "general"),
                difficulty=case_data.get("difficulty", "medium"),
                domain=case_data.get("domain", "financial"),
                metadata=case_data.get("metadata", {})
            )
            evaluation_cases.append(test_case)

        # 更新状态
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": len(evaluation_cases), "status": "开始评估..."}
        )

        # 执行评估
        report = await automated_evaluator.run_custom_evaluation(evaluation_cases, evaluation_name)

        # 生成结果
        result = {
            "evaluation_id": report.evaluation_id,
            "evaluation_name": evaluation_name,
            "overall_score": report.overall_score,
            "test_cases_count": len(evaluation_cases),
            "dimension_scores": report.dimension_scores,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "timestamp": report.timestamp.isoformat()
        }

        # 发送WebSocket通知
        websocket_service.send_message(
            "evaluation_result",
            {
                "task_id": task_id,
                "evaluation_name": evaluation_name,
                "result": result
            }
        )

        logger.info(f"Custom evaluation completed: {report.evaluation_id}")
        return result

    except Exception as e:
        logger.error(f"Custom evaluation failed: {str(e)}")
        raise


@celery_app.task(bind=True, name="app.tasks.evaluation_tasks.evaluate_query_quality")
def evaluate_query_quality(self, query: str, answer: str, contexts: List[str],
                          ground_truth: Optional[str] = None) -> Dict[str, Any]:
    """
    评估单个查询的质量
    """
    try:
        # 创建测试用例
        test_case = EvaluationTestCase(
            id=f"quality_{int(datetime.utcnow().timestamp())}",
            query=query,
            ground_truth=ground_truth,
            query_type="general"
        )

        # 使用RAGAS评估器评估
        result = await automated_evaluator.ragas_evaluator.evaluate_single_case(test_case)

        # 提取关键指标
        quality_scores = {
            "faithfulness": result.ragas_scores.get("faithfulness", 0),
            "answer_relevancy": result.ragas_scores.get("answer_relevancy", 0),
            "context_relevancy": result.ragas_scores.get("context_relevancy", 0),
            "overall_quality": (result.ragas_scores.get("faithfulness", 0) +
                              result.ragas_scores.get("answer_relevancy", 0)) / 2
        }

        # 注册指标
        for metric_name, score in quality_scores.items():
            metric_result = MetricResult(
                metric_name=metric_name,
                value=score,
                timestamp=datetime.utcnow(),
                context={"query": query[:100]}
            )
            metrics_registry.register_metric(metric_result)

        return {
            "query": query,
            "quality_scores": quality_scores,
            "execution_time": result.execution_time,
            "timestamp": result.timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"Query quality evaluation failed: {str(e)}")
        return {
            "query": query,
            "error": str(e),
            "quality_scores": {}
        }


@celery_app.task(name="app.tasks.evaluation_tasks.cleanup_old_evaluations")
def cleanup_old_evaluations():
    """
    清理旧的评估数据
    """
    logger.info("Starting cleanup of old evaluation data")

    try:
        # 清理30天前的指标数据
        metrics_registry.cleanup_old_metrics(days_to_keep=30)

        # 清理Redis中的旧评估历史
        pattern = "evaluation_history:*"
        keys = redis_client.keys(pattern)
        deleted_count = 0

        for key in keys:
            # 检查键的TTL
            ttl = redis_client.ttl(key)
            if ttl == -1:  # 没有设置过期时间
                # 解析日期并检查是否超过30天
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                if ":" in key_str:
                    date_part = key_str.split(":")[-1]
                    try:
                        eval_date = datetime.strptime(date_part, "%Y-%m-%d")
                        if datetime.utcnow() - eval_date > timedelta(days=30):
                            redis_client.delete(key)
                            deleted_count += 1
                    except ValueError:
                        pass

        logger.info(f"Cleanup completed: deleted {deleted_count} old evaluation records")
        return {"deleted_count": deleted_count}

    except Exception as e:
        logger.error(f"Evaluation cleanup failed: {str(e)}")
        raise


@celery_app.task(name="app.tasks.evaluation_tasks.generate_evaluation_report")
def generate_evaluation_report(evaluation_id: str, report_type: str = "summary") -> Dict[str, Any]:
    """
    生成评估报告
    """
    try:
        # 从Redis获取评估报告
        cache_key = f"evaluation_report:{evaluation_id}"
        report_data = redis_client.get(cache_key)

        if not report_data:
            raise ValueError(f"Evaluation report not found: {evaluation_id}")

        report = json.loads(report_data)

        if report_type == "summary":
            # 生成摘要报告
            summary = {
                "evaluation_id": evaluation_id,
                "overall_score": report.get("overall_score", 0),
                "dimension_scores": report.get("dimension_scores", {}),
                "metrics_count": len(report.get("metrics", [])),
                "summary": report.get("summary", ""),
                "recommendations": report.get("recommendations", [])[:3],  # 前3个建议
                "timestamp": report.get("timestamp")
            }
            return summary

        elif report_type == "detailed":
            # 返回详细报告
            return report

        else:
            raise ValueError(f"Unknown report type: {report_type}")

    except Exception as e:
        logger.error(f"Failed to generate evaluation report: {str(e)}")
        raise


@celery_app.task(name="app.tasks.evaluation_tasks.schedule_evaluation")
def schedule_evaluation(evaluation_type: str, schedule: str, test_cases: Optional[List[Dict]] = None):
    """
    调度定期评估
    """
    try:
        # 解析调度时间（简化处理）
        # 实际应用中应该使用更复杂的调度系统如celery-beat或APScheduler

        schedule_info = {
            "evaluation_type": evaluation_type,
            "schedule": schedule,
            "test_cases": test_cases,
            "created_at": datetime.utcnow().isoformat()
        }

        # 保存调度信息
        schedule_key = f"evaluation_schedule:{evaluation_type}"
        redis_client.setex(
            schedule_key,
            86400 * 7,  # 保存7天
            json.dumps(schedule_info, ensure_ascii=False)
        )

        logger.info(f"Evaluation scheduled: {evaluation_type} - {schedule}")
        return schedule_info

    except Exception as e:
        logger.error(f"Failed to schedule evaluation: {str(e)}")
        raise


@celery_app.task(name="app.tasks.evaluation_tasks.export_evaluation_data")
def export_evaluation_data(start_date: Optional[str] = None, end_date: Optional[str] = None,
                           format: str = "json") -> Dict[str, Any]:
    """
    导出评估数据
    """
    try:
        # 解析日期范围
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")

        # 收集评估数据
        evaluation_data = {
            "export_info": {
                "start_date": start_date,
                "end_date": end_date,
                "format": format,
                "exported_at": datetime.utcnow().isoformat()
            },
            "evaluations": [],
            "metrics_summary": {}
        }

        # 从Redis获取历史数据
        pattern = "evaluation_history:*"
        keys = redis_client.keys(pattern)

        for key in keys:
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            if ":" in key_str:
                date_part = key_str.split(":")[-1]
                if start_date <= date_part <= end_date:
                    data = redis_client.get(key)
                    if data:
                        evaluation_data["evaluations"].append(json.loads(data))

        # 生成指标摘要
        all_metrics = []
        for eval_data in evaluation_data["evaluations"]:
            all_metrics.extend(eval_data.get("metrics", []))

        if all_metrics:
            # 按指标类型分组统计
            metrics_by_type = {}
            for metric in all_metrics:
                metric_name = metric.get("metric_name")
                if metric_name not in metrics_by_type:
                    metrics_by_type[metric_name] = []
                metrics_by_type[metric_name].append(metric.get("value", 0))

            # 计算统计信息
            for metric_name, values in metrics_by_type.items():
                if values:
                    evaluation_data["metrics_summary"][metric_name] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }

        # 导出数据
        export_key = f"evaluation_export:{int(datetime.utcnow().timestamp())}"
        redis_client.setex(
            export_key,
            86400,  # 保存24小时
            json.dumps(evaluation_data, ensure_ascii=False)
        )

        return {
            "export_key": export_key,
            "record_count": len(evaluation_data["evaluations"]),
            "date_range": f"{start_date} to {end_date}"
        }

    except Exception as e:
        logger.error(f"Failed to export evaluation data: {str(e)}")
        raise


@celery_app.task(name="app.tasks.evaluation_tasks.baseline_comparison")
def baseline_comparison(evaluation_id: str, baseline_ids: List[str]) -> Dict[str, Any]:
    """
    与基线比较
    """
    try:
        # 获取当前评估结果
        current_key = f"evaluation_report:{evaluation_id}"
        current_data = redis_client.get(current_key)
        if not current_data:
            raise ValueError(f"Current evaluation not found: {evaluation_id}")

        current_report = json.loads(current_data)

        # 获取基线结果
        baseline_reports = []
        for baseline_id in baseline_ids:
            baseline_key = f"evaluation_report:{baseline_id}"
            baseline_data = redis_client.get(baseline_key)
            if baseline_data:
                baseline_reports.append(json.loads(baseline_data))

        if not baseline_reports:
            raise ValueError("No valid baseline reports found")

        # 生成比较报告
        comparison = {
            "evaluation_id": evaluation_id,
            "baseline_ids": baseline_ids,
            "comparison_timestamp": datetime.utcnow().isoformat(),
            "current_score": current_report.get("overall_score", 0),
            "baseline_scores": [report.get("overall_score", 0) for report in baseline_reports],
            "improvement": 0,
            "dimension_comparison": {},
            "recommendations": []
        }

        # 计算改进幅度
        if baseline_reports:
            baseline_avg = sum(report.get("overall_score", 0) for report in baseline_reports) / len(baseline_reports)
            improvement = current_report.get("overall_score", 0) - baseline_avg
            comparison["improvement"] = improvement
            comparison["baseline_average"] = baseline_avg

        # 维度比较
        current_dimensions = current_report.get("dimension_scores", {})
        for dimension, current_score in current_dimensions.items():
            baseline_scores = [report.get("dimension_scores", {}).get(dimension, 0) for report in baseline_reports]
            if baseline_scores:
                baseline_avg = sum(baseline_scores) / len(baseline_scores)
                comparison["dimension_comparison"][dimension] = {
                    "current": current_score,
                    "baseline_average": baseline_avg,
                    "improvement": current_score - baseline_avg
                }

        # 生成建议
        if comparison["improvement"] > 0.05:
            comparison["recommendations"].append("系统表现有所提升，继续保持")
        elif comparison["improvement"] < -0.05:
            comparison["recommendations"].append("系统性能下降，需要排查原因")
        else:
            comparison["recommendations"].append("系统表现稳定，可以考虑进一步优化")

        # 保存比较结果
        comparison_key = f"evaluation_comparison:{evaluation_id}"
        redis_client.setex(
            comparison_key,
            86400 * 7,  # 保存7天
            json.dumps(comparison, ensure_ascii=False)
        )

        logger.info(f"Baseline comparison completed: {evaluation_id}")
        return comparison

    except Exception as e:
        logger.error(f"Baseline comparison failed: {str(e)}")
        raise