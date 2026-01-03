"""
评估模块链路追踪集成
连接rag_common追踪系统
"""

from app.core.structured_logging import get_structured_logger
from datetime import datetime
from contextlib import asynccontextmanager

# 导入rag_common追踪系统
from app.services.rag_common.tracing.rag_tracer import (
    get_rag_tracer,
    TraceStage
)
from app.services.rag_common.core.logging_config import set_trace_id

logger = get_structured_logger(__name__)

class EvaluationTracer:
    """评估专用追踪器包装器"""

    def __init__(self):
        self.rag_tracer = get_rag_tracer()

    async def create_evaluation_trace(
        self,
        query: str,
        evaluation_type: str = "ragas",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        创建评估追踪

        Args:
            query: 查询文本
            evaluation_type: 评估类型 (ragas, feedback, optimization等)
            user_id: 用户ID
            metadata: 额外元数据

        Returns:
            trace_id
        """
        trace_metadata = {
            'evaluation_type': evaluation_type,
            **(metadata or {})
        }

        trace_id = self.rag_tracer.create_trace(
            query=query,
            user_id=user_id,
            metadata=trace_metadata
        )

        # 设置trace_id到日志上下文
        set_trace_id(trace_id)

        logger.info(
            f"创建评估追踪: {evaluation_type}",
            extra={
                'evaluation_type': evaluation_type,
                'query_length': len(query)
            }
        )

        return trace_id

    @asynccontextmanager
    async def trace_evaluation_stage(
        self,
        trace_id: str,
        stage_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        追踪评估阶段

        Args:
            trace_id: 追踪ID
            stage_name: 阶段名称
            metadata: 阶段元数据

        Example:
            async with tracer.trace_evaluation_stage(trace_id, "faithfulness_eval"):
                result = await evaluate_faithfulness(...)
        """
        span_id = self.rag_tracer.add_span(
            trace_id=trace_id,
            stage=TraceStage.RETRIEVAL,  # 复用RETRIEVAL阶段
            metadata=metadata or {}
        )

        logger.info(
            f"开始评估阶段: {stage_name}",
            extra={'stage': stage_name}
        )

        start_time = datetime.now()

        try:
            yield

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            self.rag_tracer.finish_span(
                trace_id,
                span_id,
                status="success",
                metadata={
                    'duration_ms': duration_ms,
                    **(metadata or {})
                }
            )

            logger.info(
                f"完成评估阶段: {stage_name}",
                extra={
                    'stage': stage_name,
                    'duration_ms': duration_ms
                }
            )

        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            self.rag_tracer.finish_span(
                trace_id,
                span_id,
                status="failed",
                metadata={
                    'error': str(e),
                    'duration_ms': duration_ms
                }
            )

            logger.error(
                f"评估阶段失败: {stage_name}",
                extra={
                    'stage': stage_name,
                    'error': str(e),
                    'duration_ms': duration_ms
                }
            )

            raise

    async def finish_evaluation_trace(
        self,
        trace_id: str,
        status: str = "completed",
        results: Optional[Dict[str, Any]] = None
    ):
        """
        完成评估追踪

        Args:
            trace_id: 追踪ID
            status: 完成状态 (completed, failed, partial)
            results: 评估结果
        """
        metadata = {}
        if results:
            # 提取关键指标
            if 'metrics' in results:
                metadata['metrics_summary'] = {
                    k: v for k, v in results['metrics'].items()
                    if isinstance(v, (int, float))
                }

            metadata.update(results)

        self.rag_tracer.finish_trace(
            trace_id,
            status=status,
            metadata=metadata
        )

        logger.info(
            f"完成评估追踪: {status}",
            extra={
                'status': status,
                'has_results': results is not None
            }
        )

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        stats = self.rag_tracer.get_statistics()

        # 添加评估特定统计
        eval_stats = {
            'total_evaluations': stats.get('total_traces', 0),
            'successful_evaluations': stats.get('status_distribution', {}).get('completed', 0),
            'failed_evaluations': stats.get('status_distribution', {}).get('failed', 0),
            'avg_evaluation_duration_ms': stats.get('avg_duration_ms', 0),
            'stage_avg_durations_ms': stats.get('stage_avg_durations_ms', {})
        }

        return eval_stats

# 全局评估追踪器
evaluation_tracer = EvaluationTracer()

# 便捷函数
async def create_evaluation_trace(
    query: str,
    evaluation_type: str = "ragas",
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """创建评估追踪(便捷函数)"""
    return await evaluation_tracer.create_evaluation_trace(
        query=query,
        evaluation_type=evaluation_type,
        user_id=user_id,
        metadata=metadata
    )

@asynccontextmanager
async def trace_evaluation_stage(
    trace_id: str,
    stage_name: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """追踪评估阶段(便捷函数)"""
    async with evaluation_tracer.trace_evaluation_stage(
        trace_id, stage_name, metadata
    ):
        yield

async def finish_evaluation_trace(
    trace_id: str,
    status: str = "completed",
    results: Optional[Dict[str, Any]] = None
):
    """完成评估追踪(便捷函数)"""
    await evaluation_tracer.finish_evaluation_trace(
        trace_id, status, results
    )

# 导出
__all__ = [
    'EvaluationTracer',
    'evaluation_tracer',
    'create_evaluation_trace',
    'trace_evaluation_stage',
    'finish_evaluation_trace',
    'get_evaluation_statistics'
]
