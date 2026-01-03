"""
评估公共核心模块
导出核心组件:错误处理、重试、追踪
"""

# 导出所有公共组件
__all__ = [
    # 错误处理
    'EvaluationError',
    'LLMTimeoutError',
    'LLMRateLimitError',
    'EvaluationMetricError',
    'CacheOperationError',
    'FeedbackProcessingError',
    'QueryEnhancementError',
    'RerankError',
    'RetrievalOptimizationError',
    'is_evaluation_error',
    'get_error_recovery_strategy',

    # 重试机制
    'retry_with_exponential_backoff',
    'retry_on_specific_errors',
    'RetryTracker',
    'retry_tracker',
    'tracked_retry',

    # 链路追踪
    'EvaluationTracer',
    'evaluation_tracer',
    'create_evaluation_trace',
    'trace_evaluation_stage',
    'finish_evaluation_trace',
]
