"""
评估公共模块
统一评估流水线的错误处理、缓存、追踪等公共组件
"""

# 核心模块

# 缓存管理

# 版本信息
__version__ = '1.0.0'
__author__ = 'Claude Code'

# 导出所有公共组件
__all__ = [
    # 核心模块
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
    'retry_with_exponential_backoff',
    'retry_on_specific_errors',
    'tracked_retry',
    'EvaluationTracer',
    'evaluation_tracer',
    'create_evaluation_trace',
    'trace_evaluation_stage',
    'finish_evaluation_trace',

    # 缓存管理
    'CachedEvaluation',
    'EvaluationCacheManager',
    'evaluation_cache_manager',
]
