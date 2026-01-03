"""
评估模块错误定义
定义评估系统中的各种异常类型和错误处理策略
"""

from typing import Optional, Type, Dict, Any
from enum import Enum


class ErrorRecoveryStrategy(Enum):
    """错误恢复策略"""
    RETRY = "retry"              # 重试
    FAIL = "fail"                # 失败
    FALLBACK = "fallback"        # 降级
    IGNORE = "ignore"            # 忽略


class EvaluationError(Exception):
    """评估错误基类"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


class LLMTimeoutError(EvaluationError):
    """LLM调用超时错误"""

    def __init__(
        self,
        message: str = "LLM调用超时",
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        details = {"timeout_seconds": timeout_seconds}
        details.update(kwargs)
        super().__init__(message, details=details)


class LLMRateLimitError(EvaluationError):
    """LLM调用频率限制错误"""

    def __init__(
        self,
        message: str = "LLM调用频率限制",
        retry_after_seconds: Optional[float] = None,
        **kwargs
    ):
        details = {"retry_after_seconds": retry_after_seconds}
        details.update(kwargs)
        super().__init__(message, details=details)


class EvaluationMetricError(EvaluationError):
    """评估指标计算错误"""

    def __init__(
        self,
        message: str = "评估指标计算失败",
        metric_name: Optional[str] = None,
        **kwargs
    ):
        details = {"metric_name": metric_name}
        details.update(kwargs)
        super().__init__(message, details=details)


class CacheOperationError(EvaluationError):
    """缓存操作错误"""

    def __init__(
        self,
        message: str = "缓存操作失败",
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = {
            "cache_key": cache_key,
            "operation": operation
        }
        details.update(kwargs)
        super().__init__(message, details=details)


class FeedbackProcessingError(EvaluationError):
    """反馈处理错误"""

    def __init__(
        self,
        message: str = "反馈处理失败",
        feedback_type: Optional[str] = None,
        **kwargs
    ):
        details = {"feedback_type": feedback_type}
        details.update(kwargs)
        super().__init__(message, details=details)


class QueryEnhancementError(EvaluationError):
    """查询增强错误"""

    def __init__(
        self,
        message: str = "查询增强失败",
        original_query: Optional[str] = None,
        **kwargs
    ):
        details = {"original_query": original_query}
        details.update(kwargs)
        super().__init__(message, details=details)


class RerankError(EvaluationError):
    """重排序错误"""

    def __init__(
        self,
        message: str = "重排序失败",
        num_documents: Optional[int] = None,
        **kwargs
    ):
        details = {"num_documents": num_documents}
        details.update(kwargs)
        super().__init__(message, details=details)


class RetrievalOptimizationError(EvaluationError):
    """检索优化错误"""

    def __init__(
        self,
        message: str = "检索优化失败",
        optimization_type: Optional[str] = None,
        **kwargs
    ):
        details = {"optimization_type": optimization_type}
        details.update(kwargs)
        super().__init__(message, details=details)


# 错误类型到恢复策略的映射
_ERROR_RECOVERY_STRATEGIES: Dict[Type[Exception], ErrorRecoveryStrategy] = {
    LLMTimeoutError: ErrorRecoveryStrategy.RETRY,
    LLMRateLimitError: ErrorRecoveryStrategy.RETRY,
    EvaluationMetricError: ErrorRecoveryStrategy.FAIL,
    CacheOperationError: ErrorRecoveryStrategy.IGNORE,
    FeedbackProcessingError: ErrorRecoveryStrategy.FAIL,
    QueryEnhancementError: ErrorRecoveryStrategy.FALLBACK,
    RerankError: ErrorRecoveryStrategy.FALLBACK,
    RetrievalOptimizationError: ErrorRecoveryStrategy.FAIL,
}


def is_evaluation_error(error: Exception) -> bool:
    """
    检查是否为评估错误

    Args:
        error: 异常对象

    Returns:
        是否为评估错误
    """
    return isinstance(error, EvaluationError)


def get_error_recovery_strategy(
    error: Exception,
    default: ErrorRecoveryStrategy = ErrorRecoveryStrategy.FAIL
) -> str:
    """
    获取错误恢复策略

    Args:
        error: 异常对象
        default: 默认策略

    Returns:
        恢复策略字符串
    """
    # 如果是评估错误，查找对应的策略
    if is_evaluation_error(error):
        error_type = type(error)
        if error_type in _ERROR_RECOVERY_STRATEGIES:
            return _ERROR_RECOVERY_STRATEGIES[error_type].value

        # 检查父类
        for error_cls, strategy in _ERROR_RECOVERY_STRATEGIES.items():
            if isinstance(error, error_cls):
                return strategy.value

    # 对于超时、连接等错误，默认重试
    error_name = type(error).__name__
    if any(keyword in error_name.lower() for keyword in ['timeout', 'connection', 'temporal']):
        return ErrorRecoveryStrategy.RETRY.value

    return default.value


# 导出所有错误类和函数
__all__ = [
    # 错误恢复策略枚举
    'ErrorRecoveryStrategy',

    # 错误类
    'EvaluationError',
    'LLMTimeoutError',
    'LLMRateLimitError',
    'EvaluationMetricError',
    'CacheOperationError',
    'FeedbackProcessingError',
    'QueryEnhancementError',
    'RerankError',
    'RetrievalOptimizationError',

    # 工具函数
    'is_evaluation_error',
    'get_error_recovery_strategy',
]
