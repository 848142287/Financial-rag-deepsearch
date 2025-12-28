"""
统一错误处理和回退机制
提供文档处理过程中的错误处理、重试和恢复功能
"""

import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """错误严重程度"""
    LOW = "low"           # 轻微错误，不影响主要功能
    MEDIUM = "medium"     # 中等错误，影响部分功能
    HIGH = "high"         # 严重错误，影响主要功能
    CRITICAL = "critical" # 致命错误，需要立即处理


class ErrorCategory(str, Enum):
    """错误分类"""
    PARSING = "parsing"           # 解析错误
    STORAGE = "storage"           # 存储错误
    NETWORK = "network"           # 网络错误
    MODEL = "model"               # 模型错误
    DATABASE = "database"         # 数据库错误
    VALIDATION = "validation"     # 验证错误
    TIMEOUT = "timeout"           # 超时错误
    RESOURCE = "resource"         # 资源错误
    UNKNOWN = "unknown"           # 未知错误


class RetryStrategy(str, Enum):
    """重试策略"""
    IMMEDIATE = "immediate"       # 立即重试
    EXPONENTIAL = "exponential"   # 指数退避
    LINEAR = "linear"             # 线性退避
    FIXED = "fixed"               # 固定间隔
    NONE = "none"                 # 不重试


@dataclass
class ErrorInfo:
    """错误信息"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    max_retries: int = 3
    next_retry_time: Optional[datetime] = None


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True


class ErrorHandler:
    """统一错误处理器"""

    def __init__(self):
        self.error_history: Dict[str, ErrorInfo] = {}
        self.error_stats: Dict[ErrorCategory, int] = {cat: 0 for cat in ErrorCategory}
        self.retry_queue: List[ErrorInfo] = []
        self.fallback_handlers: Dict[ErrorCategory, List[Callable]] = {}

    def categorize_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """分类错误并创建错误信息"""
        context = context or {}

        # 根据异常类型和上下文分类
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.MEDIUM

        exception_type = type(exception).__name__
        exception_message = str(exception).lower()

        # 网络错误
        if any(keyword in exception_message for keyword in [
            "connection", "timeout", "network", "unreachable"
        ]) or exception_type in [
            "ConnectionError", "TimeoutError", "NetworkError"
        ]:
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.MEDIUM

        # 存储错误
        elif any(keyword in exception_message for keyword in [
            "storage", "disk", "file", "path", "minio", "s3"
        ]) or exception_type in [
            "FileNotFoundError", "PermissionError", "StorageError"
        ]:
            category = ErrorCategory.STORAGE
            severity = ErrorSeverity.HIGH

        # 数据库错误
        elif any(keyword in exception_message for keyword in [
            "database", "sql", "mysql", "mongo", "redis", "connection"
        ]) or exception_type in [
            "DatabaseError", "IntegrityError", "OperationalError"
        ]:
            category = ErrorCategory.DATABASE
            severity = ErrorSeverity.HIGH

        # 模型错误
        elif any(keyword in exception_message for keyword in [
            "model", "embedding", "bge", "glm", "torch", "cuda"
        ]) or exception_type in [
            "ModelError", "CUDAError", "TorchError"
        ]:
            category = ErrorCategory.MODEL
            severity = ErrorSeverity.MEDIUM

        # 解析错误
        elif any(keyword in exception_message for keyword in [
            "parse", "pdf", "docx", "document", "corrupt"
        ]) or exception_type in [
            "ParseError", "PDFError", "DocumentError"
        ]:
            category = ErrorCategory.PARSING
            severity = ErrorSeverity.MEDIUM

        # 超时错误
        elif "timeout" in exception_message or exception_type in [
            "TimeoutError", "AsyncTimeoutError"
        ]:
            category = ErrorCategory.TIMEOUT
            severity = ErrorSeverity.MEDIUM

        # 资源错误
        elif any(keyword in exception_message for keyword in [
            "memory", "cpu", "resource", "limit"
        ]) or exception_type in [
            "MemoryError", "ResourceWarning"
        ]:
            category = ErrorCategory.RESOURCE
            severity = ErrorSeverity.HIGH

        # 验证错误
        elif any(keyword in exception_message for keyword in [
            "validation", "invalid", "format", "schema"
        ]) or exception_type in [
            "ValidationError", "ValueError"
        ]:
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.LOW

        # 生成错误ID
        error_id = f"{category.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(exception_message) % 10000}"

        return ErrorInfo(
            error_id=error_id,
            category=category,
            severity=severity,
            message=exception_message,
            exception=exception,
            stack_trace=traceback.format_exc(),
            context=context,
            max_retries=self._get_default_retries(category, severity)
        )

    def _get_default_retries(self, category: ErrorCategory, severity: ErrorSeverity) -> int:
        """获取默认重试次数"""
        if severity == ErrorSeverity.CRITICAL:
            return 0
        elif category == ErrorCategory.NETWORK:
            return 5
        elif category == ErrorCategory.TIMEOUT:
            return 3
        elif severity == ErrorSeverity.HIGH:
            return 2
        else:
            return 3

    async def handle_error(
        self,
        exception: Exception,
        context: Dict[str, Any] = None,
        retry_config: Optional[RetryConfig] = None
    ) -> ErrorInfo:
        """处理错误"""
        error_info = self.categorize_error(exception, context)

        # 更新统计
        self.error_stats[error_info.category] += 1
        self.error_history[error_info.error_id] = error_info

        # 记录错误
        self._log_error(error_info)

        # 如果还有重试机会，加入重试队列
        if error_info.retry_count < error_info.max_retries:
            retry_config = retry_config or RetryConfig()
            error_info.next_retry_time = self._calculate_retry_time(
                error_info.retry_count, retry_config
            )
            self.retry_queue.append(error_info)

        # 尝试回退处理
        await self._try_fallback_handlers(error_info)

        return error_info

    def _log_error(self, error_info: ErrorInfo):
        """记录错误日志"""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_info.severity, logging.ERROR)

        logger.log(
            log_level,
            f"[{error_info.category.value.upper()}] {error_info.message} "
            f"(ID: {error_info.error_id}, Retry: {error_info.retry_count}/{error_info.max_retries})"
        )

        if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.debug(f"Stack trace: {error_info.stack_trace}")

    def _calculate_retry_time(
        self,
        retry_count: int,
        config: RetryConfig
    ) -> datetime:
        """计算下次重试时间"""
        if config.strategy == RetryStrategy.IMMEDIATE:
            delay = 0
        elif config.strategy == RetryStrategy.FIXED:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.LINEAR:
            delay = config.base_delay * (retry_count + 1)
        elif config.strategy == RetryStrategy.EXPONENTIAL:
            delay = config.base_delay * (config.backoff_factor ** retry_count)
        else:
            delay = config.base_delay

        # 应用最大延迟限制
        delay = min(delay, config.max_delay)

        # 添加抖动
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)

        return datetime.utcnow() + timedelta(seconds=delay)

    async def _try_fallback_handlers(self, error_info: ErrorInfo):
        """尝试回退处理"""
        handlers = self.fallback_handlers.get(error_info.category, [])

        for handler in handlers:
            try:
                await handler(error_info)
                logger.info(f"Fallback handler executed successfully for {error_info.error_id}")
                break
            except Exception as e:
                logger.warning(f"Fallback handler failed for {error_info.error_id}: {e}")

    def register_fallback_handler(
        self,
        category: ErrorCategory,
        handler: Callable[[ErrorInfo], Any]
    ):
        """注册回退处理器"""
        if category not in self.fallback_handlers:
            self.fallback_handlers[category] = []
        self.fallback_handlers[category].append(handler)

    async def process_retry_queue(self) -> List[ErrorInfo]:
        """处理重试队列"""
        current_time = datetime.utcnow()
        ready_retries = [
            error for error in self.retry_queue
            if error.next_retry_time and error.next_retry_time <= current_time
        ]

        processed_retries = []
        for error_info in ready_retries:
            try:
                # 重试次数加1
                error_info.retry_count += 1

                # 从队列中移除
                self.retry_queue.remove(error_info)

                # 记录重试
                logger.info(f"Retrying operation for error {error_info.error_id} "
                          f"(attempt {error_info.retry_count}/{error_info.max_retries})")

                processed_retries.append(error_info)

            except Exception as e:
                logger.error(f"Failed to process retry for {error_info.error_id}: {e}")

        return processed_retries

    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        recent_errors = [
            error for error in self.error_history.values()
            if (datetime.utcnow() - error.timestamp).total_seconds() < 3600  # 最近1小时
        ]

        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "pending_retries": len(self.retry_queue),
            "error_by_category": dict(self.error_stats),
            "error_by_severity": self._group_by_severity(),
            "most_common_errors": self._get_most_common_errors()
        }

    def _group_by_severity(self) -> Dict[str, int]:
        """按严重程度分组"""
        severity_count = {severity.value: 0 for severity in ErrorSeverity}
        for error in self.error_history.values():
            severity_count[error.severity.value] += 1
        return severity_count

    def _get_most_common_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最常见的错误"""
        from collections import Counter

        error_messages = [error.message for error in self.error_history.values()]
        common_errors = Counter(error_messages).most_common(limit)

        return [
            {"message": message, "count": count}
            for message, count in common_errors
        ]

    def clear_old_errors(self, hours: int = 24):
        """清理旧错误记录"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        old_errors = [
            error_id for error_id, error in self.error_history.items()
            if error.timestamp < cutoff_time
        ]

        for error_id in old_errors:
            del self.error_history[error_id]

        logger.info(f"Cleared {len(old_errors)} old error records")


# 全局错误处理器实例
error_handler = ErrorHandler()


def retry_on_failure(
    retry_config: Optional[RetryConfig] = None,
    fallback_handler: Optional[Callable] = None
):
    """装饰器：在失败时重试"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            config = retry_config or RetryConfig()
            last_error = None

            for attempt in range(config.max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                except Exception as e:
                    last_error = e

                    if attempt < config.max_retries:
                        # 计算延迟
                        if config.strategy == RetryStrategy.EXPONENTIAL:
                            delay = config.base_delay * (config.backoff_factor ** attempt)
                        elif config.strategy == RetryStrategy.LINEAR:
                            delay = config.base_delay * (attempt + 1)
                        else:
                            delay = config.base_delay

                        delay = min(delay, config.max_delay)

                        if config.jitter:
                            import random
                            delay *= (0.5 + random.random() * 0.5)

                        logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}), "
                                     f"retrying in {delay:.2f}s: {e}")
                        await asyncio.sleep(delay)
                    else:
                        # 所有重试都失败，尝试回退处理
                        if fallback_handler:
                            try:
                                return await fallback_handler(e, *args, **kwargs)
                            except Exception as fallback_error:
                                logger.error(f"Fallback handler also failed: {fallback_error}")

                        # 最后一次尝试失败，抛出异常
                        raise last_error

        return async_wrapper
    return decorator


def safe_execute(
    default_return: Any = None,
    error_category: ErrorCategory = ErrorCategory.UNKNOWN,
    log_error: bool = True
):
    """装饰器：安全执行函数，捕获异常"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    await error_handler.handle_error(e, {
                        "function": func.__name__,
                        "args": str(args)[:200],
                        "kwargs": str(kwargs)[:200]
                    })
                return default_return
        return async_wrapper
    return decorator


# 便捷函数
async def handle_error_with_fallback(
    exception: Exception,
    context: Dict[str, Any],
    primary_operation: Callable,
    fallback_operation: Callable
) -> Any:
    """带回退的错误处理"""
    try:
        return await primary_operation()
    except Exception as e:
        await error_handler.handle_error(e, context)
        try:
            return await fallback_operation()
        except Exception as fallback_error:
            logger.error(f"Both primary and fallback operations failed: "
                        f"Primary: {e}, Fallback: {fallback_error}")
            raise fallback_error


# 注册一些默认的回退处理器
async def _network_fallback_handler(error_info: ErrorInfo):
    """网络错误的回退处理"""
    if "retry" in error_info.context.get("operation", "").lower():
        logger.info("Network operation failed, will retry with exponential backoff")
        # 这里可以添加网络特定的回退逻辑


async def _storage_fallback_handler(error_info: ErrorInfo):
    """存储错误的回退处理"""
    if "cache" in error_info.context.get("operation", "").lower():
        logger.info("Cache storage failed, falling back to in-memory storage")
        # 这里可以添加存储特定的回退逻辑


async def _model_fallback_handler(error_info: ErrorInfo):
    """模型错误的回退处理"""
    if "embedding" in error_info.context.get("operation", "").lower():
        logger.info("Primary embedding model failed, falling back to backup model")
        # 这里可以添加模型特定的回退逻辑


# 注册默认回退处理器
error_handler.register_fallback_handler(ErrorCategory.NETWORK, _network_fallback_handler)
error_handler.register_fallback_handler(ErrorCategory.STORAGE, _storage_fallback_handler)
error_handler.register_fallback_handler(ErrorCategory.MODEL, _model_fallback_handler)