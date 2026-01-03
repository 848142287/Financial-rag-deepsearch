"""
评估模块重试装饰器
为LLM调用等外部服务调用添加重试机制
"""

import asyncio
from app.core.structured_logging import get_structured_logger
from functools import wraps

logger = get_structured_logger(__name__)

def retry_with_exponential_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    指数退避重试装饰器

    Args:
        max_attempts: 最大重试次数
        base_delay: 基础延迟时间(秒)
        max_delay: 最大延迟时间(秒)
        exponential_base: 指数基数
        jitter: 是否添加随机抖动

    Example:
        @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
        async def evaluate_with_llm(query: str):
            return await llm_client.evaluate(query)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # 检查是否应该重试
                    recovery_strategy = get_error_recovery_strategy(e)

                    if recovery_strategy != "retry" or attempt == max_attempts - 1:
                        # 最后一次尝试或不可重试,直接抛出
                        raise

                    # 计算延迟时间
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # 添加随机抖动(±25%)
                    if jitter:
                        import random
                        delay = delay * (0.75 + random.random() * 0.5)

                    logger.warning(
                        f"{func.__name__} 失败 (尝试 {attempt + 1}/{max_attempts}): {e}",
                        extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'max_attempts': max_attempts,
                            'delay_seconds': delay,
                            'error_type': type(e).__name__,
                            'error_message': str(e)
                        }
                    )

                    await asyncio.sleep(delay)

            # 所有重试都失败
            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 同步函数的重试支持
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    recovery_strategy = get_error_recovery_strategy(e)

                    if recovery_strategy != "retry" or attempt == max_attempts - 1:
                        raise

                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    if jitter:
                        import random
                        delay = delay * (0.75 + random.random() * 0.5)

                    logger.warning(
                        f"{func.__name__} 失败 (尝试 {attempt + 1}/{max_attempts}): {e}",
                        extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'delay_seconds': delay
                        }
                    )

                    time.sleep(delay)

            raise last_exception

        # 根据函数类型返回合适的wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def retry_on_specific_errors(
    error_types: Tuple[Type[Exception], ...],
    max_attempts: int = 3,
    delay: float = 1.0
):
    """
    只对特定错误类型重试

    Args:
        error_types: 需要重试的错误类型元组
        max_attempts: 最大重试次数
        delay: 重试延迟时间

    Example:
        @retry_on_specific_errors(
            error_types=(LLMTimeoutError, LLMRateLimitError),
            max_attempts=3
        )
        async def call_llm_api():
            return await llm_client.generate()
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except error_types as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        raise

                    logger.warning(
                        f"{func.__name__} 遇到可重试错误 "
                        f"(尝试 {attempt + 1}/{max_attempts}): {e}",
                        extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'error_type': type(e).__name__
                        }
                    )

                    await asyncio.sleep(delay)

                except Exception:
                    # 其他错误直接抛出
                    raise

            raise last_exception

        return async_wrapper

    return decorator

class RetryTracker:
    """重试统计跟踪器"""

    def __init__(self):
        self.retry_counts = {}  # {function_name: retry_count}
        self.failure_counts = {}  # {function_name: failure_count}

    def record_retry(self, function_name: str, attempt: int):
        """记录重试"""
        if function_name not in self.retry_counts:
            self.retry_counts[function_name] = 0
        self.retry_counts[function_name] += 1

    def record_failure(self, function_name: str):
        """记录失败"""
        if function_name not in self.failure_counts:
            self.failure_counts[function_name] = 0
        self.failure_counts[function_name] += 1

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'retry_counts': self.retry_counts.copy(),
            'failure_counts': self.failure_counts.copy(),
            'total_retries': sum(self.retry_counts.values()),
            'total_failures': sum(self.failure_counts.values())
        }

# 全局重试跟踪器
retry_tracker = RetryTracker()

def tracked_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0
):
    """
    带统计的重试装饰器

    与retry_with_exponential_backoff相同的功能,但会记录重试统计
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    if attempt > 0:
                        retry_tracker.record_retry(func.__name__, attempt)

                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    recovery_strategy = get_error_recovery_strategy(e)

                    if recovery_strategy != "retry" or attempt == max_attempts - 1:
                        retry_tracker.record_failure(func.__name__)
                        raise

                    delay = min(base_delay * (2 ** attempt), 60.0)

                    logger.warning(
                        f"{func.__name__} 失败, {delay}秒后重试 "
                        f"(尝试 {attempt + 1}/{max_attempts})"
                    )

                    await asyncio.sleep(delay)

            retry_tracker.record_failure(func.__name__)
            raise last_exception

        return async_wrapper

    return decorator

# 导出
__all__ = [
    'retry_with_exponential_backoff',
    'retry_on_specific_errors',
    'RetryTracker',
    'retry_tracker',
    'tracked_retry'
]
