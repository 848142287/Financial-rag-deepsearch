"""
统一日志工具模块
提供所有模块应该使用的标准化日志接口
"""

import logging
from functools import wraps
import time
import asyncio

from app.core.logging_config import get_logger

def get_module_logger(module_name: str) -> logging.Logger:
    """
    获取模块日志记录器的统一接口

    Args:
        module_name: 模块名称，通常使用 __name__

    Returns:
        配置好的日志记录器

    Example:
        from app.core.logger_utils import get_module_logger
        logger = get_module_logger(__name__)
        logger.info("Processing started")
    """
    return get_logger(module_name)

def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    记录函数执行时间的装饰器

    Args:
        logger: 日志记录器，如果为None则自动获取

    Example:
        @log_execution_time()
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        # 获取logger
        _logger = logger or get_module_logger(func.__module__)

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    _logger.info(
                        f"[PERF] {func.__name__} completed in {elapsed:.2f}s"
                    )
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    _logger.error(
                        f"[PERF] {func.__name__} failed after {elapsed:.2f}s: {e}"
                    )
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    _logger.info(
                        f"[PERF] {func.__name__} completed in {elapsed:.2f}s"
                    )
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    _logger.error(
                        f"[PERF] {func.__name__} failed after {elapsed:.2f}s: {e}"
                    )
                    raise
            return sync_wrapper
    return decorator

def log_function_call(
    logger: Optional[logging.Logger] = None,
    log_level: int = logging.DEBUG,
    log_result: bool = False
):
    """
    记录函数调用和参数的装饰器

    Args:
        logger: 日志记录器
        log_level: 日志级别
        log_result: 是否记录返回值

    Example:
        @log_function_call(log_result=True)
        def my_function(arg1, arg2):
            return arg1 + arg2
    """
    def decorator(func: Callable) -> Callable:
        _logger = logger or get_module_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 记录调用
            _logger.log(
                log_level,
                f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
            )

            # 执行函数
            result = func(*args, **kwargs)

            # 记录结果
            if log_result:
                _logger.log(
                    log_level,
                    f"{func.__name__} returned {result}"
                )

            return result
        return wrapper
    return decorator

class ContextLogger:
    """
    带上下文的日志记录器
    自动在日志中添加业务上下文信息（如document_id, user_id等）
    """

    def __init__(self, base_logger: logging.Logger):
        self._logger = base_logger
        self._context = {}

    def with_context(self, **kwargs) -> 'ContextLogger':
        """
        添加上下文信息

        Example:
            logger.with_context(document_id=123, user_id=456).info("Processing")
        """
        new_context_logger = ContextLogger(self._logger)
        new_context_logger._context = {**self._context, **kwargs}
        return new_context_logger

    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """带上下文的日志记录"""
        if self._context:
            context_str = " | ".join([f"{k}={v}" for k, v in self._context.items()])
            msg = f"[{context_str}] {msg}"
        self._logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
        self._logger.exception(msg, *args, **kwargs)

def get_context_logger(module_name: str) -> ContextLogger:
    """
    获取带上下文的日志记录器

    Example:
        logger = get_context_logger(__name__)
        logger.with_context(document_id=123).info("Processing document")
    """
    return ContextLogger(get_module_logger(module_name))
