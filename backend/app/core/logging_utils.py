"""
统一日志工具集
提供便捷的日志初始化和管理功能
"""

import logging
import functools
from contextvars import ContextVar
from app.core.logging_config import get_logger

# 上下文变量：用于跨异步上下文传递追踪信息
_request_id_ctx: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
_user_id_ctx: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
_document_id_ctx: ContextVar[Optional[str]] = ContextVar('document_id', default=None)

class LoggerMixin:
    """
    日志混入类
    为类自动添加 logger 属性
    """

    @property
    def logger(self) -> logging.Logger:
        """获取或创建 logger"""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger

def with_logger(cls: type) -> type:
    """
    类装饰器：自动为类添加 logger 属性

    Example:
        @with_logger
        class MyService:
            def my_method(self):
                self.logger.info("Using injected logger")
    """
    cls.logger = property(lambda self: get_logger(self.__class__.__name__))
    return cls

def log_async_function(
    logger: Optional[logging.Logger] = None,
    level: str = "INFO",
    include_args: bool = False,
    include_result: bool = False
):
    """
    异步函数日志装饰器

    Args:
        logger: 日志记录器，默认使用函数所在模块的 logger
        level: 日志级别
        include_args: 是否记录参数
        include_result: 是否记录返回值
    """
    def decorator(func: Callable) -> Callable:
        module_logger = logger or get_logger(func.__module__)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = func.__name__

            # 记录函数调用
            log_parts = [f"调用函数: {func_name}"]
            if include_args:
                args_str = ", ".join(str(a) for a in args)
                kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                log_parts.append(f"参数: args=[{args_str}], kwargs={{{kwargs_str}}}")

            getattr(module_logger, level.lower())(" | ".join(log_parts))

            try:
                result = await func(*args, **kwargs)

                # 记录返回值
                if include_result:
                    result_str = str(result)[:200]  # 限制长度
                    getattr(module_logger, level.lower())(
                        f"函数 {func_name} 执行成功 | 返回: {result_str}"
                    )
                else:
                    getattr(module_logger, level.lower())(f"函数 {func_name} 执行成功")

                return result

            except Exception as e:
                module_logger.error(f"函数 {func_name} 执行失败: {str(e)}", exc_info=True)
                raise

        return wrapper
    return decorator

def log_function(
    logger: Optional[logging.Logger] = None,
    level: str = "INFO",
    include_args: bool = False,
    include_result: bool = False
):
    """
    同步函数日志装饰器

    Args:
        logger: 日志记录器，默认使用函数所在模块的 logger
        level: 日志级别
        include_args: 是否记录参数
        include_result: 是否记录返回值
    """
    def decorator(func: Callable) -> Callable:
        module_logger = logger or get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # 记录函数调用
            log_parts = [f"调用函数: {func_name}"]
            if include_args:
                args_str = ", ".join(str(a) for a in args)
                kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                log_parts.append(f"参数: args=[{args_str}], kwargs={{{kwargs_str}}}")

            getattr(module_logger, level.lower())(" | ".join(log_parts))

            try:
                result = func(*args, **kwargs)

                # 记录返回值
                if include_result:
                    result_str = str(result)[:200]
                    getattr(module_logger, level.lower())(
                        f"函数 {func_name} 执行成功 | 返回: {result_str}"
                    )
                else:
                    getattr(module_logger, level.lower())(f"函数 {func_name} 执行成功")

                return result

            except Exception as e:
                module_logger.error(f"函数 {func_name} 执行失败: {str(e)}", exc_info=True)
                raise

        return wrapper
    return decorator

class RequestContext:
    """请求上下文管理器"""

    @staticmethod
    def set_request_id(request_id: str):
        """设置请求ID"""
        _request_id_ctx.set(request_id)

    @staticmethod
    def set_user_id(user_id: str):
        """设置用户ID"""
        _user_id_ctx.set(user_id)

    @staticmethod
    def set_document_id(document_id: str):
        """设置文档ID"""
        _document_id_ctx.set(document_id)

    @staticmethod
    def get_request_id() -> Optional[str]:
        """获取请求ID"""
        return _request_id_ctx.get()

    @staticmethod
    def get_user_id() -> Optional[str]:
        """获取用户ID"""
        return _user_id_ctx.get()

    @staticmethod
    def get_document_id() -> Optional[str]:
        """获取文档ID"""
        return _document_id_ctx.get()

    @staticmethod
    def clear():
        """清空上下文"""
        _request_id_ctx.set(None)
        _user_id_ctx.set(None)
        _document_id_ctx.set(None)

class ContextualLogger:
    """
    带上下文的日志记录器
    自动在日志中包含请求上下文信息
    """

    def __init__(self, name: str):
        self._logger = get_logger(name)

    def _add_context(self, extra: dict) -> dict:
        """添加上下文信息到 extra"""
        extra = extra.copy() if extra else {}

        if request_id := RequestContext.get_request_id():
            extra['request_id'] = request_id
        if user_id := RequestContext.get_user_id():
            extra['user_id'] = user_id
        if document_id := RequestContext.get_document_id():
            extra['document_id'] = document_id

        return extra

    def debug(self, msg: str, **kwargs):
        self._logger.debug(msg, extra=self._add_context(kwargs))

    def info(self, msg: str, **kwargs):
        self._logger.info(msg, extra=self._add_context(kwargs))

    def warning(self, msg: str, **kwargs):
        self._logger.warning(msg, extra=self._add_context(kwargs))

    def error(self, msg: str, exc_info: bool = False, **kwargs):
        self._logger.error(msg, extra=self._add_context(kwargs), exc_info=exc_info)

    def critical(self, msg: str, exc_info: bool = False, **kwargs):
        self._logger.critical(msg, extra=self._add_context(kwargs), exc_info=exc_info)

def get_contextual_logger(name: str) -> ContextualLogger:
    """
    获取带上下文的日志记录器

    Example:
        logger = get_contextual_logger(__name__)

        # 设置上下文
        RequestContext.set_request_id("req-123")
        RequestContext.set_user_id("user-456")

        # 日志会自动包含上下文
        logger.info("Processing document")
    """
    return ContextualLogger(name)

# 便捷函数
def get_class_logger(cls: type) -> logging.Logger:
    """获取类的 logger"""
    return get_logger(cls.__name__)

def get_module_logger(module_name: str) -> logging.Logger:
    """获取模块的 logger"""
    return get_logger(module_name)
