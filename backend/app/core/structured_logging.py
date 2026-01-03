"""
结构化日志框架
提供统一的、带上下文的日志记录功能
"""

import logging
import json
import uuid
import traceback
from typing import Any, Dict, Optional
from datetime import datetime
from contextvars import ContextVar
from functools import wraps
import threading

# 上下文变量 - 用于跟踪请求/会话
REQUEST_ID_CTX: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
USER_ID_CTX: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
SESSION_ID_CTX: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


class StructuredLogger:
    """
    结构化日志记录器

    特性：
    - 自动添加请求ID、用户ID等上下文信息
    - 支持结构化字段输出
    - 自动记录耗时
    - 支持日志级别控制
    - 线程安全
    """

    def __init__(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        初始化结构化日志记录器

        Args:
            name: 日志记录器名称
            context: 额外的上下文信息
        """
        self.logger = logging.getLogger(name)
        self.base_context = context or {}
        self._lock = threading.Lock()

    def _build_log_dict(
        self,
        message: str,
        level: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        构建结构化日志字典

        Args:
            message: 日志消息
            level: 日志级别
            **kwargs: 额外的字段

        Returns:
            Dict[str, Any]: 结构化日志字典
        """
        log_dict = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': level,
            'message': message,
            'logger': self.logger.name,
        }

        # 添加上下文变量
        request_id = REQUEST_ID_CTX.get()
        if request_id:
            log_dict['request_id'] = request_id

        user_id = USER_ID_CTX.get()
        if user_id:
            log_dict['user_id'] = user_id

        session_id = SESSION_ID_CTX.get()
        if session_id:
            log_dict['session_id'] = session_id

        # 添加基础上下文
        log_dict.update(self.base_context)

        # 添加额外字段
        log_dict.update(kwargs)

        return log_dict

    def _format_log(self, log_dict: Dict[str, Any]) -> str:
        """
        格式化日志输出

        Args:
            log_dict: 日志字典

        Returns:
            str: 格式化后的日志字符串
        """
        # 核心字段放在前面
        core_fields = ['timestamp', 'level', 'request_id', 'user_id', 'message']
        ordered_dict = {}

        for field in core_fields:
            if field in log_dict:
                ordered_dict[field] = log_dict[field]

        # 添加其他字段
        for key, value in log_dict.items():
            if key not in core_fields:
                ordered_dict[key] = value

        return json.dumps(ordered_dict, ensure_ascii=False, default=str)

    def debug(self, message: str, **kwargs):
        """记录DEBUG级别日志"""
        log_dict = self._build_log_dict(message, 'DEBUG', **kwargs)
        self.logger.debug(self._format_log(log_dict))

    def info(self, message: str, **kwargs):
        """记录INFO级别日志"""
        log_dict = self._build_log_dict(message, 'INFO', **kwargs)
        self.logger.info(self._format_log(log_dict))

    def warning(self, message: str, **kwargs):
        """记录WARNING级别日志"""
        log_dict = self._build_log_dict(message, 'WARNING', **kwargs)
        self.logger.warning(self._format_log(log_dict))

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """记录ERROR级别日志"""
        log_dict = self._build_log_dict(message, 'ERROR', **kwargs)

        if exc_info:
            log_dict['exception'] = traceback.format_exc()

        self.logger.error(self._format_log(log_dict))

    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """记录CRITICAL级别日志"""
        log_dict = self._build_log_dict(message, 'CRITICAL', **kwargs)

        if exc_info:
            log_dict['exception'] = traceback.format_exc()

        self.logger.critical(self._format_log(log_dict))

    def exception(self, message: str, **kwargs):
        """记录异常日志（自动包含异常信息）"""
        self.error(message, exc_info=True, **kwargs)


def get_structured_logger(
    name: str,
    context: Optional[Dict[str, Any]] = None
) -> StructuredLogger:
    """
    获取结构化日志记录器

    Args:
        name: 日志记录器名称
        context: 额外的上下文信息

    Returns:
        StructuredLogger: 结构化日志记录器实例
    """
    return StructuredLogger(name, context)


class LoggingContext:
    """
    日志上下文管理器

    用于设置和清理上下文变量

    Usage:
        with LoggingContext(request_id="123", user_id="user1"):
            logger.info("处理请求")
            # 日志会自动包含request_id和user_id
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **extra_context
    ):
        """
        初始化日志上下文

        Args:
            request_id: 请求ID
            user_id: 用户ID
            session_id: 会话ID
            **extra_context: 额外的上下文信息
        """
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id
        self.session_id = session_id
        self.extra_context = extra_context
        self._tokens = []

    def __enter__(self):
        """设置上下文变量"""
        if self.request_id:
            self._tokens.append(REQUEST_ID_CTX.set(self.request_id))
        if self.user_id:
            self._tokens.append(USER_ID_CTX.set(self.user_id))
        if self.session_id:
            self._tokens.append(SESSION_ID_CTX.set(self.session_id))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """清理上下文变量"""
        for token in self._tokens:
            token.var.reset(token)


def with_logging_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **extra_context
):
    """
    日志上下文装饰器

    为函数添加结构化日志上下文

    Usage:
        @with_logging_context(user_id=lambda args: args[0].user_id)
        async def process_request(request):
            logger.info("处理请求")
            # 自动包含user_id
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 解析动态值
            ctx = {}
            for key, value in extra_context.items():
                if callable(value):
                    ctx[key] = value(*args, **kwargs)
                else:
                    ctx[key] = value

            request_id_val = request_id if not callable(request_id) else request_id(*args, **kwargs)
            user_id_val = user_id if not callable(user_id) else user_id(*args, **kwargs)
            session_id_val = session_id if not callable(session_id) else session_id(*args, **kwargs)

            with LoggingContext(
                request_id=request_id_val,
                user_id=user_id_val,
                session_id=session_id_val,
                **ctx
            ):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 解析动态值
            ctx = {}
            for key, value in extra_context.items():
                if callable(value):
                    ctx[key] = value(*args, **kwargs)
                else:
                    ctx[key] = value

            request_id_val = request_id if not callable(request_id) else request_id(*args, **kwargs)
            user_id_val = user_id if not callable(user_id) else user_id(*args, **kwargs)
            session_id_val = session_id if not callable(session_id) else session_id(*args, **kwargs)

            with LoggingContext(
                request_id=request_id_val,
                user_id=user_id_val,
                session_id=session_id_val,
                **ctx
            ):
                return func(*args, **kwargs)

        # 根据函数类型返回对应的包装器
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class PerformanceLogger:
    """
    性能日志记录器

    用于记录操作的耗时

    Usage:
        perf_logger = PerformanceLogger(get_structured_logger("performance"))

        with perf_logger.measure("database_query"):
            # 执行数据库查询
            results = db.query(...)

        # 自动记录耗时
    """

    def __init__(self, logger: StructuredLogger):
        """
        初始化性能日志记录器

        Args:
            logger: 结构化日志记录器
        """
        self.logger = logger

    def measure(self, operation_name: str):
        """
        性能测量上下文管理器

        Args:
            operation_name: 操作名称

        Returns:
            上下文管理器
        """
        import time

        class MeasurementContext:
            def __init__(self, perf_logger, name):
                self.perf_logger = perf_logger
                self.name = name
                self.start_time = None
                self.end_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.end_time = time.time()
                duration_ms = (self.end_time - self.start_time) * 1000

                self.perf_logger.logger.info(
                    f"Operation completed: {self.name}",
                    operation=self.name,
                    duration_ms=round(duration_ms, 2),
                    duration_s=round(self.end_time - self.start_time, 3)
                )

        return MeasurementContext(self, operation_name)


# 文档处理专用日志记录器
def get_document_logger(
    document_id: Optional[int] = None,
    processing_level: Optional[str] = None
) -> StructuredLogger:
    """
    获取文档处理专用的结构化日志记录器

    Args:
        document_id: 文档ID
        processing_level: 处理级别

    Returns:
        StructuredLogger: 配置好的日志记录器
    """
    context = {}
    if document_id:
        context['document_id'] = document_id
    if processing_level:
        context['processing_level'] = processing_level

    return get_structured_logger('document_processing', context)


# API专用日志记录器
def get_api_logger(
    endpoint: Optional[str] = None,
    method: Optional[str] = None
) -> StructuredLogger:
    """
    获取API专用的结构化日志记录器

    Args:
        endpoint: API端点
        method: HTTP方法

    Returns:
        StructuredLogger: 配置好的日志记录器
    """
    context = {}
    if endpoint:
        context['endpoint'] = endpoint
    if method:
        context['method'] = method

    return get_structured_logger('api', context)
