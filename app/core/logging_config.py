"""
统一日志配置
为Financial RAG系统提供结构化日志记录
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from app.core.config import settings


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加异常信息
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # 添加额外字段
        if hasattr(record, "document_id"):
            log_entry["document_id"] = record.document_id
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "processing_time"):
            log_entry["processing_time"] = record.processing_time

        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """彩色控制台日志格式化器"""

    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }

    def format(self, record: logging.LogRecord) -> str:
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        # 格式化时间
        record.asctime = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

        # 自定义格式
        return f"{record.asctime} | {record.levelname:8} | {record.name:20} | {record.getMessage()}"


def setup_logging(
    log_level: str = None,
    log_file: str = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_structured: bool = False
) -> None:
    """
    设置日志配置

    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        enable_console: 是否启用控制台日志
        enable_file: 是否启用文件日志
        enable_structured: 是否启用结构化日志
    """
    # 使用配置文件中的默认值
    log_level = log_level or settings.log_level
    log_file = log_file or settings.log_file

    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # 清除现有处理器
    root_logger.handlers.clear()

    # 设置日志格式
    if enable_structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # 控制台处理器
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))

        if enable_structured:
            console_handler.setFormatter(formatter)
        else:
            # 控制台使用彩色格式
            console_handler.setFormatter(ColoredFormatter())

        root_logger.addHandler(console_handler)

    # 文件处理器
    if enable_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用RotatingFileHandler支持日志轮转
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # 设置特定模块的日志级别
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # 如果是生产环境，降低一些库的日志级别
    if not settings.debug:
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
        logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器"""
    return logging.getLogger(name)


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    **kwargs
) -> None:
    """
    带上下文的日志记录

    Args:
        logger: 日志记录器
        level: 日志级别
        message: 日志消息
        **kwargs: 上下文信息（如document_id, user_id等）
    """
    log_level = getattr(logging, level.upper())

    # 创建LogRecord并添加额外属性
    record = logger.makeRecord(
        name=logger.name,
        level=log_level,
        fn="",
        lno=0,
        msg=message,
        args=(),
        exc_info=None
    )

    # 添加上下文信息
    for key, value in kwargs.items():
        setattr(record, key, value)

    logger.handle(record)


# 性能监控装饰器
def log_performance(logger: logging.Logger = None):
    """
    性能监控装饰器
    自动记录函数执行时间
    """
    import time
    import functools

    if logger is None:
        logger = get_logger(__name__)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                logger.info(
                    f"函数 {func.__name__} 执行完成",
                    extra={
                        "function": func.__name__,
                        "execution_time": execution_time,
                        "status": "success"
                    }
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                logger.error(
                    f"函数 {func.__name__} 执行失败: {str(e)}",
                    extra={
                        "function": func.__name__,
                        "execution_time": execution_time,
                        "status": "error",
                        "error": str(e)
                    }
                )

                raise

        return wrapper
    return decorator


# 异步性能监控装饰器
def log_async_performance(logger: logging.Logger = None):
    """
    异步性能监控装饰器
    """
    import time
    import functools

    if logger is None:
        logger = get_logger(__name__)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                logger.info(
                    f"异步函数 {func.__name__} 执行完成",
                    extra={
                        "function": func.__name__,
                        "execution_time": execution_time,
                        "status": "success",
                        "is_async": True
                    }
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                logger.error(
                    f"异步函数 {func.__name__} 执行失败: {str(e)}",
                    extra={
                        "function": func.__name__,
                        "execution_time": execution_time,
                        "status": "error",
                        "error": str(e),
                        "is_async": True
                    }
                )

                raise

        return wrapper
    return decorator


# 初始化日志配置
def init_logging():
    """初始化日志配置"""
    setup_logging(
        log_level=settings.log_level,
        log_file=settings.log_file,
        enable_console=settings.debug,  # 开发环境启用控制台
        enable_file=True,
        enable_structured=not settings.debug  # 生产环境使用结构化日志
    )


# 自动初始化
init_logging()