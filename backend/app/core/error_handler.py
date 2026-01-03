"""
错误处理和重试机制
"""

from functools import wraps
import asyncio
from app.core.structured_logging import get_structured_logger

# 从其他错误处理模块导入
# from app.core.errors.unified_errors import ErrorCategory  # 模块不存在，已注释
from app.core.error_handlers import handle_errors as _handle_errors

logger = get_structured_logger(__name__)

class NetworkError(Exception):
    """网络错误异常"""
    pass

def retry_on_failure(max_attempts: int = 3, retry_on: Tuple[Type[Exception], ...] = (Exception,), delay: float = 1.0):
    """
    失败重试装饰器

    Args:
        max_attempts: 最大重试次数
        retry_on: 需要重试的异常类型
        delay: 重试延迟（秒）
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"{func.__name__} 失败 (尝试 {attempt + 1}/{max_attempts}): {e}")
                        await asyncio.sleep(delay * (attempt + 1))  # 指数退避
                    else:
                        logger.error(f"{func.__name__} 在 {max_attempts} 次尝试后仍然失败")
            raise last_exception
        return wrapper
    return decorator

# 导出 handle_errors 以兼容其他模块的导入
handle_errors = _handle_errors
