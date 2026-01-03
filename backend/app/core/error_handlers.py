"""
统一异常处理模块
提供标准化的错误处理和异常装饰器，减少重复代码
"""

from app.core.structured_logging import get_structured_logger
from functools import wraps
from fastapi import HTTPException, status

logger = get_structured_logger(__name__)

class APIError(Exception):
    """
    API错误基类

    提供标准化的API错误格式

    Attributes:
        message: 错误消息
        status_code: HTTP状态码
        detail: 详细错误信息
        error_code: 自定义错误代码
    """

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: Optional[str] = None,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail or message
        self.error_code = error_code
        super().__init__(self.message)

    def to_http_exception(self) -> HTTPException:
        """转换为HTTPException"""
        return HTTPException(
            status_code=self.status_code,
            detail={
                "message": self.message,
                "detail": self.detail,
                "error_code": self.error_code
            }
        )

class ValidationError(APIError):
    """验证错误 (400)"""

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )

class NotFoundError(APIError):
    """资源未找到错误 (404)"""

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )

class ConflictError(APIError):
    """冲突错误 (409)"""

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            detail=detail
        )

class InternalServerError(APIError):
    """内部服务器错误 (500)"""

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

def handle_errors(
    error_message: str = "操作失败",
    error_code: Optional[str] = None,
    raise_on_error: bool = True,
    log_error: bool = True
):
    """
    错误处理装饰器

    统一处理函数中的异常，减少重复的try-except代码

    Args:
        error_message: 错误消息前缀
        error_code: 自定义错误代码
        raise_on_error: 是否抛出异常
        log_error: 是否记录错误日志

    Usage:
        @handle_errors(error_message="文档处理失败")
        async def process_document(doc_id: int):
            # 函数逻辑
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # 如果是HTTPException，直接抛出
                raise
            except APIError as e:
                # 如果是APIError，记录日志并转换为HTTPException
                if log_error:
                    logger.error(f"{error_message}: {str(e)}")
                if raise_on_error:
                    raise e.to_http_exception()
                return {"success": False, "error": str(e)}
            except Exception as e:
                # 其他异常
                if log_error:
                    logger.error(f"{error_message}: {str(e)}", exc_info=True)
                if raise_on_error:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"{error_message}: {str(e)}"
                    )
                return {"success": False, "error": str(e)}

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except HTTPException:
                raise
            except APIError as e:
                if log_error:
                    logger.error(f"{error_message}: {str(e)}")
                if raise_on_error:
                    raise e.to_http_exception()
                return {"success": False, "error": str(e)}
            except Exception as e:
                if log_error:
                    logger.error(f"{error_message}: {str(e)}", exc_info=True)
                if raise_on_error:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"{error_message}: {str(e)}"
                    )
                return {"success": False, "error": str(e)}

        # 根据函数类型返回对应的包装器
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def validate_required_fields(data: Dict[str, Any], required_fields: list) -> None:
    """
    验证必填字段

    Args:
        data: 数据字典
        required_fields: 必填字段列表

    Raises:
        ValidationError: 如果缺少必填字段
    """
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    if missing_fields:
        raise ValidationError(
            message="缺少必填字段",
            detail=f"缺少必填字段: {', '.join(missing_fields)}"
        )

def validate_field_length(data: Dict[str, Any], field: str, min_length: int = 0, max_length: int = 10000) -> None:
    """
    验证字段长度

    Args:
        data: 数据字典
        field: 字段名
        min_length: 最小长度
        max_length: 最大长度

    Raises:
        ValidationError: 如果字段长度不符合要求
    """
    if field not in data:
        return

    value = data[field]
    if value is None:
        return

    if not isinstance(value, (str, list)):
        return

    length = len(value)
    if length < min_length:
        raise ValidationError(
            message=f"{field}长度不足",
            detail=f"{field}长度不能少于{min_length}个字符"
        )

    if length > max_length:
        raise ValidationError(
            message=f"{field}长度超限",
            detail=f"{field}长度不能超过{max_length}个字符"
        )

def create_response(
    success: bool = True,
    message: str = "操作成功",
    data: Optional[Any] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建标准化的API响应

    Args:
        success: 操作是否成功
        message: 响应消息
        data: 响应数据
        error: 错误信息

    Returns:
        Dict[str, Any]: 标准响应格式
    """
    response = {
        "success": success,
        "message": message
    }

    if data is not None:
        response["data"] = data

    if error is not None:
        response["error"] = error

    return response

def create_success_response(
    message: str = "操作成功",
    data: Optional[Any] = None
) -> Dict[str, Any]:
    """
    创建成功响应

    Args:
        message: 成功消息
        data: 响应数据

    Returns:
        Dict[str, Any]: 成功响应
    """
    return create_response(success=True, message=message, data=data)

def create_error_response(
    message: str = "操作失败",
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建错误响应

    Args:
        message: 错误消息
        error: 详细错误信息

    Returns:
        Dict[str, Any]: 错误响应
    """
    return create_response(success=False, message=message, error=error)

# 导出所有类和函数
__all__ = [
    "APIError",
    "ValidationError",
    "NotFoundError",
    "ConflictError",
    "InternalServerError",
    "handle_errors",
    "validate_required_fields",
    "validate_field_length",
    "create_response",
    "create_success_response",
    "create_error_response"
]
