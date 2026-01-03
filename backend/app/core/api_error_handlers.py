"""
API层统一错误处理模块（增强版）
提供统一的异常处理、错误响应和日志记录
"""

from app.core.structured_logging import get_structured_logger
import traceback
import uuid
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from fastapi.responses import JSONResponse

logger = get_structured_logger(__name__)

class ErrorCode(Enum):
    """错误代码"""
    # 通用错误 (1000-1999)
    UNKNOWN_ERROR = (1000, "未知错误")
    INTERNAL_SERVER_ERROR = (1001, "服务器内部错误")
    SERVICE_UNAVAILABLE = (1002, "服务不可用")

    # 请求错误 (2000-2999)
    BAD_REQUEST = (2000, "错误的请求")
    VALIDATION_ERROR = (2001, "数据验证失败")
    MISSING_PARAMETER = (2002, "缺少必需参数")
    INVALID_PARAMETER = (2003, "无效的参数")
    INVALID_FORMAT = (2004, "格式错误")

    # 认证/授权错误 (3000-3999)
    UNAUTHORIZED = (3000, "未授权")
    FORBIDDEN = (3001, "禁止访问")
    TOKEN_EXPIRED = (3002, "令牌已过期")
    TOKEN_INVALID = (3003, "令牌无效")
    INSUFFICIENT_PERMISSIONS = (3004, "权限不足")

    # 资源错误 (4000-4999)
    NOT_FOUND = (4000, "资源不存在")
    ALREADY_EXISTS = (4001, "资源已存在")
    RESOURCE_LOCKED = (4002, "资源已锁定")
    RESOURCE_EXPIRED = (4003, "资源已过期")

    # 业务逻辑错误 (5000-5999)
    BUSINESS_LOGIC_ERROR = (5000, "业务逻辑错误")
    OPERATION_FAILED = (5001, "操作失败")
    STATE_CONFLICT = (5002, "状态冲突")
    QUOTA_EXCEEDED = (5003, "超出配额")
    RATE_LIMIT_EXCEEDED = (5004, "超出速率限制")

    # 数据库错误 (6000-6999)
    DATABASE_ERROR = (6000, "数据库错误")
    QUERY_FAILED = (6001, "查询失败")
    TRANSACTION_FAILED = (6002, "事务失败")
    CONSTRAINT_VIOLATION = (6003, "约束违反")

    # 外部服务错误 (7000-7999)
    EXTERNAL_SERVICE_ERROR = (7000, "外部服务错误")
    UPSTREAM_SERVICE_ERROR = (7001, "上游服务错误")
    TIMEOUT = (7002, "请求超时")

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message

@dataclass
class ErrorDetail:
    """错误详情"""
    field: Optional[str] = None           # 字段名
    message: str = ""                     # 错误消息
    code: Optional[str] = None            # 错误代码
    value: Optional[Any] = None           # 错误值

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'message': self.message
        }
        if self.field:
            result['field'] = self.field
        if self.code:
            result['code'] = self.code
        if self.value is not None:
            result['value'] = self.value
        return result

@dataclass
class ErrorResponse:
    """统一错误响应"""
    success: bool = False
    error_code: int = 0
    error_message: str = ""
    error_type: str = ""
    request_id: str = ""
    timestamp: str = ""
    path: str = ""
    details: List[ErrorDetail] = field(default_factory=list)
    debug_info: Optional[Dict[str, Any]] = None

    def to_dict(self, include_debug: bool = False) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'success': self.success,
            'error': {
                'code': self.error_code,
                'message': self.error_message,
                'type': self.error_type
            },
            'request_id': self.request_id,
            'timestamp': self.timestamp,
            'path': self.path
        }

        if self.details:
            result['error']['details'] = [d.to_dict() for d in self.details]

        if include_debug and self.debug_info:
            result['debug'] = self.debug_info

        return result

class AppException(Exception):
    """
    应用异常基类

    所有自定义异常都应该继承这个类
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[List[ErrorDetail]] = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or []
        self.status_code = status_code
        super().__init__(message)

    def to_error_response(self, request: Request, include_debug: bool = False) -> ErrorResponse:
        """转换为错误响应"""
        response = ErrorResponse(
            error_code=self.error_code.code,
            error_message=self.message,
            error_type=self.__class__.__name__,
            request_id=_get_request_id(request),
            timestamp=datetime.now().isoformat(),
            path=request.url.path,
            details=self.details
        )

        if include_debug:
            response.debug_info = {
                'exception_type': self.__class__.__name__,
                'traceback': traceback.format_exc()
            }

        return response

# 具体异常类

class ValidationException(AppException):
    """验证异常"""

    def __init__(
        self,
        message: str = "数据验证失败",
        details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=details,
            status_code=status.HTTP_400_BAD_REQUEST
        )

class NotFoundException(AppException):
    """资源不存在异常"""

    def __init__(self, resource_type: str = "资源", resource_id: Optional[str] = None):
        message = f"{resource_type}不存在"
        if resource_id:
            message += f": {resource_id}"

        super().__init__(
            message=message,
            error_code=ErrorCode.NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND
        )

class UnauthorizedException(AppException):
    """未授权异常"""

    def __init__(self, message: str = "未授权访问"):
        super().__init__(
            message=message,
            error_code=ErrorCode.UNAUTHORIZED,
            status_code=status.HTTP_401_UNAUTHORIZED
        )

class ForbiddenException(AppException):
    """禁止访问异常"""

    def __init__(self, message: str = "权限不足"):
        super().__init__(
            message=message,
            error_code=ErrorCode.FORBIDDEN,
            status_code=status.HTTP_403_FORBIDDEN
        )

class BusinessException(AppException):
    """业务逻辑异常"""

    def __init__(self, message: str, details: Optional[List[ErrorDetail]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.BUSINESS_LOGIC_ERROR,
            details=details,
            status_code=status.HTTP_400_BAD_REQUEST
        )

class DatabaseException(AppException):
    """数据库异常"""

    def __init__(self, message: str = "数据库操作失败"):
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

class ExternalServiceException(AppException):
    """外部服务异常"""

    def __init__(
        self,
        service_name: str,
        message: str = "外部服务调用失败"
    ):
        full_message = f"{service_name}: {message}"
        super().__init__(
            message=full_message,
            error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            status_code=status.HTTP_502_BAD_GATEWAY
        )

class RateLimitException(AppException):
    """速率限制异常"""

    def __init__(self, message: str = "超出速率限制"):
        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS
        )

# 错误处理器

async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """应用异常处理器"""
    include_debug = _should_include_debug(request)
    error_response = exc.to_error_response(request, include_debug)
    _log_error(request, exc, error_response)

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.to_dict(include_debug=include_debug)
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器"""
    include_debug = _should_include_debug(request)

    error_response = ErrorResponse(
        error_code=ErrorCode.INTERNAL_SERVER_ERROR.code,
        error_message=str(exc) if logger.isEnabledFor(logging.DEBUG) else "服务器内部错误",
        error_type=type(exc).__name__,
        request_id=_get_request_id(request),
        timestamp=datetime.now().isoformat(),
        path=request.url.path
    )

    if include_debug:
        error_response.debug_info = {
            'exception_type': type(exc).__name__,
            'traceback': traceback.format_exc()
        }

    _log_error(request, exc, error_response)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.to_dict(include_debug=include_debug)
    )

def _should_include_debug(request: Request) -> bool:
    """判断是否包含调试信息"""
    if os.getenv('DEBUG', 'false').lower() == 'true':
        return True

    debug_header = request.headers.get('X-Debug-Mode')
    if debug_header and debug_header.lower() == 'true':
        return True

    return False

def _get_request_id(request: Request) -> str:
    """获取请求ID"""
    request_id = request.headers.get('X-Request-ID')
    if request_id:
        return request_id
    return str(uuid.uuid4())

def _log_error(request: Request, exc: Exception, error_response: ErrorResponse):
    """记录错误日志"""
    log_data = {
        'request_id': error_response.request_id,
        'path': error_response.path,
        'error_code': error_response.error_code,
        'error_message': error_response.error_message,
        'error_type': error_response.error_type,
        'method': request.method,
        'client_ip': request.client.host if request.client else None
    }

    if error_response.error_code >= 500:
        logger.error(f"Server Error: {json.dumps(log_data, ensure_ascii=False)}", exc_info=exc)
    elif error_response.error_code >= 400:
        logger.warning(f"Client Error: {json.dumps(log_data, ensure_ascii=False)}")
    else:
        logger.info(f"Error: {json.dumps(log_data, ensure_ascii=False)}")

def setup_error_handlers(app):
    """
    设置FastAPI应用错误处理器

    Args:
        app: FastAPI应用实例
    """
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException

    @app.exception_handler(AppException)
    async def handle_app_exception(request: Request, exc: AppException):
        return await app_exception_handler(request, exc)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_exception(request: Request, exc: RequestValidationError):
        details = [
            ErrorDetail(
                field='.'.join(str(loc) for loc in error['loc']),
                message=error['msg'],
                code=error['type']
            )
            for error in exc.errors()
        ]

        app_exc = ValidationException(
            message="请求数据验证失败",
            details=details
        )
        return await app_exception_handler(request, app_exc)

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException):
        error_code_map = {
            400: ErrorCode.BAD_REQUEST,
            401: ErrorCode.UNAUTHORIZED,
            403: ErrorCode.FORBIDDEN,
            404: ErrorCode.NOT_FOUND,
            429: ErrorCode.RATE_LIMIT_EXCEEDED,
            500: ErrorCode.INTERNAL_SERVER_ERROR,
            502: ErrorCode.UPSTREAM_SERVICE_ERROR,
            503: ErrorCode.SERVICE_UNAVAILABLE
        }

        error_code = error_code_map.get(exc.status_code, ErrorCode.UNKNOWN_ERROR)

        app_exc = AppException(
            message=exc.detail,
            error_code=error_code,
            status_code=exc.status_code
        )
        return await app_exception_handler(request, app_exc)

    @app.exception_handler(Exception)
    async def handle_general_exception(request: Request, exc: Exception):
        return await general_exception_handler(request, exc)
