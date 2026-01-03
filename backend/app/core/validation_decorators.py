"""
验证装饰器
提供便捷的数据验证装饰器
"""

from app.core.structured_logging import get_structured_logger
from functools import wraps

from fastapi import HTTPException

logger = get_structured_logger(__name__)

def validate_request(
    required_fields: Optional[List[str]] = None,
    optional_fields: Optional[List[str]] = None,
    validate_body: bool = True
):
    """
    请求体验证装饰器

    Args:
        required_fields: 必填字段列表
        optional_fields: 可选字段列表
        validate_body: 是否验证请求体

    Usage:
        @validate_request(
            required_fields=['query', 'top_k'],
            optional_fields=['filters']
        )
        async def search_endpoint(request: Request):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 尝试从参数中获取request或body
            body = None
            for arg in args:
                if isinstance(arg, dict):
                    body = arg
                    break

            if not body and validate_body:
                # 如果没有找到body，尝试从kwargs中获取
                body = kwargs.get('body') or kwargs.get('request')

            if body and isinstance(body, dict):
                validator = CompositeValidator()

                # 验证字段
                result = validator.add_validator(
                    DataValidator.validate_dict_fields,
                    body,
                    "request_body",
                    required_fields,
                    optional_fields
                )

                if not validator.is_valid():
                    errors = validator.get_errors()
                    error_msgs = [e.message for e in errors]
                    logger.warning(f"请求验证失败: {error_msgs}")
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "请求验证失败",
                            "details": error_msgs
                        }
                    )

            return await func(*args, **kwargs)
        return wrapper
    return decorator

def validate_file_upload(
    max_size_mb: float = 100.0,
    allowed_extensions: Optional[List[str]] = None,
    require_filename: bool = True
):
    """
    文件上传验证装饰器

    Args:
        max_size_mb: 最大文件大小（MB）
        allowed_extensions: 允许的文件扩展名
        require_filename: 是否要求文件名

    Usage:
        @validate_file_upload(
            max_size_mb=50.0,
            allowed_extensions=['.pdf', '.docx', '.txt']
        )
        async def upload_endpoint(file: UploadFile):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 查找UploadFile参数
            from fastapi import UploadFile

            upload_file = None
            for arg in args:
                if isinstance(arg, UploadFile):
                    upload_file = arg
                    break

            if not upload_file:
                upload_file = kwargs.get('file')

            if upload_file:
                validator = CompositeValidator()

                # 验证文件名
                if require_filename:
                    result = validator.add_validator(
                        DataValidator.validate_string_length,
                        upload_file.filename or "",
                        "filename",
                        min_length=1,
                        max_length=255,
                        required=True
                    )

                # 验证文件扩展名
                if allowed_extensions and upload_file.filename:
                    import os
                    ext = os.path.splitext(upload_file.filename)[1].lower()
                    if ext not in allowed_extensions:
                        logger.warning(f"不支持的文件类型: {ext}")
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": "不支持的文件类型",
                                "allowed_extensions": allowed_extensions,
                                "received_extension": ext
                            }
                        )

                # 读取文件大小
                if hasattr(upload_file.file, 'seek'):
                    upload_file.file.seek(0, 2)  # 移动到文件末尾
                    size_bytes = upload_file.file.tell()
                    upload_file.file.seek(0)  # 重置到开头

                    size_mb = size_bytes / (1024 * 1024)
                    if size_mb > max_size_mb:
                        logger.warning(f"文件过大: {size_mb:.2f}MB")
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": f"文件过大，最大允许 {max_size_mb}MB",
                                "file_size_mb": round(size_mb, 2),
                                "max_size_mb": max_size_mb
                            }
                        )

            return await func(*args, **kwargs)
        return wrapper
    return decorator

def validate_query_params(
    max_length: int = 1000,
    min_length: int = 1,
    require_query: bool = True
):
    """
    查询参数验证装饰器

    Args:
        max_length: 查询字符串最大长度
        min_length: 查询字符串最小长度
        require_query: 是否要求查询参数

    Usage:
        @validate_query_params(max_length=500)
        async def search_endpoint(query: str):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            query = kwargs.get('query')

            if query is not None:
                validator = CompositeValidator()

                result = validator.add_validator(
                    DataValidator.validate_string_length,
                    query,
                    "query",
                    min_length=min_length,
                    max_length=max_length,
                    required=require_query
                )

                if not result.is_valid:
                    logger.warning(f"查询参数验证失败: {result.message}")
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "查询参数无效",
                            "message": result.message
                        }
                    )

            elif require_query:
                logger.warning("缺少必需的查询参数")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "缺少必需的查询参数"
                    }
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator

def validate_pagination(
    default_page: int = 1,
    default_page_size: int = 10,
    max_page_size: int = 100
):
    """
    分页参数验证装饰器

    Args:
        default_page: 默认页码
        default_page_size: 默认每页大小
        max_page_size: 最大每页大小

    Usage:
        @validate_pagination(default_page_size=20)
        async def list_endpoint(page: int = 1, page_size: int = 10):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            page = kwargs.get('page', default_page)
            page_size = kwargs.get('page_size', default_page_size)

            validator = CompositeValidator()

            # 验证页码
            validator.add_validator(
                DataValidator.validate_numeric_range,
                page,
                "page",
                min_value=1,
                required=True
            )

            # 验证每页大小
            validator.add_validator(
                DataValidator.validate_numeric_range,
                page_size,
                "page_size",
                min_value=1,
                max_value=max_page_size,
                required=True
            )

            if not validator.is_valid():
                errors = validator.get_errors()
                error_msgs = [e.message for e in errors]
                logger.warning(f"分页参数验证失败: {error_msgs}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "分页参数无效",
                        "details": error_msgs
                    }
                )

            # 更新kwargs中的值（确保在有效范围内）
            kwargs['page'] = max(1, page)
            kwargs['page_size'] = max(1, min(page_size, max_page_size))

            return await func(*args, **kwargs)
        return wrapper
    return decorator

def sanitize_input(fields: Optional[List[str]] = None):
    """
    输入清理装饰器（防止XSS和注入攻击）

    Args:
        fields: 需要清理的字段列表，如果为None则清理所有字符串字段

    Usage:
        @sanitize_input(fields=['title', 'description'])
        async def create_endpoint(title: str, description: str):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import html
            import re

            def clean_string(value: str) -> str:
                """清理字符串"""
                if not isinstance(value, str):
                    return value

                # HTML转义
                value = html.escape(value)

                # 移除潜在的危险字符
                value = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', value)

                return value.strip()

            # 清理kwargs中的字符串字段
            for key, value in kwargs.items():
                if fields is None or key in fields:
                    if isinstance(value, str):
                        kwargs[key] = clean_string(value)
                    elif isinstance(value, list):
                        kwargs[key] = [clean_string(v) if isinstance(v, str) else v for v in value]
                    elif isinstance(value, dict):
                        kwargs[key] = {
                            k: clean_string(v) if isinstance(v, str) else v
                            for k, v in value.items()
                        }

            return await func(*args, **kwargs)
        return wrapper
    return decorator
