"""
公共文件操作工具类
提供可复用的文件处理功能，消除代码重复
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class FileUtils:
    """文件操作工具类"""

    # 支持的图片格式
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}

    # 支持的文档格式
    DOCUMENT_EXTENSIONS = {
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.txt', '.md', '.rtf', '.odt', '.csv', '.json', '.xml'
    }

    @staticmethod
    def extract_file_metadata(file_path: str) -> Dict[str, Any]:
        """
        提取文件元数据

        Args:
            file_path: 文件路径

        Returns:
            Dict[str, Any]: 文件元数据
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_info = path.stat()

        metadata = {
            'file_name': path.name,
            'file_stem': path.stem,
            'file_extension': path.suffix.lower(),
            'file_size': file_info.st_size,
            'file_size_mb': round(file_info.st_size / (1024 * 1024), 2),
            'created_time': file_info.st_ctime,
            'modified_time': file_info.st_mtime,
            'is_file': path.is_file(),
            'is_directory': path.is_dir(),
            'absolute_path': str(path.absolute()),
        }

        # 添加MIME类型
        mime_type, _ = mimetypes.guess_type(file_path)
        metadata['mime_type'] = mime_type

        # 判断文件类型
        metadata['is_image'] = path.suffix.lower() in FileUtils.IMAGE_EXTENSIONS
        metadata['is_document'] = path.suffix.lower() in FileUtils.DOCUMENT_EXTENSIONS

        return metadata

    @staticmethod
    def calculate_file_hash(
        file_path: str,
        algorithm: str = 'sha256',
        chunk_size: int = 8192
    ) -> str:
        """
        计算文件哈希值

        Args:
            file_path: 文件路径
            algorithm: 哈希算法 (md5, sha1, sha256, sha512)
            chunk_size: 读取块大小

        Returns:
            str: 文件哈希值（十六进制）
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        hash_obj = hashlib.new(algorithm)

        with open(path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    @staticmethod
    def safe_read_file(
        file_path: str,
        encoding: Optional[str] = None,
        fallback_encodings: Optional[List[str]] = None
    ) -> Tuple[str, Optional[str]]:
        """
        安全读取文本文件

        Args:
            file_path: 文件路径
            encoding: 首选编码
            fallback_encodings: 备用编码列表

        Returns:
            Tuple[str, Optional[str]]: (文件内容, 实际使用的编码)
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if fallback_encodings is None:
            fallback_encodings = ['utf-8', 'gbk', 'gb2312', 'big5', 'latin-1']

        encodings = [encoding] + fallback_encodings if encoding else fallback_encodings

        for enc in encodings:
            if not enc:
                continue
            try:
                with open(path, 'r', encoding=enc) as f:
                    content = f.read()
                return content, enc
            except (UnicodeDecodeError, LookupError):
                continue

        # 如果所有编码都失败，返回错误信息
        raise ValueError(f"无法解码文件 {file_path}，尝试的编码: {encodings}")

    @staticmethod
    def safe_json_loads(text: Any) -> Optional[Dict[str, Any]]:
        """
        安全的JSON解析

        Args:
            text: 待解析的文本

        Returns:
            Optional[Dict[str, Any]]: 解析结果或None
        """
        if not isinstance(text, str):
            return None

        try:
            import json
            return json.loads(text.strip())
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    @staticmethod
    def validate_file_path(
        file_path: str,
        must_exist: bool = True,
        allow_symlinks: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        验证文件路径

        Args:
            file_path: 文件路径
            must_exist: 文件是否必须存在
            allow_symlinks: 是否允许符号链接

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        path = Path(file_path)

        # 检查路径是否为空
        if not file_path or not file_path.strip():
            return False, "文件路径为空"

        # 检查是否存在
        if must_exist and not path.exists():
            return False, f"文件不存在: {file_path}"

        # 检查是否为符号链接
        if not allow_symlinks and path.is_symlink():
            return False, f"不支持符号链接: {file_path}"

        # 如果文件存在，检查是否可读
        if path.exists() and not os.access(file_path, os.R_OK):
            return False, f"文件不可读: {file_path}"

        # 检查文件大小
        if path.exists() and path.is_file():
            file_size = path.stat().st_size
            if file_size == 0:
                return False, f"文件为空: {file_path}"

        return True, None

    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """
        获取文件扩展名（小写）

        Args:
            file_path: 文件路径

        Returns:
            str: 文件扩展名（包含点号，小写）
        """
        return Path(file_path).suffix.lower()

    @staticmethod
    def is_supported_image(file_path: str) -> bool:
        """
        检查是否为支持的图片格式

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否为支持的图片
        """
        ext = FileUtils.get_file_extension(file_path)
        return ext in FileUtils.IMAGE_EXTENSIONS

    @staticmethod
    def is_supported_document(file_path: str) -> bool:
        """
        检查是否为支持的文档格式

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否为支持的文档
        """
        ext = FileUtils.get_file_extension(file_path)
        return ext in FileUtils.DOCUMENT_EXTENSIONS

    @staticmethod
    def ensure_directory(directory: str) -> str:
        """
        确保目录存在，不存在则创建

        Args:
            directory: 目录路径

        Returns:
            str: 目录路径
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @staticmethod
    def get_relative_path(file_path: str, base_path: str) -> str:
        """
        获取相对路径

        Args:
            file_path: 文件路径
            base_path: 基础路径

        Returns:
            str: 相对路径
        """
        try:
            path = Path(file_path).resolve()
            base = Path(base_path).resolve()
            return str(path.relative_to(base))
        except ValueError:
            # 如果不在同一个目录树下，返回绝对路径
            return str(Path(file_path).resolve())


class ParseResultBuilder:
    """ParseResult构建器 - 统一结果创建"""

    @staticmethod
    def success(
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        parse_time: Optional[float] = None,
        **extra_fields
    ):
        """
        构建成功结果

        Args:
            content: 解析内容
            metadata: 元数据
            parse_time: 解析耗时
            **extra_fields: 额外字段

        Returns:
            ParseResult: 解析结果
        """
        from app.services.parsers.base import ParseResult

        result_metadata = metadata or {}
        result_metadata.update(extra_fields)

        return ParseResult(
            content=content,
            metadata=result_metadata,
            success=True,
            parse_time=parse_time
        )

    @staticmethod
    def failure(
        error_message: str,
        metadata: Optional[Dict[str, Any]] = None,
        parse_time: Optional[float] = None,
        **extra_fields
    ):
        """
        构建失败结果

        Args:
            error_message: 错误消息
            metadata: 元数据
            parse_time: 解析耗时
            **extra_fields: 额外字段

        Returns:
            ParseResult: 解析结果
        """
        from app.services.parsers.base import ParseResult

        result_metadata = metadata or {}
        result_metadata.update(extra_fields)
        result_metadata['error'] = error_message

        return ParseResult(
            content="",
            metadata=result_metadata,
            success=False,
            error_message=error_message,
            parse_time=parse_time
        )


class ConfigurationValidator:
    """配置验证器 - 统一配置处理"""

    @staticmethod
    def validate_parser_config(
        config: Dict[str, Any],
        parser_type: str,
        required_fields: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        验证解析器配置

        Args:
            config: 配置字典
            parser_type: 解析器类型
            required_fields: 必需字段列表

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        if not isinstance(config, dict):
            return False, f"{parser_type}: 配置必须是字典类型"

        if required_fields:
            missing_fields = [f for f in required_fields if f not in config]
            if missing_fields:
                return False, f"{parser_type}: 缺少必需字段 {missing_fields}"

        return True, None

    @staticmethod
    def get_with_default(
        config: Dict[str, Any],
        key: str,
        default: Any,
        value_type: Optional[type] = None
    ) -> Any:
        """
        获取配置值，支持默认值和类型验证

        Args:
            config: 配置字典
            key: 配置键
            default: 默认值
            value_type: 值类型

        Returns:
            Any: 配置值或默认值
        """
        value = config.get(key, default)

        if value is None:
            return default

        if value_type and not isinstance(value, value_type):
            logger.warning(
                f"配置 {key} 类型错误，期望 {value_type.__name__}，"
                f"实际 {type(value).__name__}，使用默认值"
            )
            return default

        return value


# 便捷函数
def safe_file_operation(operation: str, file_path: str, **kwargs):
    """
    安全的文件操作包装器

    Args:
        operation: 操作类型 (read, metadata, hash, validate)
        file_path: 文件路径
        **kwargs: 额外参数

    Returns:
        操作结果

    Raises:
        ValueError: 不支持的操作类型
    """
    operations = {
        'read': FileUtils.safe_read_file,
        'metadata': FileUtils.extract_file_metadata,
        'hash': FileUtils.calculate_file_hash,
        'validate': FileUtils.validate_file_path,
    }

    op_func = operations.get(operation)
    if not op_func:
        raise ValueError(f"不支持的操作类型: {operation}")

    return op_func(file_path, **kwargs)
