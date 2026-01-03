"""
数据验证和边界检查工具
提供统一的输入验证、数据完整性检查和边界验证
"""

import re
from app.core.structured_logging import get_structured_logger
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

logger = get_structured_logger(__name__)

T = TypeVar('T')

class ValidationSeverity(Enum):
    """验证严重级别"""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    field_name: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    message: str = ""
    actual_value: Any = None
    expected_constraint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'field_name': self.field_name,
            'severity': self.severity.value,
            'message': self.message,
            'actual_value': str(self.actual_value) if self.actual_value is not None else None,
            'expected_constraint': self.expected_constraint
        }

class ValidationError(Exception):
    """验证错误异常"""
    def __init__(self, results: List[ValidationResult]):
        self.results = results
        error_messages = [r.message for r in results if r.severity == ValidationSeverity.ERROR]
        super().__init__("; ".join(error_messages))

class DataValidator:
    """数据验证器"""

    @staticmethod
    def validate_string_length(
        value: str,
        field_name: str,
        min_length: int = 0,
        max_length: int = 10000,
        required: bool = False
    ) -> ValidationResult:
        """
        验证字符串长度

        Args:
            value: 待验证的字符串
            field_name: 字段名
            min_length: 最小长度
            max_length: 最大长度
            required: 是否必填

        Returns:
            ValidationResult
        """
        if required and not value:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} 是必填字段",
                actual_value=value,
                expected_constraint=f"min_length={min_length}, required={required}"
            )

        if value and len(value) < min_length:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} 长度小于最小值 {min_length}",
                actual_value=len(value),
                expected_constraint=f"min_length={min_length}"
            )

        if value and len(value) > max_length:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} 长度超过最大值 {max_length}",
                actual_value=len(value),
                expected_constraint=f"max_length={max_length}"
            )

        return ValidationResult(
            is_valid=True,
            field_name=field_name,
            actual_value=len(value) if value else 0
        )

    @staticmethod
    def validate_numeric_range(
        value: float,
        field_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        required: bool = False
    ) -> ValidationResult:
        """
        验证数值范围

        Args:
            value: 待验证的数值
            field_name: 字段名
            min_value: 最小值
            max_value: 最大值
            required: 是否必填

        Returns:
            ValidationResult
        """
        if required and value is None:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} 是必填字段",
                actual_value=value,
                expected_constraint=f"required={required}"
            )

        if value is None:
            return ValidationResult(is_valid=True, field_name=field_name)

        if min_value is not None and value < min_value:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} 小于最小值 {min_value}",
                actual_value=value,
                expected_constraint=f"min_value={min_value}"
            )

        if max_value is not None and value > max_value:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} 超过最大值 {max_value}",
                actual_value=value,
                expected_constraint=f"max_value={max_value}"
            )

        return ValidationResult(
            is_valid=True,
            field_name=field_name,
            actual_value=value
        )

    @staticmethod
    def validate_email(email: str, field_name: str = "email") -> ValidationResult:
        """
        验证邮箱格式

        Args:
            email: 邮箱地址
            field_name: 字段名

        Returns:
            ValidationResult
        """
        if not email:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message="邮箱地址不能为空",
                actual_value=email
            )

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message="邮箱格式不正确",
                actual_value=email,
                expected_constraint="valid email format"
            )

        return ValidationResult(
            is_valid=True,
            field_name=field_name,
            actual_value=email
        )

    @staticmethod
    def validate_file_path(
        file_path: str,
        field_name: str = "file_path",
        must_exist: bool = False,
        allowed_extensions: Optional[List[str]] = None,
        max_size_mb: Optional[float] = None
    ) -> ValidationResult:
        """
        验证文件路径

        Args:
            file_path: 文件路径
            field_name: 字段名
            must_exist: 文件是否必须存在
            allowed_extensions: 允许的扩展名列表
            max_size_mb: 最大文件大小（MB）

        Returns:
            ValidationResult
        """
        path = Path(file_path)

        # 检查文件是否存在
        if must_exist and not path.exists():
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"文件不存在: {file_path}",
                actual_value=file_path
            )

        if path.exists():
            # 检查是否是文件
            if not path.is_file():
                return ValidationResult(
                    is_valid=False,
                    field_name=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"路径不是文件: {file_path}",
                    actual_value=file_path
                )

            # 检查文件扩展名
            if allowed_extensions:
                ext = path.suffix.lower()
                if ext not in allowed_extensions:
                    return ValidationResult(
                        is_valid=False,
                        field_name=field_name,
                        severity=ValidationSeverity.ERROR,
                        message=f"不支持的文件类型: {ext}",
                        actual_value=ext,
                        expected_constraint=f"allowed_extensions={allowed_extensions}"
                    )

            # 检查文件大小
            if max_size_mb:
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > max_size_mb:
                    return ValidationResult(
                        is_valid=False,
                        field_name=field_name,
                        severity=ValidationSeverity.ERROR,
                        message=f"文件过大: {size_mb:.2f}MB (最大: {max_size_mb}MB)",
                        actual_value=size_mb,
                        expected_constraint=f"max_size_mb={max_size_mb}"
                    )

        return ValidationResult(
            is_valid=True,
            field_name=field_name,
            actual_value=file_path
        )

    @staticmethod
    def validate_list_items(
        items: List[Any],
        field_name: str,
        min_items: int = 0,
        max_items: Optional[int] = None,
        item_type: Optional[type] = None,
        required: bool = False
    ) -> ValidationResult:
        """
        验证列表

        Args:
            items: 列表
            field_name: 字段名
            min_items: 最小项数
            max_items: 最大项数
            item_type: 项类型
            required: 是否必填

        Returns:
            ValidationResult
        """
        if required and not items:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} 是必填字段",
                actual_value=items
            )

        if items is None:
            return ValidationResult(is_valid=True, field_name=field_name)

        # 检查项数
        if len(items) < min_items:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} 项数少于最小值 {min_items}",
                actual_value=len(items),
                expected_constraint=f"min_items={min_items}"
            )

        if max_items and len(items) > max_items:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} 项数超过最大值 {max_items}",
                actual_value=len(items),
                expected_constraint=f"max_items={max_items}"
            )

        # 检查项类型
        if item_type:
            for i, item in enumerate(items):
                if not isinstance(item, item_type):
                    return ValidationResult(
                        is_valid=False,
                        field_name=field_name,
                        severity=ValidationSeverity.ERROR,
                        message=f"{field_name}[{i}] 类型错误，期望 {item_type.__name__}",
                        actual_value=type(item).__name__,
                        expected_constraint=f"item_type={item_type.__name__}"
                    )

        return ValidationResult(
            is_valid=True,
            field_name=field_name,
            actual_value=len(items)
        )

    @staticmethod
    def validate_dict_fields(
        data: Dict[str, Any],
        field_name: str,
        required_fields: Optional[List[str]] = None,
        optional_fields: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        验证字典字段

        Args:
            data: 字典数据
            field_name: 字段名
            required_fields: 必填字段列表
            optional_fields: 可选字段列表

        Returns:
            ValidationResult
        """
        if not data:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} 是空的或不是字典",
                actual_value=data
            )

        # 检查必填字段
        if required_fields:
            missing_fields = [f for f in required_fields if f not in data or data[f] is None]
            if missing_fields:
                return ValidationResult(
                    is_valid=False,
                    field_name=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} 缺少必填字段: {', '.join(missing_fields)}",
                    actual_value=list(data.keys()),
                    expected_constraint=f"required_fields={required_fields}"
                )

        # 检查未知字段（警告）
        all_fields = set(required_fields or []) | set(optional_fields or [])
        unknown_fields = set(data.keys()) - all_fields
        if unknown_fields:
            logger.warning(f"{field_name} 包含未知字段: {', '.join(unknown_fields)}")

        return ValidationResult(
            is_valid=True,
            field_name=field_name,
            actual_value=list(data.keys())
        )

class CompositeValidator:
    """组合验证器"""

    def __init__(self):
        self.results: List[ValidationResult] = []

    def add_result(self, result: ValidationResult):
        """添加验证结果"""
        self.results.append(result)

    def add_validator(self, validator_func, *args, **kwargs):
        """添加验证器并执行"""
        result = validator_func(*args, **kwargs)
        self.add_result(result)
        return result

    def is_valid(self, severity: ValidationSeverity = ValidationSeverity.ERROR) -> bool:
        """
        检查是否全部通过验证

        Args:
            severity: 严重级别阈值

        Returns:
            bool: 是否全部通过
        """
        for result in self.results:
            if not result.is_valid and result.severity == severity:
                return False
        return True

    def get_errors(self) -> List[ValidationResult]:
        """获取所有错误"""
        return [r for r in self.results if not r.is_valid and r.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> List[ValidationResult]:
        """获取所有警告"""
        return [r for r in self.results if not r.is_valid and r.severity == ValidationSeverity.WARNING]

    def raise_if_invalid(self):
        """如果有错误则抛出异常"""
        if not self.is_valid():
            raise ValidationError(self.get_errors())

    def clear(self):
        """清除验证结果"""
        self.results.clear()

# 便捷函数
def validate_and_raise(validator: CompositeValidator):
    """
    验证并在有错误时抛出异常

    Args:
        validator: 组合验证器

    Raises:
        ValidationError: 当验证失败时
    """
    if not validator.is_valid():
        errors = validator.get_errors()
        logger.error(f"验证失败: {len(errors)} 个错误")
        for error in errors:
            logger.error(f"  - {error.field_name}: {error.message}")
        raise ValidationError(errors)
