"""
枚举工具类

提供大小写不敏感的枚举类型，支持向后兼容
"""

from sqlalchemy import TypeDecorator, String
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class CaseInsensitiveEnum(TypeDecorator):
    """
    大小写不敏感的枚举类型

    自动将数据库中的大写值（如 'PROCESSING'）转换为小写（如 'processing'）
    以支持向后兼容

    用法:
        status = Column(CaseInsensitiveEnum(TaskStatus, 50), default=TaskStatus.PENDING, index=True)

    特性:
        1. 从数据库读取时，自动将大写值转换为小写枚举值
        2. 写入数据库时，使用枚举的小写值
        3. 记录转换日志，便于追踪
        4. 支持所有枚举类型
    """
    impl = String

    def __init__(self, enum_class, length=50, *args, **kwargs):
        """
        初始化大小写不敏感枚举

        Args:
            enum_class: 枚举类
            length: 字符串长度（默认50）
            *args, **kwargs: 其他参数
        """
        self.enum_class = enum_class
        self.length = length
        super().__init__(length=length, *args, **kwargs)

    def process_bind_param(self, value, dialect):
        """
        写入数据库时，使用枚举的小写值

        Args:
            value: 要写入的值（可能是枚举或字符串）
            dialect: 数据库方言

        Returns:
            str: 小写的枚举值
        """
        if value is None:
            return None

        # 如果是枚举类型，直接获取其值
        if hasattr(value, 'value'):
            return value.value

        # 如果是字符串，尝试匹配枚举值（不区分大小写）
        if isinstance(value, str):
            # 尝试匹配枚举值（不区分大小写）
            for enum_value in self.enum_class:
                if value.upper() == enum_value.name or value.lower() == enum_value.value:
                    return enum_value.value

            # 如果没有匹配，记录警告并返回小写
            logger.warning(f"未知的枚举值: {value}, 返回小写形式")
            return value.lower()

        return value

    def process_result_value(self, value, dialect):
        """
        从数据库读取时，将大写值转换为小写枚举值

        Args:
            value: 从数据库读取的值
            dialect: 数据库方言

        Returns:
            str: 对应的小写枚举值
        """
        if value is None:
            return None

        # 如果是大写值，转换为对应的小写枚举值
        if isinstance(value, str):
            value_upper = value.upper()

            # 尝试匹配枚举名称（大写）
            for enum_value in self.enum_class:
                if value_upper == enum_value.name:
                    logger.info(f"将旧的大写状态值 '{value}' 转换为新的小写值 '{enum_value.value}'")
                    return enum_value.value

                # 也尝试直接匹配小写值
                if value.lower() == enum_value.value:
                    return enum_value.value

            # 如果没有匹配，返回小写值
            logger.warning(f"未知的枚举值 '{value}'，返回小写形式")
            return value.lower()

        return value
