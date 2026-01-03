"""
Parser工厂模式
统一管理和创建所有文档解析器
"""

from pathlib import Path
from typing import Dict, Type, Optional, List
from app.core.structured_logging import get_structured_logger

# 导入基类
from app.services.parsers.base import BaseFileParser

# 导入格式解析器
from app.services.parsers.formats import (
    UnifiedPDFParser,
    UnifiedExcelParser,
    UnifiedPPTParser,
    UnifiedWordParser
)

# 导入简单解析器
from app.services.parsers.csv_parser import CSVParser
from app.services.parsers.text_parser import TextParser

logger = get_structured_logger(__name__)


class ParserFactory:
    """
    Parser工厂类

    功能：
    1. 统一管理所有解析器
    2. 根据文件扩展名自动选择解析器
    3. 支持解析器配置和自定义
    4. 提供解析器注册机制
    """

    # 内置解析器注册表
    _parser_registry: Dict[str, Type[BaseFileParser]] = {}

    @classmethod
    def _register_default_parsers(cls):
        """注册默认解析器"""
        default_parsers = {
            # PDF解析器
            '.pdf': UnifiedPDFParser,

            # Office文档解析器
            '.xlsx': UnifiedExcelParser,
            '.xls': UnifiedExcelParser,
            '.pptx': UnifiedPPTParser,
            '.ppt': UnifiedPPTParser,
            '.docx': UnifiedWordParser,
            '.doc': UnifiedWordParser,

            # 简单文本解析器
            '.csv': CSVParser,
            '.txt': TextParser,
            '.md': TextParser,
            '.json': TextParser,
            '.xml': TextParser,
        }

        for ext, parser_cls in default_parsers.items():
            cls.register_parser(ext, parser_cls)

    @classmethod
    def register_parser(cls, extension: str, parser_class: Type[BaseFileParser]):
        """
        注册解析器

        Args:
            extension: 文件扩展名（如'.pdf'）
            parser_class: 解析器类
        """
        ext = extension.lower()
        if not ext.startswith('.'):
            ext = '.' + ext

        cls._parser_registry[ext] = parser_class
        logger.info(f"注册解析器: {ext} -> {parser_class.__name__}")

    @classmethod
    def unregister_parser(cls, extension: str):
        """
        注销解析器

        Args:
            extension: 文件扩展名
        """
        ext = extension.lower()
        if not ext.startswith('.'):
            ext = '.' + ext

        if ext in cls._parser_registry:
            del cls._parser_registry[ext]
            logger.info(f"注销解析器: {ext}")

    @classmethod
    def get_parser(cls, file_path: str, config: Optional[Dict] = None) -> BaseFileParser:
        """
        根据文件路径获取对应的解析器

        Args:
            file_path: 文件路径
            config: 解析器配置（可选）

        Returns:
            解析器实例

        Raises:
            ValueError: 不支持的文件类型
        """
        # 确保默认解析器已注册
        if not cls._parser_registry:
            cls._register_default_parsers()

        # 获取文件扩展名
        ext = Path(file_path).suffix.lower()

        # 查找对应的解析器
        if ext not in cls._parser_registry:
            # 尝试通用文本解析器
            if cls._is_text_file(file_path):
                return TextParser(config or {})

            raise ValueError(
                f"不支持的文件类型: {ext}\n"
                f"支持的类型: {', '.join(cls.supported_extensions())}"
            )

        parser_class = cls._parser_registry[ext]
        return parser_class(config or {})

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        获取所有支持的文件扩展名

        Returns:
            扩展名列表
        """
        # 确保默认解析器已注册
        if not cls._parser_registry:
            cls._register_default_parsers()

        return list(cls._parser_registry.keys())

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """获取支持的文件扩展名（别名方法）"""
        return cls.get_supported_extensions()

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """
        检查文件类型是否支持

        Args:
            file_path: 文件路径

        Returns:
            是否支持
        """
        ext = Path(file_path).suffix.lower()
        return ext in cls.supported_extensions()

    @classmethod
    def _is_text_file(cls, file_path: str) -> bool:
        """
        检查是否为文本文件

        Args:
            file_path: 文件路径

        Returns:
            是否为文本文件
        """
        text_extensions = {
            '.txt', '.md', '.markdown', '.rst',
            '.json', '.xml', '.yaml', '.yml',
            '.log', '.ini', '.conf', '.cfg'
        }

        ext = Path(file_path).suffix.lower()
        return ext in text_extensions

    @classmethod
    async def parse_file(cls, file_path: str, **kwargs):
        """
        便捷方法：解析文件

        Args:
            file_path: 文件路径
            **kwargs: 传递给解析器的额外参数

        Returns:
            解析结果
        """
        parser = cls.get_parser(file_path, kwargs.get('config'))
        return await parser.parse(file_path, **kwargs)

    @classmethod
    def get_parser_info(cls, extension: str) -> Dict:
        """
        获取解析器信息

        Args:
            extension: 文件扩展名

        Returns:
            解析器信息字典
        """
        if not cls._parser_registry:
            cls._register_default_parsers()

        ext = extension.lower()
        if not ext.startswith('.'):
            ext = '.' + ext

        if ext not in cls._parser_registry:
            return {'error': f'不支持的文件类型: {ext}'}

        parser_class = cls._parser_registry[ext]
        temp_parser = parser_class()

        return {
            'extension': ext,
            'parser_class': parser_class.__name__,
            'module': parser_class.__module__,
            'supported_extensions': temp_parser.supported_extensions,
            'parser_name': temp_parser.parser_name
        }

    @classmethod
    def list_all_parsers(cls) -> Dict[str, Dict]:
        """
        列出所有已注册的解析器信息

        Returns:
            {扩展名: 解析器信息}
        """
        if not cls._parser_registry:
            cls._register_default_parsers()

        parsers_info = {}
        for ext in cls._parser_registry:
            parsers_info[ext] = cls.get_parser_info(ext)

        return parsers_info


# 便捷函数
def get_parser(file_path: str, config: Optional[Dict] = None) -> BaseFileParser:
    """获取解析器（便捷函数）"""
    return ParserFactory.get_parser(file_path, config)


async def parse_file(file_path: str, **kwargs):
    """解析文件（便捷函数）"""
    return await ParserFactory.parse_file(file_path, **kwargs)


def is_supported_file(file_path: str) -> bool:
    """检查文件是否支持（便捷函数）"""
    return ParserFactory.is_supported(file_path)


# 导出
__all__ = [
    'ParserFactory',
    'get_parser',
    'parse_file',
    'is_supported_file'
]
