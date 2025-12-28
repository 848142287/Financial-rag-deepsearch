"""
解析器注册表
管理所有文件解析器的注册和查找
"""

from typing import Dict, List, Optional
from .base import BaseFileParser
import logging

logger = logging.getLogger(__name__)


class ParserRegistry:
    """解析器注册表"""

    def __init__(self):
        """初始化注册表"""
        self._parsers: Dict[str, BaseFileParser] = {}  # 按名称注册
        self._extension_map: Dict[str, List[str]] = {}  # 扩展名到解析器名称的映射
        self.logger = logging.getLogger(self.__class__.__name__)

    def register(self, parser: BaseFileParser) -> bool:
        """
        注册解析器

        Args:
            parser: 解析器实例

        Returns:
            bool: 是否注册成功
        """
        try:
            parser_name = parser.parser_name

            # 检查是否已注册同名解析器
            if parser_name in self._parsers:
                self.logger.warning(f"Parser '{parser_name}' already registered, overwriting")

            # 注册解析器
            self._parsers[parser_name] = parser

            # 更新扩展名映射
            for ext in parser.supported_extensions:
                ext_lower = ext.lower()
                if ext_lower not in self._extension_map:
                    self._extension_map[ext_lower] = []

                if parser_name not in self._extension_map[ext_lower]:
                    self._extension_map[ext_lower].append(parser_name)

            self.logger.info(f"Registered parser: {parser_name} for extensions: {parser.supported_extensions}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register parser {parser.parser_name}: {str(e)}")
            return False

    def unregister(self, parser_name: str) -> bool:
        """
        注销解析器

        Args:
            parser_name: 解析器名称

        Returns:
            bool: 是否注销成功
        """
        try:
            if parser_name not in self._parsers:
                self.logger.warning(f"Parser '{parser_name}' not found in registry")
                return False

            parser = self._parsers[parser_name]

            # 从扩展名映射中移除
            for ext in parser.supported_extensions:
                ext_lower = ext.lower()
                if ext_lower in self._extension_map:
                    if parser_name in self._extension_map[ext_lower]:
                        self._extension_map[ext_lower].remove(parser_name)

                    # 如果该扩展名没有其他解析器，删除映射
                    if not self._extension_map[ext_lower]:
                        del self._extension_map[ext_lower]

            # 从解析器字典中移除
            del self._parsers[parser_name]

            self.logger.info(f"Unregistered parser: {parser_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unregister parser {parser_name}: {str(e)}")
            return False

    def get_parser(self, parser_name: str) -> Optional[BaseFileParser]:
        """
        根据名称获取解析器

        Args:
            parser_name: 解析器名称

        Returns:
            Optional[BaseFileParser]: 解析器实例
        """
        return self._parsers.get(parser_name)

    def get_parser_by_extension(self, extension: str) -> Optional[BaseFileParser]:
        """
        根据文件扩展名获取解析器

        Args:
            extension: 文件扩展名

        Returns:
            Optional[BaseFileParser]: 解析器实例
        """
        ext_lower = extension.lower()

        # 移除扩展名前的点
        if ext_lower.startswith('.'):
            ext_lower = ext_lower[1:]

        if ext_lower in self._extension_map:
            parser_names = self._extension_map[ext_lower]
            # 返回第一个匹配的解析器
            if parser_names:
                return self._parsers.get(parser_names[0])

        return None

    def get_parsers_by_extension(self, extension: str) -> List[BaseFileParser]:
        """
        根据文件扩展名获取所有匹配的解析器

        Args:
            extension: 文件扩展名

        Returns:
            List[BaseFileParser]: 解析器实例列表
        """
        ext_lower = extension.lower()

        # 移除扩展名前的点
        if ext_lower.startswith('.'):
            ext_lower = ext_lower[1:]

        if ext_lower in self._extension_map:
            parser_names = self._extension_map[ext_lower]
            return [self._parsers[name] for name in parser_names if name in self._parsers]

        return []

    def get_all_parsers(self) -> List[BaseFileParser]:
        """
        获取所有注册的解析器

        Returns:
            List[BaseFileParser]: 解析器列表
        """
        return list(self._parsers.values())

    def get_supported_extensions(self) -> List[str]:
        """
        获取所有支持的文件扩展名

        Returns:
            List[str]: 扩展名列表
        """
        extensions = []
        for parser in self._parsers.values():
            extensions.extend(parser.supported_extensions)
        return sorted(list(set(extensions)))

    def get_parser_count(self) -> int:
        """
        获取注册的解析器数量

        Returns:
            int: 解析器数量
        """
        return len(self._parsers)

    def get_extension_count(self) -> int:
        """
        获取支持的扩展名数量

        Returns:
            int: 扩展名数量
        """
        return len(self._extension_map)

    def list_parsers(self) -> List[Dict]:
        """
        列出所有解析器信息

        Returns:
            List[Dict]: 解析器信息列表
        """
        parser_info = []
        for parser in self._parsers.values():
            info = parser.get_parser_info()
            info['extension_count'] = len(parser.supported_extensions)
            parser_info.append(info)
        return parser_info

    def list_extensions(self) -> List[Dict]:
        """
        列出所有支持的扩展名信息

        Returns:
            List[Dict]: 扩展名信息列表
        """
        extension_info = []
        for ext, parser_names in self._extension_map.items():
            extension_info.append({
                'extension': f'.{ext}',
                'parser_names': parser_names,
                'parser_count': len(parser_names)
            })
        return sorted(extension_info, key=lambda x: x['extension'])

    def has_parser(self, parser_name: str) -> bool:
        """
        检查是否注册了指定解析器

        Args:
            parser_name: 解析器名称

        Returns:
            bool: 是否已注册
        """
        return parser_name in self._parsers

    def supports_extension(self, extension: str) -> bool:
        """
        检查是否支持指定扩展名

        Args:
            extension: 文件扩展名

        Returns:
            bool: 是否支持
        """
        ext_lower = extension.lower()
        if ext_lower.startswith('.'):
            ext_lower = ext_lower[1:]
        return ext_lower in self._extension_map

    def get_registry_stats(self) -> Dict:
        """
        获取注册表统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            'total_parsers': len(self._parsers),
            'total_extensions': len(self._extension_map),
            'parsers': {
                name: {
                    'supported_extensions': parser.supported_extensions,
                    'class_name': parser.__class__.__name__
                }
                for name, parser in self._parsers.items()
            },
            'extensions': {
                f'.{ext}': parsers
                for ext, parsers in self._extension_map.items()
            }
        }

    def clear(self) -> None:
        """
        清空注册表
        """
        self._parsers.clear()
        self._extension_map.clear()
        self.logger.info("Parser registry cleared")

    def __len__(self) -> int:
        """
        返回注册的解析器数量

        Returns:
            int: 解析器数量
        """
        return len(self._parsers)

    def __contains__(self, parser_name: str) -> bool:
        """
        检查是否包含指定解析器

        Args:
            parser_name: 解析器名称

        Returns:
            bool: 是否包含
        """
        return parser_name in self._parsers

    def __iter__(self):
        """
        迭代所有解析器

        Returns:
            Iterator: 解析器迭代器
        """
        return iter(self._parsers.values())