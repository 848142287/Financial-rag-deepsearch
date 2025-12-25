"""
YAML文档解析器
处理.yml和.yaml格式的文档
"""

import logging
from typing import Dict, List, Any, Optional
from .base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available, YAML parsing will be limited")


class YAMLParser(BaseFileParser):
    """YAML文档解析器"""

    def __init__(self):
        self.supported_extensions = ['.yml', '.yaml']
        self.supported_mime_types = ['application/x-yaml', 'text/yaml']

    def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        解析YAML文档

        Args:
            file_path: 文件路径
            **kwargs: 其他参数

        Returns:
            ParseResult: 解析结果
        """
        try:
            if not YAML_AVAILABLE:
                return self._parse_basic(file_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # 将YAML转换为可读的文本内容
            content = self._format_yaml_content(data)

            # 构建元数据
            metadata = {
                'file_type': 'yaml',
                'data_type': type(data).__name__ if data else 'empty',
                'keys_count': 0,
                'depth': self._calculate_depth(data) if data else 0
            }

            if isinstance(data, dict):
                metadata['keys_count'] = len(data)
                metadata['keys'] = list(data.keys())
            elif isinstance(data, list):
                metadata['items_count'] = len(data)

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error(f"YAML文档解析失败: {e}")
            return ParseResult(
                content="",
                metadata={'file_type': 'yaml', 'error': str(e)},
                success=False,
                error_message=str(e)
            )

    def _parse_basic(self, file_path: str) -> ParseResult:
        """基础YAML文件解析（当PyYAML不可用时）"""
        import os

        file_size = os.path.getsize(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            content = f"YAML文档: {os.path.basename(file_path)}\n"
            content += f"文件大小: {file_size} 字节\n"
            content += f"内容预览:\n{raw_content[:500]}...\n"
            content += "注意: 由于缺少PyYAML库，无法进行结构化解析"

            metadata = {
                'file_type': 'yaml',
                'file_size': file_size,
                'parsing_limited': True,
                'recommendation': 'Install PyYAML: pip install PyYAML'
            }

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error(f"基础YAML解析失败: {e}")
            return ParseResult(
                content="",
                metadata={'file_type': 'yaml', 'error': str(e)},
                success=False,
                error_message=str(e)
            )

    def _format_yaml_content(self, data: Any, indent: int = 0) -> str:
        """将YAML数据格式化为可读文本"""
        prefix = "  " * indent

        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._format_yaml_content(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
            return "\n".join(lines)

        elif isinstance(data, list):
            lines = []
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}- Item {i}:")
                    lines.append(self._format_yaml_content(item, indent + 1))
                else:
                    lines.append(f"{prefix}- {item}")
            return "\n".join(lines)

        else:
            return f"{prefix}{data}"

    def _calculate_depth(self, obj: Any, current_depth: int = 0) -> int:
        """计算YAML结构的深度"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_depth(value, current_depth + 1) for value in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth

    def is_supported(self, file_path: str, mime_type: str = None) -> bool:
        """
        检查是否支持解析该文件

        Args:
            file_path: 文件路径
            mime_type: MIME类型（可选）

        Returns:
            bool: 是否支持
        """
        import os
        _, ext = os.path.splitext(file_path.lower())
        if ext in self.supported_extensions:
            return True

        if mime_type and mime_type in self.supported_mime_types:
            return True

        return False