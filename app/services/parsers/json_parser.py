"""
JSON文档解析器
处理.json格式的文档
"""

import logging
import json
from typing import Dict, List, Any, Optional
from .base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)


class JSONParser(BaseFileParser):
    """JSON文档解析器"""

    def __init__(self):
        self.supported_extensions = ['.json', '.jsonl']
        self.supported_mime_types = ['application/json', 'application/ld+json']

    def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        解析JSON文档

        Args:
            file_path: 文件路径
            **kwargs: 其他参数

        Returns:
            ParseResult: 解析结果
        """
        try:
            import os
            _, ext = os.path.splitext(file_path.lower())

            if ext == '.jsonl':
                return self._parse_jsonl(file_path)
            else:
                return self._parse_json(file_path)

        except Exception as e:
            logger.error(f"JSON文档解析失败: {e}")
            return ParseResult(
                content="",
                metadata={'file_type': 'json', 'error': str(e)},
                success=False,
                error_message=str(e)
            )

    def _parse_json(self, file_path: str) -> ParseResult:
        """解析JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 将JSON转换为可读的文本内容
            content = self._format_json_content(data)

            # 构建元数据
            metadata = {
                'file_type': 'json',
                'data_type': type(data).__name__,
                'keys_count': 0,
                'depth': self._calculate_depth(data)
            }

            if isinstance(data, dict):
                metadata['keys_count'] = len(data)
                metadata['keys'] = list(data.keys())
            elif isinstance(data, list):
                metadata['items_count'] = len(data)
                if data and isinstance(data[0], dict):
                    metadata['first_item_keys'] = list(data[0].keys())

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return ParseResult(
                content="",
                metadata={'file_type': 'json', 'error': f'JSON格式错误: {str(e)}'},
                success=False,
                error_message=f'JSON decode error: {str(e)}'
            )

    def _parse_jsonl(self, file_path: str) -> ParseResult:
        """解析JSONL文件（每行一个JSON对象）"""
        try:
            lines = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            lines.append(f"行 {line_num}: {json.dumps(obj, ensure_ascii=False, indent=2)}")
                        except json.JSONDecodeError:
                            lines.append(f"行 {line_num}: 解析失败 - {line[:100]}")

            content = "\n\n".join(lines)

            metadata = {
                'file_type': 'jsonl',
                'lines_count': len(lines),
                'format': 'json_lines'
            }

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error(f"JSONL解析失败: {e}")
            return ParseResult(
                content="",
                metadata={'file_type': 'jsonl', 'error': str(e)},
                success=False,
                error_message=str(e)
            )

    def _format_json_content(self, data: Any, indent: int = 0) -> str:
        """将JSON数据格式化为可读文本"""
        prefix = "  " * indent

        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._format_json_content(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
            return "\n".join(lines)

        elif isinstance(data, list):
            lines = []
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i}]:")
                    lines.append(self._format_json_content(item, indent + 1))
                else:
                    lines.append(f"{prefix}[{i}]: {item}")
            return "\n".join(lines)

        else:
            return f"{prefix}{data}"

    def _calculate_depth(self, obj: Any, current_depth: int = 0) -> int:
        """计算JSON结构的深度"""
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