"""
文本文档解析器
处理.txt格式的文档
"""

import logging
from typing import Dict, List, Any, Optional
from .base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)


class TextParser(BaseFileParser):
    """文本文档解析器"""

    def __init__(self):
        self.supported_extensions = ['.txt', '.log', '.conf', '.cfg', '.ini']
        self.supported_mime_types = ['text/plain', 'text/log', 'application/octet-stream']

    def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        解析文本文档

        Args:
            file_path: 文件路径
            **kwargs: 其他参数

        Returns:
            ParseResult: 解析结果
        """
        try:
            import os
            file_size = os.path.getsize(file_path)

            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'ascii']
            content = ""
            used_encoding = ""

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"使用 {encoding} 编码读取失败: {e}")
                    continue

            if not content:
                # 如果所有编码都失败，使用基础信息
                content = f"文本文件: {os.path.basename(file_path)}\n"
                content += f"文件大小: {file_size} 字节\n"
                content += "注意: 无法解码文件内容"

            # 分析文本内容
            lines = content.splitlines()
            words = content.split()
            chars = len(content)

            # 构建元数据
            metadata = {
                'file_type': 'text',
                'file_size': file_size,
                'encoding': used_encoding,
                'line_count': len(lines),
                'word_count': len(words),
                'char_count': chars,
                'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
                'max_line_length': max(len(line) for line in lines) if lines else 0
            }

            # 检测是否包含特定模式
            if any(keyword in content.lower() for keyword in ['表格', '数据', '图表', '统计']):
                metadata['contains_data'] = True

            if any(keyword in content.lower() for keyword in ['http://', 'https://', 'www.']):
                metadata['contains_urls'] = True

            # 提取前几行作为预览
            preview_lines = lines[:10] if lines else []
            metadata['preview'] = preview_lines

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error(f"文本文档解析失败: {e}")
            return ParseResult(
                content="",
                metadata={'file_type': 'text', 'error': str(e)},
                success=False,
                error_message=str(e)
            )

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