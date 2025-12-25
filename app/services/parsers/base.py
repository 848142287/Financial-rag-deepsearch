"""
基础文件解析器抽象类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """解析结果数据类"""
    content: str
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    parse_time: Optional[float] = None
    file_size: Optional[int] = None
    encoding: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'content': self.content,
            'metadata': self.metadata,
            'success': self.success,
            'error_message': self.error_message,
            'parse_time': self.parse_time,
            'file_size': self.file_size,
            'encoding': self.encoding
        }


@dataclass
class DocumentChunk:
    """文档分块数据类"""
    content: str
    chunk_index: int
    metadata: Dict[str, Any]
    source_type: str
    page_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'content': self.content,
            'chunk_index': self.chunk_index,
            'metadata': self.metadata,
            'source_type': self.source_type,
            'page_info': self.page_info
        }


class BaseFileParser(ABC):
    """文件解析器基类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名列表"""
        pass

    @property
    @abstractmethod
    def parser_name(self) -> str:
        """解析器名称"""
        pass

    @abstractmethod
    def can_parse(self, file_path: str, file_extension: str = None) -> bool:
        """检查是否能解析指定文件"""
        pass

    @abstractmethod
    async def parse(self, file_path: str, **kwargs) -> ParseResult:
        """解析文件内容"""
        pass

    async def parse_with_metadata(self, file_path: str, **kwargs) -> ParseResult:
        """带元数据的文件解析"""
        import os
        import time
        from pathlib import Path

        start_time = time.time()
        file_path_obj = Path(file_path)

        try:
            # 获取文件基本信息
            file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0

            # 执行解析
            result = await self.parse(file_path, **kwargs)

            # 添加基础元数据
            result.metadata.update({
                'parser_name': self.parser_name,
                'file_extension': file_path_obj.suffix.lower(),
                'file_size': file_size,
                'file_name': file_path_obj.name,
                'parsed_at': datetime.utcnow().isoformat(),
                'parsing_config': self.config
            })

            result.file_size = file_size
            result.parse_time = time.time() - start_time

            self.logger.info(f"Successfully parsed file: {file_path} with {self.parser_name}")
            return result

        except Exception as e:
            parse_time = time.time() - start_time
            error_msg = f"Failed to parse file {file_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return ParseResult(
                content="",
                metadata={'error_details': str(e)},
                success=False,
                error_message=error_msg,
                parse_time=parse_time,
                file_size=file_path_obj.stat().st_size if file_path_obj.exists() else 0
            )

    def detect_encoding(self, file_path: str) -> Optional[str]:
        """检测文件编码"""
        try:
            import chardet

            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # 读取前10KB用于检测编码
                result = chardet.detect(raw_data)
                return result.get('encoding')
        except ImportError:
            self.logger.warning("chardet not installed, using default encoding")
            return None
        except Exception as e:
            self.logger.error(f"Failed to detect encoding for {file_path}: {str(e)}")
            return None

    def validate_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """验证文件是否可解析"""
        import os
        from pathlib import Path

        # 检查文件是否存在
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        # 检查文件是否可读
        if not os.access(file_path, os.R_OK):
            return False, f"File not readable: {file_path}"

        # 检查文件大小
        file_size = Path(file_path).stat().st_size
        max_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 默认100MB

        if file_size > max_size:
            return False, f"File too large: {file_size} bytes (max: {max_size})"

        if file_size == 0:
            return False, "File is empty"

        return True, None

    def chunk_content(
        self,
        content: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """将内容分割成块"""
        if not content or chunk_size <= 0:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size

            if end >= len(content):
                # 最后一个块
                chunk_text = content[start:]
            else:
                # 尝试在合适的位置分割
                chunk_text = content[start:end]

                # 寻找最近的分割点（句号、换行等）
                split_points = ['\n\n', '\n', '。', '！', '？', '. ', '! ', '? ']
                split_pos = -1

                for point in split_points:
                    pos = chunk_text.rfind(point)
                    if pos > chunk_size * 0.8:  # 确保分割点不会太早
                        split_pos = pos + len(point)
                        break

                if split_pos > 0:
                    chunk_text = chunk_text[:split_pos]
                    end = start + split_pos

            if chunk_text.strip():
                chunk_metadata = {
                    'chunk_size': len(chunk_text),
                    'chunk_start': start,
                    'chunk_end': end,
                    **(metadata or {})
                }

                chunks.append(DocumentChunk(
                    content=chunk_text.strip(),
                    chunk_index=chunk_index,
                    metadata=chunk_metadata,
                    source_type=self.parser_name
                ))

                chunk_index += 1

            # 计算下一个块的起始位置
            start = end - chunk_overlap if end < len(content) else end

        return chunks

    def extract_metadata_from_content(self, content: str) -> Dict[str, Any]:
        """从内容中提取元数据"""
        metadata = {}

        # 基本统计信息
        metadata.update({
            'content_length': len(content),
            'word_count': len(content.split()),
            'line_count': content.count('\n') + 1,
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
        })

        # 语言检测（如果安装了相应库）
        try:
            import langdetect
            metadata['detected_language'] = langdetect.detect(content[:1000])
        except ImportError:
            pass
        except:
            pass

        return metadata

    def clean_content(self, content: str) -> str:
        """清理内容"""
        if not content:
            return ""

        # 移除多余的空白字符
        import re
        content = re.sub(r'\n\s*\n', '\n\n', content)  # 多个空行替换为两个
        content = re.sub(r'[ \t]+', ' ', content)      # 多个空格替换为一个
        content = re.sub(r'\n[ \t]+', '\n', content)   # 行首空格

        return content.strip()

    def get_parser_info(self) -> Dict[str, Any]:
        """获取解析器信息"""
        return {
            'name': self.parser_name,
            'supported_extensions': self.supported_extensions,
            'config': self.config,
            'class_name': self.__class__.__name__
        }


class ParserError(Exception):
    """解析器异常基类"""
    pass


class UnsupportedFileTypeError(ParserError):
    """不支持的文件类型异常"""
    pass


class FileParsingError(ParserError):
    """文件解析异常"""
    pass


class FileValidationError(ParserError):
    """文件验证异常"""
    pass