"""
统一的解析器接口定义
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ParserConfig:
    """统一的解析器配置"""
    enable_multimodal: bool = True
    enable_ocr: bool = False
    extract_images: bool = True
    extract_tables: bool = True
    extract_formulas: bool = True
    chunk_size: int = 2000
    chunk_overlap: int = 200
    output_format: str = 'markdown'  # markdown, json
    max_file_size: int = 100 * 1024 * 1024

    def get(self, key: str, default: Any = None) -> Any:
        """字典式访问"""
        return getattr(self, key, default)


@dataclass
class ParserResult:
    """统一的解析结果"""
    success: bool
    content: str
    metadata: Dict[str, Any]
    sections: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    parse_time: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'content': self.content,
            'metadata': self.metadata,
            'sections': self.sections,
            'tables': self.tables,
            'images': self.images,
            'parse_time': self.parse_time,
            'error': self.error,
            'warnings': self.warnings
        }


class IParser(ABC):
    """解析器接口"""

    @abstractmethod
    async def parse(
        self,
        file_path: str,
        config: ParserConfig = None
    ) -> ParserResult:
        """解析文件"""
        pass

    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """判断是否支持该文件类型"""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取解析器信息"""
        pass

    @abstractmethod
    async def validate(self, file_path: str) -> tuple[bool, Optional[str]]:
        """验证文件"""
        pass
