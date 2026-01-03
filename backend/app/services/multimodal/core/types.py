"""
多模态解析共享类型定义
避免循环导入
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class ContentType(Enum):
    """内容类型枚举"""
    CHAPTER = "chapter"
    TEXT = "text"
    IMAGE = "image"
    FORMULA = "formula"
    TABLE = "table"
    FIGURE = "figure"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    CODE = "code"
    LIST = "list"


@dataclass
class ContentBlock:
    """内容块"""
    id: str
    content_type: ContentType
    content: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    page_number: int = 0
    chapter_id: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class Chapter:
    """章节结构"""
    id: str
    title: str
    level: int  # 1: 一级标题, 2: 二级标题, ...
    start_page: int
    end_page: int
    sub_chapters: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """解析后的文档"""
    document_id: str
    title: str
    total_pages: int
    chapters: List[Chapter]
    content_blocks: List[ContentBlock]
    metadata: Dict[str, Any] = field(default_factory=dict)
    parsing_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsingConfig:
    """解析配置"""
    use_mineru: bool = True
    use_qwen_vl_ocr: bool = True
    use_qwen_vl_max: bool = True
    prefer_high_quality: bool = True
    enable_auto_repair: bool = True
    integrity_threshold: float = 0.8
    max_repair_attempts: int = 3
    parallel_processing: bool = True
    temp_dir: Optional[str] = None
