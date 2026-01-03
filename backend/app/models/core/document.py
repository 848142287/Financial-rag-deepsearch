"""
统一的文档相关数据模型

包括：DocumentChunk（文档切块）、DocumentMetadata（文档元数据）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class DocumentMetadata:
    """
    统一的文档元数据类

    替代在多个文件中重复定义的文档元数据类
    """
    source: str  # 来源文件
    file_type: str = ""  # 文件类型
    title: str = ""  # 标题
    author: str = ""  # 作者
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    page_numbers: List[int] = field(default_factory=list)
    section: str = ""  # 章节/部分
    tags: List[str] = field(default_factory=list)
    language: str = "zh"  # 语言

    # 扩展元数据
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "source": self.source,
            "file_type": self.file_type,
            "title": self.title,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "page_numbers": self.page_numbers,
            "section": self.section,
            "tags": self.tags,
            "language": self.language,
            **self.extra
        }


@dataclass
class DocumentChunk:
    """
    统一的文档切块类

    替代在6个文件中重复定义的DocumentChunk类
    """
    content: str  # 切块内容
    chunk_id: str = ""  # 切块ID
    metadata: DocumentMetadata = None
    position: int = 0  # 在文档中的位置
    token_count: int = 0  # Token数量（估算）
    embedding: Optional[List[float]] = None  # 向量嵌入

    # 层级信息（用于结构化文档）
    h1: str = ""  # 一级标题
    h2: str = ""  # 二级标题
    h3: str = ""  # 三级标题
    title_path: str = ""  # 标题路径（如 "规则 > 命名规范"）

    def __post_init__(self):
        """初始化后处理"""
        if self.metadata is None:
            self.metadata = DocumentMetadata(source="")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata.to_dict() if self.metadata else {},
            "position": self.position,
            "token_count": self.token_count,
            "h1": self.h1,
            "h2": self.h2,
            "h3": self.h3,
            "title_path": self.title_path
        }

    def get_text(self) -> str:
        """获取纯文本内容"""
        return self.content

    def get_context(self, window: int = 100) -> str:
        """获取上下文（前后window字符）"""
        # 简化实现，返回完整内容
        return self.content

    def estimate_tokens(self) -> int:
        """估算Token数量"""
        # 简单估算：中文约2字符=1token，英文约4字符=1token
        # 这里使用简化计算：1字符≈0.5token
        return len(self.content) // 2


@dataclass
class ChunkMetadata:
    """
    切块元数据（DocumentChunk的轻量版）

    用于不需要完整DocumentMetadata的场景
    """
    source: str
    h1: str = ""
    h2: str = ""
    h3: str = ""
    title_path: str = ""
    chunk_type: str = "content"  # content, heading, rule, example
    position: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "source": self.source,
            "h1": self.h1,
            "h2": self.h2,
            "h3": self.h3,
            "title_path": self.title_path,
            "chunk_type": self.chunk_type,
            "position": self.position
        }


__all__ = [
    'DocumentMetadata',
    'DocumentChunk',
    'ChunkMetadata',
]
