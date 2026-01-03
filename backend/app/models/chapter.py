"""
章节和目录数据模型
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class ChapterType(str, enum.Enum):
    PART = "part"           # 部分
    CHAPTER = "chapter"     # 章节
    SECTION = "section"     # 节
    SUBSECTION = "subsection"  # 小节
    APPENDIX = "appendix"   # 附录


class Chapter(Base):
    """文档章节表"""
    __tablename__ = "chapters"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False, index=True)
    parent_id = Column(Integer, ForeignKey("chapters.id", ondelete="SET NULL", onupdate="CASCADE"), nullable=True, index=True)

    # 章节基本信息
    chapter_type = Column(Enum(ChapterType), nullable=False, index=True)
    level = Column(Integer, nullable=False, default=1)  # 层级深度
    order_index = Column(Integer, nullable=False)  # 在文档中的顺序
    title = Column(String(1000), nullable=False)
    subtitle = Column(String(1000))

    # 章节内容
    content = Column(Text)
    summary = Column(Text)  # 章节摘要
    key_points = Column(JSON)  # 关键点列表
    tags = Column(JSON)  # 标签列表

    # 位置信息
    page_start = Column(Integer)  # 起始页码
    page_end = Column(Integer)    # 结束页码
    page_count = Column(Integer)  # 页面数量

    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 元数据
    chapter_metadata = Column(JSON)  # 额外的元数据
    confidence_score = Column(Float)  # 章节识别置信度

    # 关系
    document = relationship("Document", backref="chapters")
    parent = relationship("Chapter", remote_side=[id], backref="children")

    def __repr__(self):
        return f"<Chapter(id={self.id}, document_id={self.document_id}, title='{self.title}', level={self.level})>"


class ChapterTableOfContents(Base):
    """章节目录表"""
    __tablename__ = "chapter_toc"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False, index=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False, index=True)

    # 目录结构
    toc_path = Column(String(2000))  # 完整的目录路径 (1.1.2.3)
    toc_depth = Column(Integer, nullable=False)  # 目录深度
    toc_order = Column(Integer, nullable=False)  # 在同级目录中的顺序

    # 显示信息
    display_title = Column(String(1000))  # 显示标题
    display_number = Column(String(50))   # 章节编号 (如 "1.2.3")
    indent_level = Column(Integer)        # 缩进级别

    # 导航信息
    is_expandable = Column(Integer, default=1)  # 是否可展开 (0/1)
    is_expanded = Column(Integer, default=0)    # 是否默认展开
    has_children = Column(Integer, default=0)   # 是否有子章节

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    document = relationship("Document")
    chapter = relationship("Chapter")

    def __repr__(self):
        return f"<ChapterTableOfContents(document_id={self.document_id}, toc_path='{self.toc_path}')>"


class ChapterCrossReference(Base):
    """章节交叉引用表"""
    __tablename__ = "chapter_cross_references"

    id = Column(Integer, primary_key=True, index=True)
    source_chapter_id = Column(Integer, ForeignKey("chapters.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False, index=True)
    target_chapter_id = Column(Integer, ForeignKey("chapters.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False, index=True)

    # 引用类型
    reference_type = Column(String(50), nullable=False)  # see_also, reference, cites, related
    reference_text = Column(Text)  # 引用文本
    context = Column(Text)         # 引用上下文

    # 位置信息
    source_page = Column(Integer)
    source_position = Column(Integer)  # 在源章节中的位置

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    source_chapter = relationship("Chapter", foreign_keys=[source_chapter_id], backref="outgoing_references")
    target_chapter = relationship("Chapter", foreign_keys=[target_chapter_id], backref="incoming_references")
    document = relationship("Document")

    def __repr__(self):
        return f"<ChapterCrossReference(source={self.source_chapter_id}, target={self.target_chapter_id}, type='{self.reference_type}')>"