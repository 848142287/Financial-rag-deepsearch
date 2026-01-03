"""
分层索引的Pydantic模式
包括：文档摘要索引、章节索引、片段索引
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ChunkType(str, Enum):
    """片段类型"""
    TEXT = "text"
    TABLE = "table"
    LIST = "list"
    IMAGE = "image"
    MIXED = "mixed"


# ==================== 文档摘要索引 ====================

class DocumentSummaryIndex(BaseModel):
    """文档摘要索引"""
    document_id: str = Field(..., description="文档ID")
    summary_text: str = Field(..., description="文档整体摘要")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    entities: List[str] = Field(default_factory=list, description="实体列表（公司、人物等）")
    topics: List[str] = Field(default_factory=list, description="主题列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    # 统计信息
    doc_length: int = Field(0, description="文档长度")
    section_count: int = Field(0, description="章节数量")
    chunk_count: int = Field(0, description="片段数量")

    # 嵌入向量
    embedding: Optional[List[float]] = Field(None, description="摘要嵌入向量")

    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

    class Config:
        from_attributes = True


# ==================== 章节索引 ====================

class ChapterIndex(BaseModel):
    """章节索引"""
    chapter_id: str = Field(..., description="章节ID")
    document_id: str = Field(..., description="文档ID")
    title: str = Field(..., description="章节标题")
    level: int = Field(..., description="章节层级（1,2,3...）")
    summary: str = Field(..., description="章节摘要")
    keywords: List[str] = Field(default_factory=list, description="章节关键词")

    # 层级关系
    parent_chapter_id: Optional[str] = Field(None, description="父章节ID")
    child_chapter_ids: List[str] = Field(default_factory=list, description="子章节ID列表")

    # 位置信息
    start_page: Optional[int] = Field(None, description="起始页码")
    end_page: Optional[int] = Field(None, description="结束页码")
    start_char: int = Field(0, description="起始字符位置")
    end_char: int = Field(0, description="结束字符位置")

    # 嵌入向量
    embedding: Optional[List[float]] = Field(None, description="章节摘要嵌入向量")

    # 统计信息
    chunk_count: int = Field(0, description="包含的片段数量")

    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

    class Config:
        from_attributes = True


# ==================== 片段索引 ====================

class ChunkIndex(BaseModel):
    """片段索引"""
    chunk_id: str = Field(..., description="片段ID")
    document_id: str = Field(..., description="文档ID")
    chapter_id: Optional[str] = Field(None, description="所属章节ID")

    content: str = Field(..., description="片段内容")
    chunk_type: ChunkType = Field(ChunkType.TEXT, description="片段类型")

    # 位置信息
    chunk_index: int = Field(..., description="片段索引")
    page_number: Optional[int] = Field(None, description="页码")
    start_char: int = Field(0, description="起始字符位置")
    end_char: int = Field(0, description="结束字符位置")

    # 嵌入向量
    embedding: Optional[List[float]] = Field(None, description="内容嵌入向量")

    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

    class Config:
        from_attributes = True


# ==================== 分层索引结构 ====================

class HierarchicalIndex(BaseModel):
    """分层索引完整结构"""
    document_id: str = Field(..., description="文档ID")
    document_summary: DocumentSummaryIndex = Field(..., description="文档摘要索引")
    chapters: List[ChapterIndex] = Field(default_factory=list, description="章节索引列表")
    chunks: List[ChunkIndex] = Field(default_factory=list, description="片段索引列表")

    # 统计信息
    total_chapters: int = Field(0, description="总章节数")
    total_chunks: int = Field(0, description="总片段数")

    # 处理状态
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    processing_time: float = Field(0.0, description="处理时间（秒）")

    class Config:
        from_attributes = True


# ==================== 检索结果 ====================

class RetrievedChunk(BaseModel):
    """检索到的片段"""
    chunk_id: str
    content: str
    score: float = Field(..., description="相似度得分")
    chapter_title: Optional[str] = None
    chapter_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChapter(BaseModel):
    """检索到的章节"""
    chapter_id: str
    document_id: str
    title: str
    summary: str
    score: float = Field(..., description="相似度得分")
    level: int
    chunk_count: int


class RetrievedDocument(BaseModel):
    """检索到的文档"""
    document_id: str
    summary_text: str
    score: float = Field(..., description="相似度得分")
    keywords: List[str]
    entities: List[str]


class HierarchicalRetrievalResult(BaseModel):
    """分层检索结果"""
    query: str = Field(..., description="查询语句")

    # 三层检索结果
    documents: List[RetrievedDocument] = Field(default_factory=list, description="文档级结果")
    chapters: List[RetrievedChapter] = Field(default_factory=list, description="章节级结果")
    chunks: List[RetrievedChunk] = Field(default_factory=list, description="片段级结果")

    # 合并后的最终结果
    merged_results: List[RetrievedChunk] = Field(default_factory=list, description="合并后的最终结果")

    # 统计信息
    retrieval_time: float = Field(0.0, description="检索耗时（秒）")
    total_docs: int = Field(0, description="检索到的文档数")
    total_chapters: int = Field(0, description="检索到的章节数")
    total_chunks: int = Field(0, description="检索到的片段数")


# ==================== 请求/响应模式 ====================

class HierarchicalRetrievalRequest(BaseModel):
    """分层检索请求"""
    query: str = Field(..., description="查询语句", min_length=1)
    top_k: int = Field(5, ge=1, le=50, description="返回结果数量")
    document_ids: Optional[List[str]] = Field(None, description="限定搜索的文档ID列表")
    chapter_ids: Optional[List[str]] = Field(None, description="限定搜索的章节ID列表")

    # 检索策略
    use_summary: bool = Field(True, description="是否使用文档摘要检索")
    use_chapters: bool = Field(True, description="是否使用章节检索")
    use_chunks: bool = Field(True, description="是否使用片段检索")

    # 阈值设置
    doc_threshold: float = Field(0.6, ge=0.0, le=1.0, description="文档级相似度阈值")
    chapter_threshold: float = Field(0.5, ge=0.0, le=1.0, description="章节级相似度阈值")
    chunk_threshold: float = Field(0.4, ge=0.0, le=1.0, description="片段级相似度阈值")

    # 结果数量控制
    max_docs: int = Field(10, ge=1, le=100, description="最大文档数量")
    max_chapters_per_doc: int = Field(5, ge=1, le=20, description="每文档最大章节数")
    max_chunks_per_chapter: int = Field(5, ge=1, le=20, description="每章节最大片段数")


class HierarchicalIndexBuildRequest(BaseModel):
    """分层索引构建请求"""
    document_id: str = Field(..., description="文档ID")
    force_rebuild: bool = Field(False, description="是否强制重建")

    # 配置选项
    generate_summary: bool = Field(True, description="是否生成摘要")
    extract_chapters: bool = Field(True, description="是否提取章节")
    chunk_strategy: str = Field("intelligent", description="分块策略")


class HierarchicalIndexBuildResponse(BaseModel):
    """分层索引构建响应"""
    document_id: str
    success: bool
    message: str

    # 构建结果
    summary_index: Optional[DocumentSummaryIndex] = None
    chapter_count: int = 0
    chunk_count: int = 0

    # 统计信息
    processing_time: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)
