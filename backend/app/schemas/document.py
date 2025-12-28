"""
文档相关的Pydantic模式
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class DocumentBase(BaseModel):
    """文档基础模式"""
    title: Optional[str] = None
    filename: str


class DocumentCreate(DocumentBase):
    """文档创建模式"""
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    content_type: Optional[str] = None


class Document(DocumentBase):
    """文档响应模式"""
    id: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    content_type: Optional[str] = None
    status: str
    uploaded_by: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    chunks_count: Optional[int] = None
    entities_count: Optional[int] = None

    class Config:
        from_attributes = True


class DocumentUpdate(BaseModel):
    """文档更新模式"""
    title: Optional[str] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """文档响应模式"""
    id: int
    title: str
    filename: str
    file_size: int
    content_hash: str
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    status: str


class DocumentChunk(BaseModel):
    """文档分块模式"""
    id: str
    document_id: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    embedding_id: Optional[str] = None
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class DocumentList(BaseModel):
    """文档列表模式"""
    documents: List[Document]
    total: int
    page: int
    page_size: int
    total_pages: int


class DocumentEntity(BaseModel):
    """文档实体模式"""
    id: str
    document_id: str
    text: str
    type: str
    confidence: float
    start_char: int
    end_char: int
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


# 批量操作相关模式
class BatchUploadResponse(BaseModel):
    """批量上传响应"""
    batch_id: str = Field(..., description="批次ID")
    total_files: int = Field(..., description="总文件数")
    uploaded_files: int = Field(..., description="已上传文件数")
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    upload_results: Optional[List[Dict[str, Any]]] = Field(None, description="上传结果")


class BatchDeleteResponse(BaseModel):
    """批量删除响应"""
    batch_id: str = Field(..., description="批次ID")
    total_documents: int = Field(..., description="总文档数")
    valid_documents: int = Field(..., description="有效文档数")
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")


class DocumentListRequest(BaseModel):
    """文档列表请求"""
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=100, description="每页大小")
    status: Optional[str] = Field(None, description="状态过滤")
    file_type: Optional[str] = Field(None, description="文件类型过滤")
    search: Optional[str] = Field(None, description="搜索关键词")


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    config: Optional[Dict[str, Any]] = Field(None, description="处理配置")
    tags: Optional[List[str]] = Field(None, description="标签")


class BatchUploadRequest(BaseModel):
    """批量上传请求"""
    files: List[str] = Field(..., description="文件路径列表")
    config: Optional[Dict[str, Any]] = Field(None, description="处理配置")
    tags: Optional[List[str]] = Field(None, description="标签")