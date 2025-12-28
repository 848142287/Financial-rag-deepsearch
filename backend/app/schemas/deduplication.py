"""
文档去重相关的Pydantic模式
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DuplicateSource(BaseModel):
    """重复来源信息"""
    source: str = Field(..., description="数据源名称")
    similarity: float = Field(..., description="相似度分数")
    details: Dict[str, Any] = Field(default_factory=dict, description="详细信息")

class DocumentDuplicateCheck(BaseModel):
    """文档重复检查请求"""
    file_hash: str = Field(..., description="文件哈希")
    content_hash: str = Field(..., description="内容哈希")
    title: Optional[str] = Field(None, description="文档标题")
    size: Optional[int] = Field(None, description="文件大小（字节）")

class DocumentDuplicateResponse(BaseModel):
    """文档重复检查响应"""
    is_duplicate: bool = Field(..., description="是否重复")
    similarity_score: float = Field(..., description="最高相似度分数")
    duplicate_sources: List[DuplicateSource] = Field(default_factory=list, description="重复来源列表")
    existing_document_id: Optional[int] = Field(None, description="已存在的文档ID")
    existing_document_info: Optional[Dict[str, Any]] = Field(None, description="已存在的文档信息")
    recommendations: List[str] = Field(default_factory=list, description="处理建议")
    check_time: datetime = Field(default_factory=datetime.now, description="检查时间")

class UploadDeduplicationDecision(BaseModel):
    """上传去重决策"""
    should_block: bool = Field(..., description="是否阻止上传")
    message: str = Field(..., description="提示信息")
    severity: str = Field(default="info", description="严重程度: info/warning/error")
    allow_override: bool = Field(default=True, description="是否允许用户覆盖")
    existing_document_id: Optional[int] = Field(None, description="相关文档ID")

class DocumentUploadRequest(BaseModel):
    """文档上传请求（增强版）"""
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    skip_deduplication: bool = Field(default=False, description="跳过去重检查")
    force_upload: bool = Field(default=False, description="强制上传")

class EnhancedUploadResponse(BaseModel):
    """增强的上传响应"""
    document_id: int
    message: str
    deduplication_check: Optional[DocumentDuplicateResponse] = None
    processing_status: str = Field(default="pending")
    estimated_processing_time: Optional[int] = None
    warnings: List[str] = Field(default_factory=list)