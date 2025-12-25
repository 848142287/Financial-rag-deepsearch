"""
统一API接口规范

定义整个系统中所有API的统一接口标准、响应格式和错误处理
"""

from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

# 通用响应状态
class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"

# 通用错误代码
class ErrorCode(str, Enum):
    # 通用错误
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # 业务错误
    DOCUMENT_PARSE_ERROR = "DOCUMENT_PARSE_ERROR"
    RAG_QUERY_ERROR = "RAG_QUERY_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"

    # 系统错误
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"

# 通用API响应基础模型
class BaseResponse(BaseModel):
    """API响应基础模型"""
    status: ResponseStatus = Field(..., description="响应状态")
    message: Optional[str] = Field(None, description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")

# 成功响应
class SuccessResponse(BaseResponse):
    """成功响应"""
    status: ResponseStatus = ResponseStatus.SUCCESS
    data: Optional[Any] = Field(None, description="响应数据")

# 错误响应详情
class ErrorDetail(BaseModel):
    """错误详情"""
    code: ErrorCode = Field(..., description="错误代码")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详细信息")
    field: Optional[str] = Field(None, description="相关的字段名")

# 错误响应
class ErrorResponse(BaseResponse):
    """错误响应"""
    status: ResponseStatus = ResponseStatus.ERROR
    error: ErrorDetail = Field(..., description="错误信息")

# 分页信息
class PaginationInfo(BaseModel):
    """分页信息"""
    page: int = Field(..., ge=1, description="当前页码")
    page_size: int = Field(..., ge=1, le=100, description="每页大小")
    total: int = Field(..., ge=0, description="总记录数")
    total_pages: int = Field(..., ge=0, description="总页数")
    has_next: bool = Field(..., description="是否有下一页")
    has_prev: bool = Field(..., description="是否有上一页")

# 分页响应
class PaginatedResponse(SuccessResponse):
    """分页响应"""
    pagination: PaginationInfo = Field(..., description="分页信息")

# RAG查询相关模型
class RAGQueryMode(str, Enum):
    SIMPLE = "simple"
    ENHANCED = "enhanced"
    DEEP_SEARCH = "deep_search"
    AGENTIC = "agentic"
    FUSION = "fusion"

class RAGQueryRequest(BaseModel):
    """RAG查询请求"""
    query: str = Field(..., min_length=1, max_length=1000, description="查询内容")
    mode: RAGQueryMode = Field(RAGQueryMode.ENHANCED, description="查询模式")
    conversation_id: Optional[str] = Field(None, description="对话ID")
    document_ids: Optional[List[str]] = Field(None, description="限制搜索的文档ID列表")
    max_results: int = Field(10, ge=1, le=50, description="最大结果数")
    enable_stream: bool = Field(False, description="是否启用流式响应")
    options: Optional[Dict[str, Any]] = Field(None, description="额外选项")

class DocumentSource(BaseModel):
    """文档来源"""
    document_id: str = Field(..., description="文档ID")
    title: Optional[str] = Field(None, description="文档标题")
    content: str = Field(..., description="文档内容片段")
    score: float = Field(..., ge=0, le=1, description="相关性得分")
    source: str = Field(..., description="来源类型")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

class RAGQueryResponse(SuccessResponse):
    """RAG查询响应"""
    data: Dict[str, Any] = Field(..., description="查询结果")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "查询成功",
                "timestamp": "2024-01-01T00:00:00Z",
                "data": {
                    "answer": "基于文档的回答内容",
                    "confidence_score": 0.85,
                    "sources": [
                        {
                            "document_id": "doc_123",
                            "title": "文档标题",
                            "content": "相关内容片段...",
                            "score": 0.92,
                            "source": "vector_db",
                            "metadata": {}
                        }
                    ],
                    "query_mode": "enhanced",
                    "execution_time": 1.23,
                    "ragas_metrics": {
                        "faithfulness": 0.89,
                        "answer_relevancy": 0.85,
                        "context_precision": 0.91
                    }
                }
            }
        }

# 文档处理相关模型
class DocumentProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    file_name: str = Field(..., description="文件名")
    file_size: int = Field(..., gt=0, description="文件大小")
    file_type: str = Field(..., description="文件类型")
    content_hash: Optional[str] = Field(None, description="内容哈希")
    options: Optional[Dict[str, Any]] = Field(None, description="处理选项")

class DocumentUploadResponse(SuccessResponse):
    """文档上传响应"""
    data: Dict[str, Any] = Field(..., description="上传结果")

class DocumentProcessingProgress(BaseModel):
    """文档处理进度"""
    task_id: str = Field(..., description="任务ID")
    status: DocumentProcessingStatus = Field(..., description="处理状态")
    progress: float = Field(..., ge=0, le=100, description="进度百分比")
    current_step: str = Field(..., description="当前步骤")
    estimated_time_remaining: Optional[int] = Field(None, description="预计剩余时间(秒)")
    error_message: Optional[str] = Field(None, description="错误信息")

# 健康检查模型
class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ServiceHealth(BaseModel):
    """服务健康状态"""
    name: str = Field(..., description="服务名称")
    status: HealthStatus = Field(..., description="健康状态")
    response_time: Optional[float] = Field(None, description="响应时间(ms)")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="最后检查时间")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")

class SystemHealthResponse(SuccessResponse):
    """系统健康状态响应"""
    data: Dict[str, Any] = Field(..., description="健康状态信息")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "timestamp": "2024-01-01T00:00:00Z",
                "data": {
                    "overall_status": "healthy",
                    "services": [
                        {
                            "name": "rag_service",
                            "status": "healthy",
                            "response_time": 150.5,
                            "last_check": "2024-01-01T00:00:00Z"
                        }
                    ],
                    "version": "1.0.0",
                    "uptime": 86400
                }
            }
        }

# API路由统一前缀规范
class APIPrefix:
    """API路由前缀规范"""
    V1 = "/api/v1"

    # 核心业务模块
    RAG = "/rag"
    DOCUMENTS = "/documents"
    UPLOAD = "/upload"
    DEDUPLICATION = "/deduplication"

    # 管理模块
    ADMIN = "/admin"
    HEALTH = "/health"
    MONITORING = "/monitoring"

    # 高级功能
    AGENTIC_RAG = "/agentic-rag"
    ENHANCED_RAG = "/enhanced-rag"
    KNOWLEDGE_FUSION = "/knowledge-fusion"

# 通用请求上下文
class RequestContext(BaseModel):
    """请求上下文"""
    request_id: str = Field(..., description="请求ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    tenant_id: Optional[str] = Field(None, description="租户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    client_ip: Optional[str] = Field(None, description="客户端IP")
    user_agent: Optional[str] = Field(None, description="用户代理")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="请求时间")

# 通用配置选项
class RequestOptions(BaseModel):
    """通用请求选项"""
    timeout: Optional[int] = Field(30, ge=1, le=300, description="超时时间(秒)")
    retry_count: int = Field(0, ge=0, le=3, description="重试次数")
    enable_cache: bool = Field(True, description="是否启用缓存")
    cache_ttl: Optional[int] = Field(3600, ge=1, description="缓存TTL(秒)")
    priority: int = Field(5, ge=1, le=10, description="请求优先级")