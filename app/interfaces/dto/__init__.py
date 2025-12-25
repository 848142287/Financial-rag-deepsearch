"""
Data Transfer Objects (DTOs)

Defines data structures for transferring data between layers.
DTOs are simple data containers without business logic.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueryType(str, Enum):
    """Query type enumeration"""
    SIMPLE = "simple"
    COMPLEX = "complex"
    AGENTIC = "agentic"
    STREAMING = "streaming"


class ContentType(str, Enum):
    """Content type enumeration"""
    PDF = "pdf"
    WORD = "docx"
    EXCEL = "xlsx"
    POWERPOINT = "pptx"
    TEXT = "txt"
    HTML = "html"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class PaginationDTO:
    """Pagination information"""
    page: int = 1
    per_page: int = 10
    total: Optional[int] = None
    total_pages: Optional[int] = None
    has_next: Optional[bool] = None
    has_prev: Optional[bool] = None

    def __post_init__(self):
        if self.page < 1:
            self.page = 1
        if self.per_page < 1:
            self.per_page = 10
        if self.total is not None:
            self.total_pages = (self.total + self.per_page - 1) // self.per_page
            self.has_next = self.page < self.total_pages
            self.has_prev = self.page > 1


@dataclass
class DocumentDTO:
    """Document data transfer object"""
    id: Optional[int] = None
    file_id: Optional[str] = None
    title: str = ""
    file_name: str = ""
    content_type: ContentType = ContentType.PDF
    file_size: int = 0
    status: str = "uploading"
    processing_progress: float = 0.0
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    chunk_count: int = 0
    embedding_count: int = 0
    quality_score: Optional[float] = None
    is_confidential: bool = False
    tags: List[str] = field(default_factory=list)
    company_symbols: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "file_id": self.file_id,
            "title": self.title,
            "file_name": self.file_name,
            "content_type": self.content_type.value,
            "file_size": self.file_size,
            "status": self.status,
            "processing_progress": self.processing_progress,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "chunk_count": self.chunk_count,
            "embedding_count": self.embedding_count,
            "quality_score": self.quality_score,
            "is_confidential": self.is_confidential,
            "tags": self.tags,
            "company_symbols": self.company_symbols,
            "metadata": self.metadata,
            "created_by": self.created_by
        }


@dataclass
class QueryDTO:
    """Query data transfer object"""
    id: Optional[int] = None
    query_text: str = ""
    query_type: QueryType = QueryType.SIMPLE
    response: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None
    execution_time_ms: Optional[int] = None
    document_count: int = 0
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "query_text": self.query_text,
            "query_type": self.query_type.value,
            "response": self.response,
            "sources": self.sources,
            "confidence": self.confidence,
            "execution_time_ms": self.execution_time_ms,
            "document_count": self.document_count,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata
        }


@dataclass
class TaskDTO:
    """Task data transfer object"""
    id: Optional[str] = None
    task_type: str = ""
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    user_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "logs": self.logs
        }


@dataclass
class UserDTO:
    """User data transfer object"""
    id: Optional[int] = None
    email: str = ""
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False
    last_login: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username or self.email

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "preferences": self.preferences,
            "metadata": self.metadata
        }


@dataclass
class SearchResultDTO:
    """Search result data transfer object"""
    documents: List[DocumentDTO] = field(default_factory=list)
    total_count: int = 0
    query: str = ""
    execution_time_ms: Optional[int] = None
    facets: Optional[Dict[str, Any]] = None
    suggestions: List[str] = field(default_factory=list)
    pagination: Optional[PaginationDTO] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "total_count": self.total_count,
            "query": self.query,
            "execution_time_ms": self.execution_time_ms,
            "facets": self.facets,
            "suggestions": self.suggestions,
            "pagination": self.pagination.__dict__ if self.pagination else None
        }


@dataclass
class ProcessingStatusDTO:
    """Processing status data transfer object"""
    document_id: int
    status: str
    progress: float = 0.0
    current_step: Optional[str] = None
    steps_completed: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    estimated_time_remaining: Optional[int] = None
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "document_id": self.document_id,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "steps_completed": self.steps_completed,
            "error_message": self.error_message,
            "estimated_time_remaining": self.estimated_time_remaining,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class EmbeddingDTO:
    """Embedding data transfer object"""
    id: Optional[str] = None
    vector: List[float] = field(default_factory=list)
    model_name: str = ""
    dimension: int = 0
    document_id: Optional[int] = None
    chunk_index: Optional[int] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "created_at": self.created_at.isoformat() if self.created_at else None
            # Note: Vector is not included for size reasons
        }


@dataclass
class FileUploadDTO:
    """File upload data transfer object"""
    file_name: str = ""
    file_size: int = 0
    content_type: str = ""
    upload_url: Optional[str] = None
    upload_id: Optional[str] = None
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "file_name": self.file_name,
            "file_size": self.file_size,
            "content_type": self.content_type,
            "upload_url": self.upload_url,
            "upload_id": self.upload_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class NotificationDTO:
    """Notification data transfer object"""
    id: Optional[int] = None
    user_id: int
    message: str
    notification_type: str = "info"
    title: Optional[str] = None
    is_read: bool = False
    created_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    action_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "message": self.message,
            "notification_type": self.notification_type,
            "title": self.title,
            "is_read": self.is_read,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "metadata": self.metadata,
            "action_url": self.action_url
        }


@dataclass
class ActivityLogDTO:
    """Activity log data transfer object"""
    id: Optional[int] = None
    user_id: int
    activity_type: str
    description: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "activity_type": self.activity_type,
            "description": self.description,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class SystemStatsDTO:
    """System statistics data transfer object"""
    total_documents: int = 0
    total_users: int = 0
    total_queries: int = 0
    storage_used_gb: float = 0.0
    active_tasks: int = 0
    cache_hit_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    uptime_seconds: int = 0
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_documents": self.total_documents,
            "total_users": self.total_users,
            "total_queries": self.total_queries,
            "storage_used_gb": self.storage_used_gb,
            "active_tasks": self.active_tasks,
            "cache_hit_rate": self.cache_hit_rate,
            "avg_response_time_ms": self.avg_response_time_ms,
            "uptime_seconds": self.uptime_seconds,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }


@dataclass
class ErrorDTO:
    """Error data transfer object"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    request_id: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "request_id": self.request_id,
            "stack_trace": self.stack_trace
        }