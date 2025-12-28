"""
事件类型定义
定义系统中所有的事件类型和事件结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
import uuid
import json


class EventType(Enum):
    """事件类型枚举"""
    # 系统级事件
    SYSTEM_STARTED = "system_started"
    SYSTEM_SHUTDOWN = "system_shutdown"
    HEALTH_CHECK = "health_check"
    ERROR_OCCURRED = "error_occurred"

    # 文档处理事件
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_PARSING_STARTED = "document_parsing_started"
    DOCUMENT_PARSING_COMPLETED = "document_parsing_completed"
    DOCUMENT_PARSING_FAILED = "document_parsing_failed"
    DOCUMENT_INDEXING_STARTED = "document_indexing_started"
    DOCUMENT_INDEXING_COMPLETED = "document_indexing_completed"
    DOCUMENT_UPDATED = "document_updated"
    DOCUMENT_DELETED = "document_deleted"

    # 检索事件
    SEARCH_QUERY_RECEIVED = "search_query_received"
    SEARCH_STARTED = "search_started"
    SEARCH_COMPLETED = "search_completed"
    SEARCH_FAILED = "search_failed"
    SEARCH_CACHE_HIT = "search_cache_hit"
    SEARCH_CACHE_MISS = "search_cache_miss"

    # Agentic RAG事件
    PLANNING_STARTED = "planning_started"
    PLANNING_COMPLETED = "planning_completed"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    GENERATION_STARTED = "generation_started"
    GENERATION_COMPLETED = "generation_completed"

    # DeepSearch事件
    DEEP_SEARCH_STARTED = "deep_search_started"
    DEEP_SEARCH_ITERATION_STARTED = "deep_search_iteration_started"
    DEEP_SEARCH_ITERATION_COMPLETED = "deep_search_iteration_completed"
    DEEP_SEARCH_CONVERGED = "deep_search_converged"
    DEEP_SEARCH_COMPLETED = "deep_search_completed"

    # 评估事件
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_COMPLETED = "evaluation_completed"

    # 缓存事件
    CACHE_INVALIDATED = "cache_invalidated"
    CACHE_WARMED_UP = "cache_warmed_up"

    # 异步任务事件
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"


@dataclass
class Event:
    """基础事件类"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "data": self.data,
            "metadata": self.metadata,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建事件"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=data.get("event_type", ""),
            timestamp=timestamp or datetime.now(),
            source=data.get("source", ""),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            correlation_id=data.get("correlation_id"),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


@dataclass
class SystemEvent(Event):
    """系统级事件"""
    event_type: str = ""
    component: str = ""
    level: str = "info"  # info, warning, error, critical

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.SYSTEM_STARTED.value


@dataclass
class DocumentEvent(Event):
    """文档处理事件"""
    event_type: str = ""
    document_id: str = ""
    file_name: str = ""
    file_size: int = 0
    file_type: str = ""
    processing_stage: str = ""
    status: str = ""
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    chunk_count: Optional[int] = None

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.DOCUMENT_UPLOADED.value


@dataclass
class SearchEvent(Event):
    """搜索事件"""
    event_type: str = ""
    query: str = ""
    search_type: str = ""
    strategy: Optional[str] = None
    result_count: int = 0
    execution_time: Optional[float] = None
    cache_hit: bool = False
    quality_score: Optional[float] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.SEARCH_QUERY_RECEIVED.value


@dataclass
class DeepSearchEvent(SearchEvent):
    """DeepSearch事件"""
    iteration: int = 0
    max_iterations: int = 3
    convergence_score: float = 0.0
    strategies_used: List[str] = field(default_factory=list)
    strategy_performance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.DEEP_SEARCH_STARTED.value


@dataclass
class EvaluationEvent(Event):
    """评估事件"""
    event_type: str = ""
    evaluation_type: str = ""  # ragas, custom
    metrics: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    evaluation_time: Optional[float] = None

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.EVALUATION_STARTED.value


@dataclass
class TaskEvent(Event):
    """异步任务事件"""
    event_type: str = ""
    task_id: str = ""
    task_type: str = ""
    task_status: str = ""
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    worker_id: Optional[str] = None

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.TASK_CREATED.value


# 事件创建工厂函数
def create_system_event(event_type: EventType, component: str, level: str = "info",
                        **kwargs) -> SystemEvent:
    """创建系统事件"""
    return SystemEvent(
        event_type=event_type.value,
        component=component,
        level=level,
        **kwargs
    )


def create_document_event(event_type: EventType, document_id: str, **kwargs) -> DocumentEvent:
    """创建文档事件"""
    return DocumentEvent(
        event_type=event_type.value,
        document_id=document_id,
        **kwargs
    )


def create_search_event(event_type: EventType, query: str, **kwargs) -> SearchEvent:
    """创建搜索事件"""
    return SearchEvent(
        event_type=event_type.value,
        query=query,
        **kwargs
    )


def create_deep_search_event(event_type: EventType, query: str, **kwargs) -> DeepSearchEvent:
    """创建DeepSearch事件"""
    return DeepSearchEvent(
        event_type=event_type.value,
        query=query,
        **kwargs
    )


def create_evaluation_event(event_type: EventType, evaluation_type: str, **kwargs) -> EvaluationEvent:
    """创建评估事件"""
    return EvaluationEvent(
        event_type=event_type.value,
        evaluation_type=evaluation_type,
        **kwargs
    )


def create_task_event(event_type: EventType, task_id: str, task_type: str, **kwargs) -> TaskEvent:
    """创建任务事件"""
    return TaskEvent(
        event_type=event_type.value,
        task_id=task_id,
        task_type=task_type,
        **kwargs
    )