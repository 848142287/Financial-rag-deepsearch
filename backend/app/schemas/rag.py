"""
RAG检索相关的Pydantic模式
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """查询请求模式"""
    query: str
    conversation_id: Optional[int] = None
    document_ids: Optional[List[int]] = None
    max_results: int = 10


class RetrievalConfig(BaseModel):
    """检索配置模式"""
    retrieval_type: str = "enhanced"  # 'simple', 'enhanced', 'deep_search'
    use_vector_search: bool = True
    use_graph_search: bool = True
    use_rerank: bool = True
    top_k: int = 10
    similarity_threshold: float = 0.7


class RetrievedDocument(BaseModel):
    """检索到的文档模式"""
    id: int
    title: str
    content: str
    score: float
    source_type: str  # 'vector', 'graph', 'hybrid'
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """查询响应模式"""
    query: str
    answer: str
    sources: List[RetrievedDocument]
    response_time_ms: int
    retrieval_config: Optional[RetrievalConfig] = None
    metadata: Optional[Dict[str, Any]] = None


# TODO: TaskStatus → core.TaskStatus
class TaskStatus(BaseModel):
    """任务状态模式"""
    task_id: str
    task_type: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: float = 0.0  # 0-100
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DeepSearchRequest(BaseModel):
    """深度搜索请求模式"""
    query: str
    max_depth: int = 3
    max_documents: int = 50
    include_related_topics: bool = True


class DocumentSuggestion(BaseModel):
    """文档建议模式"""
    id: int
    title: str
    snippet: str
    relevance_score: float