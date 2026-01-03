"""
统一的检索相关数据模型

包括：RetrievalResult（检索结果）、RetrievalConfig（检索配置）
"""

from dataclasses import dataclass, field
from enum import Enum

class RetrievalMode(Enum):
    """检索模式"""
    VECTOR = "vector"  # 向量检索
    KEYWORD = "keyword"  # 关键词检索
    HYBRID = "hybrid"  # 混合检索
    GRAPH = "graph"  # 图谱检索
    SEMANTIC = "semantic"  # 语义检索

@dataclass
class RetrievalResult:
    """
    统一的检索结果类

    替代在10个文件中重复定义的RetrievalResult类
    """
    doc_id: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "retrieval"  # 来源标识

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalResult':
        """从字典创建"""
        return cls(
            doc_id=data.get("doc_id", ""),
            content=data.get("content", ""),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {}),
            source=data.get("source", "retrieval")
        )

@dataclass
class RetrievalConfig:
    """
    统一的检索配置类

    替代在7个文件中重复定义的RetrievalConfig类
    """
    top_k: int = 5
    threshold: float = 0.0
    mode: RetrievalMode = RetrievalMode.VECTOR
    enable_rerank: bool = False
    enable_hyde: bool = False
    enable_query_rewrite: bool = False

    # 高级选项
    filters: Dict[str, Any] = field(default_factory=dict)
    boost_params: Dict[str, float] = field(default_factory=dict)
    timeout: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "top_k": self.top_k,
            "threshold": self.threshold,
            "mode": self.mode.value,
            "enable_rerank": self.enable_rerank,
            "enable_hyde": self.enable_hyde,
            "enable_query_rewrite": self.enable_query_rewrite,
            "filters": self.filters,
            "boost_params": self.boost_params,
            "timeout": self.timeout
        }

    def validate(self) -> bool:
        """验证配置"""
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        return True

@dataclass
class RetrievalResponse:
    """检索响应"""
    query: str
    results: List[RetrievalResult]
    total_found: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """返回结果数量"""
        return len(self.results)

    def __iter__(self):
        """支持迭代"""
        return iter(self.results)

__all__ = [
    'RetrievalMode',
    'RetrievalResult',
    'RetrievalConfig',
    'RetrievalResponse',
]
