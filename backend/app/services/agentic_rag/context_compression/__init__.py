"""
上下文压缩模块

用于在Agentic RAG多级检索中压缩检索结果，减少上下文长度，提升生成质量

三层压缩策略:
- L1: EmbeddingsCompressor - 基于嵌入相似度的快速过滤
- L2: BGE Reranker - 重排序（已有，集成）
- L3: FinancialContextCompressor - LLM精细提取
"""

from .base_compressor import BaseCompressor, CompressionResult, Document
from .hierarchical_compressor import HierarchicalContextCompressor, get_compressor

# 可选导入 - sentence-transformers
try:
    from .embeddings_compressor import EmbeddingsCompressor
    _embeddings_available = True
except ImportError:
    _embeddings_available = False
    EmbeddingsCompressor = None

# 可选导入 - langchain
try:
    from .llm_extractor import FinancialContextCompressor
    _llm_compressor_available = True
except ImportError:
    _llm_compressor_available = False
    FinancialContextCompressor = None

__all__ = [
    "BaseCompressor",
    "CompressionResult",
    "Document",
    "HierarchicalContextCompressor",
    "get_compressor",
    "EmbeddingsCompressor",
    "FinancialContextCompressor",
    "_embeddings_available",
    "_llm_compressor_available"
]
