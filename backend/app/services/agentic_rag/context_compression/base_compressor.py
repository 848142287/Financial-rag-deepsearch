"""
上下文压缩器基础类
定义压缩器接口和通用数据结构
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Document:
    """文档数据类"""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Document(content_length={len(self.page_content)}, metadata={self.metadata.keys()})"


@dataclass
class CompressionResult:
    """压缩结果"""
    compressed_docs: List[Document]
    original_count: int
    compressed_count: int
    compression_ratio: float
    tokens_saved: int
    compression_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"CompressionResult("
            f"{self.compressed_count}/{self.original_count} docs, "
            f"ratio={self.compression_ratio:.2%}, "
            f"tokens_saved={self.tokens_saved}, "
            f"time={self.compression_time:.2f}s)"
        )


class BaseCompressor(ABC):
    """上下文压缩器基类"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化压缩器

        Args:
            config: 压缩器配置
        """
        self.config = config or {}
        self.compression_stats = {
            "total_compressions": 0,
            "total_documents_processed": 0,
            "total_tokens_saved": 0,
            "total_time": 0.0
        }

    @abstractmethod
    async def compress(
        self,
        query: str,
        documents: List[Document],
        **kwargs
    ) -> CompressionResult:
        """
        压缩文档列表

        Args:
            query: 用户查询
            documents: 检索到的文档列表
            **kwargs: 其他参数

        Returns:
            CompressionResult: 压缩结果
        """
        pass

    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量

        中文: 约1.5字符=1token
        英文: 约4字符=1token

        Args:
            text: 输入文本

        Returns:
            估算的token数量
        """
        # 简单估算：中文字符 + 英文单词数
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_words = len(text.replace('\n', ' ').split())
        return int(chinese_chars / 1.5 + english_words)

    def _estimate_tokens_saved(
        self,
        original_docs: List[Document],
        compressed_docs: List[Document]
    ) -> int:
        """
        估算节省的token数量

        Args:
            original_docs: 原始文档列表
            compressed_docs: 压缩后的文档列表

        Returns:
            节省的token数量
        """
        original_tokens = sum(
            self._estimate_tokens(doc.page_content)
            for doc in original_docs
        )
        compressed_tokens = sum(
            self._estimate_tokens(doc.page_content)
            for doc in compressed_docs
        )
        return original_tokens - compressed_tokens

    def get_stats(self) -> Dict[str, Any]:
        """
        获取压缩统计信息

        Returns:
            统计信息字典
        """
        stats = self.compression_stats.copy()
        if stats["total_compressions"] > 0:
            stats["avg_compression_ratio"] = (
                stats["total_tokens_saved"] /
                stats["total_documents_processed"]
                if stats["total_documents_processed"] > 0
                else 0
            )
            stats["avg_time"] = (
                stats["total_time"] / stats["total_compressions"]
            )
        return stats

    def reset_stats(self):
        """重置统计信息"""
        self.compression_stats = {
            "total_compressions": 0,
            "total_documents_processed": 0,
            "total_tokens_saved": 0,
            "total_time": 0.0
        }
