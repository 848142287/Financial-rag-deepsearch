"""
统一的核心数据模型

避免在各模块中重复定义相同的类
提供统一的数据结构，提高代码一致性

创建时间: 2026-01-03
"""

__all__ = [
    # Config
    'BaseConfig',

    # Retrieval
    'RetrievalResult',
    'RetrievalConfig',

    # Cache
    'CacheEntry',
    'CacheLevel',

    # Document
    'DocumentChunk',
    'DocumentMetadata',

    # Task
    'TaskStatus',
    'TaskResult',
]
