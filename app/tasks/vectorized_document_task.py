"""
向量化文档任务
已移动到 vector_tasks.py，此文件为了兼容性保留
"""

from .vector_tasks import *

# 为了向后兼容，重新导出主要函数
__all__ = [
    'vectorize_content',
    'create_document_embeddings',
    'search_similar_documents',
    'update_document_vectors'
]