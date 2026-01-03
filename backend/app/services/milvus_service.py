"""
Milvus Service兼容层
连接到新的unified_milvus_service
"""

from app.services.vectorstore.unified_milvus_service import UnifiedMilvusService

# 导出旧的类名以保持向后兼容
MilvusService = UnifiedMilvusService

__all__ = ['MilvusService', 'UnifiedMilvusService']
