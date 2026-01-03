"""
Neo4j Service兼容层
连接到新的knowledge_graph服务
"""

# 尝试从knowledge_graph导入
try:
    from app.services.knowledge_graph.neo4j_service import Neo4jService as _Neo4jService
    Neo4jService = _Neo4jService
except ImportError:
    # 如果不存在，创建一个占位符
    class Neo4jService:
        """Neo4j服务占位符"""
        def __init__(self, *args, **kwargs):
            pass

__all__ = ['Neo4jService']
