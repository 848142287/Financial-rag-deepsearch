"""
数据模型导入
"""

# from app.models.chapter import Chapter, ChapterTableOfContents, ChapterCrossReference  # 暂时注释，表不存在
# from app.models.content import (  # 暂时注释，表不存在
#     ImageContent, ChartContent, TableContent, FormulaContent, ContentRelationship
# )
# from app.models.knowledge_graph import (  # 暂时注释，表结构与模型不匹配
#     KnowledgeGraphNode, KnowledgeGraphRelation, KnowledgeGraphPath,
#     KnowledgeGraphEntity, KnowledgeGraphCluster
# )

__all__ = [
    "User",
    "Document",
    "DocumentChunk",
    "DocumentStorageIndex",
    "VectorStorage",
    "DocumentTask",
    # "Chapter",  # 暂时注释，表不存在
    # "ChapterTableOfContents",  # 暂时注释，表不存在
    # "ChapterCrossReference",  # 暂时注释，表不存在
    # "ImageContent",  # 暂时注释，表不存在
    # "ChartContent",  # 暂时注释，表不存在
    # "TableContent",  # 暂时注释，表不存在
    # "FormulaContent",  # 暂时注释，表不存在
    # "ContentRelationship",  # 暂时注释，表不存在
    # "KnowledgeGraphNode",  # 暂时注释，表结构与模型不匹配
    # "KnowledgeGraphRelation",  # 暂时注释，表结构与模型不匹配
    # "KnowledgeGraphPath",  # 暂时注释，表结构与模型不匹配
    # "KnowledgeGraphEntity",  # 暂时注释，表结构与模型不匹配
    # "KnowledgeGraphCluster",  # 暂时注释，表结构与模型不匹配
    "RetrievalLog",
    "EvaluationResult",
    "EvaluationMetric",
    "EvaluationSession",
    "DocumentSync",
    "VectorSync",
    "GraphSync",
    "EntityLink",
    "SyncConfiguration",
    "SyncLog",
    "Conversation",
    "Message",
    "RetrievalEvalTask",
    "RetrievalEvalQuestion",
    "RetrievalEvalStatus",
    "RetrievalEvalQuestionStatus",
]
